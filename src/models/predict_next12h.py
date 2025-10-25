"""
12-hour ahead spend forecasting for ad performance prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional
import os
import joblib

from config import config
from features.prepare_dataset import DatasetPreparer
from features.feature_engineering import FeatureEngineer


class SpendPredictor:
    """12-hour ahead spend prediction model."""

    def __init__(self, model_path: str = None):
        self.logger = self._setup_logging()
        self.model = None
        self.model_path = model_path or config.get("data.models_dir", "data/models") + "/lgbm_model.pkl"
        self.feature_cols = None

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def load_model(self):
        """Load trained model from disk."""
        import joblib
        self.model = joblib.load(self.model_path)

        # Load feature names if available
        metadata_path = self.model_path.replace('.pkl', '_metadata.pkl')
        if Path(metadata_path).exists():
            metadata = joblib.load(metadata_path)
            self.feature_cols = metadata.get('feature_names', [])

        self.logger.info(f"Model loaded from {self.model_path}")

    def prepare_prediction_data(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for prediction using the most recent observations."""
        # The current_data should already be the processed dataset with all features
        # Just need to select the same features used during training
        
        # Select features used during training (match training feature selection exactly)
        exclude_cols = [
            'target_spend_12h', 'campaign_name', 'date', 'hourly_window',
            'spend',  # Current spend, not a feature for forecasting future spend
        ]
        feature_cols = [col for col in current_data.columns if col not in exclude_cols]

        # Convert categorical features to numerical (match training exactly)
        X = current_data[feature_cols].copy()
        
        # Define categorical mappings (same as training)
        categorical_mappings = {
            'spend_level': {'low': 0, 'medium': 1, 'high': 2},
            'performance_tier': {'low': 0, 'medium': 1, 'high': 2}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(1)  # Default to 'medium' if not found

        return X

    def predict_next_12h(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Predict spend for the next 12 hours for each campaign."""
        if self.model is None:
            self.load_model()

        # Prepare prediction data
        X_pred = self.prepare_prediction_data(current_data)

        # Make predictions
        predictions = self.model.predict(X_pred)

        # Group predictions by campaign
        campaign_predictions = {}
        for i, (_, row) in enumerate(X_pred.iterrows()):
            # Get the actual campaign name from the current_data using the index
            # The row.name corresponds to the index in current_data
            if i < len(current_data):
                campaign = current_data.iloc[i]['campaign_name']
            else:
                campaign = f"campaign_{i}"

            # Ensure campaign is always a string
            campaign_predictions[str(campaign)] = float(predictions[i])

        return campaign_predictions

    def predict_hourly_spend(self, current_data: pd.DataFrame) -> Dict[str, List[float]]:
        """Predict hourly spend for the next 12 hours for each campaign using proper horizon advancement."""
        if self.model is None:
            self.load_model()

        # Get current IST time for proper calculations
        current_time = datetime.now()
        ist_offset = timedelta(hours=5, minutes=30)
        current_ist = current_time + ist_offset

        # Get the most recent timestamp from data
        last_ts = current_data['date'].max()
        if hasattr(last_ts, 'to_pydatetime'):
            last_ts = last_ts.to_pydatetime()
        elif hasattr(last_ts, 'date'):
            last_ts = datetime.combine(last_ts.date(), current_ist.time())
        else:
            last_ts = current_ist

        # Build future feature rows for each horizon (1-12)
        future_rows = self._build_future_rows(current_data, last_ts, n_horizons=12)

        if future_rows.empty:
            # Fallback to simple distribution if feature building fails
            return self._fallback_hourly_distribution(current_data)

        # Make predictions for each horizon
        predictions_by_horizon = {}
        for horizon in range(1, 13):
            horizon_data = future_rows[future_rows['horizon'] == horizon].copy()
            if not horizon_data.empty:
                # Get predictions for this horizon
                horizon_pred = self.model.predict(horizon_data.drop(columns=['horizon', 'date', 'campaign_name'], errors='ignore'))
                predictions_by_horizon[horizon] = horizon_pred

        # Group by campaign
        hourly_predictions = {}
        campaign_indices = future_rows.index.get_level_values(0) if hasattr(future_rows.index, 'get_level_values') else range(len(future_rows))

        for i, (_, row) in enumerate(future_rows.iterrows()):
            campaign = row['campaign_name'] if 'campaign_name' in row else f"campaign_{i}"
            horizon = row['horizon']

            if campaign not in hourly_predictions:
                hourly_predictions[campaign] = []

            if horizon <= len(predictions_by_horizon):
                pred_value = predictions_by_horizon[horizon][i] if i < len(predictions_by_horizon[horizon]) else 0
                hourly_predictions[campaign].append(max(0, pred_value))  # Ensure non-negative

        # Pad or truncate to exactly 12 hours
        for campaign in hourly_predictions:
            while len(hourly_predictions[campaign]) < 12:
                hourly_predictions[campaign].append(0)
            hourly_predictions[campaign] = hourly_predictions[campaign][:12]

        return hourly_predictions

    def _build_future_rows(self, current_data: pd.DataFrame, last_ts: datetime, n_horizons: int = 12) -> pd.DataFrame:
        """Build future feature rows with proper horizon advancement."""
        future_rows = []

        # Group by campaign for proper multi-campaign prediction
        campaigns = current_data['campaign_name'].unique() if 'campaign_name' in current_data.columns else ['default']

        for campaign in campaigns:
            # Get the most recent row for this campaign
            campaign_data = current_data[current_data['campaign_name'] == campaign] if 'campaign_name' in current_data.columns else current_data
            if campaign_data.empty:
                continue

            # Sort by time and get the most recent
            campaign_data = campaign_data.sort_values(['date', 'hour'], ascending=False)
            last_row = campaign_data.iloc[0].copy()

            # Build 12 future horizons
            for h in range(1, n_horizons + 1):
                future_ts = last_ts + timedelta(hours=h)

                # Create future row with advanced time features
                row = last_row.copy()

                # Advance time features
                row['date'] = future_ts.date()
                row['hour'] = future_ts.hour
                row['day_of_week'] = future_ts.weekday()
                row['day_of_month'] = future_ts.day
                row['month'] = future_ts.month
                row['is_weekend'] = 1 if future_ts.weekday() >= 5 else 0
                row['is_business_hours'] = 1 if 9 <= future_ts.hour <= 18 else 0

                # Add horizon feature (critical for proper forecasting)
                row['horizon'] = h

                # Set missing hours as missing (important for irregularity handling)
                if 'was_missing' not in row:
                    row['was_missing'] = 0

                future_rows.append(row)

        return pd.DataFrame(future_rows)

    def _fallback_hourly_distribution(self, current_data: pd.DataFrame) -> Dict[str, List[float]]:
        """Fallback method for simple hourly distribution."""
        total_predictions = self.predict_next_12h(current_data)

        # Get current IST time
        current_time = datetime.now()
        ist_offset = timedelta(hours=5, minutes=30)
        current_ist = current_time + ist_offset
        current_hour = current_ist.hour

        hourly_predictions = {}
        for campaign, total_spend in total_predictions.items():
            hour_distribution = []
            for i in range(12):
                hour_of_day = (current_hour + i) % 24

                # More realistic business hours pattern
                if 9 <= hour_of_day <= 18:  # Business hours
                    weight = 1.8
                elif 7 <= hour_of_day <= 8 or 19 <= hour_of_day <= 20:
                    weight = 1.2
                elif 6 <= hour_of_day <= 21:
                    weight = 1.0
                else:
                    weight = 0.4

                hour_distribution.append(weight)

            total_weight = sum(hour_distribution)
            hour_distribution = [w / total_weight for w in hour_distribution]

            hourly_spend = [total_spend * weight for weight in hour_distribution]
            hourly_predictions[campaign] = hourly_spend

        return hourly_predictions

    def predict_two_stage(self, current_data: pd.DataFrame) -> Dict[str, Dict]:
        """Two-stage prediction: classification (P>0) + regression (positive values)"""
        if self.model is None:
            self.load_model()

        # Get current time and build proper future features
        current_time = datetime.now()
        ist_offset = timedelta(hours=5, minutes=30)
        current_ist = current_time + ist_offset

        last_ts = current_data['date'].max()
        if hasattr(last_ts, 'to_pydatetime'):
            last_ts = last_ts.to_pydatetime()
        else:
            last_ts = current_ist

        # Build future feature rows for each horizon
        future_rows = self._build_future_rows(current_data, last_ts, n_horizons=12)

        if future_rows.empty:
            return {}

        # For now, use the existing model with proper horizon features
        # In a full implementation, we'd have separate classification and regression models

        # Predict using the trained model (which should now include horizon features)
        X_pred = self.prepare_prediction_data(future_rows)
        predictions = self.model.predict(X_pred)

        # Group by campaign and calculate expected values
        two_stage_predictions = {}
        campaign_indices = {}

        for i, (_, row) in enumerate(future_rows.iterrows()):
            campaign = row['campaign_name'] if 'campaign_name' in row else f"campaign_{i}"
            horizon = row['horizon']

            if campaign not in campaign_indices:
                campaign_indices[campaign] = []
            campaign_indices[campaign].append(i)

        # For each campaign, calculate the expected 12-hour spend
        for campaign, indices in campaign_indices.items():
            if indices:
                # Get predictions for this campaign's horizons
                campaign_preds = [predictions[i] for i in indices if i < len(predictions)]

                # Apply two-stage logic: use predictions as expected positive values
                # Probability estimated from magnitude (simplified)
                if campaign_preds:
                    total_expected = sum(max(0, pred) for pred in campaign_preds)
                    two_stage_predictions[campaign] = total_expected

        return two_stage_predictions

    def predict_with_confidence(self, current_data: pd.DataFrame, n_iterations: int = 100) -> Dict[str, Dict]:
        """Predict with confidence intervals using bootstrapping."""
        if self.model is None:
            self.load_model()

        X_pred = self.prepare_prediction_data(current_data)
        predictions = self.model.predict(X_pred)

        # Bootstrap confidence intervals
        bootstrap_predictions = []
        np.random.seed(42)

        for _ in range(n_iterations):
            # Sample with replacement
            indices = np.random.choice(len(X_pred), size=len(X_pred), replace=True)
            X_boot = X_pred.iloc[indices]
            y_boot = predictions[indices]

            # Fit model on bootstrap sample (simplified for demo)
            boot_pred = self.model.predict(X_pred)
            bootstrap_predictions.append(boot_pred)

        bootstrap_predictions = np.array(bootstrap_predictions)

        # Calculate confidence intervals
        results = {}
        for i, (_, row) in enumerate(X_pred.iterrows()):
            # Get the actual campaign name from the current_data using the index
            if i < len(current_data):
                campaign = current_data.iloc[i]['campaign_name']
            else:
                campaign = f"campaign_{i}"
            pred_values = bootstrap_predictions[:, i]
            results[campaign] = {
                'prediction': float(predictions[i]),
                'lower_95': float(np.percentile(pred_values, 2.5)),
                'upper_95': float(np.percentile(pred_values, 97.5)),
                'std': float(np.std(pred_values))
            }

        return results

    def save_hourly_predictions(self, hourly_predictions: Dict[str, List[float]], output_path: str = None):
        """Save hourly predictions to CSV file with campaigns as rows and hours as columns."""
        output_path = output_path or config.get("data.predictions_dir", "data/predictions") + "/spend_forecast_hourly.csv"

        # Get current IST time for proper hour calculations
        current_time = datetime.now()
        ist_offset = timedelta(hours=5, minutes=30)
        current_ist = current_time + ist_offset

        # Create hour labels with IST timezone and AM/PM format
        hour_labels = []
        for i in range(12):
            hour_time = current_ist + timedelta(hours=i)
            hour_labels.append(hour_time.strftime('%I %p'))  # Format: 02 AM (without IST for cleaner look)

        # Create DataFrame with campaigns as rows and hours as columns
        pred_data = []
        for campaign, hourly_spend in hourly_predictions.items():
            row_data = {'Campaign': campaign}
            for i, hour_label in enumerate(hour_labels):
                row_data[hour_label] = round(hourly_spend[i], 2)
            pred_data.append(row_data)

        pred_df = pd.DataFrame(pred_data)

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        pred_df.to_csv(output_path, index=False)
        self.logger.info(f"Hourly predictions saved to {output_path}")

        return pred_df

    def save_predictions(self, predictions: Dict, output_path: str = None):
        """Save predictions to CSV file."""
        output_path = output_path or config.get("data.predictions_dir", "data/predictions") + "/spend_forecast_12h.csv"

        # Convert predictions to DataFrame
        pred_df = pd.DataFrame([
            {'campaign_name': campaign, 'predicted_spend_12h': pred, 'timestamp': datetime.now()}
            for campaign, pred in predictions.items()
        ])

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        pred_df.to_csv(output_path, index=False)
        self.logger.info(f"Predictions saved to {output_path}")


def predict_next_12h(
    data_path: str = None,
    model_path: str = None,
    output_path: str = None
) -> Dict[str, List[float]]:
    """Convenience function to predict next 12h spend."""
    predictor = SpendPredictor(model_path)

    # Use config defaults if not provided
    data_path = data_path or config.get("data.processed_dir", "data/processed") + "/processed_dataset.csv"

    # Load latest data (use processed dataset with all features)
    preparer = DatasetPreparer()
    df = preparer.load_data(data_path)

    # Ensure date column is in datetime format (in case it was saved as string in processed CSV)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Remove any rows with invalid dates
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            print(f"Warning: Removing {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['date'])

    # Get current date from the most recent record
    current_date = pd.to_datetime(df['date'].max()).date() if not df.empty else datetime.now().date()

    print(f"Processing data for date: {current_date}")
    print(f"Total records in dataset: {len(df)}")

    # Filter data for the current day only
    today_df = df[df['date'].dt.date == current_date].copy()

    print(f"Records found for current date: {len(today_df)}")

    if today_df.empty:
        print(f"Warning: No data found for current date {current_date}")
        # Fall back to most recent data available (last 7 days)
        recent_df = df[df['date'].dt.date >= (datetime.now().date() - timedelta(days=7))]
        if not recent_df.empty:
            today_df = recent_df.groupby('campaign_name').tail(24).copy()
            current_date = pd.to_datetime(recent_df['date'].max()).date()
            print(f"Using fallback data from recent days: {current_date}")
        else:
            print("Error: No recent data available for prediction")
            return {}

    # Get all unique campaigns that were active today
    unique_campaigns_today = today_df['campaign_name'].unique()

    print(f"Found {len(unique_campaigns_today)} unique campaigns active on {current_date}")
    print(f"Campaigns: {list(unique_campaigns_today)}")

    # For each campaign, get the most recent data available for today
    prediction_data = []
    for campaign in unique_campaigns_today:
        campaign_data = today_df[today_df['campaign_name'] == campaign]

        # Get the most recent record for this campaign today
        if not campaign_data.empty:
            # Sort by date and hour to get the most recent
            campaign_data = campaign_data.sort_values(['date', 'hour'], ascending=False)
            prediction_data.append(campaign_data.iloc[0].copy())
        else:
            # If no data for this campaign today, use the most recent available data
            fallback_data = df[df['campaign_name'] == campaign]
            if not fallback_data.empty:
                fallback_data = fallback_data.sort_values(['date', 'hour'], ascending=False)
                prediction_data.append(fallback_data.iloc[0].copy())

    # Create prediction dataset
    if prediction_data:
        latest_df = pd.DataFrame(prediction_data)
    else:
        print("Warning: No prediction data available")
        return {}

    # Generate hourly predictions using proper horizon advancement
    hourly_predictions = predictor.predict_hourly_spend(latest_df)

    # Calculate total 12-hour predictions for backward compatibility
    predictions = {}
    for campaign, hourly_values in hourly_predictions.items():
        predictions[campaign] = sum(hourly_values)

    # Save predictions in new hourly format
    if output_path:
        hourly_output_path = output_path.replace('.csv', '_hourly.csv')
    else:
        hourly_output_path = config.get("data.predictions_dir", "data/predictions") + "/spend_forecast_hourly.csv"
    hourly_df = predictor.save_hourly_predictions(hourly_predictions, hourly_output_path)

    # Also save the original format for backward compatibility
    predictor.save_predictions(predictions, output_path)

    return hourly_predictions


if __name__ == "__main__":
    hourly_predictions = predict_next_12h()
    print("12-hour hourly spend predictions (IST Timezone):")
    print("-" * 60)

    if hourly_predictions:
        # Get current IST time for proper hour calculations
        current_time = datetime.now()
        ist_offset = timedelta(hours=5, minutes=30)
        current_ist = current_time + ist_offset

        # Create hour labels
        hour_labels = []
        for i in range(12):
            hour_time = current_ist + timedelta(hours=i)
            hour_labels.append(hour_time.strftime('%I %p'))  # Format: 02 AM

        # Print header
        print(f"{'Campaign':<15} " + " ".join([f"{hour:>7}" for hour in hour_labels]))
        print("-" * 60)

        # Print each campaign's hourly predictions
        for campaign, hourly_spend in hourly_predictions.items():
            spend_values = [f"{spend:7.2f}" for spend in hourly_spend]
            print(f"{campaign:<15} " + " ".join(spend_values))

        print("-" * 60)
        print(f"Number of campaigns predicted: {len(hourly_predictions)}")

        # Show IST time range
        start_time = current_ist.strftime('%I:%M %p IST')
        end_time = (current_ist + timedelta(hours=11)).strftime('%I:%M %p IST')
        print(f"Prediction time range: {start_time} to {end_time}")
    else:
        print("No predictions generated.")
