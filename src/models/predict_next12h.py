"""
12-hour ahead spend forecasting for ad performance prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional

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
            campaign = row.name if hasattr(row, 'name') else f"campaign_{i}"
            if isinstance(campaign, str) and campaign in current_data['campaign_name'].values:
                campaign = current_data.loc[current_data.index[i], 'campaign_name']

            campaign_predictions[campaign] = float(predictions[i])

        return campaign_predictions

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
            campaign = row.name if hasattr(row, 'name') else f"campaign_{i}"
            if isinstance(campaign, str) and campaign in current_data['campaign_name'].values:
                campaign = current_data.loc[current_data.index[i], 'campaign_name']

            pred_values = bootstrap_predictions[:, i]
            results[campaign] = {
                'prediction': float(predictions[i]),
                'lower_95': float(np.percentile(pred_values, 2.5)),
                'upper_95': float(np.percentile(pred_values, 97.5)),
                'std': float(np.std(pred_values))
            }

        return results

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
) -> Dict[str, float]:
    """Convenience function to predict next 12h spend."""
    predictor = SpendPredictor(model_path)

    # Use config defaults if not provided
    data_path = data_path or config.get("data.processed_dir", "data/processed") + "/processed_dataset.csv"

    # Load latest data (use processed dataset with all features)
    preparer = DatasetPreparer()
    df = preparer.load_data(data_path)

    # Get most recent data for each campaign (last 24 hours)
    latest_df = df.groupby('campaign_name').tail(24).copy()

    # Make predictions (data already has all features from training)
    predictions = predictor.predict_next_12h(latest_df)

    # Save predictions
    predictor.save_predictions(predictions, output_path)

    return predictions


if __name__ == "__main__":
    predictions = predict_next_12h()
    print("12-hour spend predictions:")
    for campaign, pred in predictions.items():
        print(f"{campaign}: ${pred:.2f}")
