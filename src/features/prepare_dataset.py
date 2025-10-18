"""
Dataset preparation and cleaning for ad performance forecasting
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import logging
from pathlib import Path

from config import config


class DatasetPreparer:
    """Handles data cleaning, validation, and preparation for ML modeling."""

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the dataset."""
        # Check for required columns
        required_cols = ['campaign_name', 'date', 'hourly_window', 'spend']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Remove rows with invalid dates
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            self.logger.warning(f"Removing {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['date'])

        # Remove rows with negative spend (invalid data)
        negative_spend = (df['spend'] < 0).sum()
        if negative_spend > 0:
            self.logger.warning(f"Removing {negative_spend} rows with negative spend")
            df = df[df['spend'] >= 0]

        # Fill missing numeric values with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                self.logger.warning(f"Filling {df[col].isna().sum()} missing values in {col} with 0")
                df[col] = df[col].fillna(0)

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df['hour'] = df['hourly_window'].str.split(':').str[0].astype(int)
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day

        # Create time-based features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)

        return df

    def add_campaign_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add campaign-level aggregated features."""
        # Campaign historical performance (rolling windows)
        campaign_stats = df.groupby('campaign_name').agg({
            'spend': ['mean', 'std', 'sum', 'count'],
            'impressions': ['mean', 'sum'],
            'clicks': ['mean', 'sum'],
            'ctr': ['mean'],
            'cpc': ['mean'],
        }).round(4)

        # Flatten column names
        campaign_stats.columns = ['_'.join(col).strip() for col in campaign_stats.columns.values]
        campaign_stats = campaign_stats.reset_index()

        # Merge back to original dataframe
        df = df.merge(campaign_stats, on='campaign_name', how='left')

        return df

    def add_lag_features(self, df: pd.DataFrame, lags: list = [1, 24, 168]) -> pd.DataFrame:
        """Add lag features for time series modeling."""
        df = df.sort_values(['campaign_name', 'date', 'hour'])

        for lag in lags:
            for col in ['spend', 'impressions', 'clicks']:
                df[f'{col}_lag_{lag}h'] = df.groupby('campaign_name')[col].shift(lag)

        # Fill NaN lag values with 0 (for first few periods)
        lag_cols = [col for col in df.columns if col.endswith('h')]
        df[lag_cols] = df[lag_cols].fillna(0)

        return df

    def create_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable and forecast features."""
        # Sort by time for proper forecasting
        df = df.sort_values(['campaign_name', 'date', 'hour'])

        # Create future spend target (next 12 hours)
        df['target_spend_12h'] = df.groupby('campaign_name')['spend'].shift(-12)

        # Remove rows where target is NaN (last 12 hours of each campaign)
        df = df.dropna(subset=['target_spend_12h'])

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature preparation pipeline."""
        self.logger.info("Starting feature preparation...")

        # Validate data
        df = self.validate_data(df)

        # Add temporal features
        df = self.add_temporal_features(df)

        # Add campaign features
        df = self.add_campaign_features(df)

        # Add lag features
        df = self.add_lag_features(df)

        # Create target features
        df = self.create_target_features(df)

        self.logger.info(f"Feature preparation complete. Final shape: {df.shape}")
        return df

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, split_method: str = 'campaign') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        if split_method == 'campaign':
            # Campaign-wise split: split campaigns, not individual samples
            unique_campaigns = df['campaign_name'].unique()
            n_test_campaigns = int(len(unique_campaigns) * test_size)
            
            # Randomly select campaigns for test set
            import numpy as np
            np.random.seed(42)  # For reproducibility
            test_campaigns = np.random.choice(unique_campaigns, size=n_test_campaigns, replace=False)
            train_campaigns = [camp for camp in unique_campaigns if camp not in test_campaigns]
            
            train_df = df[df['campaign_name'].isin(train_campaigns)].copy()
            test_df = df[df['campaign_name'].isin(test_campaigns)].copy()
            
            self.logger.info(f"Campaign-wise split: {len(train_campaigns)} train campaigns, {len(test_campaigns)} test campaigns")
            self.logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test samples")
            
        elif split_method == 'time':
            # Original time-based split
            df = df.sort_values(['date', 'hour'])
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            self.logger.info(f"Time-based split: {len(train_df)} train, {len(test_df)} test samples")
            
        else:
            raise ValueError("split_method must be 'campaign' or 'time'")
            
        return train_df, test_df

    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed dataset to CSV."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved processed data to {output_path}")


def prepare_dataset(input_path: str = None, output_path: str = None) -> pd.DataFrame:
    """Convenience function to prepare dataset."""
    from features.feature_engineering import FeatureEngineer
    
    preparer = DatasetPreparer()
    engineer = FeatureEngineer()

    input_path = input_path or config.get("data.raw_dir", "data/raw") + "/meta_insights_hourly_dataset.csv"
    output_path = output_path or config.get("data.processed_dir", "data/processed") + "/processed_dataset.csv"

    # Load and prepare data
    df = preparer.load_data(input_path)
    df = preparer.prepare_features(df)
    
    # Apply advanced feature engineering
    df = engineer.engineer_features(df)

    # Save processed data
    preparer.save_processed_data(df, output_path)

    return df


if __name__ == "__main__":
    prepare_dataset()
