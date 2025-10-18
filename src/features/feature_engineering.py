"""
Advanced feature engineering for ad performance forecasting
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

from config import config


class FeatureEngineer:
    """Advanced feature engineering for time series forecasting."""

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """Create rolling window features."""
        df = df.sort_values(['campaign_name', 'date', 'hour'])

        for window in windows:
            for col in ['spend', 'impressions', 'clicks', 'ctr']:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}d'] = (
                    df.groupby('campaign_name')[col]
                    .rolling(window=window*24, min_periods=1)  # 24 hours per day
                    .mean()
                    .reset_index(0, drop=True)
                )

                # Rolling std
                df[f'{col}_rolling_std_{window}d'] = (
                    df.groupby('campaign_name')[col]
                    .rolling(window=window*24, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )

        return df

    def create_expanding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create expanding window features (cumulative stats)."""
        df = df.sort_values(['campaign_name', 'date', 'hour'])

        for col in ['spend', 'impressions', 'clicks']:
            # Expanding mean
            df[f'{col}_expanding_mean'] = (
                df.groupby('campaign_name')[col]
                .expanding()
                .mean()
                .reset_index(0, drop=True)
            )

            # Expanding sum
            df[f'{col}_expanding_sum'] = (
                df.groupby('campaign_name')[col]
                .expanding()
                .sum()
                .reset_index(0, drop=True)
            )

        return df

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio and efficiency features."""
        # Cost efficiency ratios
        df['ctr_cpm_ratio'] = df['ctr'] / (df['cpm'] + 1e-6)  # Avoid division by zero
        df['cpc_ctr_ratio'] = df['cpc'] / (df['ctr'] + 1e-6)

        # Engagement ratios
        df['clicks_to_impressions'] = df['clicks'] / (df['impressions'] + 1e-6)
        df['actions_to_clicks'] = (
            df[['add_to_cart', 'landing_page_view', 'view_content']].sum(axis=1) /
            (df['clicks'] + 1e-6)
        )

        # Campaign performance ratios
        df['spend_efficiency'] = df['impressions'] / (df['spend'] + 1e-6)

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different metrics."""
        # Time-based interactions
        df['hour_spend'] = df['hour'] * df['spend']
        df['dayofweek_spend'] = df['day_of_week'] * df['spend']

        # Performance interactions
        df['ctr_cpc_interaction'] = df['ctr'] * df['cpc']
        df['impressions_clicks_interaction'] = df['impressions'] * df['clicks']

        # Business hours interactions
        df['business_hours_spend'] = df['is_business_hours'] * df['spend']
        df['weekend_spend'] = df['is_weekend'] * df['spend']

        return df

    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for key metrics."""
        key_metrics = ['spend', 'impressions', 'clicks', 'ctr', 'cpc']

        for col in key_metrics:
            for deg in range(2, degree + 1):
                df[f'{col}_pow_{deg}'] = df[col] ** deg

        return df

    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal and cyclical features."""
        # Cyclical encoding for hour and day of week
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['dayofweek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def create_campaign_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create campaign clustering features based on performance patterns."""
        # Calculate campaign-level statistics
        campaign_stats = df.groupby('campaign_name').agg({
            'spend': ['mean', 'std', 'max', 'min'],
            'ctr': ['mean', 'std'],
            'cpc': ['mean', 'std'],
            'impressions': ['mean', 'sum'],
        }).round(6)

        # Flatten column names
        campaign_stats.columns = ['_'.join(col).strip() for col in campaign_stats.columns.values]
        campaign_stats = campaign_stats.reset_index()

        # Simple clustering based on spend and performance
        campaign_stats['spend_level'] = pd.qcut(
            campaign_stats['spend_mean'],
            q=3,
            labels=['low', 'medium', 'high']
        )

        # Performance tier based on CTR and CPC
        campaign_stats['performance_tier'] = 'medium'
        high_perf_mask = (
            (campaign_stats['ctr_mean'] > campaign_stats['ctr_mean'].quantile(0.67)) &
            (campaign_stats['cpc_mean'] < campaign_stats['cpc_mean'].quantile(0.33))
        )
        low_perf_mask = (
            (campaign_stats['ctr_mean'] < campaign_stats['ctr_mean'].quantile(0.33)) |
            (campaign_stats['cpc_mean'] > campaign_stats['cpc_mean'].quantile(0.67))
        )

        campaign_stats.loc[high_perf_mask, 'performance_tier'] = 'high'
        campaign_stats.loc[low_perf_mask, 'performance_tier'] = 'low'

        # Merge back to original dataframe
        df = df.merge(campaign_stats[['campaign_name', 'spend_level', 'performance_tier']], on='campaign_name', how='left')
        
        # Convert categorical features to numerical (for LightGBM compatibility)
        if 'spend_level' in df.columns:
            df['spend_level'] = df['spend_level'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1)
        if 'performance_tier' in df.columns:
            df['performance_tier'] = df['performance_tier'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        self.logger.info("Starting advanced feature engineering...")

        # Rolling window features
        df = self.create_rolling_features(df)

        # Expanding window features
        df = self.create_expanding_features(df)

        # Ratio features
        df = self.create_ratio_features(df)

        # Interaction features
        df = self.create_interaction_features(df)

        # Polynomial features
        df = self.create_polynomial_features(df)

        # Seasonal features
        df = self.create_seasonal_features(df)

        # Campaign clustering
        df = self.create_campaign_clusters(df)

        self.logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df


def engineer_features(input_df: pd.DataFrame = None) -> pd.DataFrame:
    """Convenience function for feature engineering."""
    engineer = FeatureEngineer()
    return engineer.engineer_features(input_df)


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        df = pd.read_csv(input_file)
        df = engineer_features(df)
        print(f"Engineered features shape: {df.shape}")
    else:
        print("Usage: python -m src.features.feature_engineering <input_csv>")
