"""
Time and date utilities for ad performance forecasting
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import logging

from config import config


class TimeUtils:
    """Utilities for time series data handling and date operations."""

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string into datetime object."""
        # Try multiple common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%d',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse timestamp: {timestamp_str}")

    def get_time_windows(self, start_date: str, end_date: str, window_size: str = '1H') -> List[Tuple[str, str]]:
        """Generate time windows between start and end dates."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        windows = []
        current = start

        while current < end:
            next_time = current + pd.Timedelta(window_size)
            if next_time > end:
                next_time = end

            windows.append((current.strftime('%Y-%m-%d %H:%M:%S'), next_time.strftime('%Y-%m-%d %H:%M:%S')))
            current = next_time

        return windows

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive time-based features to DataFrame."""
        # Ensure datetime column exists
        if 'date' not in df.columns:
            raise ValueError("DataFrame must contain 'date' column")

        df['date'] = pd.to_datetime(df['date'])

        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['quarter'] = df['date'].dt.quarter

        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Business time features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
        df['is_lunch_hour'] = df['hour'].between(11, 14).astype(int)
        df['is_prime_time'] = df['hour'].between(18, 22).astype(int)

        # Holiday features (simplified - you might want to use a proper holiday library)
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)

        return df

    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Create lag features for time series data."""
        if lags is None:
            lags = [1, 24, 168]  # 1 hour, 1 day, 1 week in hours

        # Sort by time
        time_cols = ['date', 'hour']
        if 'campaign_name' in df.columns:
            time_cols.insert(0, 'campaign_name')

        df = df.sort_values(time_cols)

        # Create lag features for key metrics
        key_metrics = ['spend', 'impressions', 'clicks', 'ctr', 'cpc']

        for lag in lags:
            for metric in key_metrics:
                if metric in df.columns:
                    df[f'{metric}_lag_{lag}h'] = df.groupby('campaign_name')[metric].shift(lag)

        # Fill NaN values in lag features
        lag_cols = [col for col in df.columns if col.endswith('h')]
        df[lag_cols] = df[lag_cols].fillna(0)

        return df

    def create_rolling_features(self, df: pd.DataFrame, windows: List[str] = None) -> pd.DataFrame:
        """Create rolling window features."""
        if windows is None:
            windows = ['24H', '7D', '30D']  # 1 day, 1 week, 1 month

        # Sort by time
        df = df.sort_values(['campaign_name', 'date', 'hour'])

        # Rolling features for key metrics
        key_metrics = ['spend', 'impressions', 'clicks', 'ctr']

        for window in windows:
            for metric in key_metrics:
                if metric in df.columns:
                    # Rolling mean
                    df[f'{metric}_rolling_mean_{window}'] = (
                        df.groupby('campaign_name')[metric]
                        .rolling(window, on=df.index if 'date' not in df.columns else None)
                        .mean()
                        .reset_index(0, drop=True)
                    )

                    # Rolling std
                    df[f'{metric}_rolling_std_{window}'] = (
                        df.groupby('campaign_name')[metric]
                        .rolling(window, on=df.index if 'date' not in df.columns else None)
                        .std()
                        .reset_index(0, drop=True)
                    )

        return df

    def split_time_series(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        gap_size: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split time series data while preserving temporal order."""
        df = df.sort_values(['campaign_name', 'date', 'hour'])

        # Calculate split point
        n_total = len(df)
        n_train = int(n_total * (1 - test_size))

        # Find the last training sample
        train_end_idx = n_train

        # If we need a gap, adjust the split
        if gap_size > 0:
            # Find the date that would create the gap
            gap_start_date = df.iloc[train_end_idx]['date']
            gap_end_date = gap_start_date + timedelta(hours=gap_size)

            # Find where the gap ends in the data
            gap_end_idx = df[df['date'] >= gap_end_date].index.min()
            if gap_end_idx is not None:
                train_end_idx = gap_end_idx - 1

        train_df = df.iloc[:train_end_idx + 1].copy()
        test_df = df.iloc[train_end_idx + 1:].copy()

        self.logger.info(f"Time series split: {len(train_df)} train, {len(test_df)} test samples")

        return train_df, test_df

    def validate_time_series(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate time series data quality."""
        issues = {}

        # Check for missing timestamps
        if 'date' in df.columns:
            missing_dates = df['date'].isna().sum()
            issues['missing_dates'] = missing_dates == 0

            if missing_dates > 0:
                self.logger.warning(f"Found {missing_dates} missing dates")

        # Check for duplicate timestamps
        if 'campaign_name' in df.columns and 'date' in df.columns:
            duplicates = df.groupby(['campaign_name', 'date']).size()
            duplicate_count = (duplicates > 1).sum()
            issues['duplicate_timestamps'] = duplicate_count == 0

            if duplicate_count > 0:
                self.logger.warning(f"Found {duplicate_count} campaigns with duplicate timestamps")

        # Check for gaps in time series
        if 'campaign_name' in df.columns and 'date' in df.columns:
            df_sorted = df.sort_values(['campaign_name', 'date'])
            gaps = 0

            for campaign in df['campaign_name'].unique():
                campaign_data = df_sorted[df_sorted['campaign_name'] == campaign]
                if len(campaign_data) > 1:
                    time_diffs = campaign_data['date'].diff().dropna()
                    # Check for gaps larger than expected (assuming hourly data)
                    expected_diff = timedelta(hours=1)
                    large_gaps = time_diffs > expected_diff * 2
                    gaps += large_gaps.sum()

            issues['time_gaps'] = gaps == 0

            if gaps > 0:
                self.logger.warning(f"Found {gaps} potential time gaps in the data")

        return issues

    def resample_time_series(
        self,
        df: pd.DataFrame,
        freq: str = 'H',
        aggregation: Dict[str, str] = None
    ) -> pd.DataFrame:
        """Resample time series data to different frequency."""
        if aggregation is None:
            aggregation = {
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'ctr': 'mean',
                'cpc': 'mean',
            }

        # Set datetime index for resampling
        if 'date' in df.columns:
            df = df.set_index('date')

        # Resample and aggregate
        resampled = df.groupby('campaign_name').resample(freq).agg(aggregation)

        # Reset index
        resampled = resampled.reset_index()

        return resampled


# Convenience functions
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time features to DataFrame."""
    utils = TimeUtils()
    return utils.add_time_features(df)


def split_time_series(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series data."""
    utils = TimeUtils()
    return utils.split_time_series(df, **kwargs)


def validate_time_series(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate time series data."""
    utils = TimeUtils()
    return utils.validate_time_series(df)
