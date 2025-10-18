"""
Tests for feature engineering module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.feature_engineering import FeatureEngineer
from src.features.prepare_dataset import DatasetPreparer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        # Create sample time series data
        dates = pd.date_range('2025-01-01', periods=100, freq='H')
        campaigns = ['Campaign_A', 'Campaign_B'] * 50

        data = {
            'campaign_name': campaigns,
            'date': dates,
            'spend': np.random.uniform(10, 100, 100),
            'impressions': np.random.randint(1000, 10000, 100),
            'clicks': np.random.randint(10, 500, 100),
            'ctr': np.random.uniform(0.01, 0.05, 100),
            'cpc': np.random.uniform(0.5, 5.0, 100),
        }

        df = pd.DataFrame(data)
        return df

    def test_create_rolling_features(self, sample_data):
        """Test rolling window feature creation."""
        engineer = FeatureEngineer()
        result = engineer.create_rolling_features(sample_data)

        # Check that rolling features were added
        rolling_cols = [col for col in result.columns if 'rolling_mean' in col]
        assert len(rolling_cols) > 0

        # Check that values are reasonable
        for col in rolling_cols:
            assert not result[col].isna().all()

    def test_create_seasonal_features(self, sample_data):
        """Test seasonal feature creation."""
        engineer = FeatureEngineer()

        # Add required columns for seasonal features
        sample_data['hour'] = sample_data['date'].dt.hour
        sample_data['day_of_week'] = sample_data['date'].dt.dayofweek
        sample_data['month'] = sample_data['date'].dt.month

        result = engineer.create_seasonal_features(sample_data)

        # Check that cyclical features were added
        seasonal_cols = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
        for col in seasonal_cols:
            assert col in result.columns
            assert not result[col].isna().all()

    def test_create_ratio_features(self, sample_data):
        """Test ratio feature creation."""
        engineer = FeatureEngineer()
        result = engineer.create_ratio_features(sample_data)

        # Check that ratio features were added
        ratio_cols = ['ctr_cpm_ratio', 'cpc_ctr_ratio', 'clicks_to_impressions']
        for col in ratio_cols:
            assert col in result.columns

    def test_engineer_features_pipeline(self, sample_data):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer()
        result = engineer.engineer_features(sample_data)

        # Check that multiple feature types were added
        feature_types = ['rolling', 'expanding', 'ratio', 'interaction', 'seasonal']
        for feature_type in feature_types:
            feature_cols = [col for col in result.columns if feature_type in col.lower()]
            assert len(feature_cols) > 0


class TestDatasetPreparer:
    """Test cases for DatasetPreparer class."""

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw dataset."""
        data = {
            'campaign_name': ['Campaign_A'] * 24 + ['Campaign_B'] * 24,
            'date': pd.date_range('2025-01-01', periods=48, freq='H').tolist(),
            'hourly_window': [f'2025-01-01T{i"02d"}:00:00' for i in range(24)] * 2,
            'spend': np.random.uniform(10, 100, 48),
            'impressions': np.random.randint(1000, 10000, 48),
            'clicks': np.random.randint(10, 500, 48),
            'ctr': np.random.uniform(0.01, 0.05, 48),
            'cpc': np.random.uniform(0.5, 5.0, 48),
            'cpm': np.random.uniform(5, 50, 48),
            'cpp': np.random.uniform(1, 10, 48),
        }

        df = pd.DataFrame(data)
        return df

    def test_validate_data(self, sample_raw_data):
        """Test data validation."""
        preparer = DatasetPreparer()
        result = preparer.validate_data(sample_raw_data)

        # Check that required columns exist
        required_cols = ['campaign_name', 'date', 'hourly_window', 'spend']
        for col in required_cols:
            assert col in result.columns

        # Check that date column is datetime
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

        # Check that negative spend values are removed
        assert (result['spend'] >= 0).all()

    def test_add_temporal_features(self, sample_raw_data):
        """Test temporal feature creation."""
        preparer = DatasetPreparer()
        result = preparer.add_temporal_features(sample_raw_data)

        # Check that temporal features were added
        temporal_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours']
        for col in temporal_cols:
            assert col in result.columns

    def test_add_campaign_features(self, sample_raw_data):
        """Test campaign-level feature creation."""
        preparer = DatasetPreparer()
        result = preparer.add_campaign_features(sample_raw_data)

        # Check that campaign features were added
        campaign_cols = ['spend_mean', 'spend_std', 'impressions_mean']
        for col in campaign_cols:
            assert col in result.columns

    def test_create_target_features(self, sample_raw_data):
        """Test target variable creation."""
        preparer = DatasetPreparer()
        result = preparer.create_target_features(sample_raw_data)

        # Check that target column was added
        assert 'target_spend_12h' in result.columns

        # Check that target is shifted by 12 hours
        # (This is a simplified check - full validation would require more complex logic)
        assert not result['target_spend_12h'].isna().all()


if __name__ == "__main__":
    pytest.main([__file__])
