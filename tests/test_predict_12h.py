"""
Tests for 12-hour prediction module
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.models.predict_next12h import SpendPredictor


class TestSpendPredictor:
    """Test cases for SpendPredictor class."""

    @pytest.fixture
    def sample_prediction_data(self):
        """Create sample data for prediction."""
        # Create recent data for prediction
        dates = pd.date_range('2025-01-01 12:00:00', periods=48, freq='H')
        campaigns = ['Campaign_A', 'Campaign_B'] * 24

        data = {
            'campaign_name': campaigns,
            'date': dates,
            'hourly_window': [f'2025-01-01T{i:02d}:00:00' for i in range(12, 60)],
            'spend': np.random.uniform(10, 100, 48),
            'impressions': np.random.randint(1000, 10000, 48),
            'clicks': np.random.randint(10, 500, 48),
            'ctr': np.random.uniform(0.01, 0.05, 48),
            'cpc': np.random.uniform(0.5, 5.0, 48),
        }

        df = pd.DataFrame(data)

        # Add engineered features that would be created by the pipeline
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)

        # Add some lag features
        for lag in [1, 24]:
            for col in ['spend', 'impressions', 'clicks']:
                df[f'{col}_lag_{lag}h'] = np.random.uniform(10, 100, 48)

        return df

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = SpendPredictor()
        assert predictor.model is None
        assert predictor.model_path is not None

    def test_model_loading(self, sample_prediction_data):
        """Test model loading for prediction."""
        predictor = SpendPredictor()

        # Create a temporary model file (simplified mock)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            # Create a simple mock model
            import joblib
            mock_model = type('MockModel', (), {
                'predict': lambda self, x: np.random.uniform(50, 150, len(x)),
                'feature_name_': ['spend', 'impressions', 'clicks']
            })()

            joblib.dump(mock_model, model_path)

            # Test loading
            predictor.model_path = model_path
            predictor.load_model()

            assert predictor.model is not None

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_prepare_prediction_data(self, sample_prediction_data):
        """Test prediction data preparation."""
        predictor = SpendPredictor()

        # Create mock model for feature column access
        mock_model = type('MockModel', (), {'feature_name_': ['spend', 'impressions', 'clicks']})()
        predictor.model = mock_model

        result = predictor.prepare_prediction_data(sample_prediction_data)

        # Check that we get a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that we have the expected features
        assert len(result) == len(sample_prediction_data)

    def test_make_predictions(self, sample_prediction_data):
        """Test prediction generation."""
        predictor = SpendPredictor()

        # Create a mock model that returns predictions
        def mock_predict(X):
            # Return total 12-hour predictions (single values)
            return np.full(len(X), 75.0)  # Constant prediction for simplicity

        mock_model = type('MockModel', (), {
            'predict': mock_predict,
            'feature_name_': ['spend', 'impressions', 'clicks']
        })()

        predictor.model = mock_model

        # Test the old format (single value predictions)
        predictions = predictor.predict_next_12h(sample_prediction_data)

        # Check that we get predictions for each campaign
        assert isinstance(predictions, dict)
        assert len(predictions) > 0

        # Check that all predictions are reasonable (old format)
        for campaign, pred in predictions.items():
            assert isinstance(pred, (int, float))
            assert pred >= 0  # Spend should be non-negative

        # Test the new hourly format
        hourly_predictions = predictor.predict_hourly_spend(sample_prediction_data)

        # Check that we get hourly predictions for each campaign
        assert isinstance(hourly_predictions, dict)
        assert len(hourly_predictions) > 0

        # Check that all hourly predictions are reasonable
        for campaign, pred in hourly_predictions.items():
            assert isinstance(pred, list)
            assert len(pred) == 12  # Should have 12 hourly predictions
            assert all(p >= 0 for p in pred)  # All hourly spends should be non-negative

    def test_save_predictions(self, sample_prediction_data):
        """Test prediction saving."""
        predictor = SpendPredictor()

        # Create mock predictions
        mock_predictions = {
            'Campaign_A': 75.5,
            'Campaign_B': 82.3,
        }

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            predictor.save_predictions(mock_predictions, output_path)

            # Check that file was created
            assert os.path.exists(output_path)

            # Check that we can read the saved predictions
            saved_df = pd.read_csv(output_path)
            # For hourly format, check for 'Campaign' column and hourly columns
            assert 'Campaign' in saved_df.columns

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_hourly_predictions(self, sample_prediction_data):
        """Test hourly prediction saving."""
        predictor = SpendPredictor()

        # Create mock hourly predictions
        mock_hourly_predictions = {
            'Campaign_A': [10.0, 11.0, 12.0, 9.0, 8.0, 7.0, 6.0, 8.0, 10.0, 12.0, 11.0, 10.0],
            'Campaign_B': [15.0, 16.0, 17.0, 14.0, 13.0, 12.0, 11.0, 13.0, 15.0, 17.0, 16.0, 15.0],
        }

        with tempfile.NamedTemporaryFile(suffix='_hourly.csv', delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            hourly_df = predictor.save_hourly_predictions(mock_hourly_predictions, output_path)

            # Check that file was created
            assert os.path.exists(output_path)

            # Check that we can read the saved predictions
            saved_df = pd.read_csv(output_path)
            assert 'Campaign' in saved_df.columns

            # Check that we have the right campaigns
            assert 'Campaign_A' in saved_df['Campaign'].values
            assert 'Campaign_B' in saved_df['Campaign'].values

            # Check that we have hourly columns (should have 12 hour columns)
            hour_columns = [col for col in saved_df.columns if col != 'Campaign']
            assert len(hour_columns) == 12

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == "__main__":
    pytest.main([__file__])
