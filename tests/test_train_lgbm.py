"""
Tests for LightGBM training module
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.models.train_lgbm import LGBMTrainer


class TestLGBMTrainer:
    """Test cases for LGBMTrainer class."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training dataset."""
        np.random.seed(42)
        n_samples = 1000

        data = {
            'campaign_name': np.random.choice(['Campaign_A', 'Campaign_B', 'Campaign_C'], n_samples),
            'spend': np.random.uniform(10, 500, n_samples),
            'impressions': np.random.randint(1000, 50000, n_samples),
            'clicks': np.random.randint(5, 1000, n_samples),
            'ctr': np.random.uniform(0.005, 0.1, n_samples),
            'cpc': np.random.uniform(0.5, 10.0, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_business_hours': np.random.randint(0, 2, n_samples),
            'target_spend_12h': np.random.uniform(10, 500, n_samples),
        }

        df = pd.DataFrame(data)

        # Add some lag features
        for lag in [1, 24]:
            for col in ['spend', 'impressions', 'clicks']:
                df[f'{col}_lag_{lag}h'] = np.random.uniform(10, 500, n_samples)

        return df

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = LGBMTrainer()
        assert trainer.model is None
        assert trainer.best_params is None

    def test_prepare_features(self, sample_training_data):
        """Test feature preparation."""
        trainer = LGBMTrainer()
        X, y, feature_cols = trainer.prepare_features(sample_training_data)

        # Check that features and target are separated correctly
        assert len(X) == len(sample_training_data)
        assert len(y) == len(sample_training_data)
        assert 'target_spend_12h' not in X.columns
        assert 'campaign_name' not in X.columns

        # Check that we have features to train on
        assert len(feature_cols) > 0

    def test_model_training(self, sample_training_data):
        """Test model training with small parameter space."""
        trainer = LGBMTrainer()

        # Prepare features
        X, y, _ = trainer.prepare_features(sample_training_data)

        # Train with minimal trials for testing
        model = trainer.train_model(X, y, n_trials=2, cv_folds=2)

        # Check that model was trained
        assert trainer.model is not None
        assert trainer.best_params is not None
        assert hasattr(trainer.model, 'predict')

        # Check that we can make predictions
        predictions = trainer.model.predict(X[:10])
        assert len(predictions) == 10
        assert all(isinstance(pred, (int, float)) for pred in predictions)

    def test_model_evaluation(self, sample_training_data):
        """Test model evaluation."""
        trainer = LGBMTrainer()

        # Prepare features
        X, y, _ = trainer.prepare_features(sample_training_data)

        # Split data for evaluation
        split_idx = len(X) // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        trainer.train_model(X_train, y_train, n_trials=2, cv_folds=2)

        # Evaluate
        metrics = trainer.evaluate_model(X_train, X_test, y_train, y_test)

        # Check that all expected metrics are present
        expected_metrics = ['train_mae', 'train_rmse', 'test_mae', 'test_rmse', 'train_r2', 'test_r2']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])

    def test_model_save_load(self, sample_training_data):
        """Test model persistence."""
        trainer = LGBMTrainer()

        # Prepare and train model
        X, y, _ = trainer.prepare_features(sample_training_data)
        trainer.train_model(X, y, n_trials=2, cv_folds=2)

        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            trainer.save_model(model_path)

            # Check that files were created
            assert os.path.exists(model_path)

            # Load model
            new_trainer = LGBMTrainer()
            loaded_model = new_trainer.load_model(model_path)

            # Check that loaded model works
            assert loaded_model is not None
            predictions = loaded_model.predict(X[:10])
            assert len(predictions) == 10

        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)


if __name__ == "__main__":
    pytest.main([__file__])
