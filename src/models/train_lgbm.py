"""
LightGBM model training for ad performance forecasting
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from config import config


class LGBMTrainer:
    """LightGBM model trainer with hyperparameter optimization."""

    def __init__(self):
        self.logger = self._setup_logging()
        self.model = None
        self.best_params = None

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        # Define feature columns (exclude target and metadata)
        exclude_cols = [
            'target_spend_12h', 'campaign_name', 'date', 'hourly_window',
            'spend',  # Current spend, not a feature for forecasting future spend
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Convert categorical features to numerical (LightGBM handles them better as numerical)
        categorical_mappings = {
            'spend_level': {'low': 0, 'medium': 1, 'high': 2},
            'performance_tier': {'low': 0, 'medium': 1, 'high': 2}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in feature_cols:
                df[col] = df[col].map(mapping).fillna(1)  # Default to 'medium' if not found

        X = df[feature_cols]
        y = df['target_spend_12h']

        self.logger.info(f"Training with {len(feature_cols)} features")
        return X, y, feature_cols

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        cv_folds: int = 5,
        random_state: int = 42
    ) -> lgb.LGBMRegressor:
        """Train LightGBM model with hyperparameter optimization."""

        # Define parameter search space
        param_distributions = {
            'num_leaves': [31, 50, 100, 200],
            'max_depth': [6, 8, 10, 12, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 500, 1000],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_samples': [10, 20, 50, 100],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0],
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # Initialize base model
        base_model = lgb.LGBMRegressor(
            random_state=random_state,
            verbose=-1,  # Suppress training output
            objective='regression',
            metric='rmse',
            categorical_feature=None  # Disable categorical feature handling
        )

        # Randomized search for hyperparameter optimization
        self.logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")

        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_distributions,
            n_iter=n_trials,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )

        random_search.fit(X, y)
        self.best_params = random_search.best_params_

        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best CV score: {-random_search.best_score_:.4f}")

        # Train final model with best parameters
        self.model = random_search.best_estimator_

        return self.model

    def evaluate_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance on train and test sets."""
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Metrics
        metrics = {}

        # Training metrics
        metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
        metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
        metrics['train_r2'] = r2_score(y_train, y_train_pred)

        # Test metrics
        metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
        metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
        metrics['test_r2'] = r2_score(y_test, y_test_pred)

        # MAPE (Mean Absolute Percentage Error)
        metrics['train_mape'] = np.mean(np.abs((y_train - y_train_pred) / (y_train + 1e-6))) * 100
        metrics['test_mape'] = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-6))) * 100

        # Log metrics
        self.logger.info("Model Performance Metrics:")
        self.logger.info(f"Train - MAE: {metrics['train_mae']:.4f}, RMSE: {metrics['train_rmse']:.4f}, R²: {metrics['train_r2']:.4f}, MAPE: {metrics['train_mape']:.2f}%")
        self.logger.info(f"Test  - MAE: {metrics['test_mae']:.4f}, RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}, MAPE: {metrics['test_mape']:.2f}%")

        return metrics

    def save_model(self, model_path: str):
        """Save trained model to disk."""
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)
        self.logger.info(f"Model saved to {model_path}")

        # Save model metadata
        metadata = {
            'best_params': self.best_params,
            'feature_names': self.model.feature_name_ if hasattr(self.model, 'feature_name_') else None,
            'model_type': 'LightGBM',
        }

        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        self.logger.info(f"Model metadata saved to {metadata_path}")

    def load_model(self, model_path: str) -> lgb.LGBMRegressor:
        """Load trained model from disk."""
        self.model = joblib.load(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        return self.model


def train_lgbm_model(
    data_path: str = None,
    model_path: str = None,
    n_trials: int = None
) -> lgb.LGBMRegressor:
    """Convenience function to train LightGBM model."""
    from features.prepare_dataset import DatasetPreparer

    trainer = LGBMTrainer()

    # Use config defaults if not provided
    data_path = data_path or config.get("data.processed_dir", "data/processed") + "/processed_dataset.csv"
    model_path = model_path or config.get("data.models_dir", "data/models") + "/lgbm_model.pkl"
    n_trials = n_trials or config.get("training.n_trials", 100)

    # Load and prepare data
    preparer = DatasetPreparer()
    df = preparer.load_data(data_path)

    # Split data (campaign-wise)
    train_df, test_df = preparer.split_data(df, test_size=0.2, split_method='campaign')

    # Prepare features
    X_train, y_train, feature_cols = trainer.prepare_features(train_df)
    X_test, y_test, _ = trainer.prepare_features(test_df)

    # Train model
    trainer.train_model(X_train, y_train, n_trials=n_trials)

    # Evaluate model
    metrics = trainer.evaluate_model(X_train, X_test, y_train, y_test)

    # Save model
    trainer.save_model(model_path)

    return trainer.model, metrics


if __name__ == "__main__":
    train_lgbm_model()
