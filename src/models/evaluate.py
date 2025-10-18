"""
Model evaluation and performance analysis for ad performance forecasting
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from config import config

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""

    def __init__(self, model_path: str = None):
        self.logger = self._setup_logging()
        self.model = None
        self.model_path = model_path or config.get("data.models_dir", "data/models") + "/lgbm_model.pkl"

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def load_model(self):
        """Load trained model from disk."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.logger.info(f"Model loaded from {self.model_path}")

    def comprehensive_evaluation(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame = None,
        y_train: pd.Series = None
    ) -> Dict[str, float]:
        """Comprehensive model evaluation metrics."""
        if self.model is None:
            self.load_model()

        # Predictions
        y_pred = self.model.predict(X_test)

        # Basic metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,
        }

        # Training metrics if available
        if X_train is not None and y_train is not None:
            y_train_pred = self.model.predict(X_train)
            metrics.update({
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'train_r2': r2_score(y_train, y_train_pred),
                'train_mape': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
            })

        # Additional metrics
        metrics.update({
            'mean_target': float(y_test.mean()),
            'std_target': float(y_test.std()),
            'n_samples': len(y_test),
        })

        return metrics

    def calculate_residuals(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate residual statistics."""
        residuals = y_true - y_pred

        return {
            'mean_residual': float(residuals.mean()),
            'std_residual': float(residuals.std()),
            'max_residual': float(residuals.max()),
            'min_residual': float(residuals.min()),
            'residual_skewness': float(residuals.skew()),
            'residual_kurtosis': float(residuals.kurtosis()),
        }

    def plot_residuals(self, y_true: pd.Series, y_pred: np.ndarray, save_path: str = None):
        """Plot residual analysis charts."""
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values')

        # Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals')

        # Residuals Distribution
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')

        # Predicted vs Actual
        axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Predicted vs Actual Values')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Residual plots saved to {save_path}")

        plt.show()

    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calculate feature importance using multiple methods."""
        if self.model is None:
            self.load_model()

        # Built-in feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_,
                'method': 'built_in'
            })

        # Permutation importance
        try:
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=5, random_state=42, n_jobs=-1
            )
            perm_df = pd.DataFrame({
                'feature': X.columns,
                'importance': perm_importance.importances_mean,
                'method': 'permutation'
            })
            importance_df = pd.concat([importance_df, perm_df])
        except Exception as e:
            self.logger.warning(f"Could not calculate permutation importance: {e}")

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20, save_path: str = None):
        """Plot feature importance."""
        plt.figure(figsize=(12, 8))

        # Plot top N features
        top_features = importance_df.head(top_n)

        # Create bar plot
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')

        # Add importance values on bars
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(importance + 0.001, i, f'{importance:.4f}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")

        plt.show()

    def analyze_predictions_by_campaign(self, df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
        """Analyze model performance by campaign."""
        # Combine predictions with campaign data
        results_df = df[['campaign_name']].copy()
        results_df['actual'] = y_true
        results_df['predicted'] = y_pred
        results_df['error'] = y_true - y_pred
        results_df['abs_error'] = np.abs(y_true - y_pred)
        results_df['pct_error'] = (y_true - y_pred) / (y_true + 1e-6) * 100

        # Campaign-level metrics
        campaign_metrics = results_df.groupby('campaign_name').agg({
            'actual': ['mean', 'count'],
            'predicted': ['mean'],
            'error': ['mean', 'std'],
            'abs_error': ['mean', 'median'],
        }).round(4)

        # Flatten column names
        campaign_metrics.columns = ['_'.join(col).strip() for col in campaign_metrics.columns.values]
        campaign_metrics = campaign_metrics.reset_index()

        # Calculate MAE by campaign
        campaign_mae = results_df.groupby('campaign_name')['abs_error'].mean().reset_index()
        campaign_mae.columns = ['campaign_name', 'mae']

        return campaign_metrics, campaign_mae

    def generate_evaluation_report(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        df_test: pd.DataFrame,
        output_path: str = None
    ) -> Dict:
        """Generate comprehensive evaluation report."""
        output_path = output_path or config.get("data.models_dir", "data/models") + "/evaluation_report.txt"

        # Calculate metrics
        metrics = self.comprehensive_evaluation(X_test, y_test)

        # Calculate residuals
        y_pred = self.model.predict(X_test)
        residuals = self.calculate_residuals(y_test, y_pred)

        # Feature importance
        importance_df = self.calculate_feature_importance(X_test, y_test)

        # Campaign analysis
        campaign_metrics, campaign_mae = self.analyze_predictions_by_campaign(df_test, y_test, y_pred)

        # Generate report
        report = f"""
# Model Evaluation Report

## Overall Performance
- **MAE**: {metrics['mae']:.4f}
- **RMSE**: {metrics['rmse']:.4f}
- **R²**: {metrics['r2']:.4f}
- **MAPE**: {metrics['mape']:.2f}%

## Residual Analysis
- **Mean Residual**: {residuals['mean_residual']:.4f}
- **Residual Std**: {residuals['std_residual']:.4f}
- **Residual Skewness**: {residuals['residual_skewness']:.4f}
- **Residual Kurtosis**: {residuals['residual_kurtosis']:.4f}

## Dataset Info
- **Number of Samples**: {metrics['n_samples']}
- **Mean Target Value**: {metrics['mean_target']:.4f}
- **Target Std Dev**: {metrics['std_target']:.4f}

## Top 10 Most Important Features
"""

        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            report += f"{i+1:2d}. {row['feature']:30s} {row['importance']:.6f}\n"

        # Save report to file
        with open(output_path, 'w') as f:
            f.write(report)

        self.logger.info(f"Evaluation report saved to {output_path}")

        return {
            'metrics': metrics,
            'residuals': residuals,
            'feature_importance': importance_df,
            'campaign_metrics': campaign_metrics,
            'campaign_mae': campaign_mae
        }


def evaluate_model(
    data_path: str = None,
    model_path: str = None,
    output_dir: str = None
) -> Dict:
    """Convenience function for model evaluation."""
    from features.prepare_dataset import DatasetPreparer
    from models.train_lgbm import LGBMTrainer

    evaluator = ModelEvaluator(model_path)

    # Use config defaults if not provided
    data_path = data_path or config.get("data.processed_dir", "data/processed") + "/processed_dataset.csv"
    output_dir = output_dir or config.get("data.models_dir", "data/models")

    # Load and prepare data
    preparer = DatasetPreparer()
    df = preparer.load_data(data_path)

    # Split data (campaign-wise for evaluation)
    train_df, test_df = preparer.split_data(df, test_size=0.2, split_method='campaign')

    # Use the exact same feature preparation as training
    trainer = LGBMTrainer()
    X_train, y_train, _ = trainer.prepare_features(train_df)
    X_test, y_test, _ = trainer.prepare_features(test_df)

    # Generate comprehensive evaluation report
    results = evaluator.generate_evaluation_report(
        X_test, y_test, test_df,
        output_path=f"{output_dir}/evaluation_report.txt"
    )

    # Create residual plots
    evaluator.plot_residuals(
        y_test, evaluator.model.predict(X_test),
        save_path=f"{output_dir}/residual_analysis.png"
    )

    # Create feature importance plot
    evaluator.plot_feature_importance(
        results['feature_importance'],
        save_path=f"{output_dir}/feature_importance.png"
    )

    return results


if __name__ == "__main__":
    results = evaluate_model()
    print("Model evaluation complete. Check output files for detailed analysis.")
