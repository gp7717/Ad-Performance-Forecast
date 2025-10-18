"""
Input/Output utilities for data and model persistence
"""
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from datetime import datetime

from config import config


class DataManager:
    """Handles data loading, saving, and management operations."""

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def save_dataframe(
        self,
        df: pd.DataFrame,
        file_path: str,
        file_format: str = 'csv',
        **kwargs
    ) -> str:
        """Save DataFrame to file with automatic directory creation."""
        file_path = Path(file_path)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_format.lower() == 'csv':
            df.to_csv(file_path, index=False, **kwargs)
        elif file_format.lower() == 'parquet':
            df.to_parquet(file_path, **kwargs)
        elif file_format.lower() == 'json':
            df.to_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        self.logger.info(f"Data saved to {file_path}")
        return str(file_path)

    def load_dataframe(
        self,
        file_path: str,
        file_format: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load DataFrame from file with automatic format detection."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect format if not specified
        if file_format is None:
            if file_path.suffix.lower() == '.csv':
                file_format = 'csv'
            elif file_path.suffix.lower() == '.parquet':
                file_format = 'parquet'
            elif file_path.suffix.lower() == '.json':
                file_format = 'json'
            else:
                raise ValueError(f"Cannot auto-detect format for {file_path}")

        if file_format.lower() == 'csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_format.lower() == 'parquet':
            df = pd.read_parquet(file_path, **kwargs)
        elif file_format.lower() == 'json':
            df = pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        self.logger.info(f"Data loaded from {file_path} - Shape: {df.shape}")
        return df

    def save_model(self, model: Any, file_path: str, metadata: Dict = None) -> str:
        """Save model with metadata."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, file_path)

        # Save metadata if provided
        if metadata:
            metadata_path = file_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Model saved to {file_path}")
        return str(file_path)

    def load_model(self, file_path: str) -> Tuple[Any, Dict]:
        """Load model and metadata."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        # Load model
        model = joblib.load(file_path)

        # Load metadata if available
        metadata = {}
        metadata_path = file_path.with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        self.logger.info(f"Model loaded from {file_path}")
        return model, metadata

    def save_predictions(self, predictions: Dict[str, float], file_path: str) -> str:
        """Save predictions with timestamp."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        pred_df = pd.DataFrame([
            {
                'campaign_name': campaign,
                'predicted_spend_12h': pred,
                'timestamp': datetime.now().isoformat()
            }
            for campaign, pred in predictions.items()
        ])

        # Save to CSV
        pred_df.to_csv(file_path, index=False)

        self.logger.info(f"Predictions saved to {file_path}")
        return str(file_path)

    def list_data_files(self, directory: str, pattern: str = "*.csv") -> List[str]:
        """List data files in directory matching pattern."""
        directory = Path(directory)
        if not directory.exists():
            return []

        files = list(directory.glob(pattern))
        return [str(f) for f in sorted(files)]

    def get_latest_file(self, directory: str, pattern: str = "*.csv") -> str:
        """Get the most recently modified file matching pattern."""
        files = self.list_data_files(directory, pattern)
        if not files:
            return None

        latest_file = max(files, key=lambda f: Path(f).stat().st_mtime)
        return latest_file


class ConfigManager:
    """Configuration file management."""

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def save_config(self, config_dict: Dict, file_path: str = "config.yaml") -> str:
        """Save configuration to YAML file."""
        import yaml

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration saved to {file_path}")
        return str(file_path)

    def load_config(self, file_path: str = "config.yaml") -> Dict:
        """Load configuration from YAML file."""
        import yaml

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        self.logger.info(f"Configuration loaded from {file_path}")
        return config_dict


# Convenience functions
def save_data(df: pd.DataFrame, file_path: str, **kwargs) -> str:
    """Save DataFrame to file."""
    manager = DataManager()
    return manager.save_dataframe(df, file_path, **kwargs)


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """Load DataFrame from file."""
    manager = DataManager()
    return manager.load_dataframe(file_path, **kwargs)


def save_model(model: Any, file_path: str, **kwargs) -> str:
    """Save model to file."""
    manager = DataManager()
    return manager.save_model(model, file_path, **kwargs)


def load_model(file_path: str):
    """Load model from file."""
    manager = DataManager()
    return manager.load_model(file_path)


def save_predictions(predictions: Dict[str, float], file_path: str) -> str:
    """Save predictions to file."""
    manager = DataManager()
    return manager.save_predictions(predictions, file_path)
