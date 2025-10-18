"""
Configuration management for ad performance forecasting
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager that loads from YAML and environment variables"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Default configuration
            self._config = {
                'data': {
                    'raw_dir': 'data/raw',
                    'processed_dir': 'data/processed',
                    'models_dir': 'data/models',
                    'predictions_dir': 'data/predictions'
                },
                'model': {
                    'target_column': 'spend',
                    'feature_columns': [],
                    'forecast_horizon': 12,
                    'validation_split': 0.2
                },
                'training': {
                    'random_state': 42,
                    'n_trials': 100
                }
            }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self):
        """Save current configuration to YAML file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)


# Global config instance
config = Config()
