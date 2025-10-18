#!/usr/bin/env python3
"""
Script to train the forecasting model
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.train_lgbm import train_lgbm_model

if __name__ == "__main__":
    model, metrics = train_lgbm_model()
    print("Model training completed!")
    print(f"Best parameters: {metrics}")
    print(f"Model saved to: {Path(__file__).parent.parent / 'data' / 'models' / 'lgbm_model.pkl'}")
