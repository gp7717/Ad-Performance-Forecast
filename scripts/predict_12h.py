#!/usr/bin/env python3
"""
Script to make 12-hour spend predictions
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.predict_next12h import predict_next_12h

if __name__ == "__main__":
    predictions = predict_next_12h()
    print("12-hour spend predictions:")
    print("-" * 40)
    for campaign, spend in predictions.items():
        print(f"{campaign"30s"}: ${spend"8.2f"}")
    print("-" * 40)
    print(f"Predictions saved to: {Path(__file__).parent.parent / 'data' / 'predictions' / 'spend_forecast_12h.csv'}")
