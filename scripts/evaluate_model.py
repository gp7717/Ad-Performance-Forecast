#!/usr/bin/env python3
"""
Script to evaluate the forecasting model
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.evaluate import evaluate_model

if __name__ == "__main__":
    results = evaluate_model()
    print("Model evaluation completed!")
    print(f"Evaluation report: {Path(__file__).parent.parent / 'data' / 'models' / 'evaluation_report.txt'}")
    print(f"Residual analysis: {Path(__file__).parent.parent / 'data' / 'models' / 'residual_analysis.png'}")
    print(f"Feature importance: {Path(__file__).parent.parent / 'data' / 'models' / 'feature_importance.png'}")
