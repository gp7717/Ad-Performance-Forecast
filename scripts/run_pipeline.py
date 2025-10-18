#!/usr/bin/env python3
"""
Main entry point for the ad performance forecasting pipeline
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
