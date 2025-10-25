#!/usr/bin/env python3
"""
Script to make 12-hour spend predictions
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.predict_next12h import predict_next_12h

if __name__ == "__main__":
    hourly_predictions = predict_next_12h()
    print("12-hour hourly spend predictions (IST Timezone):")
    print("-" * 80)

    if hourly_predictions:
        # Get current IST time for proper hour calculations
        current_time = datetime.now()
        ist_offset = timedelta(hours=5, minutes=30)
        current_ist = current_time + ist_offset

        # Create hour labels
        hour_labels = []
        for i in range(12):
            hour_time = current_ist + timedelta(hours=i)
            hour_labels.append(hour_time.strftime('%I %p'))  # Format: 02 AM

        # Print header with hour labels
        print(f"{'Campaign':<20} " + " ".join([f"{hour:>8}" for hour in hour_labels]))
        print("-" * 80)

        # Print each campaign's hourly predictions
        for campaign, hourly_spend in hourly_predictions.items():
            spend_values = [f"{spend:8.2f}" for spend in hourly_spend]
            print(f"{campaign:<20} " + " ".join(spend_values))

        print("-" * 80)

        # Calculate and display totals
        total_hourly_spend = {}
        for i in range(12):
            total_hourly_spend[i] = sum(pred[i] for pred in hourly_predictions.values())

        print(f"{'Total per Hour':<20} " + " ".join([f"{total:8.2f}" for total in total_hourly_spend.values()]))
        print(f"{'Average per Hour':<20} " + " ".join([f"{total/len(hourly_predictions):8.2f}" for total in total_hourly_spend.values()]))

        print("-" * 80)
        print(f"Number of campaigns predicted: {len(hourly_predictions)}")

        # Show IST time range
        start_time = current_ist.strftime('%I:%M %p IST')
        end_time = (current_ist + timedelta(hours=11)).strftime('%I:%M %p IST')
        print(f"Prediction time range: {start_time} to {end_time}")
    else:
        print("No predictions generated.")

    print(f"Hourly predictions saved to: {Path(__file__).parent.parent / 'data' / 'predictions' / 'spend_forecast_hourly.csv'}")
    print(f"Original format saved to: {Path(__file__).parent.parent / 'data' / 'predictions' / 'spend_forecast_12h.csv'}")
