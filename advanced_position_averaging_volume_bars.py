"""
Advanced Position Averaging using Volume Bars (Lopez de Prado)

This module builds volume bars from tick/1m data and reuses the
strategy pipeline from AdvancedPositionAveraging on the resampled bars.

Key parameters:
- volume_bar_size: target cumulative volume per bar (units)
- min_bar_size: minimum number of underlying rows to avoid micro-bars

Usage:
    python advanced_position_averaging_volume_bars.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Reuse strategy pipeline
from advanced_position_averaging import AdvancedPositionAveraging


def build_volume_bars(df: pd.DataFrame, volume_bar_size: float, min_bar_size: int = 1) -> pd.DataFrame:
    """Build volume bars per Lopez de Prado.

    Accumulate transactions/rows until cumulative volume >= volume_bar_size,
    then emit one bar with:
      - Date: last timestamp in bucket
      - open: first close in bucket (or first open if present)
      - high: max(high)
      - low : min(low)
      - close: last close
      - volume: sum(volume)

    Assumptions:
      - Input df has columns: Date, open, high, low, close, volume
      - Date is datetime and df is time-sorted
    """
    required = {"Date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    df = df.sort_values("Date").reset_index(drop=True)

    out_rows = []
    bucket_idx_start = 0
    cum_volume = 0.0

    for i in range(len(df)):
        cum_volume += float(df.loc[i, "volume"])
        # If threshold reached and at least min_bar_size rows included
        if cum_volume >= volume_bar_size and (i - bucket_idx_start + 1) >= min_bar_size:
            bucket = df.iloc[bucket_idx_start : i + 1]

            # Build one aggregated bar
            first_row = bucket.iloc[0]
            last_row = bucket.iloc[-1]

            bar = {
                "Date": last_row["Date"],
                "open": float(first_row["open"]) if "open" in bucket.columns else float(first_row["close"]),
                "high": float(bucket["high"].max()),
                "low": float(bucket["low"].min()),
                "close": float(last_row["close"]),
                "volume": float(bucket["volume"].sum()),
            }
            out_rows.append(bar)

            # Reset bucket
            bucket_idx_start = i + 1
            cum_volume = 0.0

    # Optionally, emit tail if desired (commented by default to avoid partial bar lookahead)
    # if bucket_idx_start < len(df):
    #     bucket = df.iloc[bucket_idx_start:]
    #     if len(bucket) >= min_bar_size:
    #         first_row = bucket.iloc[0]
    #         last_row = bucket.iloc[-1]
    #         bar = {
    #             "Date": last_row["Date"],
    #             "open": float(first_row["open"]) if "open" in bucket.columns else float(first_row["close"]),
    #             "high": float(bucket["high"].max()),
    #             "low": float(bucket["low"].min()),
    #             "close": float(last_row["close"]),
    #             "volume": float(bucket["volume"].sum()),
    #         }
    #         out_rows.append(bar)

    bars = pd.DataFrame(out_rows)
    if len(bars) == 0:
        # Return empty with required columns
        return pd.DataFrame(columns=["Date", "open", "high", "low", "close", "volume"]) 

    return bars


class AdvancedPositionAveragingVolumeBars(AdvancedPositionAveraging):
    """Strategy runner that converts raw data into volume bars before processing."""

    def __init__(self, minute_data_path: str, initial_capital: float = 1_000_000,
                 start_date: str = "2024-01-01", volume_bar_size: float = 5_000_000,
                 min_bar_size: int = 1):
        super().__init__(minute_data_path, initial_capital, start_date)
        self.volume_bar_size = volume_bar_size
        self.min_bar_size = min_bar_size

    def load_data(self) -> bool:
        print("Loading 1-minute data and building volume bars...")
        try:
            raw = pd.read_csv(self.minute_data_path)
            raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw.sort_values("Date").reset_index(drop=True)

            # Filter from start date
            start_dt = pd.to_datetime(self.start_date)
            raw = raw[raw["Date"] >= start_dt].reset_index(drop=True)

            # Build volume bars
            bars = build_volume_bars(raw, self.volume_bar_size, self.min_bar_size)
            if len(bars) == 0:
                print("❌ No volume bars generated. Please adjust volume_bar_size/min_bar_size.")
                return False

            self.minute_data = bars.reset_index(drop=True)
            print(f"✅ Built {len(self.minute_data)} volume bars from {len(raw)} input rows")
            print(f"   Volume bar size: {self.volume_bar_size:.0f}, Min rows/bar: {self.min_bar_size}")
            print(f"   Date range: {self.minute_data['Date'].min()} to {self.minute_data['Date'].max()}")
            return True
        except Exception as e:
            print(f"❌ Error while building volume bars: {e}")
            return False


def main():
    print("Advanced Position Averaging - Volume Bars")
    print("=" * 60)

    minute_data_path = "BTCUSDT_1m_binance.csv"
    if not os.path.exists(minute_data_path):
        print(f"❌ Minute data file not found: {minute_data_path}")
        return

    # Example: ~5M units per bar; tune to your instrument/liquidity
    system = AdvancedPositionAveragingVolumeBars(
        minute_data_path=minute_data_path,
        initial_capital=1_000_000,
        start_date="2024-01-01",
        volume_bar_size=5_000_000,
        min_bar_size=5,
    )

    if system.run_full_optimization():
        print("\n🎉 Optimization on volume bars completed successfully!")
        print("Generated files:")
        print("- advanced_position_averaging_optimization_report.txt")
        print("- advanced_position_averaging_optimization_results.csv")
        print("- equity_curve_comparison.html")
    else:
        print("❌ Optimization failed (volume bars)")


if __name__ == "__main__":
    main()
