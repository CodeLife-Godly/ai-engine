"""
Pulls all asset_prices + assets from Postgres and computes technical
features per (asset, date) for ML training.

Target: next trading day's return, computed per asset — so the model
predicts "will this stock go up/down tomorrow" using only info available
as of today's close.

Run locally (from ai-engine/ root): python -m preprocessing.build_dataset
Requires: pip install pandas numpy pyarrow
"""

import sys
from pathlib import Path

# Ensure ai-engine/ root is importable when running this file directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from database.connection import get_connection
from database.config import config


def fetch_assets(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT id, symbol, sector, industry, country FROM assets;")
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def fetch_prices(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT asset_id, trading_date, open_price, high_price,
                   low_price, close_price, adjusted_close, volume
            FROM asset_prices
            ORDER BY asset_id, trading_date;
            """
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features per asset, sorted by date within each asset."""
    df = df.sort_values(["asset_id", "trading_date"]).reset_index(drop=True)

    processed_groups = []
    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.copy()
        g["asset_id"] = asset_id  # explicit, doesn't rely on apply()/groupby quirks

        g["return_1d"] = g["close_price"].pct_change(1)
        g["return_5d"] = g["close_price"].pct_change(5)
        g["return_10d"] = g["close_price"].pct_change(10)
        g["return_20d"] = g["close_price"].pct_change(20)

        g["ma_5"] = g["close_price"].rolling(5).mean()
        g["ma_10"] = g["close_price"].rolling(10).mean()
        g["ma_20"] = g["close_price"].rolling(20).mean()
        g["ma_50"] = g["close_price"].rolling(50).mean()

        g["price_vs_ma20"] = g["close_price"] / g["ma_20"] - 1

        g["volatility_5d"] = g["return_1d"].rolling(5).std()
        g["volatility_20d"] = g["return_1d"].rolling(20).std()

        g["volume_change_1d"] = g["volume"].pct_change(1)
        g["volume_ma_20"] = g["volume"].rolling(20).mean()
        g["volume_vs_avg"] = g["volume"] / g["volume_ma_20"] - 1

        # RSI (14-day)
        delta = g["close_price"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        g["rsi_14"] = 100 - (100 / (1 + rs))

        # High-low range as a volatility proxy
        g["high_low_range"] = (g["high_price"] - g["low_price"]) / g["close_price"]

        # TARGET: next day's return (shift -1 so today's row predicts tomorrow)
        g["target_next_return"] = g["close_price"].pct_change(1).shift(-1)
        g["target_next_direction"] = (g["target_next_return"] > 0).astype(int)

        processed_groups.append(g)

    return pd.concat(processed_groups, ignore_index=True)


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds market-wide and sector-relative features — every prior feature
    was single-stock-only, ignoring that stocks move together (market
    regime, sector rotation). Computed per trading_date across ALL
    assets, then joined back onto each row.
    """
    # Market-wide average 1-day return across all assets, per date
    market_daily = (
        df.groupby("trading_date")["return_1d"]
        .mean()
        .reset_index()
        .rename(columns={"return_1d": "market_return_1d"})
    )
    df = df.merge(market_daily, on="trading_date", how="left")

    # Sector-average 1-day return, per (date, sector)
    sector_daily = (
        df.groupby(["trading_date", "sector"])["return_1d"]
        .mean()
        .reset_index()
        .rename(columns={"return_1d": "sector_return_1d"})
    )
    df = df.merge(sector_daily, on=["trading_date", "sector"], how="left")

    # Relative performance — did this stock beat its market/sector that day?
    df["relative_return_vs_market"] = df["return_1d"] - df["market_return_1d"]
    df["relative_return_vs_sector"] = df["return_1d"] - df["sector_return_1d"]

    return df


def main():
    print("Connecting to database...")
    conn = get_connection()
    try:
        print("Fetching assets...")
        assets = fetch_assets(conn)
        print(f"Loaded {len(assets)} assets.")

        print("Fetching asset_prices (this may take a while for 10y x 199 tickers)...")
        prices = fetch_prices(conn)
        print(f"Loaded {len(prices)} price rows.")
    finally:
        conn.close()

    prices["trading_date"] = pd.to_datetime(prices["trading_date"])
    numeric_cols = ["open_price", "high_price", "low_price", "close_price", "adjusted_close", "volume"]
    for col in numeric_cols:
        prices[col] = pd.to_numeric(prices[col])

    print("Computing features...")
    dataset = compute_features(prices)

    # Merge in static asset metadata (sector, etc.) — useful categorical features
    dataset = dataset.merge(assets, left_on="asset_id", right_on="id", how="left", suffixes=("", "_asset"))

    print("Computing cross-sectional features (market-wide, sector-relative)...")
    dataset = add_cross_sectional_features(dataset)

    # Drop rows without enough history for rolling features, or without a target
    before = len(dataset)
    dataset = dataset.dropna(subset=["ma_50", "volatility_20d", "target_next_return"])
    print(f"Dropped {before - len(dataset)} rows lacking full feature history or target.")

    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.DATASET_DIR / "price_dataset.parquet"
    dataset.to_parquet(output_path, index=False)

    print(f"\nWrote {len(dataset)} rows to {output_path}")
    print(f"Date range: {dataset['trading_date'].min()} to {dataset['trading_date'].max()}")
    print(f"Unique assets: {dataset['asset_id'].nunique()}")


if __name__ == "__main__":
    main()