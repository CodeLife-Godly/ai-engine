"""
Generates PREDICTIVE insights (insight_type='predictive') for every asset,
using the trained LightGBM baseline model. Distinct from
generate_historical_insights.py, which explains PAST moves — this predicts
TOMORROW's direction, based on TODAY's features.

Reuses compute_features() and add_cross_sectional_features() directly from
preprocessing/build_dataset.py (imported, not duplicated) so the exact same
feature logic used in training is used at inference time — avoiding
train/serve skew from two independently-maintained implementations.

Note: this is NOT true SHAP-based per-prediction explanation. The
"contributing factors" shown are a lightweight proxy — today's actual
values of the globally most important features, described in plain
language. Flagged here explicitly rather than presented as more
sophisticated than it is.

Run (from ai-engine/ root): python -m preprocessing.generate_predictive_insights
Requires: pip install pandas numpy lightgbm supabase python-dotenv
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
from supabase import create_client
from dotenv import load_dotenv

from preprocessing.build_dataset import compute_features, add_cross_sectional_features
from training.train_baseline import FEATURE_COLUMNS

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "trained" / "baseline_lgbm.txt"
LOOKBACK_DAYS = 90  # enough history to warm up 50-day rolling features, with margin
TOP_N_FACTORS = 3   # show top N contributing features per insight

PAGE_SIZE = 1000


def fetch_all_rows(table: str, columns: str, filters=None) -> pd.DataFrame:
    all_rows = []
    offset = 0
    while True:
        query = supabase.table(table).select(columns)
        if filters:
            for col, op, val in filters:
                query = getattr(query, op)(col, val)
        resp = query.range(offset, offset + PAGE_SIZE - 1).execute()
        rows = resp.data
        if not rows:
            break
        all_rows.extend(rows)
        offset += PAGE_SIZE
        if len(rows) < PAGE_SIZE:
            break
    return pd.DataFrame(all_rows)


def fetch_recent_data():
    print("Fetching assets...")
    assets = fetch_all_rows("assets", "id,symbol,sector")
    assets = assets.rename(columns={"id": "asset_id"})

    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    print(f"Fetching prices since {cutoff}...")
    prices = fetch_all_rows(
        "asset_prices",
        "asset_id,trading_date,open_price,high_price,low_price,close_price,volume",
        filters=[("trading_date", "gte", cutoff)],
    )
    return assets, prices


def build_today_features(assets: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    prices["trading_date"] = pd.to_datetime(prices["trading_date"])
    for col in ["open_price", "high_price", "low_price", "close_price", "volume"]:
        prices[col] = pd.to_numeric(prices[col])

    print("Computing features (reusing training pipeline logic)...")
    dataset = compute_features(prices)
    dataset = dataset.merge(assets, on="asset_id", how="left")
    dataset = add_cross_sectional_features(dataset)

    # Only the MOST RECENT row per asset — today's fully-computed feature vector
    latest = (
        dataset.sort_values("trading_date")
        .groupby("asset_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    before = len(latest)
    latest = latest.dropna(subset=FEATURE_COLUMNS)
    dropped = before - len(latest)
    if dropped:
        print(f"Dropped {dropped} assets with insufficient history for full features.")

    return latest


def describe_feature(feat: str, value: float) -> str:
    descriptions = {
        "market_return_1d": f"Overall market moved {value:+.2%} today.",
        "sector_return_1d": f"This stock's sector averaged {value:+.2%} today.",
        "relative_return_vs_market": f"Stock {'outperformed' if value > 0 else 'underperformed'} the market by {abs(value):.2%} today.",
        "relative_return_vs_sector": f"Stock {'outperformed' if value > 0 else 'underperformed'} its sector by {abs(value):.2%} today.",
        "volatility_20d": f"20-day volatility is at {value:.4f}.",
        "volatility_5d": f"5-day volatility is at {value:.4f}.",
        "volume_vs_avg": f"Volume is {value:+.1%} vs its 20-day average.",
        "rsi_14": f"RSI(14) is at {value:.0f}.",
        "return_1d": f"Closed {value:+.2%} today.",
        "return_5d": f"Up {value:+.2%} over the last 5 days.",
    }
    return descriptions.get(feat, f"{feat} = {value:.4f}")


def build_predictive_insight(row, model: lgb.Booster) -> tuple:
    X = row[FEATURE_COLUMNS].values.reshape(1, -1)
    prob_up = float(model.predict(X)[0])

    direction = "up" if prob_up >= 0.5 else "down"
    confidence = prob_up if prob_up >= 0.5 else 1 - prob_up

    importances = pd.Series(model.feature_importance(), index=FEATURE_COLUMNS)
    top_features = importances.sort_values(ascending=False).head(TOP_N_FACTORS).index.tolist()

    factors = []
    for feat in top_features:
        value = row[feat]
        factors.append({
            "factor_type": "model_feature",
            "factor_name": feat,
            "factor_value": f"{value:.4f}",
            "weight": float(importances[feat] / importances.sum()),
            "explanation": describe_feature(feat, value),
        })

    summary = (
        f"Model predicts {row['symbol']} is more likely to move {direction} "
        f"tomorrow ({confidence:.0%} confidence), based on today's price/volume "
        f"patterns and market conditions. This is a statistical estimate from "
        f"historical patterns, not financial advice."
    )

    insight = {
        "asset_id": row["asset_id"],
        "insight_type": "predictive",
        "event_date": row["trading_date"].date().isoformat(),
        "title": f"{row['symbol']}: model leans {direction} for tomorrow ({confidence:.0%})",
        "summary": summary,
        "confidence": round(confidence, 4),
        "model_version": "baseline_lgbm_v2_crosssectional",
    }

    return insight, factors


def insert_insight(insight: dict, factors: list):
    resp = supabase.table("insights").insert(insight).execute()
    insight_id = resp.data[0]["id"]

    if factors:
        for f in factors:
            f["insight_id"] = insight_id
        supabase.table("insight_factors").insert(factors).execute()


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: model file not found at {MODEL_PATH}")
        sys.exit(1)

    model = lgb.Booster(model_file=str(MODEL_PATH))

    assets, prices = fetch_recent_data()
    today_features = build_today_features(assets, prices)
    print(f"Generating predictions for {len(today_features)} assets...")

    success = 0
    failed = 0
    for _, row in today_features.iterrows():
        try:
            insight, factors = build_predictive_insight(row, model)
            insert_insight(insight, factors)
            success += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED for {row.get('symbol')}: {e}")

    print(f"\nDone. Success: {success}  Failed: {failed}")


if __name__ == "__main__":
    main()