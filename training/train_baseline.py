"""
Trains a LightGBM baseline model to predict next-day stock direction
(up/down), using a TIME-BASED train/test split (never random) to avoid
lookahead leakage — the model is tested only on dates strictly after
its training window.

Run locally (from ai-engine/ root): python -m training.train_baseline
Requires: pip install lightgbm scikit-learn pandas
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

from database.config import config

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "price_vs_ma20", "volatility_5d", "volatility_20d",
    "volume_change_1d", "volume_vs_avg", "rsi_14", "high_low_range",
    "market_return_1d", "sector_return_1d",
    "relative_return_vs_market", "relative_return_vs_sector",
]

TEST_FRACTION = 0.15  # last 15% of the date range (by calendar time) held out for testing


def main():
    dataset_path = config.DATASET_DIR / "price_dataset.parquet"
    df = pd.read_parquet(dataset_path)
    df["trading_date"] = pd.to_datetime(df["trading_date"])

    unique_dates = sorted(df["trading_date"].unique())
    split_idx = int(len(unique_dates) * (1 - TEST_FRACTION))
    split_date = unique_dates[split_idx]

    train = df[df["trading_date"] < split_date]
    test = df[df["trading_date"] >= split_date]

    print(f"Split date: {split_date}")
    print(f"Train rows: {len(train)}  ({train['trading_date'].min()} to {train['trading_date'].max()})")
    print(f"Test rows:  {len(test)}  ({test['trading_date'].min()} to {test['trading_date'].max()})")

    X_train, y_train = train[FEATURE_COLUMNS], train["target_next_direction"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target_next_direction"]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    print("\nTraining...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = (preds_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_proba)

    # Baseline to compare against: naive "always predict majority class"
    naive_acc = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n=== Results ===")
    print(f"Accuracy:        {acc:.4f}")
    print(f"AUC:             {auc:.4f}")
    print(f"Naive baseline:  {naive_acc:.4f}  (always predicting majority class)")
    print(f"Improvement over naive: {acc - naive_acc:+.4f}")

    print("\n=== Top 15 feature importances ===")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).head(15))

    config.TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.TRAINED_MODEL_DIR / "baseline_lgbm.txt"
    model.booster_.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()