"""
02_feature_engineering.py
--------------------------
Prepares the real PaySim dataset (6.3M rows) for model training.

Key decisions baked in (from EDA):
  - Filter to TRANSFER + CASH_OUT only (only types with real fraud)
  - Stratified 80/20 train/test split preserving fraud ratio
  - Saves feature CSVs for fast reload during training

Run:  python notebooks/02_feature_engineering.py
Output: data/features_train.csv, data/features_test.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN  = os.path.join(ROOT, "data", "PS_20174392719_1491204439457_log.csv")
DATA_OUT = os.path.join(ROOT, "data")

sys.path.insert(0, os.path.join(ROOT, "notebooks"))
from feature_engineering_utils import (
    engineer_features, FEATURE_COLS, FRAUD_RELEVANT_TYPES, TARGET
)

if not os.path.exists(DATA_IN):
    raise FileNotFoundError(
        f"\nDataset not found at: {DATA_IN}\n"
        "Download from: https://www.kaggle.com/datasets/ealaxi/paysim1"
    )


def main():
    print("Loading real PaySim dataset (~20 sec for 6.3M rows)...")
    df = pd.read_csv(DATA_IN)
    print(f"  Loaded: {len(df):,} rows")
    print(f"  Fraud:  {df[TARGET].sum():,} ({df[TARGET].mean()*100:.4f}%)")

    # Filter to fraud-relevant types (EDA confirmed fraud ONLY here)
    print(f"\nFiltering to {FRAUD_RELEVANT_TYPES} (only types with fraud)...")
    df = df[df["type"].isin(FRAUD_RELEVANT_TYPES)].copy()
    print(f"  After filter: {len(df):,} rows")
    print(f"  Fraud:        {df[TARGET].sum():,} ({df[TARGET].mean()*100:.3f}%)")

    print("\nEngineering features...")
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df[TARGET]

    print("\nSplitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train.copy(); train_df[TARGET] = y_train.values
    test_df  = X_test.copy();  test_df[TARGET]  = y_test.values

    train_path = os.path.join(DATA_OUT, "features_train.csv")
    test_path  = os.path.join(DATA_OUT, "features_test.csv")

    print("Saving feature CSVs...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"\nFeature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"Train: {len(X_train):,} rows | fraud rate {y_train.mean()*100:.3f}%")
    print(f"Test:  {len(X_test):,} rows  | fraud rate {y_test.mean()*100:.3f}%")
    print(f"\nFeatures ({len(FEATURE_COLS)}):")
    for c in FEATURE_COLS:
        print(f"  {c}")
    print(f"\nSaved → {train_path}")
    print(f"Saved → {test_path}")
    print("\nNEXT → python notebooks/03_train_model.py")


if __name__ == "__main__":
    main()
