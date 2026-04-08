"""
feature_engineering_utils.py
-----------------------------
Shared feature logic used by BOTH the training pipeline and the Streamlit app.
This guarantees train/serve consistency — identical transformations at training
and prediction time.

Real PaySim key facts baked in here:
- Fraud only occurs in TRANSFER and CASH_OUT
- Balance drain is the #1 fraud signal
- We filter to fraud-relevant types before training
"""

import numpy as np
import pandas as pd

# Transaction types in the real PaySim dataset
TYPE_MAP = {
    "PAYMENT":  0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT":    3,
    "CASH_IN":  4,
}

# Types where fraud actually occurs in the real dataset
FRAUD_RELEVANT_TYPES = ["TRANSFER", "CASH_OUT"]

FEATURE_COLS = [
    "type_enc",
    "log_amount",
    "balance_drain_orig",
    "balance_zero_after",
    "dest_balance_increase",
    "amount_to_balance_ratio",
    "orig_balance_mismatch",
    "dest_balance_mismatch",
    "hour_of_day",
    "is_large_tx",
]

TARGET = "isFraud"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature transformations to a DataFrame.
    Works on the full dataset (training) or a single row (prediction).
    """
    df = df.copy()

    # 1. Encode transaction type
    df["type_enc"] = df["type"].map(TYPE_MAP).fillna(-1).astype(int)

    # 2. Log-transform amount (strong right skew)
    df["log_amount"] = np.log1p(df["amount"])

    # 3. Origin balance drain — #1 signal in real PaySim
    df["balance_drain_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

    # 4. Origin balance reaches exactly zero after transaction
    df["balance_zero_after"] = (df["newbalanceOrig"] == 0).astype(int)

    # 5. Destination account balance increase
    df["dest_balance_increase"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # 6. Amount relative to origin balance (how much of account was moved)
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # 7. Origin balance inconsistency — fraud often has bookkeeping errors
    #    Expected new balance = old - amount; mismatch flags manipulation
    df["orig_balance_mismatch"] = np.abs(
        (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
    )

    # 8. Destination balance inconsistency
    df["dest_balance_mismatch"] = np.abs(
        (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    )

    # 9. Hour of day (step % 24)
    df["hour_of_day"] = df["step"] % 24

    # 10. Large transaction flag (matches isFlaggedFraud threshold)
    df["is_large_tx"] = (df["amount"] >= 200_000).astype(int)

    return df
