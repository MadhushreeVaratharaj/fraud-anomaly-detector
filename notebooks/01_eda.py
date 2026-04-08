"""
01_eda.py
---------
Exploratory Data Analysis for the REAL PaySim dataset (6.3M rows).
Run from project root:  python notebooks/01_eda.py

Real dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
Expected file: data/PS_20174392719_1491204439457_log.csv

All plots saved to: notebooks/eda_plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(ROOT, "data", "PS_20174392719_1491204439457_log.csv")
OUT_DIR = os.path.join(ROOT, "notebooks", "eda_plots")
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {"Legitimate": "#378ADD", "Fraud": "#E24B4A"}
sns.set_theme(style="whitegrid", font_scale=1.05)

if not os.path.exists(DATA):
    raise FileNotFoundError(
        f"\nDataset not found at:\n  {DATA}\n\n"
        "Download from: https://www.kaggle.com/datasets/ealaxi/paysim1\n"
        "Place the CSV inside the data/ folder and re-run."
    )

# ══════════════════════════════════════════════════════════════════════════
# CELL 1 — Load
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CELL 1 — Load data (6.3M rows, ~20 sec)")
print("="*60)

df = pd.read_csv(DATA)
print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nColumn dtypes:\n", df.dtypes.to_string())
print("\nSample rows:\n", df.head(3).to_string())

# ══════════════════════════════════════════════════════════════════════════
# CELL 2 — Data quality
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CELL 2 — Data quality")
print("="*60)

print("Nulls per column:\n", df.isnull().sum().to_string())
print(f"Duplicates: {df.duplicated().sum():,}")

# ══════════════════════════════════════════════════════════════════════════
# CELL 3 — Class imbalance
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CELL 3 — Class imbalance")
print("="*60)

fraud_rate = df["isFraud"].mean() * 100
print(f"Fraud rate: {fraud_rate:.4f}%")
print(f"Legitimate: {(df['isFraud']==0).sum():,}")
print(f"Fraud:      {(df['isFraud']==1).sum():,}")

counts = df["isFraud"].value_counts()
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["Legitimate", "Fraud"], counts.values,
              color=[PALETTE["Legitimate"], PALETTE["Fraud"]],
              width=0.5, edgecolor="white")
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10_000,
            f"{count:,}\n({count/len(df)*100:.3f}%)",
            ha="center", va="bottom", fontsize=10)
ax.set_title("Class distribution — real PaySim (6.3M rows)")
ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M"))
ax.set_ylim(0, counts.max() * 1.15)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_class_imbalance.png"), dpi=150)
plt.close()
print("→ Saved 01_class_imbalance.png")

# ══════════════════════════════════════════════════════════════════════════
# CELL 4 — Transaction types
# KEY REAL-DATA FACT: fraud ONLY exists in TRANSFER and CASH_OUT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CELL 4 — Transaction types")
print("="*60)

type_fraud = (
    df.groupby("type")["isFraud"]
    .agg(total="count", fraud_count="sum")
    .assign(fraud_pct=lambda x: x["fraud_count"] / x["total"] * 100)
    .sort_values("fraud_pct", ascending=False)
)
print("Fraud rate by type:\n", type_fraud.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
vol = df["type"].value_counts()
axes[0].barh(vol.index, vol.values, color="#B5D4F4", edgecolor="white")
axes[0].set_title("Volume by type")
axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M"))

colors = [PALETTE["Fraud"] if r > 0 else "#D3D1C7" for r in type_fraud["fraud_pct"]]
axes[1].barh(type_fraud.index, type_fraud["fraud_pct"], color=colors, edgecolor="white")
axes[1].set_title("Fraud rate % — only TRANSFER & CASH_OUT have fraud")
axes[1].xaxis.set_major_formatter(mtick.PercentFormatter())
for i, v in enumerate(type_fraud["fraud_pct"]):
    if v > 0:
        axes[1].text(v+0.02, i, f"{v:.2f}%", va="center", fontsize=9, color="#A32D2D")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_type_breakdown.png"), dpi=150)
plt.close()
print("→ Saved 02_type_breakdown.png")

fraud_types = type_fraud[type_fraud["fraud_count"] > 0].index.tolist()
print(f"\nKEY FINDING: Fraud ONLY in {fraud_types}")

# ══════════════════════════════════════════════════════════════════════════
# CELL 5 — Filter to fraud-relevant types and analyse amounts
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CELL 5 — Amount distribution (TRANSFER + CASH_OUT only)")
print("="*60)

df_rel = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
print(f"Rows after type filter: {len(df_rel):,}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, scale, title in zip(
    axes, [False, True],
    ["Amount (raw)", "Amount (log scale — clearer separation)"]
):
    for label, color in PALETTE.items():
        subset = df_rel[df_rel["isFraud"] == (1 if label=="Fraud" else 0)]["amount"]
        ax.hist(subset, bins=60, alpha=0.65, color=color,
                label=f"{label} (n={len(subset):,})", density=True)
    if scale:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Density")
    ax.legend()
    sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_amount_distribution.png"), dpi=150)
plt.close()

legit_med = df_rel[df_rel["isFraud"]==0]["amount"].median()
fraud_med = df_rel[df_rel["isFraud"]==1]["amount"].median()
print(f"Median — Legitimate: ${legit_med:,.2f}")
print(f"Median — Fraud:      ${fraud_med:,.2f}  ({fraud_med/legit_med:.1f}x higher)")
print("→ Saved 03_amount_distribution.png")

# ══════════════════════════════════════════════════════════════════════════
# CELL 6 — Time pattern
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CELL 6 — Hourly patterns")
print("="*60)

df_rel["hour_of_day"] = df_rel["step"] % 24
hourly = df_rel.groupby(["hour_of_day","isFraud"]).size().reset_index(name="count")

fig, ax = plt.subplots(figsize=(11, 4))
for label, color in PALETTE.items():
    is_f = 1 if label=="Fraud" else 0
    sub = hourly[hourly["isFraud"]==is_f]
    ax.plot(sub["hour_of_day"], sub["count"], color=color,
            label=label, linewidth=2, marker="o", markersize=3)
ax.set_title("Transactions by hour (TRANSFER + CASH_OUT) — real PaySim")
ax.set_xlabel("Hour of day (0=midnight)")
ax.set_ylabel("Count")
ax.set_xticks(range(0, 24, 2))
ax.legend()
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_hourly_pattern.png"), dpi=150)
plt.close()
print("→ Saved 04_hourly_pattern.png")

# ══════════════════════════════════════════════════════════════════════════
# CELL 7 — Balance drain (strongest fraud signal in real data)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CELL 7 — Balance drain signal")
print("="*60)

df_rel["balance_drain_orig"] = df_rel["oldbalanceOrg"] - df_rel["newbalanceOrig"]
df_rel["balance_zero_after"] = (df_rel["newbalanceOrig"] == 0).astype(int)
df_rel["orig_balance_mismatch"] = np.abs(
    (df_rel["oldbalanceOrg"] - df_rel["amount"]) - df_rel["newbalanceOrig"]
)

drain_fraud = df_rel[df_rel["isFraud"]==1]["balance_drain_orig"].mean()
drain_legit = df_rel[df_rel["isFraud"]==0]["balance_drain_orig"].mean()
zero_fraud  = df_rel[df_rel["isFraud"]==1]["balance_zero_after"].mean() * 100
zero_legit  = df_rel[df_rel["isFraud"]==0]["balance_zero_after"].mean() * 100

print(f"Avg drain — Fraud: ${drain_fraud:,.2f} | Legit: ${drain_legit:,.2f}")
print(f"Balance→0 — Fraud: {zero_fraud:.1f}% | Legit: {zero_legit:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# CELL 8 — Correlation
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CELL 8 — Feature correlations")
print("="*60)

numeric = [
    "amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest",
    "newbalanceDest","balance_drain_orig","balance_zero_after",
    "orig_balance_mismatch","hour_of_day","isFraud",
]
corr = df_rel[numeric].corr()["isFraud"].drop("isFraud").sort_values(key=abs, ascending=False)
print("Correlation with isFraud:\n", corr.round(4).to_string())

fig, ax = plt.subplots(figsize=(7, 4))
colors = [PALETTE["Fraud"] if v > 0 else PALETTE["Legitimate"] for v in corr.values]
ax.barh(corr.index, corr.values, color=colors, edgecolor="white")
ax.axvline(0, color="#888780", linewidth=0.8)
ax.set_title("Feature correlation with fraud (real PaySim)")
ax.set_xlabel("Pearson r")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_correlations.png"), dpi=150)
plt.close()
print("→ Saved 05_correlations.png")

print("\n" + "="*60)
print("EDA COMPLETE — KEY FINDINGS (Real PaySim)")
print("="*60)
print(f"""
1. {len(df):,} total transactions across 30 days
2. Fraud rate: {fraud_rate:.4f}% — extreme class imbalance
3. Fraud ONLY in TRANSFER + CASH_OUT ({len(df_rel):,} rows after filter)
4. Fraud median amount {fraud_med/legit_med:.1f}x higher than legitimate
5. {zero_fraud:.0f}% of fraud drains origin balance to exactly 0

NEXT → python notebooks/02_feature_engineering.py
""")
