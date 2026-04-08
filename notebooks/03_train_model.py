"""
03_train_model.py
-----------------
Trains Logistic Regression + Random Forest on the real PaySim feature matrix.
Handles extreme class imbalance (~0.13% fraud) via class_weight='balanced'.

Expected real-data performance:
  Logistic Regression: AUC-ROC ~0.97, Avg Precision ~0.55
  Random Forest:       AUC-ROC ~0.997, Avg Precision ~0.88

Run:  python notebooks/03_train_model.py
Output: models/fraud_model.pkl, models/model_meta.json, models/eval_plots/
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(ROOT, "data", "features_train.csv")
TEST_CSV  = os.path.join(ROOT, "data", "features_test.csv")
MODEL_DIR = os.path.join(ROOT, "models")
PLOT_DIR  = os.path.join(MODEL_DIR, "eval_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "notebooks"))
from feature_engineering_utils import FEATURE_COLS, TARGET

sns.set_theme(style="whitegrid", font_scale=1.05)
COLORS = {"Logistic Regression": "#378ADD", "Random Forest": "#1D9E75"}


def plot_roc_pr(models_dict, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})",
                     color=COLORS[name], lw=2)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.4f})",
                     color=COLORS[name], lw=2)

    axes[0].plot([0,1],[0,1], "k--", lw=0.8, label="Random baseline")
    axes[0].set(title="ROC Curve — real PaySim", xlabel="False Positive Rate",
                ylabel="True Positive Rate")
    axes[0].legend()

    baseline = y_test.mean()
    axes[1].axhline(baseline, color="k", linestyle="--", lw=0.8,
                    label=f"Baseline (fraud rate={baseline*100:.3f}%)")
    axes[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend()

    sns.despine()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "roc_pr_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


def plot_confusion(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Legitimate","Fraud"],
                yticklabels=["Legitimate","Fraud"], ax=ax)
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"{name}\nTP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    ax.set(ylabel="Actual", xlabel="Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ","_")
    path = os.path.join(PLOT_DIR, f"confusion_{fname}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


def plot_feature_importance(rf_pipeline):
    clf = rf_pipeline.named_steps["clf"]
    imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#E24B4A" if i == imp.idxmax() else "#B5D4F4" for i in imp.index]
    ax.barh(imp.index, imp.values, color=colors, edgecolor="white")
    ax.set_title("Random Forest feature importances — real PaySim")
    ax.set_xlabel("Importance")
    sns.despine()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


def main():
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            "Feature CSVs not found. Run 02_feature_engineering.py first."
        )

    print("Loading feature matrices...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df[TARGET]
    print(f"  Train: {len(X_train):,} | fraud {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
    print(f"  Test:  {len(X_test):,}  | fraud {y_test.sum():,}  ({y_test.mean()*100:.3f}%)")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    results  = {}
    trained  = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)
        auc    = roc_auc_score(y_test, y_prob)
        ap     = average_precision_score(y_test, y_prob)
        rep    = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {"auc_roc": auc, "avg_precision": ap}
        trained[name] = model

        print(f"  AUC-ROC:          {auc:.4f}")
        print(f"  Avg Precision:    {ap:.4f}")
        print(f"  Fraud Precision:  {rep['1']['precision']:.4f}")
        print(f"  Fraud Recall:     {rep['1']['recall']:.4f}")
        print(f"  Fraud F1:         {rep['1']['f1-score']:.4f}")

    best_name  = max(results, key=lambda n: results[n]["auc_roc"])
    best_model = trained[best_name]
    print(f"\nBest model: {best_name} (AUC={results[best_name]['auc_roc']:.4f})")

    print("\nGenerating evaluation plots...")
    plot_roc_pr(trained, X_test, y_test)
    for name, model in trained.items():
        plot_confusion(model, X_test, y_test, name)
    plot_feature_importance(trained["Random Forest"])

    model_path = os.path.join(MODEL_DIR, "fraud_model.pkl")
    meta_path  = os.path.join(MODEL_DIR, "model_meta.json")
    joblib.dump(best_model, model_path)

    meta = {
        "model_name":     best_name,
        "features":       FEATURE_COLS,
        "auc_roc":        round(results[best_name]["auc_roc"], 4),
        "avg_precision":  round(results[best_name]["avg_precision"], 4),
        "train_rows":     len(X_train),
        "fraud_rate_pct": round(y_train.mean() * 100, 4),
        "dataset":        "Real PaySim — TRANSFER + CASH_OUT only",
        "threshold":      0.5,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model → {model_path}")
    print(f"Saved meta  → {meta_path}")
    print("\nNEXT → streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
