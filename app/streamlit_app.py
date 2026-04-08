"""
app/streamlit_app.py
--------------------
Fintech Transaction Anomaly Explorer — built on the REAL PaySim dataset.

Run locally:   streamlit run app/streamlit_app.py
Deploy free:   https://share.streamlit.io → set main file to app/streamlit_app.py

Dataset:  https://www.kaggle.com/datasets/ealaxi/paysim1
Expected: data/PS_20174392719_1491204439457_log.csv

Performance note:
  The full 6.3M-row CSV is loaded once and cached by Streamlit.
  Charts sample up to 20,000 points for browser performance.
  The prediction form uses all engineered features in real time.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── paths ───────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data", "PS_20174392719_1491204439457_log.csv")
MODEL_PATH = os.path.join(ROOT, "models", "fraud_model.pkl")
META_PATH  = os.path.join(ROOT, "models", "model_meta.json")

sys.path.insert(0, os.path.join(ROOT, "notebooks"))
from feature_engineering_utils import (
    engineer_features, FEATURE_COLS, FRAUD_RELEVANT_TYPES, TYPE_MAP
)

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Anomaly Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── load data (cached — only reads CSV once per session) ────────────────────
@st.cache_data(show_spinner="Loading 6.3M transactions from PaySim dataset...")
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df["hour_of_day"] = df["step"] % 24
    df["day"]         = df["step"] // 24
    df["label"]       = df["isFraud"].map({0: "Legitimate", 1: "Fraud"})
    df["balance_drain_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    return df

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, {}
    model = joblib.load(MODEL_PATH)
    meta  = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    return model, meta

df_full = load_data()
model, meta = load_model()

# ── missing data guard ───────────────────────────────────────────────────────
if df_full is None:
    st.error(
        "### Dataset not found\n\n"
        "Download the real PaySim CSV from Kaggle and place it in `data/`:\n\n"
        "**https://www.kaggle.com/datasets/ealaxi/paysim1**\n\n"
        "Expected filename: `PS_20174392719_1491204439457_log.csv`"
    )
    st.stop()

# ── sidebar — filters ────────────────────────────────────────────────────────
st.sidebar.title("🔍 Fraud Explorer")
st.sidebar.caption("Real PaySim · 6.3M transactions · 30 days")
st.sidebar.markdown("---")

all_types  = sorted(df_full["type"].unique().tolist())
tx_types   = st.sidebar.multiselect("Transaction types", all_types, default=all_types)

day_min, day_max = int(df_full["day"].min()), int(df_full["day"].max())
day_range = st.sidebar.slider("Day range", day_min, day_max, (day_min, day_max))

# Amount slider — cap at 99th percentile to avoid $10M outlier stretching the slider
amt_cap = int(df_full["amount"].quantile(0.99))
amount_range = st.sidebar.slider(
    "Amount filter ($)", 0, amt_cap, (0, amt_cap), step=500
)

show_fraud_only = st.sidebar.checkbox("Fraud transactions only", value=False)

st.sidebar.markdown("---")
if meta:
    st.sidebar.markdown(f"**Model:** {meta.get('model_name','RF')}")
    st.sidebar.markdown(f"**AUC-ROC:** {meta.get('auc_roc', '—')}")
    st.sidebar.markdown(f"**Avg Precision:** {meta.get('avg_precision','—')}")
    st.sidebar.markdown(f"**Trained on:** {meta.get('train_rows',0):,} rows")
    st.sidebar.markdown(f"**Fraud rate:** {meta.get('fraud_rate_pct','—')}%")
elif model is None:
    st.sidebar.warning(
        "Model not found. Run `03_train_model.py` first.\n\n"
        "Visualisations work without it — prediction form is disabled."
    )

# ── apply filters ────────────────────────────────────────────────────────────
mask = (
    df_full["type"].isin(tx_types) &
    df_full["day"].between(*day_range) &
    df_full["amount"].between(*amount_range)
)
if show_fraud_only:
    mask = mask & (df_full["isFraud"] == 1)

df = df_full[mask]

# ── header ───────────────────────────────────────────────────────────────────
st.title("💳 Fintech Transaction Anomaly Explorer")
st.caption(
    "Real PaySim synthetic fraud dataset · 6.3M transactions · 30 simulated days · "
    "[Kaggle source](https://www.kaggle.com/datasets/ealaxi/paysim1)"
)

# ── KPI cards ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
fraud_count = int(df["isFraud"].sum())
fraud_pct   = df["isFraud"].mean() * 100 if len(df) > 0 else 0
fraud_vol   = df[df["isFraud"]==1]["amount"].sum()
legit_vol   = df[df["isFraud"]==0]["amount"].sum()

k1.metric("Transactions (filtered)", f"{len(df):,}")
k2.metric("Fraud cases", f"{fraud_count:,}")
k3.metric("Fraud rate", f"{fraud_pct:.4f}%")
k4.metric("Fraud volume", f"${fraud_vol:,.0f}")
k5.metric("Legitimate volume", f"${legit_vol:,.0f}")

st.markdown("---")

# ── anomaly scatter — amount vs balance drain ─────────────────────────────────
st.subheader("Anomaly scatter — transaction amount vs origin balance drain")
st.caption("Fraud (red) clusters at high amounts with full balance drain. Sample: 20,000 points.")

# Sample for browser performance — stratified to always include all fraud rows
df_fraud  = df[df["isFraud"]==1]
df_legit  = df[df["isFraud"]==0]
n_sample  = min(20_000, len(df))
n_fraud_s = min(len(df_fraud), n_sample // 2)
n_legit_s = min(len(df_legit), n_sample - n_fraud_s)

df_plot = pd.concat([
    df_fraud.sample(n=n_fraud_s, random_state=42) if n_fraud_s > 0 else df_fraud,
    df_legit.sample(n=n_legit_s, random_state=42) if n_legit_s > 0 else df_legit,
])

fig_scatter = px.scatter(
    df_plot,
    x="amount",
    y="balance_drain_orig",
    color="label",
    color_discrete_map={"Legitimate": "#378ADD", "Fraud": "#E24B4A"},
    opacity=0.5,
    log_x=True,
    hover_data=["type", "step", "amount", "isFraud"],
    labels={
        "amount": "Transaction amount ($, log scale)",
        "balance_drain_orig": "Origin balance drained ($)",
    },
    height=430,
)
fig_scatter.update_traces(marker=dict(size=4))
fig_scatter.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend_title_text="",
)
st.plotly_chart(fig_scatter, width="stretch")

# ── two-column charts ─────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Volume by transaction type")
    type_counts = df.groupby(["type","label"]).size().reset_index(name="count")
    fig_bar = px.bar(
        type_counts, x="type", y="count", color="label",
        color_discrete_map={"Legitimate":"#378ADD","Fraud":"#E24B4A"},
        barmode="stack", height=320,
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="", xaxis_title="", yaxis_title="Count",
    )
    st.plotly_chart(fig_bar, width="stretch")
    st.caption("In the full dataset, fraud ONLY occurs in TRANSFER and CASH_OUT.")

with c2:
    st.subheader("Fraud rate by transaction type")
    fr_by_type = (
        df.groupby("type")["isFraud"]
        .agg(fraud_rate=lambda x: x.mean()*100)
        .reset_index()
        .sort_values("fraud_rate", ascending=True)
    )
    fig_rate = px.bar(
        fr_by_type, x="fraud_rate", y="type", orientation="h",
        color="fraud_rate",
        color_continuous_scale=["#B5D4F4","#E24B4A"],
        height=320,
        labels={"fraud_rate":"Fraud rate (%)","type":""},
    )
    fig_rate.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_rate, width="stretch")

# ── amount distribution ───────────────────────────────────────────────────────
st.subheader("Amount distribution — legitimate vs fraud (99th percentile cap)")
df_hist = df[df["amount"] <= df["amount"].quantile(0.99)]
fig_hist = px.histogram(
    df_hist, x="amount", color="label",
    color_discrete_map={"Legitimate":"#378ADD","Fraud":"#E24B4A"},
    nbins=80, barmode="overlay", opacity=0.65, height=300,
    labels={"amount":"Transaction amount ($)"},
)
fig_hist.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend_title_text="",
)
st.plotly_chart(fig_hist, width="stretch")

# ── hourly pattern ────────────────────────────────────────────────────────────
st.subheader("Transactions by hour of day")
hourly = df.groupby(["hour_of_day","label"]).size().reset_index(name="count")
fig_hr = px.line(
    hourly, x="hour_of_day", y="count", color="label",
    color_discrete_map={"Legitimate":"#378ADD","Fraud":"#E24B4A"},
    markers=True, height=280,
    labels={"hour_of_day":"Hour of day (0=midnight)","count":"Transactions"},
)
fig_hr.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend_title_text="",
)
st.plotly_chart(fig_hr, width="stretch")

# ── live prediction form ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔮 Live fraud probability — check any transaction")

if model is None:
    st.info(
        "Run `python notebooks/03_train_model.py` to train the model, "
        "then restart the app to enable this section."
    )
else:
    with st.form("predict_form"):
        p1, p2, p3 = st.columns(3)
        with p1:
            p_type    = st.selectbox("Transaction type", ["TRANSFER","CASH_OUT","PAYMENT","DEBIT","CASH_IN"])
            p_amount  = st.number_input("Amount ($)", min_value=0.01, max_value=10_000_000.0, value=250_000.0, step=1000.0)
            p_step    = st.number_input("Step (hour 1–744)", min_value=1, max_value=744, value=200)
        with p2:
            p_old_orig = st.number_input("Old balance — origin ($)",  min_value=0.0, value=250_000.0, step=1000.0)
            p_new_orig = st.number_input("New balance — origin ($)",  min_value=0.0, value=0.0,       step=1000.0)
        with p3:
            p_old_dest = st.number_input("Old balance — dest ($)",    min_value=0.0, value=1_000.0,   step=1000.0)
            p_new_dest = st.number_input("New balance — dest ($)",    min_value=0.0, value=251_000.0, step=1000.0)

        submitted = st.form_submit_button("Calculate fraud probability")

    if submitted:
        row = pd.DataFrame([{
            "step": p_step, "type": p_type, "amount": p_amount,
            "oldbalanceOrg": p_old_orig, "newbalanceOrig": p_new_orig,
            "oldbalanceDest": p_old_dest, "newbalanceDest": p_new_dest,
        }])
        row_feat = engineer_features(row)[FEATURE_COLS]
        proba    = float(model.predict_proba(row_feat)[0][1])
        label    = "FRAUD" if proba >= 0.5 else "LEGITIMATE"
        color    = "#E24B4A" if proba >= 0.5 else "#1D9E75"
        bg       = "#FCEBEB" if proba >= 0.5 else "#E1F5EE"

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown(
                f"""<div style="background:{bg};border:1.5px solid {color};
                border-radius:12px;padding:1.5rem;text-align:center;">
                <p style="color:{color};font-size:2rem;font-weight:600;margin:0">{label}</p>
                <p style="font-size:1.4rem;margin:0.5rem 0 0">
                  Probability: <strong style="color:{color}">{proba*100:.2f}%</strong>
                </p></div>""",
                unsafe_allow_html=True,
            )

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba*100, 2),
                number={"suffix":"%","font":{"size":28}},
                gauge={
                    "axis": {"range":[0,100]},
                    "bar":  {"color": color},
                    "steps":[
                        {"range":[0,30],  "color":"#E1F5EE"},
                        {"range":[30,60], "color":"#FAEEDA"},
                        {"range":[60,100],"color":"#FCEBEB"},
                    ],
                    "threshold":{"line":{"color":color,"width":3},"value":50},
                },
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(t=20,b=10,l=20,r=20),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, width="stretch")

            drain = p_old_orig - p_new_orig
            st.markdown("**Risk signals:**")
            signals = [
                (f"High-risk type ({p_type})",     p_type in FRAUD_RELEVANT_TYPES),
                (f"Balance drained: ${drain:,.0f}", drain >= p_amount * 0.9),
                (f"Origin balance → 0",             p_new_orig == 0.0),
                (f"Large transaction (>${p_amount:,.0f})", p_amount >= 200_000),
                (f"Amount/balance ratio {p_amount/(p_old_orig+1):.2f}x",
                 p_amount/(p_old_orig+1) > 0.9),
            ]
            for sig_text, is_risk in signals:
                st.markdown(f"{'🔴' if is_risk else '🟢'} {sig_text}")

# ── raw data table ────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander(f"View filtered transaction table ({min(len(df),1000):,} rows shown)"):
    display_cols = [
        "step","type","amount","oldbalanceOrg","newbalanceOrig",
        "oldbalanceDest","newbalanceDest","isFlaggedFraud","label",
    ]
    st.dataframe(df[display_cols].head(1000), width="stretch", hide_index=True)

st.caption(
    "Built by Madhushree Varatharaj · "
    "[GitHub](https://github.com/madhushree) · "
    "Data: PaySim synthetic fraud dataset (Kaggle)"
)
