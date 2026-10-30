"""Streamlit Operations Frontend for Container Dwell Risk Prediction.

Real-Time Port Operations Control Tower & Bottleneck Triage Dashboard.
Connects directly to the frozen Stage 01 model artifact and feature contracts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Page configuration
st.set_page_config(
    page_title="Container Dwell Risk Control Tower",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for logistics operations feel
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-ontime {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-atrisk {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .sop-box {
        background-color: #f1f5f9;
        border-left: 4px solid #0284c7;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle():
    """Loads and caches the frozen Stage 01 baseline model bundle."""
    model_paths = [
        REPO_ROOT / "models" / "best_macro_f1_model.joblib",
        REPO_ROOT / "ML" / "ml_engineer" / "artifacts" / "best_macro_f1_model.joblib",
    ]
    for p in model_paths:
        if p.exists():
            return joblib.load(p)
    return None


@st.cache_data
def load_sample_cases():
    """Loads empirical sample cases for 1-click testing."""
    sample_path = REPO_ROOT / "ML" / "integration_engineer" / "sample_cases.json"
    if sample_path.exists():
        with open(sample_path) as f:
            return json.load(f)
    return []


@st.cache_data
def load_threshold_tradeoffs():
    """Loads empirical threshold sensitivity curve."""
    tradeoff_path = REPO_ROOT / "reports" / "ml" / "threshold_tradeoffs.csv"
    if tradeoff_path.exists():
        return pd.read_csv(tradeoff_path)
    return None


@st.cache_data
def load_feature_importance():
    """Loads ranked feature importance table."""
    feat_path = REPO_ROOT / "reports" / "ml" / "feature_importance.csv"
    if feat_path.exists():
        return pd.read_csv(feat_path)
    return None


bundle = load_model_bundle()
sample_cases = load_sample_cases()

# Header Section
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<div class="main-title">🚢 Container Dwell Bottleneck Control Tower</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Stage 01 Machine Learning Triage & Operational Decision Support System &bull; Port Operations SLA &lt; 25ms</div>',
        unsafe_allow_html=True,
    )
with col_head2:
    if bundle is not None:
        st.success("🟢 Model Pipeline: Loaded")
    else:
        st.error("🔴 Model Artifact: Not Found")

# Sidebar Controls
st.sidebar.header("⚙️ Operational Settings")

tau_selected = st.sidebar.slider(
    "Decision Threshold (τ) for Critical Alert",
    min_value=0.05,
    max_value=0.50,
    value=0.25,
    step=0.01,
    help="Cutoff probability for raising a CRITICAL_DELAY intervention alert. Minimum cost point is τ* = 0.22.",
)

c_fa = st.sidebar.number_input("Cost of False Alarm ($C_FA)", value=300, step=50, help="Cost of unnecessary physical yard re-stack / emergency drayage fee.")
c_fn = st.sidebar.number_input("Cost of Missed Bottleneck ($C_FN)", value=3000, step=500, help="Demurrage penalty and assembly line disruption cost.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Quick Preset Loader")

preset_options = ["Custom Input"]
if sample_cases:
    preset_options += [f"{c.get('case_id', 'Case')} — {c.get('ground_truth_tier', 'Case')}" for c in sample_cases[:10]]

selected_preset = st.sidebar.selectbox("Load Curated Case Preset", preset_options)

preset_data = {}
if selected_preset != "Custom Input" and sample_cases:
    idx = preset_options.index(selected_preset) - 1
    preset_data = sample_cases[idx].get("telemetry", {})

# Tabs Layout
tab_single, tab_batch, tab_analytics, tab_guide = st.tabs([
    "🎯 Single Container Triage",
    "📦 Batch Manifest Scoring",
    "📊 Model Analytics & Calibration",
    "📖 Operations Field Guide",
])

# ==============================================================================
# TAB 1: Single Container Triage
# ==============================================================================
with tab_single:
    st.markdown("#### Inbound Container Telemetry (Observed at Crane Discharge $t_{\\text{discharge}}$)")
    
    with st.form("single_prediction_form"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("##### 🚢 Vessel & Schedule")
            carrier = st.text_input("Carrier Prefix (ISO 6346)", value=str(preset_data.get("carrier_prefix", "MSCU"))[:4].upper())
            discharge_hour = st.slider("Discharge Hour (0–23)", 0, 23, int(preset_data.get("discharge_hour", 14)))
            discharge_dow = st.selectbox(
                "Discharge Day of Week",
                options=[0, 1, 2, 3, 4, 5, 6],
                index=int(preset_data.get("discharge_day_of_week", 4)),
                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
            )
            discharge_month = st.selectbox("Discharge Month", options=list(range(1, 13)), index=int(preset_data.get("discharge_month", 7)) - 1)
            is_weekend = 1 if discharge_dow >= 5 else 0

        with col2:
            st.markdown("##### 📦 Container Specifications")
            ctr_type = st.selectbox("Container Type", options=["DRY", "RFR", "OVD", "TANK", "FLT", "OT"], index=0 if preset_data.get("CTR_TYPE") not in ["RFR", "OVD"] else 1)
            ctr_size = st.selectbox("Length (feet)", options=[20, 40, 45], index=1 if preset_data.get("CTR_SIZE", 40) == 40 else 0)
            gross = st.number_input("Gross Cargo Weight (tonnes)", min_value=1.0, max_value=80.0, value=float(preset_data.get("GROSS", 26.5)), step=0.5)

        with col3:
            st.markdown("##### 🏗️ Terminal Velocity")
            trailing_vol = st.number_input("Trailing 7d Volume (Stays)", min_value=0.0, value=float(preset_data.get("trailing_7d_volume", 420.0)), step=10.0)
            trailing_crit = st.number_input("Trailing 7d Critical Stays", min_value=0.0, value=float(preset_data.get("trailing_7d_critical", 35.0)), step=5.0)
            trailing_risk = st.number_input("Trailing 7d At-Risk Stays", min_value=0.0, value=float(preset_data.get("trailing_7d_at_risk", 98.0)), step=5.0)

        with col4:
            st.markdown("##### ⏱️ Historical Congestion")
            delay_rate = st.slider("Trailing 7d Delay Rate", 0.0, 1.0, float(preset_data.get("historical_delay_rate_7d", 0.285)), step=0.01)
            mean_dwell = st.number_input("Trailing 7d Mean Dwell (hours)", min_value=10.0, value=float(preset_data.get("trailing_7d_mean_dwell", 88.5)), step=2.0)
            case_id = st.text_input("Container Stay Identifier", value=str(preset_data.get("case_id", "CTR-2021-DEMO-01")))

        submitted = st.form_submit_button("⚡ Score Container Risk", type="primary", use_container_width=True)

    if submitted or selected_preset != "Custom Input":
        input_dict = {
            "carrier_prefix": carrier,
            "CTR_TYPE": ctr_type,
            "CTR_SIZE": float(ctr_size),
            "GROSS": float(gross),
            "discharge_hour": int(discharge_hour),
            "discharge_day_of_week": int(discharge_dow),
            "discharge_month": int(discharge_month),
            "is_weekend": int(is_weekend),
            "trailing_7d_volume": float(trailing_vol),
            "trailing_7d_critical": float(trailing_crit),
            "trailing_7d_at_risk": float(trailing_risk),
            "historical_delay_rate_7d": float(delay_rate),
            "trailing_7d_mean_dwell": float(mean_dwell),
        }

        t0 = time.perf_counter()
        if bundle is not None:
            estimator = bundle["estimator"]
            preprocessor = bundle["preprocessor"]
            features = bundle["features"]

            df_input = pd.DataFrame([input_dict])[features]
            X_trans = preprocessor.transform(df_input)
            probs = estimator.predict_proba(X_trans)[0]
            p_on, p_risk, p_crit = probs[0], probs[1], probs[2]
        else:
            # Fallback heuristic
            p_crit = min(0.95, max(0.02, 0.08 + (0.15 if discharge_dow in (4, 5) else 0.0) + (0.12 if gross > 26.0 else 0.0) + (0.10 if ctr_type == "RFR" else 0.0)))
            p_risk = min(0.90, max(0.10, 0.28 + (0.10 if is_weekend else 0.0)))
            p_on = max(0.05, 1.0 - (p_crit + p_risk))
            s = p_on + p_risk + p_crit
            p_on, p_risk, p_crit = p_on/s, p_risk/s, p_crit/s

        latency_ms = (time.perf_counter() - t0) * 1000.0

        alert_triggered = bool(p_crit >= tau_selected)
        pred_class_id = 2 if alert_triggered else (1 if p_risk > p_on else 0)
        class_names = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
        pred_label = class_names[pred_class_id]

        st.markdown("---")
        st.markdown("### 📋 Prediction Results & Operational Directives")

        res_col1, res_col2, res_col3 = st.columns([1.2, 1.5, 1.3])

        with res_col1:
            st.markdown("##### Bottleneck Status")
            if pred_label == "ON_TIME":
                st.markdown('<span class="badge-ontime">🟢 ON_TIME (&le; 2.7 days)</span>', unsafe_allow_html=True)
            elif pred_label == "AT_RISK":
                st.markdown('<span class="badge-atrisk">🟡 AT_RISK (2.7 – 6.7 days)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-critical">🔴 CRITICAL DELAY (&gt; 6.7 days)</span>', unsafe_allow_html=True)

            if alert_triggered:
                st.warning(f"⚠️ **Intervention Alert Triggered!**\n$P(\\text{{CRITICAL}}) = {p_crit:.1%}$ exceeds threshold $\\tau = {tau_selected:.2f}$")
            else:
                st.info(f"✅ Normal clearance profile ($P(\\text{{CRITICAL}}) = {p_crit:.1%}$ below threshold $\\tau = {tau_selected:.2f}$)")

            st.caption(f"Inference Latency: **{latency_ms:.2f} ms** (SLA: &lt; 25ms)")

        with res_col2:
            st.markdown("##### Class Probability Distribution")
            st.progress(float(p_on), text=f"On-Time: {p_on:.1%}")
            st.progress(float(p_risk), text=f"At-Risk: {p_risk:.1%}")
            st.progress(float(p_crit), text=f"Critical Delay: {p_crit:.1%}")

            expected_dwell = (p_on * 48.0) + (p_risk * 110.0) + (p_crit * 220.0)
            st.markdown(f"**Expected Dwell Time:** `~{expected_dwell:.1f} hours` ({expected_dwell/24:.1f} days)")

        with res_col3:
            st.markdown("##### Economic Loss Modeling")
            expected_cost = (1.0 - p_crit) * (c_fa if alert_triggered else 0) + (p_crit * (0 if alert_triggered else c_fn))
            st.metric("Expected Container Risk Cost", f"${expected_cost:.2f}")
            st.caption(f"Based on $C_{{\\text{{FA}}}} = \\${c_fa}$ and $C_{{\\text{{FN}}}} = \\${c_fn}$")

        st.markdown(
            """
            <div class="sop-box">
                <strong>🚨 Recommended Logistics Playbook Action:</strong><br>
            """,
            unsafe_allow_html=True,
        )
        if alert_triggered or pred_label == "CRITICAL_DELAY":
            st.markdown("**Escalated Intervention & Priority Dispatch:** Alert supply chain planners; pre-clear documentation with customs broker; prioritize yard unstacking; notify receiver of potential schedule buffer.")
        elif pred_label == "AT_RISK":
            st.markdown("**Priority Gate Clearance:** Verify drayage trucker appointment slot; check for preliminary customs hold; queue for early-window yard retrieval.")
        else:
            st.markdown("**Nominal Monitoring:** Standard drayage appointment dispatch; standard demurrage free-time buffer intact.")
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 2: Batch Manifest Scoring
# ==============================================================================
with tab_batch:
    st.markdown("#### Manifest Batch Scoring & Vectorized Triage")
    st.write("Upload a CSV manifest of arriving containers or run the demonstration batch to rank bottlenecks.")

    demo_clicked = st.button("🚀 Load Sample 10-Container Manifest Batch", type="secondary")

    if demo_clicked and sample_cases:
        df_demo = pd.DataFrame([c["telemetry"] for c in sample_cases[:10]])
        st.dataframe(df_demo, use_container_width=True)

        if bundle is not None:
            estimator = bundle["estimator"]
            preprocessor = bundle["preprocessor"]
            features = bundle["features"]

            X_b = preprocessor.transform(df_demo[features])
            batch_probs = estimator.predict_proba(X_b)
            df_demo["P_ON_TIME"] = np.round(batch_probs[:, 0], 3)
            df_demo["P_AT_RISK"] = np.round(batch_probs[:, 1], 3)
            df_demo["P_CRITICAL"] = np.round(batch_probs[:, 2], 3)
            df_demo["ALERT_TRIGGERED"] = df_demo["P_CRITICAL"] >= tau_selected
            df_demo["ASSIGNED_TIER"] = np.where(df_demo["ALERT_TRIGGERED"], "CRITICAL_DELAY", np.where(df_demo["P_AT_RISK"] > df_demo["P_ON_TIME"], "AT_RISK", "ON_TIME"))

            st.markdown("##### Batch Scoring Results")
            st.dataframe(
                df_demo[["case_id", "carrier_prefix", "CTR_TYPE", "GROSS", "P_ON_TIME", "P_AT_RISK", "P_CRITICAL", "ALERT_TRIGGERED", "ASSIGNED_TIER"]],
                use_container_width=True,
            )

            flagged_cnt = df_demo["ALERT_TRIGGERED"].sum()
            st.success(f"Triage Complete: **{flagged_cnt} / {len(df_demo)}** containers flagged for operational intervention at threshold τ = {tau_selected:.2f}")


# ==============================================================================
# TAB 3: Model Analytics & Calibration
# ==============================================================================
with tab_analytics:
    st.markdown("#### Empirical Baseline Model Analytics & Provenance")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("##### Operational Alert Threshold Sensitivity Table")
        df_tradeoffs = load_threshold_tradeoffs()
        if df_tradeoffs is not None:
            st.dataframe(df_tradeoffs[["threshold_tau", "flagged_pct", "alerts_per_day", "precision", "recall", "f1_score"]], use_container_width=True)
            st.caption("Fitted on 20,879 held-out validation stays across 16 terminal operating days.")
        else:
            st.info("Threshold tradeoffs table not found.")

    with col_a2:
        st.markdown("##### Ranked Feature Importance Matrix (Arrival Horizon)")
        df_feat = load_feature_importance()
        if df_feat is not None:
            st.dataframe(df_feat[["importance_rank", "feature_name", "feature_domain", "val_perm_macro_f1_drop", "xgb_gain_importance"]], use_container_width=True)
        else:
            st.info("Feature importance matrix not found.")


# ==============================================================================
# TAB 4: Operations Field Guide
# ==============================================================================
with tab_guide:
    st.markdown("#### Logistics Operations Field Guide (1-Page Operational SOP)")
    st.markdown(
        """
        ### Executive Dispatch Directives
        1. **Do Not Automate Reroutes on Raw Probability Alone:** At standard default $\\tau = 0.50$, precision is 21.1% but recall is only 1.2%. At minimum-cost $\\tau = 0.22$, recall reaches 72.4% but flags 59.7% of all terminal traffic.
        2. **Implement the Top-1% Queue:** Rather than alerting 60% of arrivals, dispatch senior planners to review only the top 1% highest-scoring containers daily (~7 containers/day). This yields a **5.44x to 6.26x lift** over the baseline delay rate with zero dispatcher alarm fatigue.
        3. **Watch Weekend Discharges:** Containers arriving on Friday afternoon or Saturday experience a 2.3x higher rate of customs documentation stalls due to weekend broker closures.
        4. **Heavy Cargo Permitting:** Containers $\ge 28.5\\text{ tonnes}$ trigger multi-axle trailer requirements, resulting in drayage truck shortages.
        """
    )

