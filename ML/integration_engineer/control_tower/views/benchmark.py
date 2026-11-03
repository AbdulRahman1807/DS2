"""Model Capabilities & Benchmark: Machine Learning vs Deep Learning.

Presents:
1. What each stage predicts: granularity, output format, and operational role.
2. Controlled sequence benchmark (Historical Baseline vs. GBDT vs. Neural Transformer).
3. Plain English summaries and clean, professional formatting with generous breathing room.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ML.integration_engineer.control_tower import data
from ML.integration_engineer.control_tower.theme import (
    AMBER,
    LIME,
    MUTED,
    PAPER,
    PLATE,
    RULE,
    SOFT,
)


def render() -> None:
    # Page Header
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
          <h1 style="font-size: 2.2rem; font-weight: 700; color: #F4F4EE; margin: 0 0 0.35rem;">
            Model Capabilities & Benchmark
          </h1>
          <p style="font-size: 0.95rem; color: #BDBDB4; margin: 0; line-height: 1.5;">
            Comparing Stage 1 container-level machine learning with Stage 2 global port forecasting.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================================================
    # 1. What Each Stage Predicts: Granularity, Output Format & Purpose
    # ==========================================================================
    st.markdown(
        """
        <div style="margin-top: 0.8rem; margin-bottom: 0.5rem;">
          <h2 style="font-size: 1.35rem; font-weight: 600; color: #F4F4EE; margin: 0 0 0.2rem;">
            1. Operational Scope: Stage 1 (ML) vs. Stage 2 (DL)
          </h2>
          <p style="font-size: 0.9rem; color: #8E8E86; margin: 0;">
            How each model serves a distinct operational purpose in the supply chain.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(
            f"""
            <div style="background: {PLATE}; border: 1px solid {RULE}; padding: 1.2rem 1.3rem; border-radius: 4px; height: 100%;">
              <div style="font-size: 0.82rem; font-weight: 600; color: {LIME}; text-transform: uppercase; letter-spacing: 0.04em;">
                Stage 1 · Machine Learning
              </div>
              <div style="font-size: 1.5rem; font-weight: 700; color: {PAPER}; margin: 0.3rem 0 0.5rem;">
                Container Delay Risk Triage
              </div>
              <div style="font-size: 0.9rem; line-height: 1.5; color: {SOFT}; margin-bottom: 0.8rem;">
                <b>Operational Purpose:</b> Identifies individual containers at risk of customs holds or terminal yard congestion, helping dispatchers expedite high-risk cargo and avoid demurrage penalties.
              </div>
              <div style="border-top: 1px solid {RULE}; padding-top: 0.75rem; font-size: 0.86rem; line-height: 1.6; color: {MUTED};">
                <div><b style="color: {PAPER};">What It Predicts:</b> Container dwell delay tier (On Time, At Risk, Critical Delay)</div>
                <div><b style="color: {PAPER};">Operational Level:</b> Single container shipment at terminal yard (Surabaya, Indonesia)</div>
                <div><b style="color: {PAPER};">Output Format:</b> 3 risk categories with percentage probabilities and key risk factors</div>
                <div><b style="color: {PAPER};">Input Data:</b> Container specifications, shipping carrier, gross weight, and terminal status</div>
                <div><b style="color: {PAPER};">Primary User:</b> Yard superintendents, freight dispatchers, and cargo owners</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="background: {PLATE}; border: 1px solid {RULE}; padding: 1.2rem 1.3rem; border-radius: 4px; height: 100%;">
              <div style="font-size: 0.82rem; font-weight: 600; color: {LIME}; text-transform: uppercase; letter-spacing: 0.04em;">
                Stage 2 · Deep Learning
              </div>
              <div style="font-size: 1.5rem; font-weight: 700; color: {PAPER}; margin: 0.3rem 0 0.5rem;">
                Global Port Throughput & Dwell
              </div>
              <div style="font-size: 0.9rem; line-height: 1.5; color: {SOFT}; margin-bottom: 0.8rem;">
                <b>Operational Purpose:</b> Forecasts network-wide port volume and vessel congestion 7 days in advance, allowing port authorities to schedule berths and shipping lines to optimize routes.
              </div>
              <div style="border-top: 1px solid {RULE}; padding-top: 0.75rem; font-size: 0.86rem; line-height: 1.6; color: {MUTED};">
                <div><b style="color: {PAPER};">What It Predicts:</b> 7-day vessel arrivals, 7-day cargo tonnage, and vessel dwell time</div>
                <div><b style="color: {PAPER};">Operational Level:</b> Entire commercial port across 2,065 ports globally</div>
                <div><b style="color: {PAPER};">Output Format:</b> Numeric forecast of vessel arrivals (calls) and freight tonnage</div>
                <div><b style="color: {PAPER};">Input Data:</b> 30-day historical sequence of AIS vessel tracking and ocean conditions</div>
                <div><b style="color: {PAPER};">Primary User:</b> Port authorities, ocean carriers, and global supply chain planners</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Comparison Table
    st.markdown(
        """
        <div style="margin-top: 1.6rem; margin-bottom: 0.4rem;">
          <h3 style="font-size: 1.15rem; font-weight: 600; color: #F4F4EE; margin: 0 0 0.2rem;">
            Feature Comparison Matrix
          </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    spec_matrix = pd.DataFrame([
        {
            "Comparison Point": "Operational Focus",
            "Stage 1: Machine Learning": "Individual container yard clearance and gate dispatch",
            "Stage 2: Deep Learning": "Port-wide vessel traffic, cargo flow, and berth planning",
        },
        {
            "Comparison Point": "What It Predicts",
            "Stage 1: Machine Learning": "Container delay risk category (On Time, At Risk, Critical)",
            "Stage 2: Deep Learning": "7-day total vessel arrivals, cargo tonnage, and vessel dwell",
        },
        {
            "Comparison Point": "Operating Scope",
            "Stage 1: Machine Learning": "Terminal container yard (Surabaya, Indonesia)",
            "Stage 2: Deep Learning": "Global network across 2,065 commercial ports",
        },
        {
            "Comparison Point": "Prediction Window",
            "Stage 1: Machine Learning": "Current container stay (hours to days)",
            "Stage 2: Deep Learning": "Upcoming 7 days",
        },
        {
            "Comparison Point": "Input Information",
            "Stage 1: Machine Learning": "Container details (carrier, weight, length, customs flags)",
            "Stage 2: Deep Learning": "Recent 30-day port activity and ocean weather",
        },
        {
            "Comparison Point": "Primary User",
            "Stage 1: Machine Learning": "Yard planners, freight forwarders, and dispatchers",
            "Stage 2: Deep Learning": "Port operators, shipping lines, and supply chain managers",
        },
    ])
    st.dataframe(spec_matrix, use_container_width=True, hide_index=True)

    # ==========================================================================
    # 2. Controlled Sequence Benchmark
    # ==========================================================================
    st.markdown(
        """
        <div style="margin-top: 2rem; margin-bottom: 0.4rem;">
          <h2 style="font-size: 1.35rem; font-weight: 600; color: #F4F4EE; margin: 0 0 0.2rem;">
            2. Controlled Sequence Benchmark: Macro Throughput Forecasting
          </h2>
          <p style="font-size: 0.9rem; color: #8E8E86; margin: 0;">
            Independent evaluation on 741,335 holdout sequences across 2,065 ports under realistic 2-day operational latency.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bench_data = data.controlled_benchmark_data()
    blind = bench_data.get("blind_test_2025", {})
    calls_p = blind.get("container_calls", {})
    tons_p = blind.get("cargo_tonnage", {})

    # Table A: Container Vessel Arrivals
    st.markdown(
        """
        <div style="font-weight: 600; font-size: 0.95rem; color: #F4F4EE; margin: 1rem 0 0.35rem;">
          A. Vessel Arrivals Forecast (Container Ship Berthings over 7 Days)
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_c = calls_p.get("persistence", {})
    g_c = calls_p.get("gbdt", {})
    d_c = calls_p.get("dl_transformer", {})

    df_c_display = pd.DataFrame([
        {
            "Model": "Historical Persistence Baseline",
            "Method": "Trailing 7-Day Average",
            "Average Error (MAE)": f"{p_c.get('mae', 0):.2f} calls",
            "Accuracy Score (R²)": f"{p_c.get('r2', 0):.2f}",
            "Operational Result": "Simple lookback anchor; misses dynamic congestion shifts.",
        },
        {
            "Model": "Gradient Boosted Trees (GBDT)",
            "Method": "Tabular Machine Learning",
            "Average Error (MAE)": f"★ {g_c.get('mae', 0):.2f} calls",
            "Accuracy Score (R²)": f"★ {g_c.get('r2', 0):.2f}",
            "Operational Result": "Lowest overall error: 11.5% improvement over historical baseline.",
        },
        {
            "Model": "Residual Transformer",
            "Method": "Deep Neural Network",
            "Average Error (MAE)": f"{d_c.get('mae', 0):.2f} calls",
            "Accuracy Score (R²)": f"{d_c.get('r2', 0):.2f}",
            "Operational Result": "Robust trend capture across ports: 6.3% lower loss than baseline.",
        },
    ])
    st.dataframe(df_c_display, use_container_width=True, hide_index=True)

    # Table B: Cargo Freight Tonnage
    st.markdown(
        """
        <div style="font-weight: 600; font-size: 0.95rem; color: #F4F4EE; margin: 1.3rem 0 0.35rem;">
          B. Cargo Freight Volume Forecast (Metric Tons over 7 Days)
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_t = tons_p.get("persistence", {})
    g_t = tons_p.get("gbdt", {})
    d_t = tons_p.get("dl_transformer", {})

    df_t_display = pd.DataFrame([
        {
            "Model": "Historical Persistence Baseline",
            "Method": "Trailing 7-Day Average",
            "Average Error (MAE)": f"{p_t.get('mae', 0):,.0f} tons",
            "Accuracy Score (R²)": f"{p_t.get('r2', 0):.2f}",
            "Operational Result": "Static lookback anchor; vulnerable to volume surges.",
        },
        {
            "Model": "Gradient Boosted Trees (GBDT)",
            "Method": "Tabular Machine Learning",
            "Average Error (MAE)": f"★ {g_t.get('mae', 0):,.0f} tons",
            "Accuracy Score (R²)": f"★ {g_t.get('r2', 0):.2f}",
            "Operational Result": "Lowest overall error: 10.2% improvement over historical baseline.",
        },
        {
            "Model": "Residual Transformer",
            "Method": "Deep Neural Network",
            "Average Error (MAE)": f"{d_t.get('mae', 0):,.0f} tons",
            "Accuracy Score (R²)": f"{d_t.get('r2', 0):.2f}",
            "Operational Result": "Reliable global forecasting: captures long-range seasonal patterns.",
        },
    ])
    st.dataframe(df_t_display, use_container_width=True, hide_index=True)
