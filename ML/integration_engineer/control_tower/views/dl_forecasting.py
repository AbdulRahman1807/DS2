"""DL Forecasting: Operational Port Dwell Time & 7-Day Throughput Forecaster.

Provides:
1. Expected vessel dwell times and simple interval table (1 hour, 1 day, 3 days, 7 days).
2. 7-day cumulative port throughput forecast (vessel calls and cargo tonnage).
3. Clean, professional presentation with generous spacing and plain English recommendations.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ML.integration_engineer.control_tower import dl_service
from ML.integration_engineer.control_tower.theme import (
    INK,
    LIME,
    MUTED,
    PAPER,
    PLATE,
    PLOTLY_CONFIG,
    RULE,
    SOFT,
    plotly_layout,
)


def build_dwell_table(w_pred: float) -> pd.DataFrame:
    """Builds a clear, simple dwell interval table for 1 hour, 1 day, 3 days, and 7 days."""
    w = max(1.0, float(w_pred))
    
    # Operational intervals and dwell estimates
    h1_dwell = max(1.0, round(w * 0.96, 1))
    d1_dwell = max(1.0, round(w * 0.98, 1))
    d3_dwell = max(1.0, round(w * 1.02, 1))
    d7_dwell = round(w, 1)

    if w <= 8.0:
        c_1h, r_1h = "Fast queue", "Immediate berth access likely. Clear pilot on arrival."
        c_1d, r_1d = "Steady", "Smooth turnover. Maintain standard yard equipment staging."
        c_3d, r_3d = "Fluid", "Normal volume expected. Standard quayside gang allocation."
        c_7d, r_7d = "Optimal", "Port operates smoothly across the full weekly cycle."
    elif w <= 18.0:
        c_1h, r_1h = "Normal queue", "Standard harbor queue. Follow normal pilotage schedule."
        c_1d, r_1d = "Moderate", "Pace crane crew handovers between shifts to minimize idle time."
        c_3d, r_3d = "Busy", "Review weekend berth window reservations for container vessels."
        c_7d, r_7d = "Standard", "Maintain standard weekly scheduling buffers for arriving carriers."
    else:
        c_1h, r_1h = "Heavy queue", "Delays likely. Alert harbor master and drayage dispatchers."
        c_1d, r_1d = "Congested", "Prioritize clearing departed vessels to open quay space."
        c_3d, r_3d = "High delay", "Authorize additional crane shifts to prevent vessel bunching."
        c_7d, r_7d = "Congested", "Advise shipping lines to adjust cruising speed to stagger arrivals."

    return pd.DataFrame([
        {
            "Forecast Horizon": "Next 1 Hour",
            "Likely Dwell Time": f"{h1_dwell:.1f} hours",
            "Port Condition": c_1h,
            "Operational Guidance": r_1h,
        },
        {
            "Forecast Horizon": "Next 1 Day (24 h)",
            "Likely Dwell Time": f"{d1_dwell:.1f} hours",
            "Port Condition": c_1d,
            "Operational Guidance": r_1d,
        },
        {
            "Forecast Horizon": "Next 3 Days (72 h)",
            "Likely Dwell Time": f"{d3_dwell:.1f} hours",
            "Port Condition": c_3d,
            "Operational Guidance": r_3d,
        },
        {
            "Forecast Horizon": "Next 7 Days (1 Week)",
            "Likely Dwell Time": f"{d7_dwell:.1f} hours",
            "Port Condition": c_7d,
            "Operational Guidance": r_7d,
        },
    ])


def build_comparison_figure(pred_res: Dict[str, float]) -> Optional[go.Figure]:
    """Clean, simple horizontal bar chart comparing expected throughput to typical weekly activity."""
    has_c = (pred_res["pred_container_calls"] > 0 or pred_res["naive_container_calls"] > 0)
    has_t = (pred_res["pred_cargo_tonnage"] > 0 or pred_res["naive_cargo_tonnage"] > 0)

    if not has_c and not has_t:
        return None

    categories = ["Typical Weekly Level", "Expected Next 7 Days"]

    if has_c and has_t:
        fig = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=False,
            horizontal_spacing=0.12,
            subplot_titles=[
                "Vessel Arrivals (Calls)",
                "Cargo Volume (Metric Tons)",
            ],
        )

        c_vals = [pred_res["naive_container_calls"], pred_res["pred_container_calls"]]
        fig.add_trace(
            go.Bar(
                y=categories,
                x=c_vals,
                orientation="h",
                marker=dict(color=[MUTED, LIME]),
                text=[f"{v:.1f}" for v in c_vals],
                textposition="auto",
                textfont=dict(size=12, color=INK),
                hovertemplate="%{y}: %{x:.1f} calls<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )

        t_vals = [pred_res["naive_cargo_tonnage"], pred_res["pred_cargo_tonnage"]]
        fig.add_trace(
            go.Bar(
                y=categories,
                x=t_vals,
                orientation="h",
                marker=dict(color=[MUTED, LIME]),
                text=[f"{v:,.0f} t" for v in t_vals],
                textposition="auto",
                textfont=dict(size=12, color=INK),
                hovertemplate="%{y}: %{x:,.0f} tons<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

        fig = plotly_layout(fig, height=190, margin=dict(l=150, r=20, t=36, b=24))
        fig.update_xaxes(showgrid=True, gridcolor=RULE, row=1, col=1)
        fig.update_xaxes(showgrid=True, gridcolor=RULE, row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        return fig

    if has_t:
        fig = go.Figure()
        t_vals = [pred_res["naive_cargo_tonnage"], pred_res["pred_cargo_tonnage"]]
        fig.add_trace(
            go.Bar(
                y=categories,
                x=t_vals,
                orientation="h",
                marker=dict(color=[MUTED, LIME]),
                text=[f"{v:,.0f} t" for v in t_vals],
                textposition="auto",
                textfont=dict(size=12, color=INK),
                hovertemplate="%{y}: %{x:,.0f} tons<extra></extra>",
                showlegend=False,
            )
        )
        fig = plotly_layout(fig, height=170, margin=dict(l=150, r=20, t=36, b=24))
        fig.update_xaxes(title_text="Cargo Volume (Metric Tons)", showgrid=True, gridcolor=RULE)
        fig.update_yaxes(autorange="reversed")
        return fig

    fig = go.Figure()
    c_vals = [pred_res["naive_container_calls"], pred_res["pred_container_calls"]]
    fig.add_trace(
        go.Bar(
            y=categories,
            x=c_vals,
            orientation="h",
            marker=dict(color=[MUTED, LIME]),
            text=[f"{v:.1f}" for v in c_vals],
            textposition="auto",
            textfont=dict(size=12, color=INK),
            hovertemplate="%{y}: %{x:.1f} calls<extra></extra>",
            showlegend=False,
        )
    )
    fig = plotly_layout(fig, height=170, margin=dict(l=150, r=20, t=36, b=24))
    fig.update_xaxes(title_text="Vessel Arrivals (Calls)", showgrid=True, gridcolor=RULE)
    fig.update_yaxes(autorange="reversed")
    return fig


def render() -> None:
    # Page Title & Introduction
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
          <h1 style="font-size: 2.2rem; font-weight: 700; color: #F4F4EE; margin: 0 0 0.35rem;">
            Port Dwell & Volume Forecast
          </h1>
          <p style="font-size: 0.95rem; color: #BDBDB4; margin: 0; line-height: 1.5;">
            Expected vessel dwell times and 7-day throughput for commercial ports worldwide.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================================================
    # Forecast Controls
    # ==========================================================================
    hierarchy = dl_service.get_country_port_hierarchy()
    country_list = sorted(hierarchy.keys())
    default_country_idx = country_list.index("Singapore") if "Singapore" in country_list else 0

    now_dt = datetime.datetime.now()

    c1, c2, c3, c4 = st.columns([3, 4, 2.5, 2.5], gap="medium")
    with c1:
        selected_country = st.selectbox(
            "Country",
            options=country_list,
            index=default_country_idx,
            key="dl_country_select",
        )

    available_ports = hierarchy.get(selected_country, [])
    port_labels = [p["label"] for p in available_ports]
    port_id_map = {p["label"]: p["id"] for p in available_ports}
    port_name_map = {p["label"]: p["name"] for p in available_ports}
    port_iso3_map = {p["label"]: p["iso3"] for p in available_ports}

    default_port_idx = 0
    if selected_country == "Singapore":
        for i, lbl in enumerate(port_labels):
            if "Singapore (" in lbl:
                default_port_idx = i
                break

    with c2:
        selected_port_label = st.selectbox(
            "Port",
            options=port_labels,
            index=default_port_idx if port_labels else 0,
            key="dl_port_select",
        )

    with c3:
        forecast_date = st.date_input(
            "Forecast Date",
            value=now_dt.date(),
            key="dl_date_input",
        )

    with c4:
        st.write("")
        st.write("")
        run_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

    if not port_labels:
        st.warning("No commercial ports found for the selected country.")
        return

    selected_port_id = port_id_map[selected_port_label]
    selected_port_name = port_name_map[selected_port_label]
    selected_port_iso3 = port_iso3_map[selected_port_label]

    # Explicit button submission
    if run_btn:
        with st.spinner(f"Forecasting activity for {selected_port_name}..."):
            try:
                seq_tensor, obs_start, obs_end, mode = dl_service.build_sequence_tensor(
                    selected_port_id, forecast_date
                )
                pred_res = dl_service.forecast_port_throughput(seq_tensor)
                dwell_info = dl_service.get_dwell_prediction_for_port(selected_port_id)
                meta = dl_service.get_operational_forecast_metadata(forecast_date)

                st.session_state["submitted_forecast"] = {
                    "pred_res": pred_res,
                    "dwell_info": dwell_info,
                    "meta": meta,
                    "port_id": selected_port_id,
                    "port_name": selected_port_name,
                    "port_iso3": selected_port_iso3,
                    "date_str": str(forecast_date),
                }
            except Exception as e:
                st.error(f"Forecast generation failed: {e}")
                return

    # Check for submitted forecast
    submitted = st.session_state.get("submitted_forecast")

    if submitted is None:
        st.markdown(
            f"""
            <div style="border: 1px dashed {RULE}; background: {PLATE}; border-radius: 4px; padding: 2.5rem 1rem; text-align: center; margin-top: 1.2rem;">
              <div style="font-size: 1.15rem; font-weight: 600; color: {PAPER};">
                Ready to Forecast
              </div>
              <div style="color: {MUTED}; font-size: 0.9rem; margin-top: 0.35rem;">
                Select a port and date above, then click <b>Generate Forecast</b>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # If the user changed the selection after generating, show a gentle note
    is_editing = (
        submitted["port_id"] != selected_port_id
        or submitted["date_str"] != str(forecast_date)
    )
    if is_editing:
        st.markdown(
            f"""
            <div style="background: #1A1A18; border-left: 3px solid #F2A93B; padding: 0.6rem 0.9rem; border-radius: 2px; font-size: 0.85rem; color: #F4F4EE; margin: 0.5rem 0 1rem;">
              Showing current forecast for <b>{submitted['port_name']} ({submitted['port_iso3']})</b>. 
              Click <b>Generate Forecast</b> to update for your new selection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    meta = submitted["meta"]
    dwell_info = submitted["dwell_info"]
    pred_res = submitted["pred_res"]
    p_name = submitted["port_name"]
    p_iso = submitted["port_iso3"]
    w_pred = dwell_info["pred_hours"]
    w_naive = dwell_info["naive_hours"]
    w_delta = dwell_info["delta_hours"]

    # ==========================================================================
    # Clean Forecast Context Strip
    # ==========================================================================
    fallback_note = ""
    if meta.get("is_fallback", False):
        fallback_note = f' <span style="color:#F2A93B; font-size:0.84rem;">(Using latest verified baseline records through {meta["obs_end"]})</span>'

    st.markdown(
        f"""
        <div style="background: #141413; border: 1px solid {RULE}; border-radius: 4px; padding: 0.85rem 1.2rem; margin: 0.8rem 0 1.2rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.6rem;">
          <div>
            <span style="color: #8E8E86; font-size: 0.9rem;">Selected Port:</span>
            <b style="color: #F4F4EE; font-size: 1.05rem; margin-left: 0.3rem;">{p_name} ({p_iso})</b>
            {fallback_note}
          </div>
          <div style="font-size: 0.9rem; color: #BDBDB4;">
            Forecast Window: <b style="color: {LIME};">{meta['forecast_start']} to {meta['forecast_end']}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================================================
    # 1. Expected Vessel Dwell Time & Simple Interval Table
    # ==========================================================================
    st.markdown(
        """
        <div style="margin-top: 1rem; margin-bottom: 0.4rem;">
          <h2 style="font-size: 1.35rem; font-weight: 600; color: #F4F4EE; margin: 0 0 0.2rem;">
            1. Expected Vessel Dwell Time
          </h2>
          <p style="font-size: 0.9rem; color: #8E8E86; margin: 0;">
            Estimated time vessels spend waiting and at berth for this port.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Dwell Metric Card
    delta_str = f"{w_delta:+.1f} h vs typical" if abs(w_delta) > 0.05 else "matches typical baseline"
    st.markdown(
        f"""
        <div style="background: #141413; border: 1px solid {RULE}; border-left: 4px solid {LIME}; border-radius: 4px; padding: 1.1rem 1.3rem; margin-bottom: 1rem;">
          <div style="font-size: 0.82rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; color: #8E8E86; margin-bottom: 0.3rem;">
            Average Expected Vessel Dwell
          </div>
          <div style="display: flex; align-items: baseline; gap: 0.6rem;">
            <span style="font-size: 2.6rem; font-weight: 700; color: {LIME}; line-height: 1;">
              {w_pred:.1f}
            </span>
            <span style="font-size: 1.1rem; color: #BDBDB4; font-weight: 500;">hours</span>
          </div>
          <div style="font-size: 0.88rem; color: #BDBDB4; margin-top: 0.45rem;">
            Typical baseline: <b>{w_naive:.1f} hours</b> · <span>{delta_str}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # The Simple Interval Table (1 hour, 1 day, 3 days, 7 days)
    df_intervals = build_dwell_table(w_pred)
    st.dataframe(
        df_intervals,
        use_container_width=True,
        hide_index=True,
    )

    # ==========================================================================
    # 2. Seven-Day Port Throughput Forecast
    # ==========================================================================
    st.markdown(
        """
        <div style="margin-top: 1.8rem; margin-bottom: 0.4rem;">
          <h2 style="font-size: 1.35rem; font-weight: 600; color: #F4F4EE; margin: 0 0 0.2rem;">
            2. Seven-Day Port Throughput Forecast
          </h2>
          <p style="font-size: 0.9rem; color: #8E8E86; margin: 0;">
            Projected total vessel arrivals and cargo volume over the upcoming 7 days.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pred_c = pred_res["pred_container_calls"]
    naive_c = pred_res["naive_container_calls"]
    pred_t = pred_res["pred_cargo_tonnage"]
    naive_t = pred_res["naive_cargo_tonnage"]

    t_col1, t_col2 = st.columns(2, gap="medium")
    with t_col1:
        st.markdown(
            f"""
            <div style="background: #141413; border: 1px solid {RULE}; border-radius: 4px; padding: 1.1rem 1.3rem;">
              <div style="font-size: 0.82rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; color: #8E8E86; margin-bottom: 0.3rem;">
                7-Day Vessel Arrivals
              </div>
              <div style="display: flex; align-items: baseline; gap: 0.5rem;">
                <span style="font-size: 2.2rem; font-weight: 700; color: #F4F4EE; line-height: 1;">
                  {pred_c:.1f}
                </span>
                <span style="font-size: 0.95rem; color: #BDBDB4;">vessel calls</span>
              </div>
              <div style="font-size: 0.86rem; color: #8E8E86; margin-top: 0.45rem;">
                Typical weekly arrivals: <b style="color: #F4F4EE;">{naive_c:.1f} calls</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_col2:
        st.markdown(
            f"""
            <div style="background: #141413; border: 1px solid {RULE}; border-radius: 4px; padding: 1.1rem 1.3rem;">
              <div style="font-size: 0.82rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; color: #8E8E86; margin-bottom: 0.3rem;">
                7-Day Cargo Volume
              </div>
              <div style="display: flex; align-items: baseline; gap: 0.5rem;">
                <span style="font-size: 2.2rem; font-weight: 700; color: #F4F4EE; line-height: 1;">
                  {pred_t:,.0f}
                </span>
                <span style="font-size: 0.95rem; color: #BDBDB4;">metric tons</span>
              </div>
              <div style="font-size: 0.86rem; color: #8E8E86; margin-top: 0.45rem;">
                Typical weekly volume: <b style="color: #F4F4EE;">{naive_t:,.0f} tons</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Throughput Comparison Chart
    fig = build_comparison_figure(pred_res)
    if fig is not None:
        st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
        st.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)
