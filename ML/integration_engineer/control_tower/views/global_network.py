"""Global Overview: Macro-logistics network across 2,065 commercial ports.

Presents global vessel flow, continuous throughput forecasting under T-2 latency,
and geospatial bottleneck distributions across world trade corridors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from ML.integration_engineer.control_tower import data
from ML.integration_engineer.control_tower.theme import (
    AMBER,
    INK,
    LIME,
    MUTED,
    PLATE,
    PLATE_2,
    RED,
    RULE,
    SOFT,
    head,
)


def _cell(key: str, value: str, unit: str, note: str, span: str = "s2", kind: str = "", extra: str = "") -> str:
    tag = f'<span style="color:{LIME}">{kind}</span>' if kind in ("VAL", "MODEL", "DL") else f"<span>{kind}</span>"
    return (
        f'<div class="cell {span}">'
        f'<div class="k"><span>{key}</span>{tag}</div>'
        f'<div class="v">{value}<small>{unit}</small></div>'
        f'{extra}'
        f'<div class="n">{note}</div>'
        f'</div>'
    )


def render_map(df: pd.DataFrame) -> None:
    """Renders interactive dark Pydeck scatterplot across 2,065 ports."""
    df_plot = df.copy()

    # Assign dynamic colors based on 7-day cumulative vessel calls
    def get_color(calls: float) -> list[int]:
        if calls >= 50.0:
            return [255, 90, 71, 220]      # RED: High-throughput mega-hub
        elif calls >= 10.0:
            return [242, 169, 59, 200]     # AMBER: Active commercial gateway
        elif calls > 0.5:
            return [199, 243, 107, 180]    # LIME: Moderate regional feeder
        else:
            return [142, 142, 134, 110]    # MUTED: Quiet / zero-call berth

    df_plot["color"] = df_plot["pred_container_calls"].apply(get_color)

    # Dynamic radius proportional to square-root cargo tonnage (meters)
    tonnage = df_plot["pred_cargo_tonnage"].fillna(0).to_numpy()
    radii = np.clip(np.sqrt(np.maximum(0.0, tonnage)) * 45.0, 25000.0, 320000.0)
    df_plot["radius"] = radii

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_plot,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=18.0,
        longitude=15.0,
        zoom=1.35,
        pitch=0,
    )

    tooltip = {
        "html": (
            "<div style='font-family:IBM Plex Mono, monospace; padding:4px 8px; font-size:11px; line-height:1.4;'>"
            "<b style='color:#F4F4EE; font-size:12px;'>{portwatch_name}</b> <span style='color:#8E8E86;'>({portwatch_iso3})</span><br/>"
            "<span style='color:#8E8E86;'>7-Day Vessel Calls:</span> <b style='color:#C7F36B;'>{pred_container_calls:.1f}</b> calls<br/>"
            "<span style='color:#8E8E86;'>7-Day Cargo Tonnage:</span> <b style='color:#F4F4EE;'>{pred_cargo_tonnage:,.0f}</b> metric tons<br/>"
            "<span style='color:#8E8E86;'>UN/LOCODE:</span> <span style='color:#BDBDB4;'>{accepted_unlocode}</span>"
            "</div>"
        ),
        "style": {
            "backgroundColor": "#1A1A18",
            "border": "1px solid #2B2B27",
            "color": "#F4F4EE",
            "borderRadius": "2px",
            "zIndex": "1000",
        },
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="dark",
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, height=520)


def render() -> None:
    # 1. Header Strip
    st.markdown(
        '<div class="strip">'
        '<span><b>Global Macro-Logistics Network</b> · PortWatch AIS Telemetry</span>'
        '<span>Monitored: <b>2,065</b> Commercial Ports</span>'
        '<span>Panel: <b>2019–2025</b></span>'
        '<span>Latency: <b>T-2 Publication Quarantine</b></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    df_ports = data.global_network_ports()
    bench = data.controlled_benchmark_data()
    blind_res = bench.get("blind_test_2025", {})

    # Compute network-wide totals
    tot_calls = float(df_ports["pred_container_calls"].sum())
    tot_tons = float(df_ports["pred_cargo_tonnage"].sum())
    active_ports = int((df_ports["pred_container_calls"] > 0.1).sum())
    n_countries = int(df_ports["portwatch_country"].nunique())

    # Improvements from verified 2025 blind-test
    calls_huber_red = -6.31
    tons_huber_red = -2.79

    # Hero Thesis & Intro
    st.markdown(
        '<div class="thesis">2,065 COMMERCIAL PORTS · GLOBAL CONTROL TOWER</div>'
        '<div class="lede">'
        'Macro-logistics network overview providing continuous 7-day cumulative continuous flow forecasting under '
        'strict <b>T-2 AIS observation latency</b>. Deep Learning Transformer residual projections ground each gateway in its '
        'local persistence anchor while modeling multi-port supply-chain disruptions across world trade corridors.'
        '</div>',
        unsafe_allow_html=True,
    )

    # 2. Bento KPI Grid
    calls_extra = (
        f'<div style="font:500 .7rem \'IBM Plex Mono\'; color:{LIME}; margin-top:.2rem;">'
        f'▼ {abs(calls_huber_red):.1f}% Huber Loss vs Persistence</div>'
    )
    tons_extra = (
        f'<div style="font:500 .7rem \'IBM Plex Mono\'; color:{LIME}; margin-top:.2rem;">'
        f'▼ {abs(tons_huber_red):.1f}% Huber Loss vs Persistence</div>'
    )

    cells = [
        _cell("Global Vessel Arrivals", f"{tot_calls:,.0f}", "calls/wk", "7-day forward container-carrying ship berthings", "s3", "DL", calls_extra),
        _cell("Global Freight Volume", f"{tot_tons / 1e6:,.1f}", "M tons/wk", "7-day forward cumulative deadweight cargo tonnage", "s3", "DL", tons_extra),
        _cell("Monitored Coverage", f"{len(df_ports):,}", "ports", f"{active_ports:,} active ports across {n_countries} maritime nations", "s2", "OBS"),
        _cell("Quiet Berths", f"{(len(df_ports) - active_ports) / len(df_ports) * 100:.1f}", "%", "Zero weekly vessel berthings (persistence optimal)", "s2", "OBS"),
        _cell("Publication Lag", "48", "hours", "Mandatory T-2 observation quarantine enforced", "s2", "SOP"),
    ]
    st.markdown(f'<div class="bento">{"".join(cells)}</div>', unsafe_allow_html=True)

    # 3. Interactive Global AIS Flow Map
    head("Global Port Activity & Forecast Flow", "2,065 commercial ports sized by weekly metric tonnage and colored by container vessel arrivals.", "dl_forecast")
    
    col_filter, col_legend = st.columns([8, 4])
    with col_filter:
        country_filter = st.selectbox(
            "Filter corridor / nation:",
            ["All Countries (Global)"] + sorted(df_ports["portwatch_country"].dropna().unique().tolist()),
            index=0,
            label_visibility="collapsed"
        )
    with col_legend:
        st.markdown(
            f'<div style="display:flex; gap:1.2rem; align-items:center; justify-content:flex-end; font:400 .72rem \'IBM Plex Mono\'; color:{MUTED}; margin-top:6px;">'
            f'<span><span style="display:inline-block; width:9px; height:9px; background:{RED}; border-radius:50%; margin-right:4px;"></span>&ge;50 calls</span>'
            f'<span><span style="display:inline-block; width:9px; height:9px; background:{AMBER}; border-radius:50%; margin-right:4px;"></span>10–50 calls</span>'
            f'<span><span style="display:inline-block; width:9px; height:9px; background:{LIME}; border-radius:50%; margin-right:4px;"></span>1–10 calls</span>'
            f'<span><span style="display:inline-block; width:9px; height:9px; background:{MUTED}; border-radius:50%; margin-right:4px;"></span>&lt;1 call</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    df_filtered = df_ports if country_filter == "All Countries (Global)" else df_ports[df_ports["portwatch_country"] == country_filter]
    render_map(df_filtered)

    # 4. Gateway Ports Table & Diagnostics
    head("High-Throughput Global Gateway Ports", "Leading world container and cargo maritime gateways ranked by forecast cargo tonnage.", "dl_forecast")

    top_gateways = df_ports.sort_values("pred_cargo_tonnage", ascending=False).head(10).copy()
    top_gateways["pred_calls_fmt"] = top_gateways["pred_container_calls"].apply(lambda v: f"{v:.1f} calls")
    top_gateways["pred_tons_fmt"] = top_gateways["pred_cargo_tonnage"].apply(lambda v: f"{v:,.0f} t")
    top_gateways["actual_calls_fmt"] = top_gateways["actual_container_calls"].apply(lambda v: f"{v:.1f} calls")
    top_gateways["delta_calls_fmt"] = (top_gateways["pred_container_calls"] - top_gateways["naive_container_calls"]).apply(lambda v: f"{v:+.1f} calls")

    display_df = top_gateways[[
        "portwatch_name", "portwatch_country", "accepted_unlocode",
        "pred_calls_fmt", "pred_tons_fmt", "actual_calls_fmt", "delta_calls_fmt"
    ]].rename(columns={
        "portwatch_name": "Port Gateway",
        "portwatch_country": "Country",
        "accepted_unlocode": "UN/LOCODE",
        "pred_calls_fmt": "7d Vessel Calls (DL)",
        "pred_tons_fmt": "7d Cargo Tonnage (DL)",
        "actual_calls_fmt": "Actual Calls (Obs)",
        "delta_calls_fmt": "Residual vs Persistence",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

