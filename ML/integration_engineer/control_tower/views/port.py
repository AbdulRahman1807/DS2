"""Port intelligence: where and when pressure concentrated at the terminal."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ML.integration_engineer.control_tower import data, visuals
from ML.integration_engineer.control_tower.theme import AMBER, INK, LIME, MUTED, PAPER, PLOTLY_CONFIG, RED, RULE, SOFT, head, plotly_layout


def constellation(vessels: list) -> go.Figure:
    v = pd.DataFrame(vessels)
    v["t"] = pd.to_datetime(v["t"])
    fig = go.Figure(go.Scatter(
        x=v["t"], y=v["median"], mode="markers",
        marker=dict(size=(v["n"] ** .5) * .95, sizemin=4, color=v["crit"], cmin=0, cmax=.45,
                    colorscale=[[0, "#2c2c28"], [.25, "#6b5a2a"], [.5, AMBER], [1, RED]], line=dict(color=INK, width=1), opacity=.95,
                    colorbar=dict(title=dict(text="Critical share", font=dict(size=10)), thickness=8, len=.7, outlinewidth=0, tickformat=".0%", tickfont=dict(size=10))),
        customdata=v[["n", "crit", "ontime", "precleared"]],
        hovertemplate="Berthed %{x|%d %b %H:%M}<br>%{customdata[0]:,} containers<br>Median dwell %{y:.0f} h<br>%{customdata[1]:.0%} critical · %{customdata[2]:.0%} on time<br>%{customdata[3]:.0%} customs-cleared before discharge<extra></extra>",
    ))
    for level, c, lab in [(65.11, AMBER, "65 h"), (160.9, RED, "161 h")]:
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=level, y1=level, line=dict(color=c, width=1, dash="dot"))
        fig.add_annotation(xref="paper", x=1, y=math.log10(level), text=lab, showarrow=False, font=dict(color=c, size=10), xanchor="right", yanchor="bottom")
    fig = plotly_layout(fig, height=400)
    fig.update_yaxes(type="log", title_text="Median dwell of the call (h, log)", tickvals=[24, 48, 96, 192, 384], range=[1.1, 2.75], autorange=False)
    fig.update_xaxes(tickformat="%d %b")
    return fig


def survival_figure(p: dict) -> go.Figure:
    fig = go.Figure()
    colors = {"Customs cleared a day before discharge": LIME, "Quarantine flag known at discharge": PAPER, "No customs clearance or flag yet": RED}
    for name, s in p["survival"].items():
        fig.add_trace(go.Scatter(x=p["survival_grid"], y=s["s"], mode="lines", name=f'{name} ({s["n"]:,})', line=dict(color=colors[name], width=2),
                                 hovertemplate="%{x:.0f} h: %{y:.1%} still in yard<extra>" + name + "</extra>"))
    fig.add_vline(x=65.11, line=dict(color=AMBER, width=1, dash="dot"))
    fig.add_vline(x=160.9, line=dict(color=RED, width=1, dash="dot"))
    fig = plotly_layout(fig, height=340, legend=dict(orientation="v", x=.56, y=.98, yanchor="top", bgcolor="rgba(10,10,10,.7)"))
    fig.update_xaxes(type="log", title_text="Hours since discharge (log)", tickvals=[6, 12, 24, 48, 96, 192, 384])
    fig.update_yaxes(title_text="Share still in the yard", tickformat=".0%", range=[0, 1.02])
    return fig


def portwatch_small(pw: list) -> go.Figure:
    d = pd.DataFrame(pw)
    d["d"] = pd.to_datetime(d["d"])
    fig = go.Figure(go.Scatter(x=d["d"], y=d["imports"], mode="lines", line=dict(color=LIME, width=2), name="Imports",
                               hovertemplate="%{x|%d %b}<br>%{y:,.0f} tonnes/day (7-day mean)<extra></extra>"))
    fig = plotly_layout(fig, height=190, margin=dict(l=46, r=10, t=8, b=28), showlegend=False)
    fig.update_xaxes(tickformat="%d %b")
    fig.update_yaxes(tickformat="~s", title_text="t / day")
    return fig


def render() -> None:
    p = data.port()
    st.markdown('<div class="thesis" style="font-size:clamp(1.8rem,3vw,2.8rem)">Where the pressure <i>built</i>, and when it let go</div>'
                '<div class="lede">Everything on this page comes from one terminal’s 2021 event log. Yard capacity and live occupancy aren’t in the data, so none is shown.</div>', unsafe_allow_html=True)

    a, b = st.columns([1, 1], gap="large")
    with a:
        head("Terminal clock", "When do containers actually move? Each ring is a weekday, each slice an hour; brighter means more containers.", "observed")
        side = st.segmented_control("Activity", ["Gate-outs", "Discharges"], default="Gate-outs", label_visibility="collapsed")
        matrix, title = (p["clock_in"], "Discharges") if side == "Discharges" else (p["clock_out"], "Gate-outs")
        visuals.show_svg(visuals.clock_svg(matrix, title), 500)
    with b:
        head("Calendar of later delay", "Which discharge days produced the most critical stays? Hover for restrictions and holidays.", "observed")
        visuals.show_svg(visuals.calendar_svg(p["calendar"]), 395)
        st.markdown('<div class="note">Colour shows how arrivals on that day <b>ended up</b>, which nobody could see on the day. Arrivals from about 9 to 21 July fared worst.</div>', unsafe_allow_html=True)

    head("Vessel-call constellation", "Are bad outcomes spread evenly, or do some ships carry them? One dot per call with 30+ containers; size is containers, colour is critical share.", "observed")
    st.plotly_chart(constellation(p["vessels"]), config=PLOTLY_CONFIG, width="stretch")
    st.markdown('<div class="note">Vessel names are anonymised in the source; calls are identified by berthing time. Differences between calls explain about 20% of the variation in dwell.</div>', unsafe_allow_html=True)

    c, d = st.columns([7, 5], gap="large")
    with c:
        head("Yard blocks over time", "Which blocks held the stuck containers? Rows are blocks with 400+ stays, sorted by critical share; empty cells have under 15 stays. A schematic, not the yard layout.", "observed")
        visuals.show_svg(visuals.bay_svg(p["blocks"], p["block_weeks"]), 470)
    with d:
        head("The customs split", "How much does clearing customs before the ship unloads change a container’s path?", "observed")
        st.plotly_chart(survival_figure(p), config=PLOTLY_CONFIG, width="stretch")
        head("Port-wide imports", "Outside check: did port-wide container imports dip with the terminal’s? AIS estimates, usable only with a 7-day lag.", "lagged")
        st.plotly_chart(portwatch_small(p["portwatch"]), config=PLOTLY_CONFIG, width="stretch")
