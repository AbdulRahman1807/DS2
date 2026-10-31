"""Overview: terminal throughput, dwell patterns, and bottleneck triage metrics."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ML.integration_engineer.control_tower import data, visuals
from ML.integration_engineer.control_tower.theme import AMBER, LIME, MUTED, PLOTLY_CONFIG, RED, RULE, SOFT, head, plotly_layout


def _cell(key: str, value: str, unit: str, note: str, span: str = "s2", kind: str = "", extra: str = "") -> str:
    tag = f'<span style="color:{LIME}">{kind}</span>' if kind in ("VAL", "MODEL") else f"<span>{kind}</span>"
    return f'<div class="cell {span}"><div class="k"><span>{key}</span>{tag}</div><div class="v">{value}<small>{unit}</small></div>{extra}<div class="n">{note}</div></div>'


def tide_figure(ov: dict) -> go.Figure:
    t = pd.DataFrame(ov["tide"])
    t["t"] = pd.to_datetime(t["t"])
    f = pd.DataFrame(ov["flow"])
    f["d"] = pd.to_datetime(f["d"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.72, .28], vertical_spacing=.03)
    hover = "%{x|%d %b %H:%M}<br>%{customdata}: %{y:,} containers<extra></extra>"
    for col, name, c in [("fresh", "In yard up to 65 h", LIME), ("aging", "65–161 h", AMBER), ("over", "Over 161 h", RED)]:
        fig.add_trace(go.Scatter(x=t["t"], y=t[col], name=name, stackgroup="yard", mode="lines", line=dict(width=0, color=c), fillcolor=c,
                                 customdata=[name] * len(t), hovertemplate=hover), row=1, col=1)
    fig.add_trace(go.Bar(x=f["d"], y=f["in"], name="Discharged per day", marker_color=SOFT, marker_line_width=0,
                         hovertemplate="%{x|%d %b}<br>Discharged: %{y:,}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(x=f["d"], y=-f["out"], name="Gated out per day", marker_color="#5f5f58", marker_line_width=0, customdata=f["out"],
                         hovertemplate="%{x|%d %b}<br>Gated out: %{customdata:,}<extra></extra>"), row=2, col=1)
    fig.update_layout(barmode="relative", bargap=.15)
    for h in ov["holidays"]:
        fig.add_vline(x=pd.Timestamp(h["d"]).timestamp() * 1000, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=1)
        fig.add_annotation(x=h["d"], y=1, yref="y domain", text=h["name"].split(" (")[0], showarrow=False, font=dict(size=10, color=SOFT), xanchor="left", xshift=3, yanchor="top", row=1, col=1)
    em = next(p for p in ov["ppkm"] if p["label"] == "PPKM Darurat")
    fig.add_shape(type="line", x0=em["start"], x1=em["end"], y0=1.02, y1=1.02, yref="y domain", line=dict(color=AMBER, width=3), row=1, col=1)
    fig.add_annotation(x=em["start"], y=1.03, yref="y domain", text="Emergency COVID restrictions (PPKM Darurat)", showarrow=False, font=dict(size=10, color=AMBER), xanchor="left", yanchor="bottom", row=1, col=1)
    fig = plotly_layout(fig, height=430, hovermode="x unified", legend=dict(orientation="h", y=-.09, x=0, yanchor="top", bgcolor="rgba(0,0,0,0)"), margin=dict(l=48, r=10, t=26, b=30))
    fig.update_yaxes(title_text="Containers in yard", row=1, col=1)
    fig.update_yaxes(title_text="Per day", row=2, col=1, tickformat="~s")
    fig.update_xaxes(tickformat="%d %b", dtick=7 * 24 * 3600 * 1000, tick0="2021-07-05", row=2, col=1)
    return fig


def render() -> None:
    ov, m = data.overview(), data.model()
    k = ov["kpi"]
    h = m["horizons"]
    crit = k["mix"][2]
    st.markdown(f'<div class="strip"><span><b>Terminal Petikemas Surabaya</b> · Tanjung Perak, Indonesia</span>'
                f'<span>Discharges: {k["first_discharge"]} to {k["last_discharge"]}</span>'
                f'<span>Cohort: {k["stays_cohort"]:,} stays</span></div>', unsafe_allow_html=True)

    left, right = st.columns([6, 5], gap="large")
    with left:
        ports = data.ports()
        visuals.show(visuals.globe_html(ports, {"n": f'{k["stays_cohort"]:,}', "crit": f"{crit*100:.1f}%"}, 590), 594)
    with right:
        st.markdown(f'<div class="thesis">{k["stays_cohort"]:,} container stays</div>'
                    f'<div class="lede">Terminal dwell progression from discharge to +48h triage. Macro-F1 advances from <b>{h["A"]["mean_fold_macro_f1"]:.2f}</b> at arrival to <b>{h["B48"]["mean_fold_macro_f1"]:.2f}</b> at 48 hours.</div>', unsafe_allow_html=True)
        mix = k["mix"]
        mix_bar = f'<div class="mixbar"><span style="width:{mix[0]*100}%;background:{LIME}"></span><span style="width:{mix[1]*100}%;background:{AMBER}"></span><span style="width:{mix[2]*100}%;background:{RED}"></span></div>'
        lift = h["B24"]["top1pct_precision"] / h["B24"]["critical_prevalence"]
        cells = [
            _cell("Dwell Distribution", f"{crit*100:.1f}", "% critical", f'On time {mix[0]*100:.0f}% · At risk {mix[1]*100:.0f}% · Critical {mix[2]*100:.0f}%', "s6", "OBS", mix_bar),
            _cell("Median Dwell", f'{k["dwell_median"]:.0f}', "h", f'Mean: {k["dwell_mean"]:.0f} h', "s2", "OBS"),
            _cell("Worst Week", f'{k["peak_week"]["crit"]*100:.0f}', "% crit", f'Discharged {visuals._fmt_week(k["peak_week"]["wk"]).title()}', "s2", "OBS"),
            _cell("Vessel Calls", f'{k["vessel_calls"]}', "calls", "Unique berthing calls", "s2", "OBS"),
            _cell("+24h Macro-F1", f'{h["B24"]["mean_fold_macro_f1"]:.3f}', "", f'Out-of-fold validation', "s3", "VAL"),
            _cell("+24h Critical Recall", f'{h["B24"]["per_class"]["CRITICAL_DELAY"]["recall"]*100:.0f}', "%", f'Precision: {h["B24"]["per_class"]["CRITICAL_DELAY"]["precision"]*100:.0f}%', "s3", "VAL"),
            _cell("Top-1% Queue Lift", f"{lift:.1f}", "× base rate", f'{h["B24"]["top1pct_precision"]*100:.0f}% precision vs {h["B24"]["critical_prevalence"]*100:.0f}% baseline prevalence', "s6", "VAL"),
        ]
        st.markdown(f'<div class="bento">{"".join(cells)}</div>', unsafe_allow_html=True)

    head("Terminal Replay", "Spatial container movement and yard dwell progression.", "observed")
    visuals.show(visuals.replay_html(data.replay(), 520), 524)

    head("Yard Inventory & Daily Flow", "Active containers in yard by dwell tier and daily gate movements.", "observed")
    st.plotly_chart(tide_figure(ov), config=PLOTLY_CONFIG, width="stretch")

    head("Weekly Dwell Distribution", "Progression of dwell hours across discharge weeks.", "observed")
    visuals.show_svg(visuals.ridgeline_svg(ov), 470)
