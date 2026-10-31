"""Model intelligence: evaluation metrics, confusion matrices, calibration, and feature dynamics."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ML.integration_engineer.control_tower import data, labels, visuals
from ML.integration_engineer.control_tower.theme import AMBER, CLASS_COLOR, CLASS_LABEL, INK, LIME, MUTED, PAPER, PLOTLY_CONFIG, RED, RULE, SOFT, head, plotly_layout

FAMILY_MARK = {"Direct classifier": ("circle", PAPER), "Stage-time stacking": ("diamond", LIME), "Survival": ("square", AMBER), "Ordinal": ("triangle-up", SOFT), "Regression": ("x", MUTED), "Legacy 13 features": ("circle-open", RED)}


def strip_figure(strips: dict, best: dict) -> go.Figure:
    fig = go.Figure()
    y_of = {"A": "At discharge", "B12": "+12 h", "B24": "+24 h", "B48": "+48 h"}
    seen = set()
    for h, rows in strips.items():
        for r in rows:
            sym, c = FAMILY_MARK[r["family"]]
            show = r["family"] not in seen
            seen.add(r["family"])
            fig.add_trace(go.Scatter(x=[r["macro_f1"]], y=[y_of[h]], mode="markers", name=r["family"], legendgroup=r["family"], showlegend=show,
                                     marker=dict(symbol=sym, size=11, color=c, line=dict(color=INK, width=1)),
                                     hovertemplate=f'{r["id"]}<br>{r["family"]}<br>Macro-F1 %{{x:.4f}}<br>{r["note"]}<extra></extra>'))
    fig = plotly_layout(fig, height=290, margin=dict(l=84, r=12, t=8, b=78), legend=dict(orientation="h", y=-.3, x=0, yanchor="top", bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title_text="Mean-fold Macro-F1", range=[.29, .58], dtick=.05)
    fig.update_yaxes(categoryorder="array", categoryarray=["+48 h", "+24 h", "+12 h", "At discharge"], gridcolor=RULE)
    return fig


def reliability_figure(rel: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1, dash="dot"), name="Perfect", hoverinfo="skip"))
    for name in CLASS_LABEL:
        pts = rel[name]["points"]
        fig.add_trace(go.Scatter(x=[p["pred"] for p in pts], y=[p["obs"] for p in pts], mode="lines+markers", name=CLASS_LABEL[name],
                                 line=dict(color=CLASS_COLOR[name], width=2), marker=dict(size=[max(5, min(14, p["n"] ** .33)) for p in pts], color=CLASS_COLOR[name], line=dict(color=INK, width=1)),
                                 customdata=[p["n"] for p in pts], hovertemplate="Score %{x:.2f} → observed %{y:.1%}<br>%{customdata:,} stays<extra>" + CLASS_LABEL[name] + "</extra>"))
    fig = plotly_layout(fig, height=345)
    fig.update_xaxes(title_text="Predicted score", range=[0, 1], tickformat=".1f")
    fig.update_yaxes(title_text="Observed frequency", range=[0, 1], tickformat=".0%")
    return fig


def render() -> None:
    m = data.model()
    H = m["horizons"]
    st.markdown('<div class="thesis" style="font-size:clamp(1.8rem,3vw,2.8rem)">Model Performance & Evaluation</div>'
                '<div class="lede">Temporal cross-validation metrics across discharge ($t_0$), +12h, +24h, and +48h prediction horizons.</div>', unsafe_allow_html=True)

    head("Performance Progression by Horizon", "Validation Macro-F1 across operational milestones.", "validation")
    visuals.show_svg(visuals.staircase_svg(H, m["oracles"]), 310)

    h = st.segmented_control("Horizon", ["At discharge", "+12 h", "+24 h", "+48 h"], default="+24 h")
    key = {"At discharge": "A", "+12 h": "B12", "+24 h": "B24", "+48 h": "B48"}.get(h or "+24 h")
    v = H[key]
    pc = v["per_class"]

    st.markdown(f"""<div class="bento" style="margin-top:.4rem">
<div class="cell"><div class="k">Macro-F1</div><div class="v">{v["mean_fold_macro_f1"]:.3f}</div><div class="n">3-fold CV mean</div></div>
<div class="cell"><div class="k">Balanced Accuracy</div><div class="v">{v["balanced_accuracy"]:.3f}</div><div class="n">Unweighted mean recall</div></div>
<div class="cell"><div class="k">Accuracy</div><div class="v">{v["accuracy"]:.3f}</div><div class="n">Weighted F1: {v["weighted_f1"]:.3f}</div></div>
<div class="cell"><div class="k">Critical Recall</div><div class="v">{pc["CRITICAL_DELAY"]["recall"]*100:.0f}<small>%</small></div><div class="n">Precision: {pc["CRITICAL_DELAY"]["precision"]*100:.0f}%</div></div>
<div class="cell"><div class="k">Critical PR-AUC</div><div class="v">{v["critical_pr_auc"]:.3f}</div><div class="n">Base prevalence: {v["critical_prevalence"]*100:.1f}%</div></div>
<div class="cell"><div class="k">Active Cohort</div><div class="v">{v["n"]:,}</div><div class="n">Stays in yard at {h}</div></div>
</div>""", unsafe_allow_html=True)

    a, b = st.columns([5, 7], gap="large")
    with a:
        head("Confusion Matrix", "Actual vs predicted outcome distribution.", "validation")
        visuals.show_svg(visuals.confusion_svg(v["confusion"]), 340)
        rows = "".join(f'<tr><td style="color:{CLASS_COLOR[c]}">{CLASS_LABEL[c]}</td><td>{pc[c]["precision"]:.3f}</td><td>{pc[c]["recall"]:.3f}</td><td>{pc[c]["f1"]:.3f}</td><td>{pc[c]["support"]:,}</td></tr>' for c in CLASS_LABEL)
        st.markdown(f'<table class="mono" style="width:100%;font-size:.76rem;border-collapse:collapse;text-align:right"><tr style="color:{MUTED}"><td style="text-align:left">Class</td><td>Precision</td><td>Recall</td><td>F1</td><td>Stays</td></tr>{rows}</table>', unsafe_allow_html=True)
    with b:
        head("Calibration Curve", "Empirical reliability across probability bins.", "validation")
        st.plotly_chart(reliability_figure(v["reliability"]), config=PLOTLY_CONFIG, width="stretch")
        rel = v["reliability"]
        st.markdown(f'<div class="note">Brier score: <b>{rel["brier"]:.3f}</b> · ECE: <b>{rel["CRITICAL_DELAY"]["ece"]:.3f}</b> (Critical tier).</div>', unsafe_allow_html=True)

    head("Decision Boundary Simplex", "Separation of container stays across 3-class probability space.", "validation")
    visuals.show_svg(visuals.simplex_triptych(v["simplex"]), 310)

    c, d = st.columns([6, 6], gap="large")
    with c:
        head("Model Family Comparison", "Evaluation across tabular model architectures.", "validation")
        st.plotly_chart(strip_figure(m["strips"], m["best"]), config=PLOTLY_CONFIG, width="stretch")
    with d:
        head("Dynamic Feature Importance", "Feature gain evolution across horizons.", "validation")
        visuals.show_svg(visuals.bump_svg(m["bump"], labels.FEATURE_LABEL), 380)
