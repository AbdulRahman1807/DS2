"""Model intelligence: what was evaluated, how well, and how far to trust it."""

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
    fig.update_xaxes(title_text="Mean-fold Macro-F1 (development folds)", range=[.29, .58], dtick=.05)
    fig.update_yaxes(categoryorder="array", categoryarray=["+48 h", "+24 h", "+12 h", "At discharge"], gridcolor=RULE)
    return fig


def reliability_figure(rel: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1, dash="dot"), name="Perfect", hoverinfo="skip"))
    for name in CLASS_LABEL:
        pts = rel[name]["points"]
        fig.add_trace(go.Scatter(x=[p["pred"] for p in pts], y=[p["obs"] for p in pts], mode="lines+markers", name=CLASS_LABEL[name],
                                 line=dict(color=CLASS_COLOR[name], width=2), marker=dict(size=[max(5, min(14, p["n"] ** .33)) for p in pts], color=CLASS_COLOR[name], line=dict(color=INK, width=1)),
                                 customdata=[p["n"] for p in pts], hovertemplate="Score %{x:.2f} → happened %{y:.1%}<br>%{customdata:,} stays<extra>" + CLASS_LABEL[name] + "</extra>"))
    fig = plotly_layout(fig, height=345)
    fig.update_xaxes(title_text="Model score", range=[0, 1], tickformat=".1f")
    fig.update_yaxes(title_text="How often it happened", range=[0, 1], tickformat=".0%")
    return fig


def render() -> None:
    m, live = data.model(), data.live()
    H = m["horizons"]
    st.markdown('<div class="thesis" style="font-size:clamp(1.8rem,3vw,2.8rem)">More time, more information, <i>better answers</i>. Not more model.</div>', unsafe_allow_html=True)
    head("The information staircase", "How good can dwell prediction get as information arrives? Solid steps are real models; dashed steps show the ceiling if future events were known.", "validation")
    visuals.show_svg(visuals.staircase_svg(H, m["oracles"]), 310)

    h = st.segmented_control("Prediction horizon", ["At discharge", "+12 h", "+24 h", "+48 h"], default="+24 h")
    key = {"At discharge": "A", "+12 h": "B12", "+24 h": "B24", "+48 h": "B48"}.get(h or "+24 h")
    v = H[key]
    pc = v["per_class"]

    st.markdown(f"""<div class="bento" style="margin-top:.2rem">
<div class="cell"><div class="k">Macro-F1</div><div class="v">{v["mean_fold_macro_f1"]:.3f}</div><div class="n">Mean of 3 folds; pooled {v["pooled_macro_f1"]:.3f}, 95% {v["ci95"][0]:.2f}–{v["ci95"][1]:.2f}</div></div>
<div class="cell"><div class="k">Balanced accuracy</div><div class="v">{v["balanced_accuracy"]:.3f}</div><div class="n">Average recall across the three classes</div></div>
<div class="cell"><div class="k">Accuracy</div><div class="v">{v["accuracy"]:.3f}</div><div class="n">Weighted F1 {v["weighted_f1"]:.3f}</div></div>
<div class="cell"><div class="k">Critical recall</div><div class="v">{pc["CRITICAL_DELAY"]["recall"]*100:.0f}<small>%</small></div><div class="n">Precision {pc["CRITICAL_DELAY"]["precision"]*100:.0f}%, F1 {pc["CRITICAL_DELAY"]["f1"]:.3f}</div></div>
<div class="cell"><div class="k">Critical PR-AUC</div><div class="v">{v["critical_pr_auc"]:.3f}</div><div class="n">Base rate here {v["critical_prevalence"]*100:.1f}%</div></div>
<div class="cell"><div class="k">Stays scored</div><div class="v">{v["n"]:,}</div><div class="n">Still in the yard at this point; median {v["median_lead_time_h"]:.0f} h before exit</div></div>
</div>""", unsafe_allow_html=True)

    a, b = st.columns([5, 7], gap="large")
    with a:
        head("Confusion mosaic", "What does the model confuse? Row height is class size; bright cells are correct.", "validation")
        visuals.show_svg(visuals.confusion_svg(v["confusion"]), 340)
        rows = "".join(f'<tr><td style="color:{CLASS_COLOR[c]}">{CLASS_LABEL[c]}</td><td>{pc[c]["precision"]:.3f}</td><td>{pc[c]["recall"]:.3f}</td><td>{pc[c]["f1"]:.3f}</td><td>{pc[c]["support"]:,}</td></tr>' for c in CLASS_LABEL)
        st.markdown(f'<table class="mono" style="width:100%;font-size:.76rem;border-collapse:collapse;text-align:right"><tr style="color:{MUTED}"><td style="text-align:left">Class</td><td>Precision</td><td>Recall</td><td>F1</td><td>Stays</td></tr>{rows}</table>', unsafe_allow_html=True)
    with b:
        head("Calibration", "When the model says 60%, does it happen 60% of the time? Below the dotted line means overconfident.", "validation")
        st.plotly_chart(reliability_figure(v["reliability"]), config=PLOTLY_CONFIG, width="stretch")
        rel = v["reliability"]
        st.markdown(f'<div class="note">Brier score <b>{rel["brier"]:.3f}</b> · Expected calibration error <b>{rel["ON_TIME"]["ece"]:.3f}</b> on time, <b>{rel["AT_RISK"]["ece"]:.3f}</b> at risk, <b>{rel["CRITICAL_DELAY"]["ece"]:.3f}</b> critical. '
                    'These models train with balanced class weights, so <b>their scores run high for delay classes</b>. Read them as a ranking, not as odds.</div>', unsafe_allow_html=True)

    head("Decision triangles", "How cleanly does the model separate outcomes? Each triangle holds stays of one true outcome. The closer a stay sits to the matching corner, the better. Overlap is the problem.", "validation")
    visuals.show_svg(visuals.simplex_triptych(v["simplex"]), 310)

    c, d = st.columns([6, 6], gap="large")
    with c:
        head("Every model tried", "Does the algorithm matter? Each mark is one registered experiment. Horizon moves scores far more than model family.", "validation")
        st.plotly_chart(strip_figure(m["strips"], m["best"]), config=PLOTLY_CONFIG, width="stretch")
        best_rows = "".join(f'<tr><td style="text-align:left">{ {"A":"At discharge","B12":"+12 h","B24":"+24 h","B48":"+48 h"}[k] }</td><td style="text-align:left">{b_["id"]}</td><td>{b_["macro_f1"]:.4f}</td></tr>' for k, b_ in m["best"].items())
        st.markdown(f'<table class="mono" style="width:100%;font-size:.74rem;border-collapse:collapse;text-align:right"><tr style="color:{MUTED}"><td style="text-align:left">Best so far</td><td style="text-align:left">Experiment</td><td>Macro-F1</td></tr>{best_rows}</table>'
                    '<div class="note" style="margin-top:.4rem">Picked by comparing on the same folds, so slightly optimistic. Gaps under 0.03 are within noise. No configuration is frozen yet, and the holdout hasn’t been scored.</div>', unsafe_allow_html=True)
    with d:
        head("What the model sees", "Which signals matter, and when? Ranks by share of tree gain at each horizon: delivery and customs status climb as they become knowable.", "validation")
        visuals.show_svg(visuals.bump_svg(m["bump"], labels.FEATURE_LABEL), 380)
        fam = "".join(f'<span style="margin-right:.8rem;white-space:nowrap"><i style="display:inline-block;width:10px;height:3px;background:{c};vertical-align:middle;margin-right:4px"></i>{k}</span>' for k, c in visuals.FAMILY_COLOR.items())
        st.markdown(f'<div class="note">{fam}<br>Gain favours many-valued fields such as container owner, so read ranks as direction rather than exact weight.</div>', unsafe_allow_html=True)

    head("The model behind Container triage", "What is deployed today, and why it isn’t the best model above.", "validation")
    dep = live["deployed"]
    e, f = st.columns([5, 7], gap="large")
    with e:
        st.markdown(f"""<dl class="kv"><dt>Model</dt><dd>EXP-MOD-07, {dep["model"]}</dd><dt>Inputs</dt><dd>{len(dep["features"])} arrival-time fields</dd>
<dt>Macro-F1, Stage 01 validation</dt><dd>{dep["macro_f1_val"]:.3f} on {dep["n_val"]:,} stays</dd><dt>Same features, corrected protocol</dt><dd>{m["legacy_baseline"]["corrected"]:.3f}</dd>
<dt>Latency, one container</dt><dd>{dep["single_latency_ms_p50"]:.1f} ms median, measured when this app’s data was built</dd></dl>""", unsafe_allow_html=True)
    with f:
        st.markdown(f'<div class="note"><b>Why this one:</b> it’s the only pipeline saved as a runnable file. The better models above exist as saved validation predictions, so the +12, +24 and +48 h views replay those predictions rather than score new input. '
                    f'<b>Its Stage 01 score is flattered</b> by a validation split affected by how the source was cut; on the corrected protocol the same features score {m["legacy_baseline"]["corrected"]:.2f}.</div>', unsafe_allow_html=True)
    g = pd.DataFrame(m["calibration_stage01"]).rename(columns={"method": "Method", "brier": "Brier", "log_loss": "Log loss", "ece": "ECE", "critical_ece": "Critical ECE"})
    st.markdown('<div class="note" style="margin-top:.5rem"><b>Stage 01 calibration study</b> (Evaluation Engineer, XGBoost baseline, legacy test split):</div>', unsafe_allow_html=True)
    st.dataframe(g, hide_index=True, width="stretch")
