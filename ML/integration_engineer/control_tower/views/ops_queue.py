"""Operations queue: prioritized container triage queue for terminal expeditors."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ML.integration_engineer.control_tower import data
from ML.integration_engineer.control_tower.theme import CLASS_COLOR, CLASS_LABEL, RED, AMBER, LIME, head
from ML.integration_engineer.control_tower.views.triage import CLASSES, HORIZON_KEYS, dossier, status_at


def build_queue(d: pd.DataFrame, day: pd.Timestamp, horizon: str) -> pd.DataFrame:
    key, hours = HORIZON_KEYS[horizon]
    q = d[d[f"{key}_p0"].notna()].copy()
    q["t_pred"] = q["discharge_start_ts"] + pd.Timedelta(hours=hours)
    q = q[q["t_pred"].dt.normalize() == day]
    q["p_on"], q["p_risk"], q["p_crit"] = q[f"{key}_p0"], q[f"{key}_p1"], q[f"{key}_p2"]
    q["pred"] = q[["p_on", "p_risk", "p_crit"]].to_numpy().argmax(1)
    q["risk"] = q["pred"].map(lambda i: CLASS_LABEL[CLASSES[i]])
    return q.sort_values("p_crit", ascending=False).reset_index(drop=True)


def reason(row: pd.Series) -> str:
    statuses = status_at(row, row["t_pred"])
    return " · ".join(statuses[:3])


ACTION = {0: "Monitor", 1: "Verify customs & gate slot", 2: "Escalate to yard planner"}


def render() -> None:
    d = data.load_dossier()
    m = data.model()
    st.markdown('<div class="thesis" style="font-size:clamp(1.8rem,3vw,2.8rem)">Operations Queue</div>'
                '<div class="lede">Prioritized bottleneck triage queue based on predicted critical delay probability.</div>', unsafe_allow_html=True)
    days = sorted(d["discharge_day"].unique())
    c1, c2, c3, c4 = st.columns([3, 3, 2, 3])
    day = pd.Timestamp(c1.selectbox("Date", days[3:], index=min(8, len(days) - 4), format_func=lambda x: pd.Timestamp(x).strftime("%a %d %b %Y")))
    horizon = c2.segmented_control("Horizon", list(HORIZON_KEYS), default="+24 h") or "+24 h"
    top_n = c3.selectbox("Queue size", [10, 25, 50, 100], index=1)
    search = c4.text_input("Find container", placeholder="Container number").strip().upper()
    risk_filter = st.pills("Tier", list(CLASS_LABEL.values()), selection_mode="multi", default=list(CLASS_LABEL.values()))
    reveal = st.toggle("Reveal actual outcome", value=False)

    full = build_queue(d, day, horizon)
    if full.empty:
        st.markdown('<div class="unavailable"><b>No active stays reached this prediction horizon on selected date.</b></div>', unsafe_allow_html=True)
        return
    q = full[full["risk"].isin(risk_filter)] if risk_filter else full
    if search:
        q = q[q["case_id"].str.contains(search, regex=False)]
    show = q.head(top_n)
    if show.empty:
        st.markdown('<div class="unavailable"><b>No matching containers found.</b></div>', unsafe_allow_html=True)
        return

    k = m["horizons"][HORIZON_KEYS[horizon][0]]
    n_day = len(full)
    n_crit_pred = int((full["pred"] == 2).sum())
    summary = [
        f'<div class="cell s2"><div class="k">Daily Stays at Horizon</div><div class="v">{n_day:,}</div><div class="n">{day:%d %b %Y}</div></div>',
        f'<div class="cell s2"><div class="k">Predicted Critical</div><div class="v">{n_crit_pred:,}</div><div class="n">{n_crit_pred/n_day*100:.1f}% of cohort</div></div>',
    ]
    if reveal:
        hits = int((show["y_true"] == 2).sum())
        base = (full["y_true"] == 2).mean()
        summary.append(f'<div class="cell s2"><div class="k">Queue Hit Rate</div><div class="v">{hits}<small>of {len(show)}</small></div><div class="n">Actual critical ({base*100:.0f}% day avg)</div></div>')
    else:
        summary.append(f'<div class="cell s2"><div class="k">Top-1% Queue Precision</div><div class="v">{k["top1pct_precision"]*100:.0f}<small>%</small></div><div class="n">vs {k["critical_prevalence"]*100:.1f}% baseline rate</div></div>')
    st.markdown(f'<div class="bento">{"".join(summary)}</div>', unsafe_allow_html=True)

    head("Prioritized Inspection Queue", "Ranked by critical delay risk score. Select a row to inspect container details.", "model")
    view = pd.DataFrame({
        "Rank": range(1, len(show) + 1), "Container": show["container_id"], "Predicted": show["risk"], "Critical score": show["p_crit"], "At-risk score": show["p_risk"], "On-time score": show["p_on"],
        "Scored at": show["t_pred"].dt.strftime("%d %b %H:%M"), "Carrier": show["carrier_prefix"], "Vessel call": show["vessel_call_id"], "Key Status": show.apply(reason, axis=1), "Action": show["pred"].map(ACTION),
    })
    if reveal:
        view.insert(3, "Actual", show["y_true"].map(lambda i: CLASS_LABEL[CLASSES[int(i)]]).to_numpy())
        view.insert(4, "Dwell, h", show["dwell_hours"].round(0).to_numpy())

    def tint(v):
        return {"Critical delay": f"color:{RED};font-weight:600", "At risk": f"color:{AMBER}", "On time": f"color:{LIME}"}.get(v, "")

    styler = view.style.map(tint, subset=[c for c in ("Predicted", "Actual") if c in view])
    event = st.dataframe(styler, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(view)), on_select="rerun", selection_mode="single-row",
                         column_config={
                             "Critical score": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1, color="#FF5A47"),
                             "At-risk score": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1, color="#F2A93B"),
                             "On-time score": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1, color="#C7F36B"),
                             "Key Status": st.column_config.TextColumn(width="large"), "Rank": st.column_config.NumberColumn(width="small"),
                         })
    selected = event.selection.rows[0] if event.selection.rows else 0
    row = show.iloc[selected]
    head(f"Container Dossier: {row['case_id']}", f"Detailed event log and status at {horizon}.", "model")
    dossier(row, horizon, reveal)
