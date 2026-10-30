"""Operations queue: replay what a planner would have reviewed on a given day."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ML.integration_engineer.control_tower import data, visuals
from ML.integration_engineer.control_tower.theme import AMBER, CLASS_COLOR, CLASS_LABEL, LIME, MUTED, RED, SOFT, head
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


ACTION = {0: "Monitor", 1: "Check gate slot and customs hold", 2: "Escalate to planner"}


def render() -> None:
    d = data.load_dossier()
    m = data.model()
    st.markdown('<div class="thesis" style="font-size:clamp(1.8rem,3vw,2.8rem)">Who needed attention <i>that day</i></div>'
                '<div class="lede">A replay of real validation days: the containers the model ranked highest for critical delay, with only what was known then. These aren’t live containers.</div>', unsafe_allow_html=True)
    days = sorted(d["discharge_day"].unique())
    c1, c2, c3, c4 = st.columns([3, 3, 2, 3])
    day = pd.Timestamp(c1.selectbox("Replay day", days[3:], index=min(8, len(days) - 4), format_func=lambda x: pd.Timestamp(x).strftime("%a %d %b %Y")))
    horizon = c2.segmented_control("Prediction point", list(HORIZON_KEYS), default="+24 h") or "+24 h"
    top_n = c3.selectbox("Queue length", [10, 25, 50, 100], index=1)
    search = c4.text_input("Search container", placeholder="Container number").strip().upper()
    risk_filter = st.pills("Predicted outcome", list(CLASS_LABEL.values()), selection_mode="multi", default=list(CLASS_LABEL.values()))
    reveal = st.toggle("Reveal what happened", value=False, help="Off shows what a planner would have seen that day.")

    full = build_queue(d, day, horizon)
    if full.empty:
        st.markdown('<div class="unavailable"><b>No stays reached this prediction point on that day.</b> Pick another day: the validation windows are 26 Jul–1 Aug, 5–11 Aug and 15–21 Aug 2021.</div>', unsafe_allow_html=True)
        return
    q = full[full["risk"].isin(risk_filter)] if risk_filter else full
    if search:
        q = q[q["case_id"].str.contains(search, regex=False)]
    show = q.head(top_n)
    if show.empty:
        st.markdown('<div class="unavailable"><b>No containers match.</b> Clear the search or select more outcomes.</div>', unsafe_allow_html=True)
        return

    k = m["horizons"][HORIZON_KEYS[horizon][0]]
    n_day = len(full)
    n_crit_pred = int((full["pred"] == 2).sum())
    summary = [f'<div class="cell s2"><div class="k">Scored that day</div><div class="v">{n_day:,}</div><div class="n">Stays reaching this point on {day:%d %b}</div></div>',
               f'<div class="cell s2"><div class="k">Predicted critical</div><div class="v">{n_crit_pred:,}</div><div class="n">{n_crit_pred/n_day*100:.0f}% of the day. Too many to chase, so rank them.</div></div>']
    if reveal:
        hits = int((show["y_true"] == 2).sum())
        base = (full["y_true"] == 2).mean()
        summary.append(f'<div class="cell s2"><div class="k">Queue hit rate</div><div class="v">{hits}<small>of {len(show)}</small></div><div class="n">Turned critical, against {base*100:.0f}% across that day’s stays.</div></div>')
    else:
        summary.append(f'<div class="cell s2"><div class="k">Top-1% hit rate, all folds</div><div class="v">{k["top1pct_precision"]*100:.0f}<small>%</small></div><div class="n">Against a {k["critical_prevalence"]*100:.0f}% base rate at this point.</div></div>')
    st.markdown(f'<div class="bento">{"".join(summary)}</div>', unsafe_allow_html=True)

    head("Queue", "Ranked by critical-delay score. Select a row to open its file.", "model")
    view = pd.DataFrame({
        "Rank": range(1, len(show) + 1), "Container": show["container_id"], "Predicted": show["risk"], "Critical score": show["p_crit"], "At-risk score": show["p_risk"], "On-time score": show["p_on"],
        "Scored at": show["t_pred"].dt.strftime("%d %b %H:%M"), "Owner": show["carrier_prefix"], "Vessel call": show["vessel_call_id"], "Why it’s here": show.apply(reason, axis=1), "Action": show["pred"].map(ACTION),
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
                             "Why it’s here": st.column_config.TextColumn(width="large"), "Rank": st.column_config.NumberColumn(width="small"),
                         })
    selected = event.selection.rows[0] if event.selection.rows else 0
    row = show.iloc[selected]
    head(f"File: {row['case_id']}", "What the model said, what was known, and what to do.", "model")
    dossier(row, horizon, reveal)
