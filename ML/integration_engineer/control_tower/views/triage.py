"""Container triage: score a new container with the deployed model, or investigate a real stay."""

from __future__ import annotations

import html
from typing import List

import pandas as pd
import streamlit as st

from ML.integration_engineer.control_tower import data, inference, labels, visuals
from ML.integration_engineer.control_tower.theme import AMBER, CLASS_COLOR, CLASS_LABEL, CLASS_RANGE, LIME, MUTED, RED, SOFT, head

HORIZON_KEYS = {"At discharge": ("A", 0), "+12 h": ("B12", 12), "+24 h": ("B24", 24), "+48 h": ("B48", 48)}
CLASSES = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
MILESTONES = [
    ("avail_HAS_QUARANTINE_FLAG", "Quarantine flag"), ("avail_SP_RED_FLAG_PIB", "Customs red flag"), ("avail_CUSTOMS_DEL", "Customs released"),
    ("avail_FIRST_JOB_CUSTOMS_BEHANDLE", "Physical inspection started"), ("avail_QUARANTINE_RELEASE", "Quarantine released"), ("avail_JOB_DEL", "Delivery job created"), ("avail_TRUCK_IN", "Truck at the gate"),
]


def verdict_block(name: str, proba: List[float], meta: str) -> None:
    c = CLASS_COLOR[name]
    bars = "".join(f'<div class="pbar"><span>{CLASS_LABEL[k]}</span><div class="t"><span style="width:{p*100:.1f}%;background:{CLASS_COLOR[k]}"></span></div><span class="p">{p*100:.1f}%</span></div>' for k, p in zip(CLASSES, proba))
    st.markdown(f'<div class="verdict" style="--c:{c}"><div class="lab">{CLASS_LABEL[name]}</div><div class="sub">{CLASS_RANGE[name]} · {html.escape(meta)}</div></div>{bars}', unsafe_allow_html=True)


def action_block(class_id: int) -> None:
    try:
        action, steps = inference.playbook(class_id)
    except Exception:
        st.markdown('<div class="unavailable"><b>Recommended action unavailable.</b> The operating procedure module could not load.</div>', unsafe_allow_html=True)
        return
    st.markdown(f'<div class="note" style="margin-top:.5rem"><b>Recommended action: {html.escape(action)}.</b> {html.escape(steps)}</div>', unsafe_allow_html=True)


def journey_svg(row: pd.Series, t: pd.Timestamp, reveal: bool) -> str:
    d0, out = row["discharge_start_ts"], row["truck_out_end_ts"]
    span = max(((out - d0) / pd.Timedelta(hours=1)) if reveal else max((t - d0) / pd.Timedelta(hours=1) * 1.5, 72), 72)
    W, H, L, R = 760, 158, 12, 16
    x = lambda h: L + min(max(h, -12), span) / span * (W - L - R) if h >= 0 else L
    out_svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Timeline of this container\'s events">']
    y0 = 70
    now_h = (t - d0) / pd.Timedelta(hours=1)
    out_svg.append(f'<line x1="{L}" x2="{W-R}" y1="{y0}" y2="{y0}" stroke="#2B2B27" stroke-width="2"/>')
    out_svg.append(f'<line x1="{L}" x2="{x(now_h):.1f}" y1="{y0}" y2="{y0}" stroke="{LIME}" stroke-width="2"/>')
    for lim, c, lab in [(65.11, AMBER, "65 h"), (160.9, RED, "161 h")]:
        if lim <= span:
            out_svg.append(f'<line x1="{x(lim):.1f}" x2="{x(lim):.1f}" y1="{y0-8}" y2="{y0+8}" stroke="{c}"/><text x="{x(lim):.1f}" y="{y0+20}" font-size="9.5" text-anchor="middle" style="fill:{c}">{lab}</text>')
    events = []
    for col, lab in MILESTONES:
        ts = row[col]
        if pd.notna(ts):
            events.append(((ts - d0) / pd.Timedelta(hours=1), lab, ts))
    events.sort(key=lambda e: e[0])
    for i, (h, lab, ts) in enumerate(events):
        known = ts <= t
        if not known and not reveal:
            continue
        cx = x(max(h, 0))
        up = i % 2 == 0
        yy = y0 - 34 - (i % 4) * 7 if up else y0 + 38 + (i % 4) * 7
        c = LIME if known else SOFT
        anchor = "end" if cx > W * .78 else "start"
        when = "before discharge" if h < 0 else f"+{h:.0f} h"
        out_svg.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y0}" y2="{yy+ (4 if up else -9)}" stroke="{c}" stroke-opacity=".5"/><circle cx="{cx:.1f}" cy="{y0}" r="3.5" fill="{c}"/>'
                       f'<text x="{cx+(3 if anchor=="start" else -3):.1f}" y="{yy}" font-size="10" text-anchor="{anchor}" style="fill:{c}" data-tip="{lab}\n{ts:%d %b %H:%M}\n{when}">{lab} {when}</text>')
    out_svg.append(f'<line x1="{x(now_h):.1f}" x2="{x(now_h):.1f}" y1="{y0-12}" y2="{y0+12}" stroke="{LIME}" stroke-width="2"/>')
    out_svg.append(f'<text x="{x(now_h):.1f}" y="12" font-size="10.5" text-anchor="{"end" if x(now_h) > W*.8 else "middle"}" style="fill:{LIME}">Prediction point, {t:%d %b %H:%M}</text>')
    if reveal:
        out_svg.append(f'<rect x="{W-R-5}" y="{y0-5}" width="10" height="10" fill="{CLASS_COLOR[CLASSES[int(row["y_true"])]]}"/><text x="{W-R}" y="{y0-12}" font-size="10" text-anchor="end" style="fill:#F4F4EE">Gate-out +{row["dwell_hours"]:.0f} h</text>')
    return "".join(out_svg) + "</svg>"


def status_at(row: pd.Series, t: pd.Timestamp) -> List[str]:
    """Plain-language operational status at time t, from real event timestamps only."""
    out = []
    if pd.notna(row["avail_CUSTOMS_DEL"]) and row["avail_CUSTOMS_DEL"] <= t:
        out.append("Customs released")
    else:
        out.append("Customs pending")
    if pd.notna(row["avail_JOB_DEL"]) and row["avail_JOB_DEL"] <= t:
        out.append("Delivery job created")
    else:
        out.append("No delivery job yet")
    if pd.notna(row["avail_HAS_QUARANTINE_FLAG"]) and row["avail_HAS_QUARANTINE_FLAG"] <= t:
        out.append("Quarantine released" if pd.notna(row["avail_QUARANTINE_RELEASE"]) and row["avail_QUARANTINE_RELEASE"] <= t else "Quarantine hold")
    if pd.notna(row["avail_SP_RED_FLAG_PIB"]) and row["avail_SP_RED_FLAG_PIB"] <= t:
        out.append("Customs red flag")
    if pd.notna(row["avail_FIRST_JOB_CUSTOMS_BEHANDLE"]) and row["avail_FIRST_JOB_CUSTOMS_BEHANDLE"] <= t:
        out.append("Physical inspection")
    return out


def dossier(row: pd.Series, horizon: str, reveal: bool) -> None:
    """Shared investigation panel for a real validation stay (also used by the operations queue)."""
    key, hours = HORIZON_KEYS[horizon]
    m = data.model()
    d0 = row["discharge_start_ts"]
    t = d0 + pd.Timedelta(hours=hours)
    if pd.isna(row[f"{key}_p0"]):
        st.markdown(f'<div class="unavailable"><b>No prediction at this point.</b> Container departed after {row["dwell_hours"]:.0f} h (prior to {horizon}).</div>', unsafe_allow_html=True)
        return
    proba = [float(row[f"{key}_p{i}"]) for i in range(3)]
    pred = int(max(range(3), key=lambda i: proba[i]))
    name = CLASSES[pred]
    a, b = st.columns([7, 4], gap="medium")
    with a:
        verdict_block(name, proba, f"{horizon} prediction")
        action_block(pred)
    with b:
        visuals.show_svg(visuals.simplex_single(m["horizons"][key]["simplex"], proba, name), 250)
    visuals.show_svg(journey_svg(row, t, reveal), 215)
    st.markdown('<div class="statuses">' + "".join(f'<span class="st on">{html.escape(s)}</span>' for s in status_at(row, t)) + "</div>", unsafe_allow_html=True)
    if reveal:
        true = CLASSES[int(row["y_true"])]
        hit = "Prediction correct." if true == name else f"Actual was {CLASS_LABEL[true]}."
        st.markdown(f'<div class="note" style="margin-top:.5rem"><b style="color:{CLASS_COLOR[true]}">Actual outcome: {CLASS_LABEL[true]} ({row["dwell_hours"]:.0f} h).</b> {hit}</div>', unsafe_allow_html=True)


def score_mode() -> None:
    ref = data.live()
    if inference.load_bundle() is None:
        st.markdown('<div class="unavailable"><b>Model bundle unavailable.</b> Run build_assets.py to restore.</div>', unsafe_allow_html=True)
        return
    left, right = st.columns([3, 8], gap="large")
    days = sorted(ref["trailing_by_day"])
    with left:
        with st.form("score"):
            carrier = st.selectbox("Container owner prefix", ref["carriers"], help="First three letters of container code.")
            c1, c2 = st.columns(2)
            ctr_type = c1.selectbox("Type", list(ref["types"]))
            ctr_size = c2.selectbox("Length, ft", ref["sizes"], index=1)
            gross = st.number_input("Gross weight, tonnes", min_value=1.0, max_value=45.0, value=float(ref["gross_median"]), step=.5)
            day = st.date_input("Discharge date", value=pd.Timestamp("2021-08-16").date(), min_value=pd.Timestamp(days[0]).date(), max_value=pd.Timestamp(days[-1]).date())
            hour = st.slider("Discharge hour", 0, 23, 10)
            submitted = st.form_submit_button("Score container", width="stretch")
    with right:
        head("Container Dwell Risk Score", "Real-time risk scoring and operational triage directive.", "model")
        if not submitted and "last_score" not in st.session_state:
            st.markdown('<div class="unavailable"><b>Enter container telemetry on the left and select Score container.</b></div>', unsafe_allow_html=True)
            return
        if submitted:
            errors, notes = inference.validate_inputs(carrier, ctr_type, int(ctr_size), float(gross), day, int(hour), ref)
            if errors:
                st.session_state.pop("last_score", None)
                for e in errors:
                    st.error(e)
                return
            feats = inference.build_features(carrier, ctr_type, int(ctr_size), float(gross), day, int(hour), ref)
            st.session_state["last_score"] = {"res": inference.score(feats), "feats": feats, "notes": notes}
        s = st.session_state["last_score"]
        res, feats, notes = s["res"], s["feats"], s["notes"]
        a, b = st.columns([7, 4], gap="medium")
        with a:
            verdict_block(res["pred_name"], res["proba"], "At discharge")
            action_block(res["pred"])
        with b:
            visuals.show_svg(visuals.simplex_single(data.model()["horizons"]["A"]["simplex"], res["proba"], res["pred_name"]), 250)
        
        head("Key Risk Factors", "Top feature contributions influencing this score.", "model")
        peak = max(abs(r["contribution"]) for r in res["explanation"]) or 1
        rows = []
        for r in res["explanation"][:8]:
            w = abs(r["contribution"]) / peak * 50
            left_pos = 50 if r["contribution"] >= 0 else 50 - w
            color = CLASS_COLOR[res["pred_name"]] if r["contribution"] >= 0 else "#5f5f58"
            val = r["value"]
            shown = f"{val:,.2f}".rstrip("0").rstrip(".") if isinstance(val, float) else str(val)
            rows.append(f'<div class="shap"><span>{labels.FEATURE_LABEL.get(r["feature"], r["feature"])}: <b style="color:#F4F4EE;font-weight:500">{html.escape(shown)}</b></span><div class="bar"><span style="left:{left_pos:.1f}%;width:{w:.1f}%;background:{color}"></span><div class="mid"></div></div><span style="text-align:right">{"+" if r["contribution"]>=0 else "−"}{abs(r["contribution"]):.3f}</span></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.markdown(f"""<dl class="kv" style="margin-top:.7rem"><dt>Prediction time</dt><dd>{res["timestamp"]}</dd><dt>Model</dt><dd>{inference.MODEL_VERSION}</dd><dt>Latency</dt><dd>{res["latency_ms"]:.1f} ms</dd></dl>""", unsafe_allow_html=True)
        if notes:
            notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
            st.markdown(f'<div class="note" style="margin-top:.4rem"><b>Notice:</b><ul style="margin:.2rem 0 0 1rem">{notes_html}</ul></div>', unsafe_allow_html=True)


def replay_mode() -> None:
    d = data.load_dossier()
    left, right = st.columns([3, 8], gap="large")
    with left:
        q = st.text_input("Find a container", placeholder="e.g. TCLU or MSCU").strip().upper()
        pool = d[d["case_id"].str.contains(q, regex=False)] if q else d
        lens = st.selectbox("Filter", ["Highest critical score at +24 h", "Stayed longest", "Critical, and the model missed it", "Any stay"])
        scored = pool[pool["B24_p2"].notna()]
        if lens == "Highest critical score at +24 h":
            pool = scored.sort_values("B24_p2", ascending=False)
        elif lens == "Stayed longest":
            pool = pool.sort_values("dwell_hours", ascending=False)
        elif lens == "Critical, and the model missed it":
            pool = scored[(scored["y_true"] == 2) & (scored[["B24_p0", "B24_p1", "B24_p2"]].to_numpy().argmax(1) != 2)].sort_values("B24_p2")
        if pool.empty:
            st.markdown('<div class="unavailable"><b>No matching stays.</b> Clear search filter.</div>', unsafe_allow_html=True)
            return
        case = st.selectbox("Container stay", pool["case_id"].head(400).tolist())
        horizon = st.segmented_control("Prediction point", list(HORIZON_KEYS), default="+24 h") or "+24 h"
        reveal = st.toggle("Reveal actual outcome", value=False)
        row = d[d["case_id"] == case].iloc[0]
        st.markdown(f"""<dl class="kv"><dt>Carrier</dt><dd>{row["carrier_prefix"]}</dd><dt>Size / Type</dt><dd>{row["CTR_SIZE"]:.0f} ft {row["CTR_TYPE"]}</dd><dt>Gross weight</dt><dd>{row["gross_t"]:.1f} t</dd>
<dt>Yard location</dt><dd>Block {row["YARD_BLOCK"]}, slot {row["YARD_SLOT"]:.0f}</dd><dt>Discharge</dt><dd>{row["discharge_start_ts"]:%d %b %H:%M}</dd><dt>Vessel call</dt><dd>{row["vessel_call_id"]}</dd></dl>""", unsafe_allow_html=True)
    with right:
        head(case, f"Historical validation stay · Scored at {horizon}", "model")
        dossier(row, horizon, reveal)


def render() -> None:
    st.markdown('<div class="thesis" style="font-size:clamp(1.8rem,3vw,2.8rem)">Container Triage</div>', unsafe_allow_html=True)
    mode = st.segmented_control("Mode", ["Score a container", "Replay a real stay"], default="Score a container", label_visibility="collapsed") or "Score a container"
    if mode == "Score a container":
        score_mode()
    else:
        replay_mode()
