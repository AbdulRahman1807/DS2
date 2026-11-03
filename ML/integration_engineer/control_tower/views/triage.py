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
    st.markdown(
        f'<div class="verdict" style="--c:{c}; margin-bottom: 0.8rem;">'
        f'<div class="lab" style="font-size: 2.2rem; font-weight: 700; color:{c};">{CLASS_LABEL[name]}</div>'
        f'<div class="sub" style="font-size: 0.92rem; color: #BDBDB4; margin-top: 0.25rem;">{CLASS_RANGE.get(name, "")} · {html.escape(meta)}</div>'
        f'</div>{bars}',
        unsafe_allow_html=True,
    )


def action_block(class_id: int) -> None:
    try:
        action, steps = inference.playbook(class_id)
    except Exception:
        action, steps = "Standard Monitoring", "Process container through standard terminal gate procedures."
    
    plain_actions = {
        0: ("Standard Gate Dispatch", "Normal turnaround expected. Dispatch trucker appointment according to schedule; free demurrage buffer is healthy."),
        1: ("Priority Gate & Customs Check", "Moderate delay risk. Confirm trucker appointment and verify customs clearance early to prevent yard congestion."),
        2: ("Immediate Escalation & Fast-Track", "High delay risk. Flag for dispatcher attention, expedite customs documents, and prioritize yard unstacking."),
    }
    title, desc = plain_actions.get(class_id, (action, steps))
    c = [LIME, AMBER, RED][class_id]

    st.markdown(
        f"""
        <div style="background: #141413; border: 1px solid #2B2B27; border-left: 4px solid {c}; padding: 0.9rem 1.1rem; border-radius: 4px; margin-top: 0.8rem;">
          <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: {c}; margin-bottom: 0.25rem;">
            Recommended Operational Action
          </div>
          <div style="font-size: 1.05rem; font-weight: 600; color: #F4F4EE; margin-bottom: 0.35rem;">
            {html.escape(title)}
          </div>
          <div style="font-size: 0.9rem; line-height: 1.5; color: #BDBDB4;">
            {html.escape(desc)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


PRESETS = {
    "ON_TIME": {
        "title": "On Time",
        "help": "Autofill scenario: 40ft Reefer, fast turnaround profile (expected: ON_TIME)",
        "carrier": "APR",
        "ctr_type": "RFR",
        "ctr_size": 40,
        "gross": 27.3,
        "day": pd.Timestamp("2021-08-06").date(),
        "hour": 2,
    },
    "AT_RISK": {
        "title": "At Risk",
        "help": "Autofill scenario: 20ft Dry, intermediate dwell risk (expected: AT_RISK)",
        "carrier": "EMC",
        "ctr_type": "DRY",
        "ctr_size": 20,
        "gross": 7.2,
        "day": pd.Timestamp("2021-08-15").date(),
        "hour": 17,
    },
    "CRITICAL_DELAY": {
        "title": "Critical Delay",
        "help": "Autofill scenario: 40ft Tank, severe dwell delay risk (expected: CRITICAL_DELAY)",
        "carrier": "TCN",
        "ctr_type": "TNK",
        "ctr_size": 40,
        "gross": 15.0,
        "day": pd.Timestamp("2021-07-09").date(),
        "hour": 14,
    },
}


def apply_preset(name: str, ref: dict) -> None:
    p = PRESETS[name]
    st.session_state["score_carrier"] = p["carrier"]
    st.session_state["score_type"] = p["ctr_type"]
    st.session_state["score_size"] = p["ctr_size"]
    st.session_state["score_gross"] = float(p["gross"])
    st.session_state["score_day"] = p["day"]
    st.session_state["score_hour"] = int(p["hour"])
    feats = inference.build_features(p["carrier"], p["ctr_type"], p["ctr_size"], p["gross"], p["day"], p["hour"], ref)
    errors, notes = inference.validate_inputs(p["carrier"], p["ctr_type"], p["ctr_size"], p["gross"], p["day"], p["hour"], ref)
    if not errors:
        st.session_state["last_score"] = {"res": inference.score(feats), "feats": feats, "notes": notes}


def score_mode() -> None:
    ref = data.live()
    if inference.load_bundle() is None:
        st.markdown('<div class="unavailable"><b>Model bundle unavailable.</b> Run build_assets.py to restore.</div>', unsafe_allow_html=True)
        return

    # Initialize form defaults in session_state if not already present
    if "score_carrier" not in st.session_state:
        st.session_state["score_carrier"] = ref["carriers"][0]
    if "score_type" not in st.session_state:
        st.session_state["score_type"] = list(ref["types"])[0]
    if "score_size" not in st.session_state:
        st.session_state["score_size"] = ref["sizes"][1]
    if "score_gross" not in st.session_state:
        st.session_state["score_gross"] = float(ref["gross_median"])
    if "score_day" not in st.session_state:
        st.session_state["score_day"] = pd.Timestamp("2021-08-16").date()
    if "score_hour" not in st.session_state:
        st.session_state["score_hour"] = 10

    left, right = st.columns([3, 8], gap="large")
    with left:
        st.markdown('<div class="sec-hd" style="margin-bottom:0.4rem"><span class="kicker">PRESETS</span><div class="title" style="font-size:1.05rem">Autofill Scenario</div></div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p1.button("🟢 On Time", help=PRESETS["ON_TIME"]["help"], use_container_width=True, on_click=apply_preset, args=("ON_TIME", ref))
        p2.button("🟡 At Risk", help=PRESETS["AT_RISK"]["help"], use_container_width=True, on_click=apply_preset, args=("AT_RISK", ref))
        p3.button("🔴 Critical", help=PRESETS["CRITICAL_DELAY"]["help"], use_container_width=True, on_click=apply_preset, args=("CRITICAL_DELAY", ref))

        with st.form("score"):
            carrier = st.selectbox("Container owner prefix", ref["carriers"], key="score_carrier", help="First three letters of container code.")
            c1, c2 = st.columns(2)
            ctr_type = c1.selectbox("Type", list(ref["types"]), key="score_type")
            ctr_size = c2.selectbox("Length, ft", ref["sizes"], key="score_size")
            gross = st.number_input("Gross weight, tonnes", min_value=1.0, max_value=45.0, step=.5, key="score_gross")
            day = st.date_input(
                "Discharge date",
                min_value=inference.DATE_LIMITS[0],
                max_value=inference.DATE_LIMITS[1],
                key="score_day",
                help="Supported range: 2020–2027. Dates outside Jul–Sep 2021 automatically apply terminal baseline congestion metrics.",
            )
            hour = st.slider("Discharge hour", 0, 23, key="score_hour")
            submitted = st.form_submit_button("Score container", width="stretch")
    with right:
        head("Delay Risk Assessment", "Predicted container clearance tier and recommended action.", "model")
        if submitted:
            errors, notes = inference.validate_inputs(carrier, ctr_type, int(ctr_size), float(gross), day, int(hour), ref)
            if errors:
                st.session_state.pop("last_score", None)
                for e in errors:
                    st.error(e)
                return
            feats = inference.build_features(carrier, ctr_type, int(ctr_size), float(gross), day, int(hour), ref)
            st.session_state["last_score"] = {"res": inference.score(feats), "feats": feats, "notes": notes}
        if "last_score" not in st.session_state:
            st.markdown('<div class="unavailable"><b>Enter container telemetry or select an autofill preset on the left, then select Score container.</b></div>', unsafe_allow_html=True)
            return
        s = st.session_state["last_score"]
        res, feats, notes = s["res"], s["feats"], s["notes"]
        a, b = st.columns([7, 4], gap="medium")
        with a:
            verdict_block(res["pred_name"], res["proba"], "At discharge")
            action_block(res["pred"])
        with b:
            visuals.show_svg(visuals.simplex_single(data.model()["horizons"]["A"]["simplex"], res["proba"], res["pred_name"]), 250)
        
        head("Top Influencing Factors", "Key shipment characteristics impacting this delay risk prediction.", "model")
        peak = max(abs(r["contribution"]) for r in res["explanation"]) or 1
        rows = []
        for r in res["explanation"][:6]:
            w = abs(r["contribution"]) / peak * 50
            left_pos = 50 if r["contribution"] >= 0 else 50 - w
            color = CLASS_COLOR[res["pred_name"]] if r["contribution"] >= 0 else "#5f5f58"
            val = r["value"]
            shown = f"{val:,.2f}".rstrip("0").rstrip(".") if isinstance(val, float) else str(val)
            effect_label = "+Increases delay" if r["contribution"] >= 0 else "−Reduces delay"
            rows.append(
                f'<div class="shap"><span>{labels.FEATURE_LABEL.get(r["feature"], r["feature"])}: <b style="color:#F4F4EE;font-weight:500">{html.escape(shown)}</b></span>'
                f'<div class="bar"><span style="left:{left_pos:.1f}%;width:{w:.1f}%;background:{color}"></span><div class="mid"></div></div>'
                f'<span style="text-align:right;font-size:0.78rem;color:{color}">{effect_label}</span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.markdown(f"""<dl class="kv" style="margin-top:.8rem"><dt>Prediction time</dt><dd>{res["timestamp"]}</dd><dt>Model</dt><dd>{inference.MODEL_VERSION}</dd><dt>Response latency</dt><dd>{res["latency_ms"]:.1f} ms</dd></dl>""", unsafe_allow_html=True)
        if notes:
            notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
            st.markdown(f'<div class="note" style="margin-top:.5rem"><b>Operational Note:</b><ul style="margin:.2rem 0 0 1rem">{notes_html}</ul></div>', unsafe_allow_html=True)


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
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
          <h1 style="font-size: 2.2rem; font-weight: 700; color: #F4F4EE; margin: 0 0 0.35rem;">
            Container Delay Risk Triage
          </h1>
          <p style="font-size: 0.95rem; color: #BDBDB4; margin: 0; line-height: 1.5;">
            Predict container dwell delay risk and receive immediate operational guidance.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode = st.segmented_control("Mode", ["Score a container", "Replay a real stay"], default="Score a container", label_visibility="collapsed") or "Score a container"
    if mode == "Score a container":
        score_mode()
    else:
        replay_mode()
