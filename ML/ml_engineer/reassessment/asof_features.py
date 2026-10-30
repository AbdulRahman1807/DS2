"""As-Of Feature Builder.

Assigned Role: ML Research (Stage 03 Forensic Reassessment)

Builds features for a stay at an arbitrary prediction timestamp t using only events whose
availability timestamp (see cohort._availability) is <= t. Three groups:
- Container: static manifest/discharge attributes, milestone states and ages at t.
- Vessel call: as-of shares of the same call that are cleared / flagged / released / departed.
- Terminal: yard inventory, uncleared inventory and trailing departure statistics at t.

Terminal and call aggregates are computed over *all* reconstructed stays (not only the cohort),
because the terminal operating system sees every container physically in the yard.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ML.ml_engineer.reassessment.cohort import EXTRACT_GATE_OUT_START, P50_HOURS, P85_HOURS

STATUS_MILESTONES = [
    "HAS_QUARANTINE_FLAG",
    "SP_RED_FLAG_PIB",
    "FIRST_STACK_QUARANTINE_BLK",
    "CUSTOMS_DEL",
    "QUARANTINE_RELEASE",
    "FIRST_JOB_CUSTOMS_BEHANDLE",
    "FIRST_STACK_BEHANDLE",
    "LAST_COMPLETE_BEHANDLE",
    "STACK_MAIN_YARD",
    "STACK_TRK_BF_MAIN_YARD",
    "JOB_PLP",
    "JOB_DEL",
    "TRUCK_IN",
]
CALL_STATUS_MILESTONES = ["CUSTOMS_DEL", "HAS_QUARANTINE_FLAG", "JOB_DEL"]
CATEGORICAL = ["carrier_prefix", "CTR_TYPE", "YARD_BLOCK", "doc_type_known"]

# Feature groups used for ablations.
CUSTOMS_GROUP = [
    "known_CUSTOMS_DEL", "age_CUSTOMS_DEL", "customs_rel_discharge_h",
    "known_HAS_QUARANTINE_FLAG", "age_HAS_QUARANTINE_FLAG",
    "known_SP_RED_FLAG_PIB", "age_SP_RED_FLAG_PIB",
    "call_share_CUSTOMS_DEL", "call_share_HAS_QUARANTINE_FLAG", "YARD_BLOCK",
]


def _count_le(sorted_values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_values, query, side="right")


def _count_lt(sorted_values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_values, query, side="left")


def _ns(ts: pd.Series) -> np.ndarray:
    """Timestamps as int64 nanoseconds with NaT mapped to +inf-like sentinel."""
    values = ts.to_numpy(dtype="datetime64[ns]").astype("int64").astype("float64")
    values[ts.isna().to_numpy()] = np.inf
    return values


def _window_stats(
    event_ts: np.ndarray, values: Dict[str, np.ndarray], t: np.ndarray, window_ns: float
) -> Dict[str, np.ndarray]:
    """Counts and means of `values` over events with timestamp in [t - window, t)."""
    order = np.argsort(event_ts, kind="stable")
    ts_sorted = event_ts[order]
    finite = np.isfinite(ts_sorted)
    ts_sorted = ts_sorted[finite]
    hi = _count_lt(ts_sorted, t)
    lo = _count_lt(ts_sorted, t - window_ns)
    count = (hi - lo).astype(float)
    out = {"count": count}
    for name, v in values.items():
        csum = np.concatenate([[0.0], np.cumsum(v[order][finite])])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(count > 0, (csum[hi] - csum[lo]) / count, np.nan)
    return out


def terminal_state(all_cases: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """Yard-wide state at each prediction time (all inputs observable at t)."""
    d0 = _ns(all_cases["discharge_start_ts"])
    out = _ns(all_cases["truck_out_end_ts"])
    cust = _ns(all_cases["avail_CUSTOMS_DEL"])
    job = _ns(all_cases["avail_JOB_DEL"])
    dwell = all_cases["dwell_hours"].to_numpy()

    d0_s, out_s = np.sort(d0), np.sort(out)
    inventory = _count_le(d0_s, t) - _count_le(out_s, t)
    cleared_or_gone = np.sort(np.minimum(np.maximum(d0, np.where(np.isfinite(cust), cust, out)), out))
    released_or_gone = np.sort(np.minimum(np.maximum(d0, np.where(np.isfinite(job), job, out)), out))
    cleared_in_yard = _count_le(cleared_or_gone, t) - _count_le(out_s, t)
    released_in_yard = _count_le(released_or_gone, t) - _count_le(out_s, t)

    hour, day = 3600e9, 86400e9
    feats = {
        "yard_inventory": inventory,
        "yard_uncleared": inventory - cleared_in_yard,
        "yard_unreleased": inventory - released_in_yard,
        "discharges_24h": _count_lt(d0_s, t) - _count_lt(d0_s, t - day),
    }
    feats["yard_uncleared_share"] = feats["yard_uncleared"] / np.maximum(inventory, 1)

    dep_values = {
        "mean_dwell": dwell,
        "crit_rate": (dwell > P85_HOURS).astype(float),
        "ontime_rate": (dwell <= P50_HOURS).astype(float),
    }
    dep24 = _window_stats(out, {}, t, day)
    dep7 = _window_stats(out, dep_values, t, 7 * day)
    post_lag = np.where(np.isfinite(cust), np.maximum(cust - d0, 0.0) / hour, np.nan)
    clr7 = _window_stats(np.where(np.isfinite(post_lag), cust, np.inf), {"lag": np.nan_to_num(post_lag)}, t, 7 * day)

    # Departures before 2021-07-01 are absent from the extract: mask windows that reach before it.
    observable_24 = t - day >= EXTRACT_GATE_OUT_START.value
    observable_7 = t - 7 * day >= EXTRACT_GATE_OUT_START.value
    feats["departures_24h"] = np.where(observable_24, dep24["count"], np.nan)
    feats["departures_7d"] = np.where(observable_7, dep7["count"], np.nan)
    for name in dep_values:
        feats[f"dep7d_{name}"] = np.where(observable_7, dep7[name], np.nan)
    feats["clear7d_post_discharge_lag_h"] = np.where(observable_7, clr7["lag"], np.nan)
    feats["clearances_7d"] = np.where(observable_7, clr7["count"], np.nan)
    return pd.DataFrame(feats)


def call_state(all_cases: pd.DataFrame, query: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """Shares of the same vessel call that have reached each milestone by t."""
    result = pd.DataFrame(index=np.arange(len(query)), dtype=float)
    cols = ["call_size", "call_discharge_rank_frac", "call_share_departed"] + [
        f"call_share_{m}" for m in CALL_STATUS_MILESTONES
    ]
    for c in cols:
        result[c] = np.nan

    groups = all_cases.groupby("vessel_call_id").indices
    q_calls = query["vessel_call_id"].to_numpy()
    q_d0 = _ns(query["discharge_start_ts"])
    for call, q_idx in pd.Series(np.arange(len(query))).groupby(q_calls).groups.items():
        q_idx = np.asarray(q_idx)
        members = all_cases.iloc[groups[call]]
        size = len(members)
        tq = t[q_idx]
        result.loc[q_idx, "call_size"] = size
        result.loc[q_idx, "call_discharge_rank_frac"] = (
            _count_le(np.sort(_ns(members["discharge_start_ts"])), q_d0[q_idx]) / size
        )
        result.loc[q_idx, "call_share_departed"] = _count_le(np.sort(_ns(members["truck_out_end_ts"])), tq) / size
        for m in CALL_STATUS_MILESTONES:
            result.loc[q_idx, f"call_share_{m}"] = _count_le(np.sort(_ns(members[f"avail_{m}"])), tq) / size
    return result


def container_state(query: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """Static attributes plus milestone status and age at t."""
    hour = 3600e9
    d0 = _ns(query["discharge_start_ts"])
    t_ts = pd.to_datetime(t.astype("int64"))
    feats = pd.DataFrame(index=np.arange(len(query)))
    feats["carrier_prefix"] = query["carrier_prefix"].to_numpy()
    feats["CTR_TYPE"] = query["CTR_TYPE"].to_numpy()
    feats["YARD_BLOCK"] = query["YARD_BLOCK"].to_numpy()
    feats["CTR_SIZE"] = query["CTR_SIZE"].to_numpy()
    feats["GROSS"] = query["GROSS"].to_numpy()
    feats["YARD_SLOT"] = query["YARD_SLOT"].to_numpy()
    d0_ts = pd.to_datetime(query["discharge_start_ts"]).reset_index(drop=True)
    feats["discharge_hour"] = d0_ts.dt.hour.to_numpy()
    feats["discharge_dow"] = d0_ts.dt.dayofweek.to_numpy()
    feats["pred_hour"] = t_ts.hour.to_numpy()
    feats["pred_dow"] = t_ts.dayofweek.to_numpy()
    feats["elapsed_h"] = (t - d0) / hour
    feats["baplie_lead_h"] = (d0 - _ns(query["avail_BAPLIE"])) / hour
    feats["atb_to_discharge_h"] = (d0 - _ns(query["avail_VESSEL_ATB"])) / hour

    for m in STATUS_MILESTONES:
        avail = _ns(query[f"avail_{m}"])
        known = avail <= t
        feats[f"known_{m}"] = known.astype(int)
        feats[f"age_{m}"] = np.where(known, (t - avail) / hour, np.nan)
    cust = _ns(query["avail_CUSTOMS_DEL"])
    feats["customs_rel_discharge_h"] = np.where(cust <= t, (cust - d0) / hour, np.nan)
    doc = query["JOB_DEL_DOCTYPE"].fillna("NONE").to_numpy()
    feats["doc_type_known"] = np.where(feats["known_JOB_DEL"].to_numpy() == 1, doc, "PENDING")
    return feats


def build_asof_features(all_cases: pd.DataFrame, query: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    """Feature matrix for `query` rows evaluated at prediction timestamps `t` (aligned to query)."""
    query = query.reset_index(drop=True)
    t_ns = _ns(pd.Series(pd.to_datetime(t.to_numpy())))
    parts = [
        container_state(query, t_ns),
        call_state(all_cases, query, t_ns),
        terminal_state(all_cases, t_ns),
    ]
    return pd.concat(parts, axis=1)


def feature_columns(frame: pd.DataFrame, exclude: List[str] | None = None) -> List[str]:
    exclude = set(exclude or [])
    return [c for c in frame.columns if c not in exclude]
