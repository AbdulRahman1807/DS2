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
CATEGORICAL = ["carrier_prefix", "CTR_TYPE", "YARD_BLOCK", "doc_type_known", "yard_zone", "slot_depth_tier"]

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


# Container-serial distance scopes; same-day customs concordance 0.74 / 0.52 / 0.39 vs 0.27 far apart.
SIBLING_WINDOWS = {"s20": 20, "s1k": 1_000, "s10k": 10_000}
SIBLING_MILESTONES = ["CUSTOMS_DEL", "JOB_DEL", "HAS_QUARANTINE_FLAG", "TRUCK_IN"]
CALENDAR_PATH = "data/reference/indonesia_calendar_2021.csv"
PORTWATCH_PATH = "data/raw/portwatch_daily_ports.parquet"
PORTWATCH_LAG_DAYS = 7  # PortWatch publishes weekly; only values at least a week old are usable


def sibling_state(all_cases: pd.DataFrame, query: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """As-of status of probable consignment-mates: same vessel call and owner prefix ('grp'),
    narrowed by container-serial distance (SIBLING_WINDOWS). Close serials usually belong to the
    same declaration, so their observed customs / delivery progress at t is informative."""
    hour = 3600e9
    key = ["vessel_call_id", "carrier_prefix"]
    left = pd.DataFrame({
        "q": np.arange(len(query)),
        "vessel_call_id": query["vessel_call_id"].to_numpy(),
        "carrier_prefix": query["carrier_prefix"].to_numpy(),
        "serial": pd.to_numeric(query["container_id"].str[4:10], errors="coerce").to_numpy(),
        "case_id": query["case_id"].to_numpy(),
        "t": t,
    })
    right = pd.DataFrame({
        "vessel_call_id": all_cases["vessel_call_id"].to_numpy(),
        "carrier_prefix": all_cases["carrier_prefix"].to_numpy(),
        "s_serial": pd.to_numeric(all_cases["container_id"].str[4:10], errors="coerce").to_numpy(),
        "s_case": all_cases["case_id"].to_numpy(),
        "s_out": _ns(all_cases["truck_out_end_ts"]),
        **{f"s_{m}": _ns(all_cases[f"avail_{m}"]) for m in SIBLING_MILESTONES},
    })
    pairs = left.merge(right, on=key)
    pairs = pairs[pairs["s_case"] != pairs["case_id"]]
    dist = (pairs["serial"] - pairs["s_serial"]).abs().to_numpy()
    flags = {"departed": pairs["s_out"] <= pairs["t"]}
    for m in SIBLING_MILESTONES:
        flags[m] = pairs[f"s_{m}"] <= pairs["t"]
    flag_df = pd.DataFrame(flags).astype(float)
    jobdel_age = np.where(flags["JOB_DEL"], (pairs["t"] - pairs["s_JOB_DEL"]) / hour, np.nan)
    flag_df["jobdel_age"] = jobdel_age
    flag_df["q"] = pairs["q"].to_numpy()

    out = pd.DataFrame(index=np.arange(len(query)))
    scopes = [("grp", flag_df)] + [(name, flag_df[dist <= w]) for name, w in SIBLING_WINDOWS.items()]
    for scope, frame in scopes:
        g = frame.groupby("q")
        out[f"sib_{scope}_n"] = g.size().reindex(out.index).fillna(0).to_numpy()
        for col in ["departed"] + SIBLING_MILESTONES:
            out[f"sib_{scope}_share_{col}"] = g[col].mean().reindex(out.index).to_numpy()
        out[f"sib_{scope}_max_jobdel_age_h"] = g["jobdel_age"].max().reindex(out.index).to_numpy()
    return out


def calendar_state(t: np.ndarray, path: str = CALENDAR_PATH) -> pd.DataFrame:
    """Restriction level, holidays and working hours ahead (all announced in advance)."""
    cal = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    t_ts = pd.to_datetime(t.astype("int64"))
    day = t_ts.normalize()
    feats = pd.DataFrame(index=np.arange(len(t)))
    feats["ppkm_level"] = pd.to_numeric(cal["ppkm_level"].reindex(day), errors="coerce").to_numpy()
    feats["is_holiday_today"] = cal["is_public_holiday"].reindex(day).to_numpy()
    working = cal["is_working_day"]
    for days_ahead in (1, 2, 3):
        window = sum(working.reindex(day + pd.Timedelta(days=k)).fillna(1).to_numpy() for k in range(1, days_ahead + 1))
        feats[f"working_days_next_{days_ahead}d"] = window
    level_change = cal["ppkm_level"].ne(cal["ppkm_level"].shift())
    last_change = pd.Series(cal.index.where(level_change), index=cal.index).ffill()
    feats["days_since_ppkm_change"] = ((day - pd.to_datetime(last_change.reindex(day).to_numpy())).days).to_numpy()
    return feats


def portwatch_state(t: np.ndarray, path: str = PORTWATCH_PATH) -> pd.DataFrame:
    """Port-wide AIS activity for Surabaya, using only data at least PORTWATCH_LAG_DAYS old."""
    pw = pd.read_parquet(path, columns=["date", "portcalls_container", "import_container", "portcalls"])
    pw = pw.set_index(pd.to_datetime(pw["date"]).dt.normalize()).drop(columns="date").sort_index()
    roll7 = pw.rolling("7D").mean()
    roll28 = pw.rolling("28D").mean()
    asof_day = pd.to_datetime(t.astype("int64")).normalize() - pd.Timedelta(days=PORTWATCH_LAG_DAYS)
    feats = pd.DataFrame(index=np.arange(len(t)))
    for col in pw.columns:
        r7 = roll7[col].reindex(asof_day, method="ffill").to_numpy()
        r28 = roll28[col].reindex(asof_day, method="ffill").to_numpy()
        feats[f"pw_{col}_7d"] = r7
        feats[f"pw_{col}_7d_vs_28d"] = r7 / np.where(r28 > 0, r28, np.nan)
    return feats


# Functional yard zones (ported from the yard/vessel feature investigation). Quarantine/reefer
# blocks match the quarantine-flag allocation observed in the audit; the rest follow block letters.
REEFER_QUARANTINE_BLOCKS = {"Y", "YY", "Z", "ZZ", "W1", "W3"}
MAIN_IMPORT_BLOCKS = {"I", "J", "K", "L", "M", "N", "O", "P"}
OVERFLOW_BLOCKS = {"H", "S", "BB", "XP", "XO", "XXP", "D"}


def yard_state(all_cases: pd.DataFrame, query: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """Yard zone, slot-depth tier and as-of inventory of the container's own block."""
    blocks = query["YARD_BLOCK"].astype("string").str.strip()
    zone = np.select(
        [blocks.isin(REEFER_QUARANTINE_BLOCKS), blocks.isin(MAIN_IMPORT_BLOCKS), blocks.isin(OVERFLOW_BLOCKS), blocks.isna()],
        ["ZONE_REEFER_QUARANTINE", "ZONE_MAIN_IMPORT", "ZONE_OVERFLOW", "ZONE_UNKNOWN"],
        "ZONE_SECONDARY",
    )
    slot = pd.to_numeric(query["YARD_SLOT"], errors="coerce").to_numpy()
    depth = np.select(
        [np.isnan(slot), slot <= 40, slot <= 80, slot <= 120], ["SLOT_UNKNOWN", "SLOT_FRONT", "SLOT_MID", "SLOT_DEEP"], "SLOT_FAR"
    )

    d0_all, out_all = _ns(all_cases["discharge_start_ts"]), _ns(all_cases["truck_out_end_ts"])
    blocks_all = all_cases["YARD_BLOCK"].fillna("UNKNOWN").astype(str).to_numpy()
    q_blocks = query["YARD_BLOCK"].fillna("UNKNOWN").astype(str).to_numpy()
    total = np.maximum(_count_le(np.sort(d0_all), t) - _count_le(np.sort(out_all), t), 1)
    block_inv = np.zeros(len(query))
    for b in np.unique(q_blocks):
        idx = np.where(q_blocks == b)[0]
        m = blocks_all == b
        block_inv[idx] = _count_le(np.sort(d0_all[m]), t[idx]) - _count_le(np.sort(out_all[m]), t[idx])
    return pd.DataFrame({
        "yard_zone": zone, "slot_depth_tier": depth,
        "block_inventory": block_inv, "block_inventory_share": block_inv / total,
    })


def manifest_state(all_cases: pd.DataFrame, query: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """Vessel-call manifest composition as known at t: size/owner mix over call members whose
    stowage plan (BAPLIE) is available by t; mean gross weight over members discharged by t
    (GROSS is recorded on the DISCHARGE event; values above 100 are kilograms and are converted)."""
    result = pd.DataFrame(index=np.arange(len(query)), columns=["call_share_40", "call_mean_gross", "call_n_carriers"], dtype=float)
    groups = all_cases.groupby("vessel_call_id").indices
    q_calls = query["vessel_call_id"].to_numpy()
    for call, q_idx in pd.Series(np.arange(len(query))).groupby(q_calls).groups.items():
        q_idx = np.asarray(q_idx)
        members = all_cases.iloc[groups[call]]
        baplie = _ns(members["avail_BAPLIE"])
        d0 = _ns(members["discharge_start_ts"])
        size40 = (members["CTR_SIZE"] == 40).to_numpy()
        gross = members["GROSS"].to_numpy(dtype=float)
        gross = np.where(gross > 100, gross / 1000.0, gross)  # 18.9% of rows are recorded in kg, the rest in tonnes
        owners = members["carrier_prefix"].to_numpy()
        for i in q_idx:
            known = baplie <= t[i]
            disch = d0 <= t[i]
            result.iloc[i] = [
                size40[known].mean() if known.any() else np.nan,
                np.nanmean(gross[disch]) if disch.any() else np.nan,
                len(set(owners[known])) if known.any() else np.nan,
            ]
    return result


def manifest_state_whole_call(all_cases: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """Original whole-call definition (uses members discharged after t). For verification only."""
    stats = all_cases.groupby("vessel_call_id").agg(
        call_share_40=("CTR_SIZE", lambda x: float((x == 40).mean())),
        call_mean_gross=("GROSS", "mean"),
        call_n_carriers=("carrier_prefix", "nunique"),
    )
    return query[["vessel_call_id"]].merge(stats, on="vessel_call_id", how="left").drop(columns="vessel_call_id").astype(float)


def build_asof_features(
    all_cases: pd.DataFrame, query: pd.DataFrame, t: pd.Series, extended: bool = False, extra: tuple = ()
) -> pd.DataFrame:
    """Feature matrix for `query` rows evaluated at prediction timestamps `t` (aligned to query).
    `extended` adds consignment-sibling, calendar and PortWatch blocks (Stage 03 data additions);
    `extra` may contain "yard" and/or "manifest"."""
    query = query.reset_index(drop=True)
    t_ns = _ns(pd.Series(pd.to_datetime(t.to_numpy())))
    parts = [
        container_state(query, t_ns),
        call_state(all_cases, query, t_ns),
        terminal_state(all_cases, t_ns),
    ]
    if extended:
        parts += [sibling_state(all_cases, query, t_ns), calendar_state(t_ns), portwatch_state(t_ns)]
    if "yard" in extra:
        parts.append(yard_state(all_cases, query, t_ns))
    if "manifest" in extra:
        parts.append(manifest_state(all_cases, query, t_ns))
    return pd.concat(parts, axis=1)


def feature_columns(frame: pd.DataFrame, exclude: List[str] | None = None) -> List[str]:
    exclude = set(exclude or [])
    return [c for c in frame.columns if c not in exclude]
