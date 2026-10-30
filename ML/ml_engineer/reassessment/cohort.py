"""Selection-Aware Case Table, Evaluation Cohort and Pre-Registered Temporal Folds.

Assigned Role: ML Research (Stage 03 Forensic Reassessment)

Why this module exists:
- The source extract contains only stays whose TRUCK_OUT falls in [2021-07-01, 2021-09-30].
  The sample is therefore selected on the outcome: early discharges are present only if they
  dwelled long (left truncation) and late discharges are present only if they left quickly
  (right censoring). Stage 01/02 partitions inherited both biases.
- Every milestone receives an explicit *availability* timestamp. CUSTOMS_DEL carries
  date-only precision (00:00 marks the start of the clearance day: 59% of JOB_DEL events
  follow within 24h, 0.15% precede it), so it is only treated as known at date + 24h.

Cohort definition (fixed before any Stage 03 model was fitted):
- discharge_start_ts >= 2021-07-01: no left truncation (any such stay gates out inside the window).
- discharge_start_ts <  2021-10-01 - 600h: >= 600h follow-up, so < 2.4% of CRITICAL stays
  (dwell > 600h) can be missing from the extract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

CLASS_NAMES = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
P50_HOURS = 65.1102
P85_HOURS = 160.8993

EXTRACT_GATE_OUT_START = pd.Timestamp("2021-07-01")
EXTRACT_GATE_OUT_END = pd.Timestamp("2021-10-01")  # exclusive
FOLLOW_UP_HOURS = 600.0
COHORT_START = EXTRACT_GATE_OUT_START
COHORT_END = EXTRACT_GATE_OUT_END - pd.Timedelta(hours=FOLLOW_UP_HOURS)  # 2021-09-06 00:00

# Labels of a stay are determined by discharge + 160.9h at the latest, so a 7-day gap between the
# last training discharge and the first evaluated discharge means every training label was knowable.
LABEL_EMBARGO = pd.Timedelta(days=7)

MILESTONES = [
    "BAPLIE",
    "VESSEL_ATB",
    "DISCHARGE",
    "STACK",
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
    "UNSTACK_TO_TRUCK",
    "TRUCK_OUT",
]

# Activities recorded with a date stamp only: the event is known no earlier than the end of that day.
ALWAYS_DATE_PRECISION = {"CUSTOMS_DEL"}
MIDNIGHT_DATE_PRECISION = {"SP_RED_FLAG_PIB"}

# Pre-registered folds. Train on discharges in [COHORT_START, train_end); evaluate on
# discharges in [eval_start, eval_end). TEST is held out and is not scored during development.
DEV_FOLDS: List[Dict[str, str]] = [
    {"name": "F1", "train_end": "2021-07-19", "eval_start": "2021-07-26", "eval_end": "2021-08-02"},
    {"name": "F2", "train_end": "2021-07-29", "eval_start": "2021-08-05", "eval_end": "2021-08-12"},
    {"name": "F3", "train_end": "2021-08-08", "eval_start": "2021-08-15", "eval_end": "2021-08-22"},
]
HOLDOUT_TEST: Dict[str, str] = {
    "name": "TEST", "train_end": "2021-08-20", "eval_start": "2021-08-27", "eval_end": "2021-09-06",
}


def assign_target(dwell_hours: pd.Series) -> pd.Series:
    """Original Stage 01 three-class target with the documented thresholds."""
    return pd.Series(
        np.select([dwell_hours <= P50_HOURS, dwell_hours <= P85_HOURS], [0, 1], 2),
        index=dwell_hours.index,
    ).astype(int)


def _availability(end: pd.DataFrame, baplie: pd.Series) -> pd.DataFrame:
    """Timestamp at which each milestone is observable to the terminal."""
    avail = end.copy()
    for act in ALWAYS_DATE_PRECISION & set(avail.columns):
        avail[act] = avail[act] + pd.Timedelta(hours=24)
    for act in MIDNIGHT_DATE_PRECISION & set(avail.columns):
        at_midnight = avail[act].dt.hour.eq(0) & avail[act].dt.minute.eq(0) & avail[act].dt.second.eq(0)
        avail.loc[at_midnight, act] = avail.loc[at_midnight, act] + pd.Timedelta(hours=24)
    # Dates more than 90 days before the stowage plan (e.g. 1954-07-14) are recording errors.
    implausible = avail.lt(baplie - pd.Timedelta(days=90), axis=0)
    return avail.mask(implausible)


def build_case_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per stay with attributes, milestone availability timestamps, dwell and target."""
    ev = events
    start = ev.pivot_table(index="case_id", columns="activity", values="Start Timestamp", aggfunc="min")
    end = ev.pivot_table(index="case_id", columns="activity", values="End Timestamp", aggfunc="min")
    end = end.reindex(columns=MILESTONES)

    def first_value(col: str) -> pd.Series:
        return ev.loc[ev[col].notna()].drop_duplicates("case_id").set_index("case_id")[col]

    cases = pd.DataFrame(index=start.index)
    cases["container_id"] = ev.drop_duplicates("case_id").set_index("case_id")["container_id"]
    cases["carrier_prefix"] = ev.drop_duplicates("case_id").set_index("case_id")["carrier_prefix"]
    cases["call_date"] = ev.drop_duplicates("case_id").set_index("case_id")["call_date"]
    for col in ["CTR_SIZE", "CTR_TYPE", "GROSS", "YARD_BLOCK", "YARD_SLOT", "JOB_DEL_DOCTYPE"]:
        cases[col] = first_value(col)
    cases["YARD_SLOT"] = pd.to_numeric(cases["YARD_SLOT"], errors="coerce")

    cases["discharge_start_ts"] = start["DISCHARGE"]
    cases["truck_out_end_ts"] = end["TRUCK_OUT"]
    cases = cases.dropna(subset=["discharge_start_ts", "truck_out_end_ts"])
    cases["dwell_hours"] = (
        cases["truck_out_end_ts"] - cases["discharge_start_ts"]
    ).dt.total_seconds() / 3600.0
    cases["target_class"] = assign_target(cases["dwell_hours"])
    # One berthing timestamp per vessel call (235 calls, identical to call_date grouping).
    cases["vessel_call_id"] = end["VESSEL_ATB"].reindex(cases.index).dt.strftime("%Y-%m-%d %H:%M")

    avail = _availability(end.reindex(cases.index), end["BAPLIE"].reindex(cases.index))
    for act in MILESTONES:
        cases[f"avail_{act}"] = avail[act]

    d0 = cases["discharge_start_ts"]
    cases["in_cohort"] = (d0 >= COHORT_START) & (d0 < COHORT_END)
    cases.index.name = "case_id"
    return cases.reset_index()


def load_case_table(path: str = "data/interim/cleaned_event_log.parquet") -> pd.DataFrame:
    return build_case_table(pd.read_parquet(path))


def fold_masks(cases: pd.DataFrame, fold: Dict[str, str]) -> tuple[pd.Series, pd.Series]:
    """Train/eval masks on discharge_start_ts inside the selection-aware cohort."""
    d0 = cases["discharge_start_ts"]
    cohort = cases["in_cohort"]
    train = cohort & (d0 < pd.Timestamp(fold["train_end"]))
    evaluate = cohort & (d0 >= pd.Timestamp(fold["eval_start"])) & (d0 < pd.Timestamp(fold["eval_end"]))
    gap = pd.Timestamp(fold["eval_start"]) - pd.Timestamp(fold["train_end"])
    if gap < LABEL_EMBARGO:
        raise ValueError(f"Fold {fold['name']} violates the {LABEL_EMBARGO} label embargo")
    return train, evaluate


def save_case_table(cases: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_parquet(path, index=False)
    return path
