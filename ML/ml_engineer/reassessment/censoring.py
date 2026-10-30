"""Censoring-Aware Label Construction for Arrival-Based Cohorts.

Assigned Role: ML Research (Stage 03 Data Recovery)

For an extract taken at `cutoff`, every arrival in the observation window enters with one of:
- exact class: the stay ended before the cutoff;
- CRITICAL: still in the yard more than 160.9 h after discharge (dwell > 160.9 h is already true);
- partial label {AT_RISK, CRITICAL}: still in the yard, 65.1 h < elapsed <= 160.9 h;
- unknown: still in the yard, elapsed <= 65.1 h.
A missing TRUCK_OUT is never read as ON_TIME or CRITICAL by itself; only elapsed time decides.

Also provides a Kaplan-Meier estimate of class probabilities from right-censored dwell, and
the exit-window extraction used for the published dataset (to quantify its bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ML.ml_engineer.reassessment.cohort import P50_HOURS, P85_HOURS

UNKNOWN = -1
NOT_ON_TIME = -2  # partial label: AT_RISK or CRITICAL_DELAY


def label_status(discharge: pd.Series, truck_out: pd.Series, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Class information available for each arrival when the extract is taken at `cutoff`."""
    hour = pd.Timedelta(hours=1)
    exited = truck_out.notna() & (truck_out <= cutoff)
    dwell = (truck_out - discharge) / hour
    elapsed = (cutoff - discharge) / hour
    exact = pd.Series(np.select([dwell <= P50_HOURS, dwell <= P85_HOURS], [0, 1], 2), index=discharge.index)
    label = np.where(
        exited, exact,
        np.where(elapsed > P85_HOURS, 2, np.where(elapsed > P50_HOURS, NOT_ON_TIME, UNKNOWN)),
    )
    return pd.DataFrame({
        "observed_exit": exited,
        "censored_dwell_h": np.where(exited, dwell, elapsed),
        "label": label.astype(int),
        "label_known": label >= 0,
    }, index=discharge.index)


def complete_label_arrival_window(cutoff: pd.Timestamp) -> pd.Timestamp:
    """Latest discharge time whose class is determined at `cutoff` for every stay (if unfinished stays are kept)."""
    return cutoff - pd.Timedelta(hours=P85_HOURS)


def exit_window_extract(discharge: pd.Series, truck_out: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Selection used by the published dataset: keep stays whose gate-out falls in [start, end)."""
    return truck_out.notna() & (truck_out >= start) & (truck_out < end) & (discharge < end)


def kaplan_meier_class_probabilities(duration_h: np.ndarray, event: np.ndarray) -> np.ndarray:
    """P(ON_TIME), P(AT_RISK), P(CRITICAL) from right-censored dwell via the Kaplan-Meier estimator."""
    order = np.argsort(duration_h, kind="stable")
    d, e = np.asarray(duration_h)[order], np.asarray(event)[order].astype(bool)
    at_risk = len(d) - np.arange(len(d))
    surv_steps = np.cumprod(np.where(e, 1.0 - 1.0 / at_risk, 1.0))

    def survival(t: float) -> float:
        idx = np.searchsorted(d, t, side="right") - 1
        return 1.0 if idx < 0 else float(surv_steps[idx])

    s50, s85 = survival(P50_HOURS), survival(P85_HOURS)
    return np.array([1.0 - s50, s50 - s85, s85])
