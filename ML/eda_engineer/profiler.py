"""Telemetry Profiling & Data Quality Audit Module.

Assigned Role: EDA Engineer

Responsibilities:
- Profile event cardinality, distinct cases, and container counts
- Audit activity frequencies, milestone transitions, and customs/quarantine events
- Calculate empirical dwell time percentiles across container categories
- Surface data quality anomalies (unparseable timestamps, midnight signatures, negative durations)
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99)

# Prediction-time feature eligibility tiers aligned with ML/data_engineer/feature_contracts.py
PREDICTION_ELIGIBILITY: Dict[str, str] = {
    # Eligible at container discharge milestone (ARRIVAL_TIME scenario)
    "carrier_prefix": "ARRIVAL_TIME_ELIGIBLE",
    "CTR_SIZE": "ARRIVAL_TIME_ELIGIBLE",
    "CTR_TYPE": "ARRIVAL_TIME_ELIGIBLE",
    "GROSS": "ARRIVAL_TIME_ELIGIBLE",
    "YARD_BLOCK": "ARRIVAL_TIME_ELIGIBLE",
    "YARD_SLOT": "ARRIVAL_TIME_ELIGIBLE",
    "discharge_hour": "ARRIVAL_TIME_ELIGIBLE",
    "discharge_day_of_week": "ARRIVAL_TIME_ELIGIBLE",
    "discharge_month": "ARRIVAL_TIME_ELIGIBLE",
    "is_weekend": "ARRIVAL_TIME_ELIGIBLE",
    "trailing_7d_volume": "ARRIVAL_TIME_ELIGIBLE",
    "trailing_7d_critical": "ARRIVAL_TIME_ELIGIBLE",
    "trailing_7d_at_risk": "ARRIVAL_TIME_ELIGIBLE",
    "historical_delay_rate_7d": "ARRIVAL_TIME_ELIGIBLE",
    "trailing_7d_mean_dwell": "ARRIVAL_TIME_ELIGIBLE",
    "weather_severity_index": "ARRIVAL_TIME_ELIGIBLE",  # (constant 0 in 2021)
    "has_active_disruption": "ARRIVAL_TIME_ELIGIBLE",   # (constant 0 in 2021)
    # Conditional pre-arrival (requires explicit pre-arrival EDI notification assumption)
    "has_quarantine_flag": "CONDITIONAL_PRE_ARRIVAL",
    "has_red_flag_pib": "CONDITIONAL_PRE_ARRIVAL",
    # Post-arrival / mid-stay / post-discharge diagnostic variables (FORBIDDEN at arrival)
    "JOB_DEL_DOCTYPE": "POST_DISCHARGE_DIAGNOSTIC",
    "total_events_in_stay": "POST_DISCHARGE_DIAGNOSTIC",
    "has_behandle_inspection": "POST_DISCHARGE_DIAGNOSTIC",
}


class EventLogProfiler:
    def __init__(
        self,
        case_col: str = "case_id",
        activity_col: str = "activity",
        start_ts_col: str = "Start Timestamp",
        end_ts_col: str = "End Timestamp",
        discharge_act: str = "DISCHARGE",
        truck_out_act: str = "TRUCK_OUT",
        plausible_min: str = "2021-01-01",
        plausible_max: str = "2022-12-31",
    ):
        self.case_col = case_col
        self.activity_col = activity_col
        self.start_ts_col = start_ts_col
        self.end_ts_col = end_ts_col
        self.discharge_act = discharge_act
        self.truck_out_act = truck_out_act
        self.plausible_min = pd.Timestamp(plausible_min)
        self.plausible_max = pd.Timestamp(plausible_max)

    # ------------------------------------------------------------------ cardinality
    def cardinality(self, events: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
        """Headline counts: events, cases, containers, carriers, activities, splits."""
        per_container = events.drop_duplicates(self.case_col).groupby("container_id").size()
        has_exit = set(events.loc[events[self.activity_col] == self.truck_out_act, self.case_col])
        call_dates = pd.to_datetime(cases["call_date"])
        rows = [
            ("event_rows", len(events)),
            ("distinct_cases", events[self.case_col].nunique()),
            ("distinct_containers", events["container_id"].nunique()),
            ("containers_with_multiple_stays", int((per_container > 1).sum())),
            ("distinct_carrier_prefixes", events["carrier_prefix"].nunique()),
            ("distinct_activities", events[self.activity_col].nunique()),
            ("cases_without_truck_out", events[self.case_col].nunique() - len(has_exit)),
            ("labeled_stays", len(cases)),
            ("first_call_date", call_dates.min().date()),
            ("last_call_date", call_dates.max().date()),
            ("distinct_call_dates", call_dates.nunique()),
        ]
        for split, n in cases["split_tag"].value_counts().items():
            rows.append((f"stays_in_{split}", int(n)))
        return pd.DataFrame(rows, columns=["metric", "value"])

    def call_date_alignment(self, cases: pd.DataFrame) -> pd.DataFrame:
        """Days between the call_date embedded in case_id and the actual discharge date."""
        gap = (pd.to_datetime(cases["discharge_start_ts"]).dt.normalize() - pd.to_datetime(cases["call_date"])).dt.days
        out = gap.value_counts().sort_index().rename("stays").to_frame()
        out["share_pct"] = (out["stays"] / len(gap) * 100).round(3)
        return out.rename_axis("discharge_minus_call_days").reset_index()

    # ------------------------------------------------------------------ activities
    def activity_frequencies(self, events: pd.DataFrame) -> pd.DataFrame:
        """Per-activity event counts, case coverage and share of events missing a start timestamp."""
        total_cases = events[self.case_col].nunique()
        g = events.groupby(self.activity_col)
        out = pd.DataFrame({
            "events": g.size(),
            "cases": g[self.case_col].nunique(),
            "median_event_minutes": g["event_duration_minutes"].median().round(1),
            "pct_missing_start": (g[self.start_ts_col].apply(lambda s: s.isna().mean()) * 100).round(3),
        })
        out["coverage_pct"] = (out["cases"] / total_cases * 100).round(2)
        out["events_per_case"] = (out["events"] / out["cases"]).round(3)
        return out.sort_values("cases", ascending=False).reset_index()

    def milestone_timing(self, events: pd.DataFrame) -> pd.DataFrame:
        """Start-time offset of each activity relative to DISCHARGE start (hours).

        `pct_before_discharge` shows which activities are knowable at the moment of
        discharge and which only happen in the yard; implausible timestamps are excluded.
        Date-only timestamps (e.g., midnight 00:00:00) are flagged and evaluated at
        calendar-day precision to avoid intra-day discharge hour artifacts.
        """
        ev = events[[self.case_col, self.activity_col, self.start_ts_col, self.end_ts_col]].copy()
        ts = ev[self.start_ts_col].fillna(ev[self.end_ts_col])
        ev["ts"] = ts.where((ts >= self.plausible_min) & (ts <= self.plausible_max))
        discharge = (
            ev[ev[self.activity_col] == self.discharge_act]
            .drop_duplicates(self.case_col)
            .set_index(self.case_col)["ts"]
        )
        ev = ev[ev[self.activity_col] != self.discharge_act].copy()
        dis_ts = ev[self.case_col].map(discharge)
        ev["offset_h"] = (ev["ts"] - dis_ts).dt.total_seconds() / 3600.0
        ev = ev.dropna(subset=["offset_h"])

        # Date-only precision check (events where timestamp is exactly midnight 00:00:00)
        ev["is_midnight"] = (ev["ts"] == ev["ts"].dt.normalize())
        ev["calendar_day_diff"] = (ev["ts"].dt.normalize() - dis_ts.dt.normalize()).dt.days

        out_rows = []
        for act_name, group in ev.groupby(self.activity_col):
            n_act = len(group)
            pct_before = round(float((group["offset_h"] < 0).mean() * 100), 1)
            p10 = round(float(group["offset_h"].quantile(0.10)), 1)
            med = round(float(group["offset_h"].median()), 1)
            p90 = round(float(group["offset_h"].quantile(0.90)), 1)

            is_date_only = bool(group["is_midnight"].mean() > 0.90)
            precision = "date_only" if is_date_only else "datetime"
            pct_prior_day = round(float((group["calendar_day_diff"] < 0).mean() * 100), 1)
            pct_same_day = round(float((group["calendar_day_diff"] == 0).mean() * 100), 1)

            if is_date_only:
                timing_note = (
                    "Date-only midnight (00:00:00); sub-day negative offset reflects discharge hour-of-day "
                    "rather than true pre-discharge completion"
                )
            else:
                timing_note = "Continuous datetime precision"

            out_rows.append({
                self.activity_col: act_name,
                "n": n_act,
                "pct_before_discharge": pct_before,
                "p10_h": p10,
                "median_h": med,
                "p90_h": p90,
                "timestamp_precision": precision,
                "pct_calendar_prior": pct_prior_day,
                "pct_calendar_same_day": pct_same_day,
                "timing_note": timing_note,
            })

        out = pd.DataFrame(out_rows)
        return out.sort_values("median_h").reset_index(drop=True)

    # ------------------------------------------------------------------ quality ledger
    def timestamp_quality(self, events: pd.DataFrame) -> pd.DataFrame:
        """Ledger of timestamp and ordering defects, each with count, share of events and a note."""
        n = len(events)
        start, end = events[self.start_ts_col], events[self.end_ts_col]
        act = events[self.activity_col]
        rows: List[Dict] = []

        def add(check: str, count: int, denom: int, note: str) -> None:
            rows.append({"check": check, "count": int(count), "share_pct": round(100 * count / max(denom, 1), 4), "note": note})

        add("null_start_timestamp", start.isna().sum(), n, "Start missing; end is still present.")
        add("null_end_timestamp", end.isna().sum(), n, "End missing.")
        bad_start = (start < self.plausible_min) | (start > self.plausible_max)
        bad_end = (end < self.plausible_min) | (end > self.plausible_max)
        add("implausible_start_year", bad_start.sum(), n,
            f"Start outside {self.plausible_min.date()}..{self.plausible_max.date()}; activities: "
            f"{act[bad_start].value_counts().to_dict()}")
        add("implausible_end_year", bad_end.sum(), n, "End outside the plausible window.")
        add("end_before_start", (end < start).sum(), n, "Negative event duration.")

        midnight = events.assign(_mid=(end == end.dt.normalize()) & end.notna()).groupby(self.activity_col)["_mid"].mean()
        for name in midnight[midnight > 0.99].index:
            add(f"date_only_end_timestamp:{name}", (act == name).sum(), n,
                "Every End is exactly 00:00:00, so the event has date precision only.")

        has_start = start.notna()
        ties = events[has_start].duplicated([self.case_col, self.start_ts_col], keep=False).sum()
        add("tied_start_timestamps_within_case", ties, n, "Rows sharing a start time inside one case.")

        duration_raw = events["Duration"] if "Duration" in events.columns else pd.Series(dtype=object)
        if len(duration_raw):
            parsed = pd.to_timedelta(duration_raw.astype(str).where(duration_raw.notna()), errors="coerce")
            add("duration_column_null", duration_raw.isna().sum(), n, "Pre-computed Duration is empty.")
            unparsed = duration_raw.notna() & parsed.isna()
            excel_like = duration_raw[unparsed].astype(str).str.startswith("1900-").sum()
            add("duration_column_unparseable", unparsed.sum(), n,
                f"Not HH:MM:SS; {excel_like} of these are Excel durations over 24h stored as 1900-01-0X dates. "
                "Use event_duration_minutes instead.")

        starts = events[has_start]
        first = lambda a: starts[starts[self.activity_col] == a].groupby(self.case_col)[self.start_ts_col].min()  # noqa: E731
        unstack, truck_out, discharge = first("UNSTACK_TO_TRUCK"), first(self.truck_out_act), first(self.discharge_act)
        both = pd.concat([unstack.rename("u"), truck_out.rename("t")], axis=1).dropna()
        add("truck_out_starts_before_unstack", (both["t"] < both["u"]).sum(), len(both),
            "TRUCK_OUT start precedes UNSTACK_TO_TRUCK start; the steps overlap, so do not assume sequence "
            "(denominator is cases having both).")
        release = events[(act == "CUSTOMS_DEL") & end.notna()].groupby(self.case_col)[self.end_ts_col].min()
        rel = pd.concat([release.rename("r"), discharge.rename("d")], axis=1).dropna()
        rel = rel[(rel["r"] >= self.plausible_min)]
        # CUSTOMS_DEL carries date precision only, so compare calendar dates, not timestamps
        add("customs_release_date_before_discharge_date", (rel["r"].dt.normalize() < rel["d"].dt.normalize()).sum(), len(rel),
            "Customs release dated before the discharge date; clearance may legitimately precede discharge, so not necessarily an error.")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ dwell
    @staticmethod
    def _quantile_row(s: pd.Series) -> Dict[str, float]:
        row = {"n": int(len(s)), "mean": round(float(s.mean()), 2)}
        for q in PERCENTILES:
            row[f"p{int(q * 100)}"] = round(float(s.quantile(q)), 2)
        return row

    def dwell_percentiles(self, cases: pd.DataFrame, by: Optional[Sequence[str]] = None, min_n: int = 30) -> pd.DataFrame:
        """Dwell-hour percentile table: overall and for each grouping column."""
        rows = [{"group_by": "ALL", "level": "all", **self._quantile_row(cases["dwell_hours"])}]
        for col in by or []:
            for level, sub in cases.groupby(col, dropna=False, observed=True):
                if len(sub) >= min_n:
                    rows.append({"group_by": col, "level": str(level), **self._quantile_row(sub["dwell_hours"])})
        return pd.DataFrame(rows)

    def missingness(self, cases: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Share of nulls per column, overall and by split."""
        out = cases[list(columns)].isna().mean().rename("overall").to_frame()
        for split, sub in cases.groupby("split_tag"):
            out[split] = sub[list(columns)].isna().mean()
        return (out * 100).round(3).reset_index().rename(columns={"index": "column"})

    def threshold_diagnostics(self, cases: pd.DataFrame, p_on_time: float, p_critical: float, seed: int = 7) -> Dict:
        """Does the p50/p85 pair sit on real structure in log-dwell? KDE modes, troughs, cutoff density."""
        d = cases["dwell_hours"]
        sample = np.log10(d[d > 0].sample(min(len(d), 30000), random_state=seed).to_numpy())
        grid = np.linspace(sample.min(), sample.max(), 600)
        density = stats.gaussian_kde(sample)(grid)
        peaks, _ = find_peaks(density, prominence=0.05 * density.max())
        troughs, _ = find_peaks(-density, prominence=0.05 * density.max())
        at = lambda hours: float(np.interp(np.log10(hours), grid, density) / density.max())  # noqa: E731
        return {
            "modes_hours": [round(float(10 ** grid[i]), 1) for i in peaks],
            "troughs_hours": [round(float(10 ** grid[i]), 1) for i in troughs],
            "relative_density_at_p_on_time": round(at(p_on_time), 3),
            "relative_density_at_p_critical": round(at(p_critical), 3),
            "log10_skew": round(float(stats.skew(sample)), 3),
            "grid": grid, "density": density,
        }

    # ------------------------------------------------------------------ signal screen
    def feature_signal(self, train: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]) -> pd.DataFrame:
        """Univariate association of features with dwell (train only), annotated with prediction-time eligibility tiers."""
        rows = []
        for col in numeric:
            s = train[[col, "dwell_hours"]].dropna()
            tier = PREDICTION_ELIGIBILITY.get(col, "UNCLASSIFIED")
            is_arr = (tier == "ARRIVAL_TIME_ELIGIBLE")
            if s[col].nunique() < 2:
                rows.append({
                    "feature": col,
                    "kind": "numeric",
                    "association": np.nan,
                    "n": len(s),
                    "note": "constant in train",
                    "eligibility_tier": tier,
                    "arrival_eligible": is_arr,
                })
                continue
            rho = stats.spearmanr(s[col], s["dwell_hours"])[0]
            rows.append({
                "feature": col,
                "kind": "numeric",
                "association": round(float(rho), 4),
                "n": len(s),
                "note": "Spearman rho vs dwell",
                "eligibility_tier": tier,
                "arrival_eligible": is_arr,
            })
        for col in categorical:
            s = train[[col, "dwell_hours"]].dropna()
            tier = PREDICTION_ELIGIBILITY.get(col, "UNCLASSIFIED")
            is_arr = (tier == "ARRIVAL_TIME_ELIGIBLE")
            top = s[col].value_counts().loc[lambda v: v >= 30].index
            s = s[s[col].isin(top)]
            if s[col].nunique() < 2:
                rows.append({
                    "feature": col,
                    "kind": "categorical",
                    "association": np.nan,
                    "n": len(s),
                    "note": "single level",
                    "eligibility_tier": tier,
                    "arrival_eligible": is_arr,
                })
                continue
            groups = [g["dwell_hours"].to_numpy() for _, g in s.groupby(col, observed=True)]
            h = stats.kruskal(*groups)[0]
            eps2 = max((h - len(groups) + 1) / (len(s) - len(groups)), 0.0)
            rows.append({
                "feature": col,
                "kind": "categorical",
                "association": round(float(eps2), 4),
                "n": len(s),
                "note": "Kruskal epsilon^2 vs dwell",
                "eligibility_tier": tier,
                "arrival_eligible": is_arr,
            })
        out = pd.DataFrame(rows)
        out["abs_assoc"] = out["association"].abs()
        return out.sort_values("abs_assoc", ascending=False).drop(columns="abs_assoc").reset_index(drop=True)

    @staticmethod
    def filter_eligible_signals(signal_df: pd.DataFrame, scenario: str = "ARRIVAL_TIME") -> pd.DataFrame:
        """Filters feature signal results to only include predictors valid under the declared scenario.

        Args:
            signal_df: DataFrame output by feature_signal().
            scenario: 'ARRIVAL_TIME' (strictly arrival-safe) or 'IN_YARD_DYNAMIC' or 'ALL'.
        """
        if scenario == "ARRIVAL_TIME":
            return signal_df[signal_df["arrival_eligible"] == True].copy().reset_index(drop=True)
        elif scenario == "IN_YARD_DYNAMIC":
            return signal_df[signal_df["eligibility_tier"].isin(
                ["ARRIVAL_TIME_ELIGIBLE", "CONDITIONAL_PRE_ARRIVAL", "POST_DISCHARGE_DIAGNOSTIC"]
            )].copy().reset_index(drop=True)
        return signal_df.copy().reset_index(drop=True)

    @staticmethod
    def split_feature_signals(signal_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separates feature signal analysis into eligible arrival predictors vs retrospective diagnostics."""
        arrival_df = signal_df[signal_df["arrival_eligible"] == True].copy().reset_index(drop=True)
        diagnostic_df = signal_df[signal_df["arrival_eligible"] == False].copy().reset_index(drop=True)
        return arrival_df, diagnostic_df
