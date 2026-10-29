"""Noise Floor Variations & Reporting Drift Analysis Module.

Assigned Role: EDA Engineer

Responsibilities:
- Analyze seasonal demand cycles and noise floor variations in vessel call rates
- Evaluate reporting drift and temporal shifts across the 2021 operational window
- Investigate missing call date gaps in the dense operational window
- Correlate customs and quarantine events with baseline dwell distribution shifts

The event log is selected by GATE-OUT date, not by arrival, so stays are truncated at
both ends of the call-date axis. Every drift statement is therefore checked against a
"clean band" of call dates that are fully observable on both sides.
"""

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from scipy import stats

LABELS = {0: "ON_TIME", 1: "AT_RISK", 2: "CRITICAL_DELAY"}


class DriftAnalyzer:
    def __init__(self, dense_start: str = "2021-06-10", horizon_quantile: float = 0.99):
        self.dense_start = pd.Timestamp(dense_start)
        self.horizon_quantile = horizon_quantile

    # ------------------------------------------------------------------ observation window
    def observation_window(self, cases: pd.DataFrame) -> Dict:
        """Gate-out window of the extract, the dwell horizon and the fully observable call-date band."""
        out = pd.to_datetime(cases["truck_out_end_ts"])
        dis = pd.to_datetime(cases["discharge_start_ts"])
        call = pd.to_datetime(cases["call_date"])
        w_start, w_end = out.min().normalize(), out.max().normalize()
        ref = cases.loc[(call >= w_start) & (call <= w_end - pd.Timedelta(days=31)), "dwell_hours"]
        horizon_h = float(ref.quantile(self.horizon_quantile))
        clean_end = w_end - pd.Timedelta(days=int(np.ceil(horizon_h / 24.0)))
        return {
            "first_gate_out": out.min(), "last_gate_out": out.max(),
            "first_discharge": dis.min(), "last_discharge": dis.max(),
            "window_start": w_start, "window_end": w_end,
            "window_end_exclusive": w_end + pd.Timedelta(days=1),
            "horizon_hours": round(horizon_h, 1),
            "clean_start": w_start, "clean_end": clean_end,
            "reference_dwell": np.sort(ref.to_numpy()),
        }

    @staticmethod
    def clean_mask(cases: pd.DataFrame, band: Dict) -> pd.Series:
        call = pd.to_datetime(cases["call_date"])
        return (call >= band["clean_start"]) & (call <= band["clean_end"])

    @staticmethod
    def observable_share(cases: pd.DataFrame, band: Dict) -> pd.Series:
        """Per-stay probability that the extract could have observed a stay discharged at that time.

        A stay is visible only if its gate-out lies inside the window. Using the clean-band dwell
        distribution as reference, the share is F(window_end - discharge) on the right edge and
        1 - F(window_start - discharge) on the left edge.
        """
        ref = band["reference_dwell"]
        dis = pd.to_datetime(cases["discharge_start_ts"])
        right_h = (band["window_end_exclusive"] - dis).dt.total_seconds() / 3600.0
        left_h = (band["window_start"] - dis).dt.total_seconds() / 3600.0
        obs_right = np.searchsorted(ref, right_h.clip(lower=0), side="right") / len(ref)
        obs_left = np.where(left_h > 0, 1.0 - np.searchsorted(ref, left_h.clip(lower=0), side="left") / len(ref), 1.0)
        return pd.Series(np.minimum(obs_right, obs_left), index=cases.index)

    def truncation_profile(self, cases: pd.DataFrame, band: Dict) -> pd.DataFrame:
        """Weekly call cohorts with the estimated share of their stays that the extract could observe."""
        df = cases.copy()
        df["observable_share"] = self.observable_share(cases, band)
        df["call_week"] = pd.to_datetime(df["call_date"]).dt.to_period("W-SUN").dt.start_time
        g = df.groupby("call_week")
        out = pd.DataFrame({
            "stays": g.size(),
            "median_dwell_h": g["dwell_hours"].median().round(1),
            "p85_dwell_h": g["dwell_hours"].quantile(0.85).round(1),
            "mean_observable_share": g["observable_share"].mean().round(3),
        })
        out["est_missing_pct"] = ((1 - out["mean_observable_share"]) * 100).round(1)
        out["in_clean_band"] = [
            bool(wk >= band["clean_start"] and wk + pd.Timedelta(days=6) <= band["clean_end"]) for wk in out.index
        ]
        return out.reset_index()

    def cutoff_comparison(self, cases: pd.DataFrame, band: Dict, p_on: float, p_crit: float) -> pd.DataFrame:
        """Dwell cutoffs and label mix: train-fitted cutoffs versus a clean-band refit, by split."""
        rows = []
        clean = cases[self.clean_mask(cases, band)]
        for scope, sub in [("train (as built)", cases[cases["split_tag"] == "train"]),
                           ("clean band, all splits", clean),
                           ("clean band, train only", clean[clean["split_tag"] == "train"])]:
            rows.append({"scope": scope, "n": len(sub),
                         "p50_h": round(float(sub["dwell_hours"].quantile(0.50)), 2),
                         "p85_h": round(float(sub["dwell_hours"].quantile(0.85)), 2)})
        out = pd.DataFrame(rows)
        out["note"] = f"label cutoffs in use: p50={p_on:.2f}h p85={p_crit:.2f}h"
        return out

    def split_label_mix(self, cases: pd.DataFrame, band: Dict) -> pd.DataFrame:
        """Share of each class by split, on all stays and on the clean band only."""
        frames = []
        for scope, sub in [("all stays", cases), ("clean band", cases[self.clean_mask(cases, band)])]:
            sub = sub[sub["split_tag"].isin(["train", "val", "test"])]
            mix = pd.crosstab(sub["split_tag"], sub["target_class"], normalize="index").rename(columns=LABELS)
            mix.insert(0, "n", sub.groupby("split_tag").size())
            mix.insert(0, "scope", scope)
            frames.append(mix.reset_index())
        out = pd.concat(frames, ignore_index=True).rename(columns={"split_tag": "split"})
        return out.round(4)

    def weekly_dwell(self, cases: pd.DataFrame, band: Dict, p_on: float, p_crit: float) -> pd.DataFrame:
        """Weekly call-cohort dwell quantiles and critical rate, flagged for clean-band membership."""
        df = cases.copy()
        df["call_week"] = pd.to_datetime(df["call_date"]).dt.to_period("W-SUN").dt.start_time
        g = df.groupby("call_week")
        out = pd.DataFrame({
            "stays": g.size(),
            "p50_h": g["dwell_hours"].quantile(0.50),
            "p85_h": g["dwell_hours"].quantile(0.85),
            "pct_above_p_on_time": g["dwell_hours"].apply(lambda s: (s > p_on).mean() * 100),
            "pct_critical": g["dwell_hours"].apply(lambda s: (s > p_crit).mean() * 100),
        }).round(2)
        out["in_clean_band"] = [
            bool(wk >= band["clean_start"] and wk + pd.Timedelta(days=6) <= band["clean_end"]) for wk in out.index
        ]
        return out.reset_index()

    def clean_band_trend(self, cases: pd.DataFrame, band: Dict) -> Dict:
        """Is there real dwell drift once truncation is removed? Spearman of call date vs dwell, plus weekly slope."""
        sub = cases[self.clean_mask(cases, band)]
        day = (pd.to_datetime(sub["call_date"]) - band["clean_start"]).dt.days
        rho, p = stats.spearmanr(day, sub["dwell_hours"])
        wk = self.weekly_dwell(sub, band, 0, 0)
        wk = wk[wk["stays"] >= 200]
        slope = float(np.polyfit(np.arange(len(wk)), wk["p50_h"], 1)[0]) if len(wk) >= 3 else np.nan
        return {"n": len(sub), "spearman_rho": round(float(rho), 4), "p_value": float(p),
                "weekly_median_slope_h_per_week": round(slope, 2),
                "weekly_median_range_h": (round(float(wk["p50_h"].min()), 1), round(float(wk["p50_h"].max()), 1))}

    # ------------------------------------------------------------------ call-volume rhythm
    def daily_calls(self, cases: pd.DataFrame) -> pd.DataFrame:
        call = pd.to_datetime(cases["call_date"])
        daily = call.value_counts().sort_index()
        full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        out = daily.reindex(full, fill_value=0).rename_axis("call_date").rename("stays").reset_index()
        out["weekday"] = out["call_date"].dt.day_name()
        out["is_dense"] = out["call_date"] >= self.dense_start
        return out

    def missing_call_dates(self, daily: pd.DataFrame, band: Dict) -> pd.DataFrame:
        gaps = daily[(daily["stays"] == 0) & daily["is_dense"]].copy()
        gaps["in_clean_band"] = (gaps["call_date"] >= band["clean_start"]) & (gaps["call_date"] <= band["clean_end"])
        return gaps[["call_date", "weekday", "in_clean_band"]].reset_index(drop=True)

    def weekday_rhythm(self, daily: pd.DataFrame, band: Dict) -> pd.DataFrame:
        """Call volume by weekday inside the clean band (days with no calls included as zero)."""
        d = daily[(daily["call_date"] >= band["clean_start"]) & (daily["call_date"] <= band["clean_end"])]
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        g = d.groupby("weekday")["stays"]
        out = pd.DataFrame({"days": g.size(), "mean_stays": g.mean().round(1), "median_stays": g.median(),
                            "zero_call_days": g.apply(lambda s: int((s == 0).sum()))})
        return out.reindex(order).reset_index()

    def cycles(self, cases: pd.DataFrame, band: Dict) -> pd.DataFrame:
        """Stay count and median dwell by discharge weekday x hour (clean band)."""
        sub = cases[self.clean_mask(cases, band)]
        g = sub.groupby(["discharge_day_of_week", "discharge_hour"])["dwell_hours"]
        return pd.DataFrame({"stays": g.size(), "median_dwell_h": g.median().round(1)}).reset_index()

    # ------------------------------------------------------------------ noise floor
    def departure_daily(self, cases: pd.DataFrame) -> pd.DataFrame:
        """Daily series indexed by gate-out date. Unlike arrival cohorts it is not truncated."""
        df = cases.copy()
        df["dep"] = pd.to_datetime(df["truck_out_end_ts"]).dt.normalize()
        df["delayed"] = (df["target_class"] >= 1).astype(int)
        g = df.groupby("dep")
        out = pd.DataFrame({"volume": g.size(), "delayed": g["delayed"].sum(),
                            "mean_dwell_h": g["dwell_hours"].mean(), "median_dwell_h": g["dwell_hours"].median()})
        out = out.reindex(pd.date_range(out.index.min(), out.index.max(), freq="D"))
        out["volume"] = out["volume"].fillna(0)
        out["delayed"] = out["delayed"].fillna(0)
        out["delay_share"] = out["delayed"] / out["volume"].replace(0, np.nan)
        return out.rename_axis("departure_date").reset_index()

    def noise_floor(self, dep: pd.DataFrame, window: int = 7, detrend_days: int = 21) -> pd.DataFrame:
        """Day-to-day variation left after removing the local level (centered rolling median).

        observed_sigma is the robust spread of the residual. For delay_share, binomial_sigma is
        the spread pure sampling noise would give at the typical daily volume; a ratio above 1
        means real operational variation sits on top of sampling noise.
        """
        d = dep.set_index("departure_date")
        rows = []

        def resid_sigma(s: pd.Series) -> float:
            r = (s - s.rolling(detrend_days, center=True, min_periods=detrend_days // 2).median()).dropna()
            return float(1.4826 * np.median(np.abs(r - np.median(r))))

        trailing = pd.DataFrame({
            "volume_7d": d["volume"].rolling(window).sum(),
            "delay_share_7d": d["delayed"].rolling(window).sum() / d["volume"].rolling(window).sum().replace(0, np.nan),
        })
        series = {"daily_volume": d["volume"], "daily_delay_share": d["delay_share"],
                  "daily_median_dwell_h": d["median_dwell_h"], "trailing_7d_delay_share": trailing["delay_share_7d"],
                  "trailing_7d_volume": trailing["volume_7d"]}
        typical_n = float(d["volume"].median())
        for name, s in series.items():
            s = s.dropna()
            sigma = resid_sigma(s)
            level = float(s.median())
            row = {"metric": name, "typical_level": round(level, 3), "observed_sigma": round(sigma, 3),
                   "min_detectable_shift_2sigma": round(2 * sigma, 3), "binomial_sigma": np.nan, "overdispersion": np.nan}
            if name == "daily_delay_share":
                b = float(np.sqrt(level * (1 - level) / typical_n))
                row.update(binomial_sigma=round(b, 3), overdispersion=round(sigma / b, 2))
            elif name == "trailing_7d_delay_share":
                b = float(np.sqrt(level * (1 - level) / (typical_n * window)))
                row.update(binomial_sigma=round(b, 3), overdispersion=round(sigma / b, 2))
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ reporting drift
    def reporting_drift(self, events: pd.DataFrame, cases: pd.DataFrame, band: Dict,
                        case_col: str = "case_id", activity_col: str = "activity") -> pd.DataFrame:
        """Weekly share of cases that record each activity (rows: call week; clean-band weeks only)."""
        call = cases.set_index(case_col)["call_date"]
        ev = events[events[case_col].isin(call.index)][[case_col, activity_col]].drop_duplicates()
        ev["call_week"] = pd.to_datetime(ev[case_col].map(call)).dt.to_period("W-SUN").dt.start_time
        weeks = self.weekly_dwell(cases, band, 0, 0)
        keep = weeks.loc[weeks["in_clean_band"] & (weeks["stays"] >= 200), "call_week"]
        n_cases = cases.assign(call_week=pd.to_datetime(cases["call_date"]).dt.to_period("W-SUN").dt.start_time)
        n_cases = n_cases.groupby("call_week").size()
        cov = ev.groupby(["call_week", activity_col]).size().unstack(fill_value=0)
        cov = cov.div(n_cases, axis=0).loc[lambda x: x.index.isin(keep)] * 100
        return cov.round(2)

    @staticmethod
    def coverage_variation(cov: pd.DataFrame, weekly_n: pd.Series, min_mean_pct: float = 1.0) -> pd.DataFrame:
        """Week-to-week spread of each activity's coverage against what sampling noise alone allows.

        ratio = observed sd / binomial sd. A ratio near 1 means weeks differ only by chance; well above 1
        means coverage really changes between weeks (cargo mix or recording practice; the log cannot say which).
        """
        n = float(weekly_n.reindex(cov.index).mean())
        rows = []
        for act in cov.columns:
            p = cov[act].mean() / 100.0
            if p * 100 < min_mean_pct:
                continue
            binom = float(np.sqrt(p * (1 - p) / n) * 100)
            sd = float(cov[act].std(ddof=1))
            rows.append({"activity": act, "mean_pct": round(p * 100, 1), "min_pct": round(float(cov[act].min()), 1),
                         "max_pct": round(float(cov[act].max()), 1), "observed_sd_pp": round(sd, 2),
                         "binomial_sd_pp": round(binom, 2), "ratio": round(sd / binom, 1) if binom > 0 else np.nan})
        return pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)

    def weekday_dwell(self, cases: pd.DataFrame, band: Dict) -> pd.DataFrame:
        """Dwell by discharge weekday (clean band, model splits)."""
        sub = cases[self.clean_mask(cases, band) & cases["split_tag"].isin(["train", "val", "test"])]
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        g = sub.groupby("discharge_day_of_week")["dwell_hours"]
        out = pd.DataFrame({"stays": g.size(), "median_h": g.median().round(1), "p85_h": g.quantile(0.85).round(1),
                            "pct_critical": (sub.groupby("discharge_day_of_week")["target_class"].apply(lambda s: (s == 2).mean()) * 100).round(1)})
        out.index = [names[i] for i in out.index]
        return out.rename_axis("discharge_weekday").reset_index()

    @staticmethod
    def departure_episode(dep: pd.DataFrame, window: int = 7) -> Dict:
        """Where departures slowed most and where the delayed share of departing stays peaked."""
        d = dep.set_index("departure_date")
        vol7 = d["volume"].rolling(window).mean().dropna()
        share7 = (d["delayed"].rolling(window).sum() / d["volume"].rolling(window).sum().replace(0, np.nan)).dropna()
        return {"min_vol7_date": vol7.idxmin(), "min_vol7": round(float(vol7.min()), 0), "median_vol7": round(float(vol7.median()), 0),
                "peak_share_date": share7.idxmax(), "peak_share": round(float(share7.max()), 3),
                "late_share": round(float(share7.tail(30).median()), 3)}

    def attribute_drift(self, cases: pd.DataFrame, band: Dict, columns: Sequence[str]) -> pd.DataFrame:
        """Weekly null rate of case attributes (clean band)."""
        sub = cases[self.clean_mask(cases, band)].copy()
        sub["call_week"] = pd.to_datetime(sub["call_date"]).dt.to_period("W-SUN").dt.start_time
        return (sub.groupby("call_week")[list(columns)].apply(lambda g: g.isna().mean()) * 100).round(3)

    # ------------------------------------------------------------------ distribution shift
    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:  # binary or near-constant: compare category shares instead
            return DriftAnalyzer._psi_cat(pd.Series(expected).astype(str), pd.Series(actual).astype(str))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
        e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
        return float(np.sum((a - e) * np.log(a / e)))

    @staticmethod
    def _psi_cat(expected: pd.Series, actual: pd.Series, top: int = 10) -> float:
        keep = expected.value_counts().head(top).index
        e = expected.where(expected.isin(keep), "OTHER").value_counts(normalize=True)
        a = actual.where(actual.isin(keep), "OTHER").value_counts(normalize=True)
        idx = e.index.union(a.index)
        e, a = e.reindex(idx, fill_value=0).clip(lower=1e-4), a.reindex(idx, fill_value=0).clip(lower=1e-4)
        return float(np.sum((a - e) * np.log(a / e)))

    def split_shift(self, cases: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]) -> pd.DataFrame:
        """PSI (and KS for numerics) of val and test against train for every feature."""
        tr = cases[cases["split_tag"] == "train"]
        rows = []
        for col in list(numeric) + list(categorical):
            is_num = col in numeric
            row = {"feature": col, "kind": "numeric" if is_num else "categorical",
                   "constant_in_train": bool(tr[col].nunique(dropna=True) <= 1)}
            for name in ("val", "test"):
                other = cases[cases["split_tag"] == name]
                if is_num:
                    a, b = tr[col].dropna().to_numpy(float), other[col].dropna().to_numpy(float)
                    row[f"psi_{name}"] = round(self._psi(a, b), 4)
                    row[f"ks_{name}"] = round(float(stats.ks_2samp(a, b)[0]), 4)
                else:
                    row[f"psi_{name}"] = round(self._psi_cat(tr[col].astype(str), other[col].astype(str)), 4)
                    row[f"ks_{name}"] = np.nan
            rows.append(row)
        out = pd.DataFrame(rows)
        out["max_psi"] = out[["psi_val", "psi_test"]].max(axis=1)
        return out.sort_values("max_psi", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ label timing
    def label_maturity(self, cases: pd.DataFrame) -> pd.DataFrame:
        """Train/val stays whose outcome becomes known only after the next split begins."""
        rows = []
        out_ts = pd.to_datetime(cases["truck_out_end_ts"])
        for name, nxt in [("train", "val"), ("val", "test")]:
            boundary = pd.to_datetime(cases.loc[cases["split_tag"] == nxt, "call_date"]).min()
            sub = cases[cases["split_tag"] == name]
            late = sub[out_ts[sub.index] >= boundary]
            rows.append({"split": name, "next_split_starts": boundary.date(), "stays": len(sub),
                         "outcome_known_after_boundary": len(late),
                         "pct_of_split": round(100 * len(late) / len(sub), 2),
                         "pct_critical_among_late": round(100 * (late["target_class"] == 2).mean(), 1),
                         "pct_critical_overall": round(100 * (sub["target_class"] == 2).mean(), 1)})
        return pd.DataFrame(rows)

    def cold_start(self, cases: pd.DataFrame) -> Dict:
        """Rows whose trailing metrics rest on an empty window."""
        tr = cases[cases["split_tag"] == "train"]
        empty = tr["trailing_7d_volume"].fillna(0) == 0
        return {"rows": int(empty.sum()), "pct_of_train": round(100 * float(empty.mean()), 2),
                "call_date_min": tr.loc[empty, "call_date"].min(), "call_date_max": tr.loc[empty, "call_date"].max(),
                "null_trailing_critical": int(tr["trailing_7d_critical"].isna().sum())}

    # ------------------------------------------------------------------ customs / quarantine
    def flag_effects(self, cases: pd.DataFrame, band: Dict, flags: Sequence[str]) -> pd.DataFrame:
        """Dwell with and without each customs/quarantine flag (clean band), plus stability across splits."""
        sub = cases[self.clean_mask(cases, band) & cases["split_tag"].isin(["train", "val", "test"])]
        rows = []
        for f in flags:
            on, off = sub[sub[f] == 1], sub[sub[f] == 0]
            u = stats.mannwhitneyu(on["dwell_hours"], off["dwell_hours"], alternative="two-sided")[1] if len(on) and len(off) else np.nan
            row = {"flag": f, "stays_with": len(on), "share_pct": round(100 * len(on) / len(sub), 2),
                   "median_with_h": round(float(on["dwell_hours"].median()), 1),
                   "median_without_h": round(float(off["dwell_hours"].median()), 1),
                   "median_ratio": round(float(on["dwell_hours"].median() / off["dwell_hours"].median()), 2),
                   "pct_critical_with": round(100 * float((on["target_class"] == 2).mean()), 1),
                   "pct_critical_without": round(100 * float((off["target_class"] == 2).mean()), 1),
                   "mannwhitney_p": float(u)}
            for s in ("train", "val", "test"):
                a, b = on[on["split_tag"] == s], off[off["split_tag"] == s]
                row[f"ratio_{s}"] = round(float(a["dwell_hours"].median() / b["dwell_hours"].median()), 2) if len(a) and len(b) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)
