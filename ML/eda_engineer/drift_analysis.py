"""Exploratory Data Analysis - Drift, Non-Stationarity and Observation Horizon Analysis.

Assigned Role: EDA Engineer

Responsibilities:
- Extract-window truncation and observation horizon diagnostics (gate-out date selection).
- Identification of the unbiased "clean band" (2021-07-01 to 2021-09-12).
- Non-stationarity and operational drift (July congestion crisis vs calm September).
- Lagged departure velocity vs daily dwell dynamics.
- Operational noise floor of 7-day trailing delay rate vs binomial sampling noise.
- Milestone reporting coverage drift across weeks.
- Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests between splits and synthetic data.
- Vessel call rhythms, Thursday liner schedule gaps, and diurnal discharge cycles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class DriftAnalyzer:
    """Analyzes temporal drift, extract truncation, noise floors, and distribution shift."""

    def __init__(self, dense_start: str = "2021-06-10", horizon_quantile: float = 0.99):
        self.dense_start = pd.Timestamp(dense_start)
        self.horizon_quantile = horizon_quantile

    # ------------------------------------------------------------------ observation window & clean band
    def observation_window(self, cases: pd.DataFrame) -> Dict:
        """Gate-out window of the extract, the dwell horizon, and the fully observable call-date band."""
        out = pd.to_datetime(cases["truck_out_end_ts"])
        dis = pd.to_datetime(cases["discharge_start_ts"])
        call = pd.to_datetime(cases["call_date"])
        w_start, w_end = out.min().normalize(), out.max().normalize()
        
        # Reference dwell inside the dense interior window
        ref = cases.loc[(call >= w_start) & (call <= w_end - pd.Timedelta(days=31)), "dwell_hours"]
        if len(ref) == 0:
            ref = cases["dwell_hours"].dropna()
            
        horizon_h = float(ref.quantile(self.horizon_quantile))
        clean_end = w_end - pd.Timedelta(days=int(np.ceil(horizon_h / 24.0)))
        
        return {
            "first_gate_out": out.min(),
            "last_gate_out": out.max(),
            "first_discharge": dis.min(),
            "last_discharge": dis.max(),
            "window_start": w_start,
            "window_end": w_end,
            "window_end_exclusive": w_end + pd.Timedelta(days=1),
            "horizon_hours": round(horizon_h, 1),
            "clean_start": w_start,
            "clean_end": clean_end,
            "reference_dwell": np.sort(ref.to_numpy()),
        }

    @staticmethod
    def clean_mask(cases: pd.DataFrame, band: Dict) -> pd.Series:
        """Boolean mask for stays falling strictly inside the clean call-date band."""
        call = pd.to_datetime(cases["call_date"]).dt.normalize()
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
            "est_missing_pct": ((1.0 - g["observable_share"].mean()) * 100).round(1),
            "in_clean_band": g["call_week"].first().between(band["clean_start"], band["clean_end"]),
        }).reset_index()
        out["call_week"] = out["call_week"].dt.strftime("%Y-%m-%d")
        return out

    # ------------------------------------------------------------------ weekly dwell & congestion
    def weekly_dwell(self, cases: pd.DataFrame, p50: float, p85: float) -> pd.DataFrame:
        """Weekly call cohorts with median dwell, p85 dwell, and percentage of stays exceeding p85."""
        df = cases.copy()
        df["call_week"] = pd.to_datetime(df["call_date"]).dt.to_period("W-SUN").dt.start_time
        g = df.groupby("call_week")
        out = pd.DataFrame({
            "stays": g.size(),
            "median_dwell_h": g["dwell_hours"].median().round(1),
            "p85_dwell_h": g["dwell_hours"].quantile(0.85).round(1),
            "pct_critical": (g["dwell_hours"].apply(lambda s: (s > p85).mean()) * 100).round(1),
            "pct_at_risk": (g["dwell_hours"].apply(lambda s: ((s > p50) & (s <= p85)).mean()) * 100).round(1),
            "pct_on_time": (g["dwell_hours"].apply(lambda s: (s <= p50).mean()) * 100).round(1),
        }).reset_index()
        out["call_week"] = out["call_week"].dt.strftime("%Y-%m-%d")
        return out

    # ------------------------------------------------------------------ rolling dynamics
    def rolling_dwell_dynamics(self, cases: pd.DataFrame, band: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Relate prior completed-departure velocity to daily dwell in the clean call-date band."""
        sub = cases[self.clean_mask(cases, band)].copy()
        sub["call_date"] = pd.to_datetime(sub["call_date"]).dt.normalize()
        # Restrict to when trailing 7-day history is available
        sub = sub[sub["call_date"] >= pd.Timestamp("2021-07-08")]
        
        sub["trailing_delay_rate"] = (
            (sub["trailing_7d_critical"] + sub["trailing_7d_at_risk"])
            / sub["trailing_7d_volume"].replace(0, np.nan)
        )
        daily = sub.groupby("call_date").agg(
            stays=("dwell_hours", "size"),
            dwell_p50_h=("dwell_hours", "median"),
            dwell_p85_h=("dwell_hours", lambda values: values.quantile(0.85)),
            trailing_7d_volume=("trailing_7d_volume", "median"),
            trailing_7d_delay_rate=("trailing_delay_rate", "median"),
        ).reset_index()

        comparisons = [
            ("trailing_7d_volume", "dwell_p50_h"),
            ("trailing_7d_volume", "dwell_p85_h"),
            ("trailing_7d_delay_rate", "dwell_p50_h"),
            ("trailing_7d_delay_rate", "dwell_p85_h"),
        ]
        rows = []
        for feature, outcome in comparisons:
            paired = daily[[feature, outcome]].dropna()
            rho, p_value = stats.spearmanr(paired[feature], paired[outcome]) if len(paired) >= 3 else (np.nan, np.nan)
            rows.append({
                "prior_departure_metric": feature,
                "daily_dwell_metric": outcome,
                "n_call_dates": len(paired),
                "spearman_rho": round(float(rho), 4) if pd.notna(rho) else np.nan,
                "p_value_descriptive": float(p_value) if pd.notna(p_value) else np.nan,
            })
        return daily, pd.DataFrame(rows)

    # ------------------------------------------------------------------ daily departures & noise floor
    @staticmethod
    def departures_daily(cases: pd.DataFrame) -> pd.DataFrame:
        """Daily completed departures indexed by gate-out date (free from extract edge censoring)."""
        df = cases.copy()
        df["dep_date"] = pd.to_datetime(df["truck_out_end_ts"]).dt.normalize()
        df["_is_delayed"] = df["target_class"] != 0 if "target_class" in df.columns else df["dwell_hours"] > 65.11
        out = df.groupby("dep_date").agg(
            departure_date=("dep_date", "first"),
            volume=("dep_date", "size"),
            delayed=("_is_delayed", "sum"),
            mean_dwell_h=("dwell_hours", lambda s: round(float(s.mean()), 2)),
            median_dwell_h=("dwell_hours", lambda s: round(float(s.median()), 2)),
        ).reset_index(drop=True)
        out["delay_share"] = out["delayed"] / out["volume"].replace(0, np.nan)
        return out.sort_values("departure_date").reset_index(drop=True)

    def noise_floor(self, dep: pd.DataFrame, window: int = 7, detrend_days: int = 21) -> pd.DataFrame:
        """Day-to-day variation left after removing the local level (centered rolling median).

        observed_sigma is the robust spread of the residual. For delay_share, binomial_sigma is
        the spread pure sampling noise would give at typical daily volume; a ratio above 1
        means real operational variation sits on top of sampling noise.
        """
        d = dep.set_index("departure_date")
        rows = []

        def resid_sigma(s: pd.Series) -> float:
            r = (s - s.rolling(detrend_days, center=True, min_periods=max(3, detrend_days // 2)).median()).dropna()
            return float(1.4826 * np.median(np.abs(r - np.median(r)))) if len(r) > 0 else np.nan

        trailing = pd.DataFrame({
            "volume_7d": d["volume"].rolling(window).sum(),
            "delay_share_7d": d["delayed"].rolling(window).sum() / d["volume"].rolling(window).sum().replace(0, np.nan),
        })
        series = {
            "daily_volume": d["volume"],
            "daily_delay_share": d["delay_share"],
            "daily_median_dwell_h": d["median_dwell_h"],
            "trailing_7d_delay_share": trailing["delay_share_7d"],
            "trailing_7d_volume": trailing["volume_7d"],
        }
        typical_n = float(d["volume"].median())
        for name, s in series.items():
            s = s.dropna()
            sigma = resid_sigma(s)
            level = float(s.median())
            row = {
                "metric": name,
                "typical_level": round(level, 3),
                "observed_sigma": round(sigma, 3),
                "min_detectable_shift_2sigma": round(2 * sigma, 3),
                "binomial_sigma": np.nan,
                "overdispersion": np.nan,
            }
            if name == "daily_delay_share":
                b = float(np.sqrt(level * (1 - level) / typical_n)) if typical_n > 0 and 0 < level < 1 else np.nan
                row.update(binomial_sigma=round(b, 3) if pd.notna(b) else np.nan,
                           overdispersion=round(sigma / b, 2) if pd.notna(b) and b > 0 else np.nan)
            elif name == "trailing_7d_delay_share":
                b = float(np.sqrt(level * (1 - level) / (typical_n * window))) if typical_n > 0 and 0 < level < 1 else np.nan
                row.update(binomial_sigma=round(b, 3) if pd.notna(b) else np.nan,
                           overdispersion=round(sigma / b, 2) if pd.notna(b) and b > 0 else np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def departure_episode(dep: pd.DataFrame, window: int = 7) -> Dict:
        """Where departures slowed most and where the delayed share of departing stays peaked."""
        d = dep.set_index("departure_date")
        vol7 = d["volume"].rolling(window).mean().dropna()
        share7 = (d["delayed"].rolling(window).sum() / d["volume"].rolling(window).sum().replace(0, np.nan)).dropna()
        return {
            "min_vol7_date": str(vol7.idxmin().date()) if len(vol7) else "–",
            "min_vol7": round(float(vol7.min()), 0) if len(vol7) else np.nan,
            "median_vol7": round(float(vol7.median()), 0) if len(vol7) else np.nan,
            "peak_share_date": str(share7.idxmax().date()) if len(share7) else "–",
            "peak_share": round(float(share7.max()), 3) if len(share7) else np.nan,
            "late_share": round(float(share7.tail(30).median()), 3) if len(share7) else np.nan,
        }

    # ------------------------------------------------------------------ reporting drift
    def reporting_drift(
        self,
        events: pd.DataFrame,
        cases: pd.DataFrame,
        band: Dict,
        case_col: str = "case_id",
        activity_col: str = "activity",
    ) -> pd.DataFrame:
        """Weekly share of cases that record each activity (clean-band weeks only)."""
        clean_cases = cases[self.clean_mask(cases, band)].copy()
        clean_cases["call_week"] = pd.to_datetime(clean_cases["call_date"]).dt.to_period("W-SUN").dt.start_time
        
        merged = events.merge(clean_cases[[case_col, "call_week"]], on=case_col, how="inner")
        
        weekly_totals = clean_cases.groupby("call_week").size()
        pivot = merged.groupby(["call_week", activity_col])[case_col].nunique().unstack(fill_value=0)
        
        cov = pivot.div(weekly_totals, axis=0) * 100.0
        cov.attrs["weekly_totals"] = weekly_totals
        return cov

    @staticmethod
    def coverage_variation(cov: pd.DataFrame, weekly_n: pd.Series, min_mean_pct: float = 1.0) -> pd.DataFrame:
        """Week-to-week spread of each activity's coverage against what sampling noise alone allows."""
        n = float(weekly_n.reindex(cov.index).mean()) if len(cov.index) else 1000.0
        rows = []
        for act in cov.columns:
            p = cov[act].mean() / 100.0
            if p * 100 < min_mean_pct:
                continue
            binom = float(np.sqrt(p * (1 - p) / n) * 100) if n > 0 and 0 < p < 1 else np.nan
            sd = float(cov[act].std(ddof=1)) if len(cov) > 1 else 0.0
            ratio = round(sd / binom, 1) if pd.notna(binom) and binom > 0 else np.nan
            rows.append({
                "activity": act,
                "mean_pct": round(p * 100, 1),
                "min_pct": round(float(cov[act].min()), 1),
                "max_pct": round(float(cov[act].max()), 1),
                "observed_sd_pp": round(sd, 2),
                "binomial_sd_pp": round(binom, 2) if pd.notna(binom) else np.nan,
                "ratio": ratio,
            })
        return pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ rhythms and cycles
    def call_rhythm(self, cases: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Daily vessel call rhythm, zero-call days, and weekday averages."""
        df = cases.copy()
        df["call_date"] = pd.to_datetime(df["call_date"]).dt.normalize()
        
        min_date = df[df["call_date"] >= self.dense_start]["call_date"].min()
        max_date = df["call_date"].max()
        all_dates = pd.date_range(min_date, max_date, freq="D")
        
        daily = df.groupby("call_date").size().reindex(all_dates, fill_value=0).reset_index()
        daily.columns = ["call_date", "stays"]
        daily["weekday"] = daily["call_date"].dt.day_name()
        daily["is_zero"] = daily["stays"] == 0
        
        # Weekday aggregates
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        w_agg = daily.groupby("weekday").agg(
            days=("stays", "size"),
            mean_stays=("stays", lambda s: round(float(s.mean()), 1)),
            median_stays=("stays", lambda s: round(float(s.median()), 1)),
            zero_call_days=("is_zero", "sum"),
        ).reindex(weekday_order).reset_index()
        
        return daily, w_agg

    def discharge_cycles(self, cases: pd.DataFrame, band: Dict) -> pd.DataFrame:
        """2D discharge weekday x hour distribution and median dwell (clean band)."""
        sub = cases[self.clean_mask(cases, band)].copy()
        if "discharge_hour" not in sub.columns:
            sub["discharge_hour"] = pd.to_datetime(sub["discharge_start_ts"]).dt.hour
        if "discharge_day_of_week" not in sub.columns:
            sub["discharge_day_of_week"] = pd.to_datetime(sub["discharge_start_ts"]).dt.dayofweek
            
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sub["weekday_name"] = sub["discharge_day_of_week"].apply(lambda i: names[int(i)] if 0 <= i <= 6 else "–")
        
        counts = sub.pivot_table(index="weekday_name", columns="discharge_hour", values="case_id",
                                 aggfunc="count", fill_value=0).reindex(names)
        dwells = sub.pivot_table(index="weekday_name", columns="discharge_hour", values="dwell_hours",
                                 aggfunc="median").reindex(names)
        dwells = dwells.where(counts >= 30, np.nan).round(1)
        return dwells

    def weekday_dwell(self, cases: pd.DataFrame, band: Dict) -> pd.DataFrame:
        """Dwell duration and critical delay rate by discharge weekday (clean band)."""
        sub = cases[self.clean_mask(cases, band) & cases["split_tag"].isin(["train", "val", "test"])].copy()
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if "discharge_day_of_week" not in sub.columns:
            sub["discharge_day_of_week"] = pd.to_datetime(sub["discharge_start_ts"]).dt.dayofweek
            
        g = sub.groupby("discharge_day_of_week")["dwell_hours"]
        out = pd.DataFrame({
            "stays": g.size(),
            "median_h": g.median().round(1),
            "p85_h": g.quantile(0.85).round(1),
            "pct_critical": (sub.groupby("discharge_day_of_week")["target_class"].apply(lambda s: (s == 2).mean()) * 100).round(1),
        })
        out.index = [names[i] for i in out.index]
        return out.rename_axis("discharge_weekday").reset_index()

    # ------------------------------------------------------------------ distribution shift (PSI/KS)
    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index for continuous or binary distributions."""
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return DriftAnalyzer._psi_cat(pd.Series(expected).astype(str), pd.Series(actual).astype(str))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
        e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
        return float(np.sum((a - e) * np.log(a / e)))

    @staticmethod
    def _psi_cat(expected: pd.Series, actual: pd.Series, top: int = 10) -> float:
        """Population Stability Index for categorical distributions."""
        keep = expected.value_counts().head(top).index
        e = expected.where(expected.isin(keep), "OTHER").value_counts(normalize=True)
        a = actual.where(actual.isin(keep), "OTHER").value_counts(normalize=True)
        idx = e.index.union(a.index)
        e, a = e.reindex(idx, fill_value=0).clip(lower=1e-4), a.reindex(idx, fill_value=0).clip(lower=1e-4)
        return float(np.sum((a - e) * np.log(a / e)))

    def split_shift(self, cases: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]) -> pd.DataFrame:
        """PSI and KS drift tests comparing validation and test splits against train."""
        tr = cases[cases["split_tag"] == "train"]
        rows = []
        for col in list(numeric) + list(categorical):
            is_num = col in numeric
            row = {
                "feature": col,
                "kind": "numeric" if is_num else "categorical",
                "constant_in_train": bool(tr[col].nunique(dropna=True) <= 1),
            }
            for name in ("val", "test"):
                other = cases[cases["split_tag"] == name]
                if is_num:
                    a, b = tr[col].dropna().to_numpy(float), other[col].dropna().to_numpy(float)
                    if len(a) > 0 and len(b) > 0:
                        row[f"psi_{name}"] = round(self._psi(a, b), 4)
                        row[f"ks_{name}"] = round(float(stats.ks_2samp(a, b)[0]), 4)
                    else:
                        row[f"psi_{name}"] = np.nan
                        row[f"ks_{name}"] = np.nan
                else:
                    row[f"psi_{name}"] = round(self._psi_cat(tr[col].astype(str), other[col].astype(str)), 4)
                    row[f"ks_{name}"] = np.nan
            rows.append(row)
        out = pd.DataFrame(rows)
        out["max_psi"] = out[["psi_val", "psi_test"]].max(axis=1)
        return out.sort_values("max_psi", ascending=False).reset_index(drop=True)

    @staticmethod
    def synthetic_fidelity_drift(real_df: pd.DataFrame, synth_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Evaluates PSI and KS drift between real training stays and synthetic stays."""
        rows = []
        for feat in features:
            if feat not in real_df.columns or feat not in synth_df.columns:
                continue
            r_vals = real_df[feat].dropna()
            s_vals = synth_df[feat].dropna()
            if len(r_vals) < 10 or len(s_vals) < 10:
                continue
            is_num = pd.api.types.is_numeric_dtype(r_vals)
            if is_num:
                a, b = r_vals.to_numpy(float), s_vals.to_numpy(float)
                psi_val = DriftAnalyzer._psi(a, b)
                ks_stat, p_val = stats.ks_2samp(a, b)
            else:
                psi_val = DriftAnalyzer._psi_cat(r_vals.astype(str), s_vals.astype(str))
                ks_stat, p_val = np.nan, np.nan
                
            rows.append({
                "feature": feat,
                "kind": "numeric" if is_num else "categorical",
                "psi_synthetic_vs_real": round(float(psi_val), 4),
                "ks_stat": round(float(ks_stat), 4) if pd.notna(ks_stat) else np.nan,
                "ks_p_value": float(p_val) if pd.notna(p_val) else np.nan,
                "fidelity_verdict": "HIGH_FIDELITY" if psi_val < 0.10 else ("MODERATE_FIDELITY" if psi_val < 0.25 else "TARGETED_REDISTRIBUTION"),
            })
        return pd.DataFrame(rows).sort_values("psi_synthetic_vs_real").reset_index(drop=True)
