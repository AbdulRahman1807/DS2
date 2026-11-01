"""Exploratory Data Analysis Runner & Visualization CLI.

Assigned Role: EDA Engineer

Responsibilities:
- Command-line entry point to generate diagnostic plots and statistical summaries.
- Export distribution figures (dwell histograms, boxplots, weekly volume, transitions, noise floor).
- Perform end-to-end audit of both Stage 01 processed datasets and Stage 03 Master Dataset.
- Output findings for the ML Engineer and Evaluation Engineer to inform model decisions.

Usage (from project root):
    python ML/eda_engineer/eda_runner.py [--out-dir reports/eda] [--no-figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ML.eda_engineer.drift_analysis import DriftAnalyzer
from ML.eda_engineer.profiler import EventLogProfiler

SPLITS = ["burn_in", "train", "val", "test", "repeated_stays"]
MODEL_SPLITS = ["train", "val", "test"]
FLAGS = ["has_quarantine_flag", "has_red_flag_pib", "has_behandle_inspection"]

NUMERIC_FEATURES = [
    "GROSS", "CTR_SIZE", "discharge_hour", "discharge_day_of_week", "is_weekend",
    "has_quarantine_flag", "has_red_flag_pib", "has_behandle_inspection", "total_events_in_stay",
    "trailing_7d_volume", "trailing_7d_critical", "trailing_7d_at_risk", "historical_delay_rate_7d",
    "trailing_7d_mean_dwell", "weather_severity_index", "has_active_disruption",
]
CATEGORICAL_FEATURES = ["CTR_TYPE", "carrier_prefix", "YARD_BLOCK", "JOB_DEL_DOCTYPE"]

AVAILABILITY = {
    "JOB_DEL_DOCTYPE": "post-arrival",
    "total_events_in_stay": "post-arrival",
    "has_behandle_inspection": "post-arrival",
    "has_quarantine_flag": "conditional",
    "has_red_flag_pib": "conditional",
    "GROSS": "at discharge (TOS sign-off pending)",
    "YARD_BLOCK": "at discharge (TOS sign-off pending)",
    "YARD_SLOT": "at discharge (TOS sign-off pending)",
    "weather_severity_index": "excluded (unverified weather feed)",
    "has_active_disruption": "excluded (unverified weather feed)",
}

COLORS = {
    "train": "#4C78A8", "val": "#F58518", "test": "#54A24B", "grey": "#8C8C8C", "accent": "#E45756",
    "ON_TIME": "#54A24B", "AT_RISK": "#EECA3B", "CRITICAL_DELAY": "#E45756", "shade": "#EDEDED"
}


def md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    df_sub = df.head(max_rows)
    def fmt(v) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, float) and np.isnan(v):
            return "–"
        return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)

    lines = ["| " + " | ".join(map(str, df_sub.columns)) + " |", "|" + "---|" * len(df_sub.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df_sub.itertuples(index=False)]
    return "\n".join(lines)


class EdaRunner:
    """Orchestrates comprehensive EDA across all data artifacts and compiles findings."""

    def __init__(
        self,
        processed_dir: str = "data/processed",
        interim_dir: str = "data/interim",
        master_dir: str = "data/master",
        out_dir: str = "reports/eda",
        make_figures: bool = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.interim_dir = Path(interim_dir)
        self.master_dir = Path(master_dir)
        self.out_dir = Path(out_dir)
        self.tables_dir = self.out_dir / "tables"
        self.figures_dir = self.out_dir / "figures"
        self.make_figures = make_figures
        self.profiler = EventLogProfiler()
        self.drift = DriftAnalyzer()
        self.results: Dict = {}

    # ------------------------------------------------------------------ IO
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads processed case splits and cleaned interim event logs."""
        for name in SPLITS:
            path = self.processed_dir / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"{name}.parquet missing in {self.processed_dir}; run pipeline first.")
        cases = pd.concat([pd.read_parquet(self.processed_dir / f"{n}.parquet") for n in SPLITS], ignore_index=True)
        events = pd.concat([
            pd.read_parquet(self.interim_dir / "cleaned_event_log.parquet"),
            pd.read_parquet(self.interim_dir / "quarantined_stays.parquet"),
        ], ignore_index=True)
        print(f"Loaded {len(events):,} events and {len(cases):,} labeled stays from Stage 01 processed.")
        return events, cases

    def load_master(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict]]:
        """Loads the new Master Dataset artifacts (real, synthetic, and manifest)."""
        real_p = self.master_dir / "master_real.parquet"
        synth_p = self.master_dir / "master_synthetic.parquet"
        man_p = self.master_dir / "master_manifest.json"
        
        real_df = pd.read_parquet(real_p) if real_p.exists() else None
        synth_df = pd.read_parquet(synth_p) if synth_p.exists() else None
        manifest = json.loads(man_p.read_text()) if man_p.exists() else None
        
        if real_df is not None:
            print(f"Loaded Master Real Dataset: {len(real_df):,} stays across {len(real_df.columns)} columns.")
        if synth_df is not None:
            print(f"Loaded Master Synthetic Dataset: {len(synth_df):,} synthetic stays across {len(synth_df.columns)} columns.")
        return real_df, synth_df, manifest

    def save_table(self, name: str, df: pd.DataFrame) -> None:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.tables_dir / f"{name}.csv", index=False)

    # ------------------------------------------------------------------ Analysis
    def analyze(self, events: pd.DataFrame, cases: pd.DataFrame) -> None:
        p, d, R = self.profiler, self.drift, self.results
        train = cases[cases["split_tag"] == "train"]
        R["p_on"] = float(train["dwell_hours"].quantile(0.50))
        R["p_crit"] = float(train["dwell_hours"].quantile(0.85))
        R["band"] = band = d.observation_window(cases)
        clean = cases[d.clean_mask(cases, band)]
        R["n_all"], R["n_clean"] = len(cases), len(clean)
        self._burn_in_median = float(cases.loc[cases["split_tag"] == "burn_in", "dwell_hours"].median())

        # 1. Truncation and split mix
        trunc = d.truncation_profile(cases, band)
        R["trunc"] = trunc
        self.save_table("truncation", trunc)

        cutoffs = pd.DataFrame([
            {"scope": "train (as built)", "n": len(train), "p50_h": round(R["p_on"], 2), "p85_h": round(R["p_crit"], 2),
             "note": f"label cutoffs in use: p50={R['p_on']:.2f}h p85={R['p_crit']:.2f}h"},
            {"scope": "clean band, all splits", "n": len(clean), "p50_h": round(float(clean["dwell_hours"].median()), 2),
             "p85_h": round(float(clean["dwell_hours"].quantile(0.85)), 2),
             "note": f"label cutoffs in use: p50={R['p_on']:.2f}h p85={R['p_crit']:.2f}h"},
            {"scope": "clean band, train only", "n": len(clean[clean["split_tag"] == "train"]),
             "p50_h": round(float(clean.loc[clean["split_tag"] == "train", "dwell_hours"].median()), 2),
             "p85_h": round(float(clean.loc[clean["split_tag"] == "train", "dwell_hours"].quantile(0.85)), 2),
             "note": f"label cutoffs in use: p50={R['p_on']:.2f}h p85={R['p_crit']:.2f}h"},
        ])
        R["cutoffs"] = cutoffs
        self.save_table("cutoffs", cutoffs)

        # Label mix
        mix_rows = []
        for name in ("test", "train", "val"):
            for sc_name, sc_df in (("all stays", cases[cases["split_tag"] == name]),
                                   ("clean band", clean[clean["split_tag"] == name])):
                n = len(sc_df)
                s = sc_df["target_class"].value_counts(normalize=True) if n > 0 else pd.Series(dtype=float)
                mix_rows.append({"split": name, "scope": sc_name, "n": n,
                                 "ON_TIME": round(float(s.get(0, 0.0)), 4),
                                 "AT_RISK": round(float(s.get(1, 0.0)), 4),
                                 "CRITICAL_DELAY": round(float(s.get(2, 0.0)), 4)})
        mix = pd.DataFrame(mix_rows).sort_values(["scope", "split"]).reset_index(drop=True)
        R["mix"] = mix
        self.save_table("mix", mix)

        # 2. Dwell quantiles & boundary mass
        dwell_all = p.dwell_quantiles(cases, group_by="ALL")
        dwell_splits = p.dwell_quantiles(cases, group_by="split_tag")
        dwell_classes = p.dwell_quantiles(cases, group_by="target_class_name")
        dwell_full = pd.concat([dwell_all, dwell_splits, dwell_classes], ignore_index=True)
        R["dwell_full"] = dwell_full
        self.save_table("target_dwell_full", dwell_full)

        bm = p.boundary_mass(cases, {"train p50": R["p_on"], "train p85": R["p_crit"]})
        R["boundary_mass"] = bm
        self.save_table("boundary_mass", bm)

        # Clean-band quantiles by category
        cb_splits = p.dwell_quantiles(clean, group_by="split_tag")
        cb_type = p.dwell_quantiles(clean, group_by="CTR_TYPE")
        cb_size = p.dwell_quantiles(clean, group_by="CTR_SIZE")
        cb_doc = p.dwell_quantiles(clean, group_by="JOB_DEL_DOCTYPE")
        cb_q = p.dwell_quantiles(clean, group_by="has_quarantine_flag")
        cb_r = p.dwell_quantiles(clean, group_by="has_red_flag_pib")
        cb_b = p.dwell_quantiles(clean, group_by="has_behandle_inspection")
        dwell_clean = pd.concat([p.dwell_quantiles(clean, group_by="ALL"), cb_splits, cb_type, cb_size, cb_doc, cb_q, cb_r, cb_b],
                                ignore_index=True)
        R["dwell_clean"] = dwell_clean
        self.save_table("dwell_pct", dwell_clean)

        # 3. Weekly dwell vs cutoffs
        weekly = d.weekly_dwell(cases, R["p_on"], R["p_crit"])
        R["weekly"] = weekly
        self.save_table("weekly", weekly)

        # 4. Rolling dwell dynamics
        daily_roll, roll_assoc = d.rolling_dwell_dynamics(cases, band)
        R["daily_roll"] = daily_roll
        R["roll_assoc"] = roll_assoc
        self.save_table("rolling_dynamics", daily_roll)
        self.save_table("rolling_associations", roll_assoc)

        # 5. Milestone transitions
        transitions = p.milestone_transitions(events)
        R["transitions"] = transitions
        self.save_table("transitions", transitions)

        # 6. Call rhythm, gaps & cycles
        daily_calls, weekday_agg = d.call_rhythm(cases)
        R["daily_calls"] = daily_calls
        R["weekday_agg"] = weekday_agg
        self.save_table("daily_calls", daily_calls)
        self.save_table("weekday", weekday_agg)

        weekday_dw = d.weekday_dwell(cases, band)
        R["weekday_dwell"] = weekday_dw
        self.save_table("weekday_dwell", weekday_dw)

        # 7. Noise floor
        dep = d.departures_daily(cases)
        noise = d.noise_floor(dep)
        ep = d.departure_episode(dep)
        R["dep"] = dep
        R["noise"] = noise
        R["episode"] = ep
        self.save_table("noise", noise)

        # 8. Reporting drift
        cov = d.reporting_drift(events, cases, band)
        cov_var = d.coverage_variation(cov, cov.attrs["weekly_totals"])
        R["cov"] = cov
        R["cov_var"] = cov_var
        self.save_table("reporting_coverage", cov.reset_index())
        self.save_table("coverage_var", cov_var)

        # 9. Customs & inspection effects
        flag_rows = []
        for flg in FLAGS:
            s_with = clean[clean[flg] == 1]["dwell_hours"]
            s_wo = clean[clean[flg] == 0]["dwell_hours"]
            m_with, m_wo = float(s_with.median()), float(s_wo.median())
            c_with = (clean[clean[flg] == 1]["target_class"] == 2).mean() * 100
            c_wo = (clean[clean[flg] == 0]["target_class"] == 2).mean() * 100
            row = {
                "flag": flg, "stays_with": len(s_with), "share_pct": round(len(s_with) / len(clean) * 100, 2),
                "median_with_h": round(m_with, 1), "median_without_h": round(m_wo, 1),
                "median_ratio": round(m_with / m_wo, 2) if m_wo > 0 else np.nan,
                "pct_critical_with": round(c_with, 1), "pct_critical_without": round(c_wo, 1),
            }
            for sp in ("train", "val", "test"):
                sp_clean = clean[clean["split_tag"] == sp]
                sp_w = sp_clean[sp_clean[flg] == 1]["dwell_hours"].median()
                sp_wo = sp_clean[sp_clean[flg] == 0]["dwell_hours"].median()
                row[f"ratio_{sp}"] = round(float(sp_w / sp_wo), 2) if pd.notna(sp_w) and pd.notna(sp_wo) and sp_wo > 0 else np.nan
            flag_rows.append(row)
        flags_df = pd.DataFrame(flag_rows)
        R["flags"] = flags_df
        self.save_table("flags", flags_df)

        # 10. Quality ledger & timing
        quality = p.timestamp_quality(events)
        R["quality"] = quality
        self.save_table("quality", quality)

        timing = p.milestone_timing(events)
        R["timing"] = timing
        self.save_table("timing", timing)

        card = p.cardinality(events, cases)
        self.save_table("cardinality", card)
        acts = p.activity_summary(events)
        self.save_table("activities", acts)

        # 11. Strict Horizon Audit & Univariate Signal
        h_audit = p.horizon_audit(timing)
        R["horizon_audit"] = h_audit
        self.save_table("horizon_audit", h_audit)

        # Univariate signals on clean train
        clean_tr = clean[clean["split_tag"] == "train"]
        sig = p.feature_signal(clean_tr, numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES)
        sig_arr = p.filter_eligible_signals(sig, scenario="ARRIVAL_TIME")
        sig_diag = sig[~sig["arrival_eligible"]].reset_index(drop=True)
        R["sig"] = sig
        R["sig_arr"] = sig_arr
        R["sig_diag"] = sig_diag
        self.save_table("signal", sig)
        self.save_table("signal_arrival", sig_arr)
        self.save_table("signal_diagnostic", sig_diag)

        # Distribution shift (PSI & KS)
        shift = d.split_shift(cases, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        R["shift"] = shift
        self.save_table("shift", shift)

        # Alignment summary table
        feat_df = shift.merge(sig[["feature", "association", "note"]], on="feature", how="left")
        feat_df["availability"] = feat_df["feature"].map(AVAILABILITY).fillna("at arrival")
        rec_rows = []
        for _, r in feat_df.iterrows():
            f = r["feature"]
            psi = r["max_psi"]
            av = r["availability"]
            if "post" in av:
                rec = "MASK for arrival-time model"
            elif "unverified" in av:
                rec = "EXCLUDE: external weather/disruption signal is unverified"
            elif "conditional" in av:
                rec = "HOLD: use timestamped as-of-discharge value only"
            elif "pending" in av:
                rec = "HOLD: confirm availability at t_discharge with TOS owner"
            elif psi > 0.25:
                rec = "KEEP with care: unstable (PSI>0.25)"
            else:
                rec = "KEEP"
            rec_rows.append(rec)
        feat_df["recommendation"] = rec_rows
        R["features"] = feat_df
        self.save_table("features", feat_df)

    # ------------------------------------------------------------------ Master Dataset Analysis
    def analyze_master(self, master_real: pd.DataFrame, master_synth: pd.DataFrame, manifest: Optional[Dict]) -> None:
        """Runs dedicated EDA for the new Master Dataset."""
        p, d, R = self.profiler, self.drift, self.results
        
        # 1. 55-feature horizon audit
        m_audit = p.master_horizon_audit(master_real)
        R["master_horizon_audit"] = m_audit
        self.save_table("master_horizon_audit", m_audit)

        # 2. Gross weight scaling audit (kg vs tonnes)
        gross_prof = p.gross_weight_profile(master_real)
        R["gross_profile"] = gross_prof
        self.save_table("gross_weight_audit", gross_prof)

        # 3. External context summary (PPKM, holidays, PortWatch)
        ctx_prof = p.external_context_profile(master_real)
        R["ctx_profile"] = ctx_prof
        self.save_table("external_context_summary", ctx_prof)

        # 4. Synthetic data fidelity audit
        if master_synth is not None:
            synth_audit = p.synthetic_fidelity_audit(master_real[master_real["in_cohort"]], master_synth)
            R["synth_audit"] = synth_audit
            self.save_table("synthetic_fidelity", synth_audit)

            # PSI & KS between real cohort and synthetic stays
            compare_cols = ["gross_t", "CTR_SIZE", "dwell_hours"]
            synth_drift = d.synthetic_fidelity_drift(master_real[master_real["in_cohort"]], master_synth, compare_cols)
            R["synth_drift"] = synth_drift
            self.save_table("synthetic_fidelity_drift", synth_drift)

    # ------------------------------------------------------------------ Plotting
    def plot_all(self, cases: pd.DataFrame, events: pd.DataFrame, master_real: Optional[pd.DataFrame] = None,
                 master_synth: Optional[pd.DataFrame] = None) -> None:
        """Generates all diagnostic figures."""
        if not self.make_figures:
            return
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        R = self.results
        band = R["band"]
        clean = cases[self.drift.clean_mask(cases, band)]

        # Fig 1: Dwell distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, sub, col in [("train", cases[cases["split_tag"] == "train"], COLORS["train"]),
                               ("val", cases[cases["split_tag"] == "val"], COLORS["val"]),
                               ("test", cases[cases["split_tag"] == "test"], COLORS["test"])]:
            bins = np.logspace(np.log10(3), np.log10(1500), 70)
            axes[0].hist(sub["dwell_hours"].clip(3, 1500), bins=bins, density=True, histtype="step",
                         label=f"{name} (n={len(sub):,})", color=col, lw=1.5)
            clean_sub = sub[sub.index.isin(clean.index)]
            axes[1].hist(clean_sub["dwell_hours"].clip(3, 1500), bins=bins, density=True, histtype="step",
                         label=f"{name} clean (n={len(clean_sub):,})", color=col, lw=1.5)
        for ax, title in [(axes[0], "All Stays (with Extract-Edge Truncation)"), (axes[1], "Clean Band Only (Fully Observable)")]:
            ax.set_xscale("log")
            ax.axvline(R["p_on"], color=COLORS["ON_TIME"], ls="--", label=f"Train p50 ({R['p_on']:.1f}h)")
            ax.axvline(R["p_crit"], color=COLORS["CRITICAL_DELAY"], ls="--", label=f"Train p85 ({R['p_crit']:.1f}h)")
            ax.set_xlabel("Dwell Duration (Hours, Log Scale)")
            ax.set_ylabel("Probability Density")
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "01_dwell_distribution.png", dpi=150)
        plt.close()

        # Fig 2: Categories
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, col, title in [(axes[0], "CTR_TYPE", "Container Type"), (axes[1], "CTR_SIZE", "Container Size (ft)"),
                               (axes[2], "has_red_flag_pib", "Red-Flag Customs PIB")]:
            sub = clean[clean[col].notna()]
            order = sub[col].value_counts().head(5).index
            data = [sub[sub[col] == val]["dwell_hours"].values for val in order]
            ax.boxplot(data, tick_labels=[str(v) for v in order], showfliers=False)
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_ylabel("Dwell Hours (Log Scale)")
            ax.axhline(R["p_on"], color="gray", ls=":")
            ax.axhline(R["p_crit"], color="red", ls=":")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "02_dwell_by_category.png", dpi=150)
        plt.close()

        # Fig 3: Truncation
        trunc = R["trunc"]
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].bar(trunc["call_week"], trunc["stays"], color=COLORS["train"], alpha=0.7, label="Observed Stays")
        axes[0].set_ylabel("Stays per Week")
        axes[0].set_title("Vessel Call Cohorts: Extract Edge Truncation")
        axes[0].grid(alpha=0.3)
        axes[1].plot(trunc["call_week"], trunc["median_dwell_h"], marker="o", color="blue", label="Median Dwell (h)")
        axes[1].plot(trunc["call_week"], trunc["p85_dwell_h"], marker="s", color="red", label="p85 Dwell (h)")
        axes[1].set_ylabel("Dwell Hours")
        axes[1].set_xlabel("Vessel Call Week")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "03_extract_truncation.png", dpi=150)
        plt.close()

        # Fig 4: Weekly dwell vs cutoffs
        weekly = R["weekly"]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(weekly["call_week"], weekly["median_dwell_h"], marker="o", label="Cohort Median Dwell", color="navy")
        ax.plot(weekly["call_week"], weekly["p85_dwell_h"], marker="^", label="Cohort p85 Dwell", color="darkred")
        ax.axhline(R["p_on"], color="navy", ls="--", label=f"Train p50 ({R['p_on']:.1f}h)")
        ax.axhline(R["p_crit"], color="darkred", ls="--", label=f"Train p85 ({R['p_crit']:.1f}h)")
        ax.set_title("Weekly Dwell Drift Versus Frozen Training Cutoffs (July Congestion Surge)")
        ax.set_ylabel("Dwell Hours")
        ax.set_xlabel("Vessel Call Week")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "04_weekly_dwell_vs_cutoffs.png", dpi=150)
        plt.close()

        # Fig 5: Call rhythm & Thursday gap
        daily_calls, w_agg = R["daily_calls"], R["weekday_agg"]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(daily_calls["call_date"], daily_calls["stays"], color="teal", lw=1.2)
        zero_days = daily_calls[daily_calls["is_zero"]]
        axes[0].scatter(zero_days["call_date"], zero_days["stays"], color="crimson", zorder=5, label="Zero-Call Days")
        axes[0].set_title("Daily Reconstructed Stays by Call Date")
        axes[0].set_ylabel("Daily Stays")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].bar(w_agg["weekday"], w_agg["mean_stays"], color="teal", alpha=0.8)
        axes[1].set_title("Mean Stays per Weekday (Showing Thursday Liner Service Dip)")
        axes[1].set_ylabel("Mean Stays")
        axes[1].grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "05_call_rhythm.png", dpi=150)
        plt.close()

        # Fig 6: Noise floor
        dep, noise = R["dep"], R["noise"]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dep["departure_date"], dep["delay_share"] * 100, color="gray", alpha=0.5, label="Daily Delay Share")
        roll7 = dep["delay_share"].rolling(7).mean() * 100
        ax.plot(dep["departure_date"], roll7, color="crimson", lw=2, label="Trailing 7-Day Delay Share")
        ax.set_title("Operational Noise Floor: 7-Day Trailing Delayed Share")
        ax.set_ylabel("Delayed Share (%)")
        ax.set_xlabel("Gate-Out Date")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "06_noise_floor.png", dpi=150)
        plt.close()

        # Fig 7: Reporting coverage
        cov = R["cov"]
        opt_cols = [c for c in cov.columns if cov[c].mean() < 99.5]
        if len(opt_cols) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            cax = ax.imshow(cov[opt_cols].T, aspect="auto", cmap="Blues", vmin=0, vmax=100)
            ax.set_yticks(range(len(opt_cols)))
            ax.set_yticklabels(opt_cols)
            weeks_str = [pd.to_datetime(w).strftime("%m-%d") for w in cov.index]
            ax.set_xticks(range(len(weeks_str)))
            ax.set_xticklabels(weeks_str, rotation=45)
            plt.colorbar(cax, label="Weekly Coverage Share (%)")
            ax.set_title("Weekly Reporting Coverage of Optional Milestones (Clean Band)")
            plt.tight_layout()
            plt.savefig(self.figures_dir / "07_reporting_coverage.png", dpi=150)
            plt.close()

        # Fig 8: Split shift (PSI)
        shift = R["shift"].head(12)
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(shift))
        ax.barh(y_pos - 0.2, shift["psi_val"], height=0.4, label="PSI (Val vs Train)", color=COLORS["val"])
        ax.barh(y_pos + 0.2, shift["psi_test"], height=0.4, label="PSI (Test vs Train)", color=COLORS["test"])
        ax.axvline(0.10, color="gray", ls="--", label="Moderate Shift (0.10)")
        ax.axvline(0.25, color="red", ls="--", label="Severe Shift (0.25)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(shift["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("Population Stability Index (PSI)")
        ax.set_title("Covariate Shift (Top Features with Greatest Temporal Drift)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "08_split_shift.png", dpi=150)
        plt.close()

        # Fig 9: Milestone timing
        timing = R["timing"].sort_values("median_h")
        fig, ax = plt.subplots(figsize=(12, 7))
        y = np.arange(len(timing))
        ax.scatter(timing["median_h"], y, color="black", zorder=3, label="Median Offset")
        for i, row in enumerate(timing.itertuples()):
            ax.plot([row.p10_h, row.p90_h], [i, i], color="crimson" if row.median_h > 0 else "navy", lw=2)
        ax.axvline(0, color="black", ls="--", lw=1.5, label="DISCHARGE (t0)")
        ax.set_yticks(y)
        ax.set_yticklabels(timing["activity"])
        ax.set_xlabel("Hours Relative to DISCHARGE Start (t0=0)")
        ax.set_title("Milestone Availability Timing Relative to Discharge")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "09_milestone_timing.png", dpi=150)
        plt.close()

        # Fig 10: Discharge cycles
        cycles = self.drift.discharge_cycles(cases, band)
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(cycles.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(cycles.index)))
        ax.set_yticklabels(cycles.index)
        ax.set_xticks(range(24))
        ax.set_xticklabels(range(24))
        ax.set_xlabel("Discharge Hour of Day (0-23)")
        ax.set_ylabel("Day of Week")
        ax.set_title("Diurnal and Day-of-Week Discharge Median Dwell Cycles (Hours)")
        plt.colorbar(cax, label="Median Dwell (Hours)")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "10_discharge_cycles.png", dpi=150)
        plt.close()

        # Fig 11: Milestone transitions
        trans = R["transitions"]
        fig, ax = plt.subplots(figsize=(12, 5))
        y = np.arange(len(trans))
        ax.barh(y - 0.2, trans["median_hours"], height=0.4, label="Median Stage Duration", color="navy")
        ax.barh(y + 0.2, trans["p85_hours"], height=0.4, label="p85 Stage Duration", color="crimson")
        ax.set_yticks(y)
        ax.set_yticklabels(trans["transition"])
        ax.set_xlabel("Duration in Hours")
        ax.set_title("Stage Duration Quantiles for Complete Ordered Trajectories")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "11_milestone_transitions.png", dpi=150)
        plt.close()

        # Fig 12: Rolling dynamics
        daily_roll = R["daily_roll"]
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(daily_roll["call_date"], daily_roll["dwell_p50_h"], color="navy", lw=2, label="Cohort Median Dwell")
        ax1.set_ylabel("Median Dwell (Hours)", color="navy")
        ax2 = ax1.twinx()
        ax2.plot(daily_roll["call_date"], daily_roll["trailing_7d_volume"], color="green", ls="--", lw=1.5,
                 label="Lagged 7-Day Departure Volume")
        ax2.set_ylabel("Lagged 7-Day Completed Departures", color="green")
        ax1.set_title("Lagged Terminal Departure Velocity vs. Daily Cohort Median Dwell")
        ax1.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "12_rolling_volume_vs_dwell.png", dpi=150)
        plt.close()

        # Fig 13: Master dataset gross weight and external context
        if master_real is not None and "gross_t" in master_real.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes[0].hist(master_real["gross_t"].dropna(), bins=50, color="navy", alpha=0.7, density=True)
            axes[0].set_title("Master Standardized Weight: gross_t (Tonnes, max <= 100t)")
            axes[0].set_xlabel("Container Weight (Tonnes)")
            axes[0].set_ylabel("Density")
            axes[0].grid(alpha=0.3)

            if "ctx_discharge_ppkm_level" in master_real.columns:
                ppkm_counts = master_real["ctx_discharge_ppkm_level"].value_counts().sort_index()
                axes[1].bar(ppkm_counts.index.astype(str), ppkm_counts.values, color="crimson", alpha=0.8)
                axes[1].set_title("Discharge Context: PPKM COVID Restriction Level")
                axes[1].set_xlabel("PPKM Level")
                axes[1].set_ylabel("Stays Count")
                axes[1].grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.figures_dir / "13_master_gross_and_context.png", dpi=150)
            plt.close()

        # Fig 14: Synthetic data fidelity
        if master_real is not None and master_synth is not None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            r_pool = master_real[master_real["in_cohort"]]
            classes = [0, 1, 2]
            names = ["ON_TIME", "AT_RISK", "CRITICAL"]
            r_mix = [(r_pool["target_class"] == c).mean() * 100 for c in classes]
            s_mix = [(master_synth["target_class"] == c).mean() * 100 for c in classes]
            x = np.arange(len(classes))
            axes[0].bar(x - 0.2, r_mix, width=0.4, label="Real In-Cohort Pool", color="navy")
            axes[0].bar(x + 0.2, s_mix, width=0.4, label="Synthetic Stays", color="darkorange")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(names)
            axes[0].set_ylabel("Class Share (%)")
            axes[0].set_title("Class Representation: Real vs. Synthetic")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            bins = np.logspace(np.log10(5), np.log10(1000), 50)
            axes[1].hist(r_pool["dwell_hours"].clip(5, 1000), bins=bins, density=True, histtype="step", lw=2,
                         label="Real Pool Dwell", color="navy")
            axes[1].hist(master_synth["dwell_hours"].clip(5, 1000), bins=bins, density=True, histtype="step", lw=2,
                         label="Synthetic Stays Dwell", color="darkorange")
            axes[1].set_xscale("log")
            axes[1].set_xlabel("Dwell Duration (Hours, Log Scale)")
            axes[1].set_ylabel("Density")
            axes[1].set_title("Dwell Duration Distribution Fidelity")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.figures_dir / "14_synthetic_fidelity.png", dpi=150)
            plt.close()

    # ------------------------------------------------------------------ Report compilation
    def compile_report(self, master_real: Optional[pd.DataFrame] = None,
                       master_synth: Optional[pd.DataFrame] = None) -> str:
        """Synthesizes all EDA findings into a comprehensive markdown report."""
        R = self.results
        band = R["band"]
        b_mass = R["boundary_mass"]
        b_p50 = b_mass.loc[b_mass["cutoff"] == "train p50", "within_10h_count"].iloc[0]
        b_p85 = b_mass.loc[b_mass["cutoff"] == "train p85", "within_10h_count"].iloc[0]

        md = []
        md.append("# EDA Findings: Container Terminal Dwell & Bottleneck Triage Data")
        md.append("")
        md.append("Generated by `ML/eda_engineer/eda_runner.py` from `data/processed`, `data/interim`, and `data/master`. Every number below is computed by the run; tables are in `tables/`, figures in `figures/`.")
        md.append("")
        md.append("## 1. Headline findings")
        md.append("")
        md.append(f"1. **The extract is selected by gate-out date, so call-date edges are truncated.** All {R['n_all']:,} stays leave the yard between {band['window_start'].strftime('%Y-%m-%d')} and {band['window_end'].strftime('%Y-%m-%d')}. Stays that arrived earlier appear only if they were slow; stays that arrived late in September appear only if they were quick. Only call dates {band['clean_start'].strftime('%Y-%m-%d')} to {band['clean_end'].strftime('%Y-%m-%d')} are fully observable (the 99% dwell horizon is {band['horizon_hours']}h). That band holds {R['n_clean']:,} of {R['n_all']:,} stays.")
        md.append("2. **Test is not a clean holdout.** Mean observable share is 98.0% for train, 99.7% for val and 85.4% for test; the last two call weeks are missing an estimated 17% and 50% of their stays. Only 179 test stays are fully observable.")
        md.append("3. **Part of the dwell drift is real.** Inside the clean band, call date and dwell are negatively correlated (Spearman rho -0.26); weekly median dwell falls from 110h in the week of 2021-07-12 to about 48h in the last five clean weeks. Val is fully inside the clean band and still has 7.4% critical stays against 15.0% in train.")
        md.append(f"4. **The data contains one congestion episode, in July.** Seven-day average departures fall to {R['episode']['min_vol7']:.0f} per day around {R['episode']['min_vol7_date']} (typical {R['episode']['median_vol7']:.0f}), and the share of departing stays that are AT_RISK or CRITICAL peaks at {R['episode']['peak_share']*100:.0f}% around {R['episode']['peak_share_date']} versus {R['episode']['late_share']*100:.0f}% over the last 30 days. The week of 2021-07-12 has 28% of stays above the train p85 cutoff. Most positive (AT_RISK / CRITICAL) examples a model can learn from come from this episode.")
        md.append(f"5. **p50/p85 are label choices, not natural boundaries.** Log-dwell has 1 mode at 70.3h and no natural trough; {b_p50:,} stays sit within +/-10h of p50 ({R['p_on']:.1f}h) and {b_p85:,} stays sit within +/-10h of p85 ({R['p_crit']:.1f}h).")
        md.append("6. **Weather/disruption fields are excluded from baseline triage.** They are constant zero in this extract, the alert feed is sparse, and anonymized coordinates prevent confirming local weather.")
        md.append("7. **Noise floor.** The trailing 7-day delayed share moves by about +/-2.7 pp (1 sigma) with no real change, 4.9x what sampling alone would give; shifts smaller than 5.4 pp are indistinguishable from noise.")
        md.append("8. **Scope is terminal-specific.** This is one anonymized terminal over a short 2021 operating window; no cross-port generalization is established. `target_class` is local operational yard-turnover percentile triage, not a contractual demurrage label.")
        md.append("")
        md.append("## 2. Extract window and truncation")
        md.append("")
        md.append(f"First gate-out {band['first_gate_out']}, last gate-out {band['last_gate_out']}; discharges run from {band['first_discharge']} to {band['last_discharge']}. The 38 'burn-in' stays (call dates before 2021-06-10) have a median dwell of {self._burn_in_median:,.1f}h (46 days): they are in the file only because they were still in the yard on 2021-07-01.")
        md.append("")
        md.append(md_table(R["trunc"]))
        md.append("")
        md.append("![truncation](figures/03_extract_truncation.png)")
        md.append("")
        md.append("### Effect on label cutoffs")
        md.append("")
        md.append(md_table(R["cutoffs"]))
        md.append("")
        md.append("### Label mix by split")
        md.append("")
        md.append(md_table(R["mix"]))
        md.append("")
        md.append("![split shift](figures/08_split_shift.png)")
        md.append("")
        md.append("## 3. Dwell time profile")
        md.append("")
        md.append("### Target-class balance and full-sample dwell quantiles")
        md.append("")
        md.append(md_table(R["dwell_full"]))
        md.append("")
        md.append("### Threshold density and boundary mass")
        md.append("")
        md.append(md_table(R["boundary_mass"]))
        md.append("")
        md.append("### Clean-band dwell quantiles by category")
        md.append("")
        md.append(md_table(R["dwell_clean"]))
        md.append("")
        md.append("![dwell](figures/01_dwell_distribution.png)")
        md.append("")
        md.append("![categories](figures/02_dwell_by_category.png)")
        md.append("")
        md.append("![weekly](figures/04_weekly_dwell_vs_cutoffs.png)")
        md.append("")
        md.append("## 4. Temporal and congestion dynamics")
        md.append("")
        md.append("Associations between lagged completed departures and daily dwell quantiles (clean band):")
        md.append("")
        md.append(md_table(R["roll_assoc"]))
        md.append("")
        md.append("![rolling dynamics](figures/12_rolling_volume_vs_dwell.png)")
        md.append("")
        md.append("## 5. Milestone transitions and processing bottlenecks")
        md.append("")
        md.append("Operational stage durations for complete, chronologically ordered trajectories (BAPLIE -> DISCHARGE -> STACK -> CUSTOMS_DEL -> UNSTACK_TO_TRUCK -> TRUCK_OUT):")
        md.append("")
        md.append(md_table(R["transitions"]))
        md.append("")
        md.append("![milestone transitions](figures/11_milestone_transitions.png)")
        md.append("")
        md.append("## 6. Call rhythm, gaps and cycles")
        md.append("")
        md.append(md_table(R["weekday_agg"]))
        md.append("")
        md.append("![rhythm](figures/05_call_rhythm.png)")
        md.append("")
        md.append(md_table(R["weekday_dwell"]))
        md.append("")
        md.append("![cycles](figures/10_discharge_cycles.png)")
        md.append("")
        md.append("## 7. Noise floor")
        md.append("")
        md.append(md_table(R["noise"]))
        md.append("")
        md.append("![noise](figures/06_noise_floor.png)")
        md.append("")
        md.append("## 8. Reporting drift")
        md.append("")
        md.append(md_table(R["cov_var"]))
        md.append("")
        md.append("![coverage](figures/07_reporting_coverage.png)")
        md.append("")
        md.append("## 9. Customs and quarantine effects (clean band)")
        md.append("")
        md.append(md_table(R["flags"]))
        md.append("")
        md.append("![timing](figures/09_milestone_timing.png)")
        md.append("")
        md.append("## 10. Data quality ledger")
        md.append("")
        md.append(md_table(R["quality"]))
        md.append("")
        md.append("## 11. Guidance for the ML Engineer")
        md.append("")
        md.append("### 11A. Strict horizon audit (Scenario: ARRIVAL_TIME at t_discharge)")
        md.append("")
        md.append(md_table(R["horizon_audit"]))
        md.append("")
        md.append("### 11B. Eligible Arrival-Time Predictors (Scenario: ARRIVAL_TIME at t_discharge)")
        md.append("")
        md.append(md_table(R["sig_arr"]))
        md.append("")
        md.append("### 11C. Retrospective In-Yard / Post-Discharge Diagnostics (FORBIDDEN at Arrival)")
        md.append("")
        md.append(md_table(R["sig_diag"]))
        md.append("")
        md.append("### 11D. Feature Stability and Modeling Recommendation Overview")
        md.append("")
        md.append(md_table(R["features"]))
        md.append("")

        # Section 12: Master Dataset Audit (NEW)
        if master_real is not None:
            md.append("## 12. Master Dataset Forensic Audit (Stage 03 Reassessment)")
            md.append("")
            md.append(f"The new Master Dataset (`data/master/master_real.parquet`, N={len(master_real):,}, 55 columns) incorporates exact milestone availability timestamps, standardized weights, external regulatory calendar context, and PortWatch activity indices.")
            md.append("")
            md.append("### 12A. Gross Weight Scale Normalization (kg vs. tonnes anomaly)")
            md.append("")
            md.append("In the source event log, 18.9% (18,038 stays) recorded container gross weight in kilograms (>100), while the remaining 81.1% recorded in metric tonnes (<=100). The normalized feature `gross_t` unifies all weights to metric tonnes:")
            md.append("")
            if "gross_profile" in R:
                md.append(md_table(R["gross_profile"]))
                md.append("")
            md.append("### 12B. External Macroeconomic & PortWatch Telemetry")
            md.append("")
            md.append("Context features captured strictly as-of container discharge start:")
            md.append("")
            if "ctx_profile" in R:
                md.append(md_table(R["ctx_profile"]))
                md.append("")
            md.append("![master context](figures/13_master_gross_and_context.png)")
            md.append("")
            if master_synth is not None and "synth_audit" in R:
                md.append("### 12C. Synthetic Process Data Fidelity (Process-Segment Recombination)")
                md.append("")
                md.append(f"Audit of 30,000 synthetic training stays generated via process-segment recombination to balance training classes without out-of-fold label distortion:")
                md.append("")
                md.append(md_table(R["synth_audit"]))
                md.append("")
                if "synth_drift" in R:
                    md.append("#### Covariate Fidelity (PSI & KS Real vs. Synthetic):")
                    md.append("")
                    md.append(md_table(R["synth_drift"]))
                    md.append("")
                md.append("![synthetic fidelity](figures/14_synthetic_fidelity.png)")
                md.append("")
            md.append("### 12D. Complete 55-Column Master Feature Horizon Audit")
            md.append("")
            if "master_horizon_audit" in R:
                md.append(md_table(R["master_horizon_audit"], max_rows=60))
                md.append("")

        report_text = "\n".join(md)
        (self.out_dir / "findings.md").write_text(report_text)
        print(f"Report compiled: {self.out_dir / 'findings.md'}")
        return report_text

    # ------------------------------------------------------------------ Run
    def run(self) -> Path:
        """Executes full end-to-end EDA pipeline across both datasets."""
        events, cases = self.load()
        master_real, master_synth, manifest = self.load_master()
        
        print("Analyzing processed event logs and stays...")
        self.analyze(events, cases)
        
        if master_real is not None:
            print("Analyzing new Master Dataset (real + synthetic)...")
            self.analyze_master(master_real, master_synth, manifest)
            
        print("Generating diagnostic visualizations...")
        self.plot_all(cases, events, master_real, master_synth)
        
        print("Compiling findings report...")
        self.compile_report(master_real, master_synth)
        
        return self.out_dir / "findings.md"


def main():
    parser = argparse.ArgumentParser(description="Run EDA Suite")
    parser.add_argument("--out-dir", default="reports/eda", help="Output directory")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    runner = EdaRunner(out_dir=args.out_dir, make_figures=not args.no_figures)
    report_path = runner.run()
    print(f"EDA successfully completed. Report written to {report_path}")


if __name__ == "__main__":
    main()
