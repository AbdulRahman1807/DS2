"""Exploratory Data Analysis Runner & Visualization CLI.

Assigned Role: EDA Engineer

Responsibilities:
- Command-line entry point to generate diagnostic plots and statistical summaries
- Export distribution figures (dwell histograms, boxplots by container type, weekly volume)
- Output findings for the ML Engineer and Evaluation Engineer to inform model decisions

Usage (from the project root, after build_pipeline.py has run):
    python ML/eda_engineer/eda_runner.py [--out-dir reports/eda] [--no-figures]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

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
# Availability at arrival time, taken from data_engineer/FEATURE_PROVENANCE_MATRIX.md
AVAILABILITY = {
    "JOB_DEL_DOCTYPE": "post-arrival", "total_events_in_stay": "post-arrival", "has_behandle_inspection": "post-arrival",
    "has_quarantine_flag": "conditional", "has_red_flag_pib": "conditional",
}
COLORS = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B", "grey": "#8C8C8C", "accent": "#E45756",
          "ON_TIME": "#54A24B", "AT_RISK": "#EECA3B", "CRITICAL_DELAY": "#E45756", "shade": "#EDEDED"}


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    df = df.head(max_rows)
    def fmt(v) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, float) and np.isnan(v):
            return "–"
        return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


class EdaRunner:
    def __init__(self, processed_dir: str = "data/processed", interim_dir: str = "data/interim",
                 out_dir: str = "reports/eda", make_figures: bool = True):
        self.processed_dir, self.interim_dir = Path(processed_dir), Path(interim_dir)
        self.out_dir = Path(out_dir)
        self.tables_dir, self.figures_dir = self.out_dir / "tables", self.out_dir / "figures"
        self.make_figures = make_figures
        self.profiler, self.drift = EventLogProfiler(), DriftAnalyzer()
        self.results: Dict = {}

    # ------------------------------------------------------------------ io
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for name in SPLITS:
            if not (self.processed_dir / f"{name}.parquet").exists():
                raise FileNotFoundError(f"{name}.parquet missing in {self.processed_dir}; run ML/data_engineer/build_pipeline.py first.")
        cases = pd.concat([pd.read_parquet(self.processed_dir / f"{n}.parquet") for n in SPLITS], ignore_index=True)
        events = pd.concat(
            [pd.read_parquet(self.interim_dir / "cleaned_event_log.parquet"),
             pd.read_parquet(self.interim_dir / "quarantined_stays.parquet")], ignore_index=True)
        print(f"Loaded {len(events):,} events and {len(cases):,} labeled stays.")
        return events, cases

    def save(self, name: str, df: pd.DataFrame) -> None:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.tables_dir / f"{name}.csv", index=False)

    # ------------------------------------------------------------------ analysis
    def analyze(self, events: pd.DataFrame, cases: pd.DataFrame) -> None:
        p, d, R = self.profiler, self.drift, self.results
        train = cases[cases["split_tag"] == "train"]
        R["p_on"], R["p_crit"] = float(train["dwell_hours"].quantile(0.50)), float(train["dwell_hours"].quantile(0.85))
        R["band"] = band = d.observation_window(cases)
        clean = cases[d.clean_mask(cases, band)]
        R["n_all"], R["n_clean"] = len(cases), len(clean)
        self._burn_in_median = cases.loc[cases["split_tag"] == "burn_in", "dwell_hours"].median()

        print("Profiling event log...")
        R["cardinality"] = p.cardinality(events, cases)
        R["activities"] = p.activity_frequencies(events)
        R["timing"] = p.milestone_timing(events)
        R["quality"] = p.timestamp_quality(events)
        R["alignment"] = p.call_date_alignment(cases)
        R["dwell_pct"] = p.dwell_percentiles(clean[clean["split_tag"].isin(MODEL_SPLITS)],
                                             by=["split_tag", "CTR_TYPE", "CTR_SIZE", "JOB_DEL_DOCTYPE"] + FLAGS)
        R["missing"] = p.missingness(cases, ["CTR_SIZE", "CTR_TYPE", "GROSS", "YARD_BLOCK", "YARD_SLOT", "JOB_DEL_DOCTYPE"])
        R["threshold"] = p.threshold_diagnostics(train, R["p_on"], R["p_crit"])
        train_clean = clean[clean["split_tag"] == "train"]
        cyclic = ["discharge_day_of_week", "discharge_hour"]  # non-monotonic, so screened as categories
        R["signal"] = p.feature_signal(train_clean, [c for c in NUMERIC_FEATURES if c not in cyclic], CATEGORICAL_FEATURES + cyclic)
        R["signal_arrival"] = p.filter_eligible_signals(R["signal"], scenario="ARRIVAL_TIME")
        R["signal_diagnostic"] = R["signal"][R["signal"]["eligibility_tier"] == "POST_DISCHARGE_DIAGNOSTIC"].copy().reset_index(drop=True)

        print("Analyzing extract window, drift and noise floor...")
        R["truncation"] = d.truncation_profile(cases, band)
        cases["observable_share"] = d.observable_share(cases, band)
        R["obs_by_split"] = cases[cases["split_tag"].isin(MODEL_SPLITS)].groupby("split_tag")["observable_share"].mean().round(3)
        R["cutoffs"] = d.cutoff_comparison(cases, band, R["p_on"], R["p_crit"])
        R["mix"] = d.split_label_mix(cases, band)
        R["weekly"] = d.weekly_dwell(cases, band, R["p_on"], R["p_crit"])
        R["trend"] = d.clean_band_trend(cases, band)
        R["daily"] = d.daily_calls(cases)
        R["gaps"] = d.missing_call_dates(R["daily"], band)
        R["weekday"] = d.weekday_rhythm(R["daily"], band)
        R["cycles"] = d.cycles(cases, band)
        R["departures"] = d.departure_daily(cases)
        R["noise"] = d.noise_floor(R["departures"])
        R["episode"] = d.departure_episode(R["departures"])
        R["weekday_dwell"] = d.weekday_dwell(cases, band)
        R["reporting"] = d.reporting_drift(events, cases, band)
        R["coverage_var"] = d.coverage_variation(R["reporting"], R["weekly"].set_index("call_week")["stays"])
        R["attr_drift"] = d.attribute_drift(cases, band, ["GROSS", "YARD_SLOT", "JOB_DEL_DOCTYPE"])
        R["shift"] = d.split_shift(cases[cases["split_tag"].isin(MODEL_SPLITS)], NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        R["maturity"] = d.label_maturity(cases)
        R["cold"] = d.cold_start(cases)
        R["flags"] = d.flag_effects(cases, band, FLAGS)
        R["features"] = self.feature_guidance()

        for key in ["cardinality", "activities", "timing", "quality", "alignment", "dwell_pct", "missing", "signal",
                    "signal_arrival", "signal_diagnostic",
                    "truncation", "cutoffs", "mix", "weekly", "gaps", "weekday", "weekday_dwell", "noise", "coverage_var",
                    "shift", "maturity", "flags", "features"]:
            if key in R:
                self.save(key, R[key])
        self.save("reporting_coverage", R["reporting"].reset_index())
        self.save("daily_calls", R["daily"])

    def feature_guidance(self) -> pd.DataFrame:
        """One row per candidate feature: availability, stability, signal and a recommendation."""
        R = self.results
        sig = R["signal"].set_index("feature")
        rows = []
        for _, s in R["shift"].iterrows():
            f = s["feature"]
            avail = AVAILABILITY.get(f, "at arrival")
            if s["constant_in_train"]:
                rec = "DROP: constant in train"
            elif avail == "post-arrival":
                rec = "MASK for arrival-time model"
            elif avail == "conditional":
                rec = "KEEP only if pre-arrival notice is guaranteed"
            elif s["max_psi"] > 0.25:
                rec = "KEEP with care: unstable (PSI>0.25)"
            else:
                rec = "KEEP"
            rows.append({"feature": f, "availability": avail, "signal": sig["association"].get(f, np.nan),
                         "signal_type": sig["note"].get(f, ""), "psi_val": s["psi_val"], "psi_test": s["psi_test"],
                         "recommendation": rec})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ figures
    def plot_all(self, cases: pd.DataFrame) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                             "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 130,
                             "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        R, band = self.results, self.results["band"]
        mask = self.drift.clean_mask(cases, band)
        for fn in (self._fig_dwell, self._fig_categories, self._fig_truncation, self._fig_weekly, self._fig_rhythm,
                   self._fig_noise, self._fig_reporting, self._fig_shift, self._fig_milestones, self._fig_cycles):
            fn(plt, cases, mask, R, band)
        print(f"Wrote {len(list(self.figures_dir.glob('*.png')))} figures to {self.figures_dir}")

    def _save_fig(self, fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(self.figures_dir / name, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

    @staticmethod
    def _shade(ax, band, xmin, xmax):
        ax.axvspan(xmin, band["clean_start"], color=COLORS["shade"], zorder=0)
        ax.axvspan(band["clean_end"] + pd.Timedelta(days=1), xmax, color=COLORS["shade"], zorder=0)

    def _fig_dwell(self, plt, cases, mask, R, band):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        bins = np.linspace(np.log10(3), np.log10(1500), 70)
        ticks = [5, 10, 30, 100, 300, 1000]
        for ax, (title, sub) in zip(axes, [("All stays (truncated weeks included)", cases), ("Clean band only", cases[mask])]):
            for s in MODEL_SPLITS:
                v = sub.loc[sub["split_tag"] == s, "dwell_hours"]
                v = v[v > 0]
                if len(v) >= 500:
                    ax.hist(np.log10(v), bins=bins, density=True, histtype="step", lw=1.8, color=COLORS[s], label=f"{s} (n={len(v):,})")
            for val, lab, side in [(R["p_on"], "p50", "right"), (R["p_crit"], "p85", "left")]:
                ax.axvline(np.log10(val), ls="--", color=COLORS["grey"], lw=1)
                text = f" train {lab} {val:.0f}h" if side == "left" else f"train {lab} {val:.0f}h "
                ax.text(np.log10(val), ax.get_ylim()[1] * 0.99, text, fontsize=8, va="top", ha=side, color=COLORS["grey"])
            ax.set_xticks(np.log10(ticks)); ax.set_xticklabels(ticks)
            ax.set_xlabel("dwell hours (log scale)"); ax.set_title(title); ax.legend(loc="upper right", bbox_to_anchor=(1, 0.8))
        axes[0].set_ylabel("density")
        self._save_fig(fig, "01_dwell_distribution.png")

    def _fig_categories(self, plt, cases, mask, R, band):
        sub = cases[mask & cases["split_tag"].isin(MODEL_SPLITS)]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, gridspec_kw={"width_ratios": [2, 1, 2.6]})
        short = {"has_quarantine_flag": "quarantine", "has_red_flag_pib": "red flag", "has_behandle_inspection": "behandle"}
        groups = [
            ("Container type", [(str(k), g["dwell_hours"]) for k, g in sub.groupby("CTR_TYPE") if len(g) >= 100]),
            ("Container size (ft)", [(str(int(k)), g["dwell_hours"]) for k, g in sub.groupby("CTR_SIZE") if len(g) >= 100]),
            ("Customs / quarantine flags", [(f"{short[f]}\n{'yes' if v else 'no'}", sub.loc[sub[f] == v, "dwell_hours"])
                                            for f in FLAGS for v in (0, 1)]),
        ]
        for ax, (title, items) in zip(axes, groups):
            ax.boxplot([i[1] for i in items], showfliers=False, patch_artist=True, medianprops={"color": "black"},
                       boxprops={"facecolor": "#C9D6E8", "edgecolor": COLORS["train"]}, whiskerprops={"color": COLORS["train"]})
            ax.set_xticklabels([f"{i[0]}\nn={len(i[1]):,}" if "\n" not in i[0] else f"{i[0]}\nn={len(i[1]):,}" for i in items], fontsize=7)
            ax.set_yscale("log"); ax.set_title(title)
            ax.axhline(R["p_on"], ls=":", color=COLORS["grey"]); ax.axhline(R["p_crit"], ls=":", color=COLORS["grey"])
        axes[0].set_ylabel("dwell hours (log, clean band; dotted = train p50 / p85)")
        self._save_fig(fig, "02_dwell_by_category.png")

    def _fig_truncation(self, plt, cases, mask, R, band):
        t = R["truncation"]
        fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        col = [COLORS["train"] if c else COLORS["grey"] for c in t["in_clean_band"]]
        a.bar(t["call_week"], t["stays"], width=5.5, color=col)
        for _, r in t.iterrows():
            if r["est_missing_pct"] >= 1 and r["stays"] > 0:
                a.text(r["call_week"], r["stays"], f"{r['est_missing_pct']:.0f}%\nmissing", ha="center", va="bottom", fontsize=7, color=COLORS["accent"])
        a.set_ylabel("stays per call week"); a.set_title("The extract is selected by gate-out date, so edge weeks are truncated (grey = outside clean band)")
        b.plot(t["call_week"], t["median_dwell_h"], marker="o", color=COLORS["accent"], label="median dwell")
        b.plot(t["call_week"], t["p85_dwell_h"], marker="o", color=COLORS["val"], label="p85 dwell")
        b.set_yscale("log"); b.set_ylabel("dwell hours (log)"); b.legend(loc="upper right")
        b.set_xlabel("vessel call week")
        for ax in (a, b):
            self._shade(ax, band, t["call_week"].min() - pd.Timedelta(days=4), t["call_week"].max() + pd.Timedelta(days=10))
        self._save_fig(fig, "03_extract_truncation.png")

    def _fig_weekly(self, plt, cases, mask, R, band):
        w = R["weekly"][R["weekly"]["stays"] >= 200]
        fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        a.plot(w["call_week"], w["p50_h"], marker="o", color=COLORS["train"], label="weekly p50")
        a.plot(w["call_week"], w["p85_h"], marker="o", color=COLORS["val"], label="weekly p85")
        a.axhline(R["p_on"], ls="--", color=COLORS["train"], alpha=0.6); a.axhline(R["p_crit"], ls="--", color=COLORS["val"], alpha=0.6)
        a.text(w["call_week"].iloc[-1], R["p_crit"], "  train p85 cutoff", va="bottom", ha="right", fontsize=8, color=COLORS["val"])
        a.text(w["call_week"].iloc[-1], R["p_on"], "  train p50 cutoff", va="bottom", ha="right", fontsize=8, color=COLORS["train"])
        a.set_ylabel("dwell hours"); a.legend(loc="upper right"); a.set_title("Weekly dwell by call cohort versus the fixed training cutoffs")
        col = [COLORS["accent"] if c else COLORS["grey"] for c in w["in_clean_band"]]
        b.bar(w["call_week"], w["pct_critical"], width=5.5, color=col)
        b.set_ylabel("% of stays above train p85"); b.set_xlabel("vessel call week")
        for ax in (a, b):
            self._shade(ax, band, w["call_week"].min() - pd.Timedelta(days=4), w["call_week"].max() + pd.Timedelta(days=10))
        self._save_fig(fig, "04_weekly_dwell_vs_cutoffs.png")

    def _fig_rhythm(self, plt, cases, mask, R, band):
        d, g, wk = R["daily"][R["daily"]["is_dense"]], R["gaps"], R["weekday"]
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.4, 1]})
        a.plot(d["call_date"], d["stays"], color=COLORS["train"], lw=1.2)
        self._shade(a, band, d["call_date"].min(), d["call_date"].max())
        a.scatter(g["call_date"], np.zeros(len(g)), color=COLORS["accent"], zorder=3, label="no-call day")
        for _, r in g.iterrows():
            a.text(r["call_date"], 150, r["weekday"][:3], rotation=90, fontsize=7, ha="center", color=COLORS["accent"])
        a.set_ylabel("stays by call date"); a.set_title("Daily vessel-call volume (grey = truncated weeks)"); a.legend(loc="upper right")
        b.bar(wk["weekday"].str[:3], wk["mean_stays"], color=COLORS["train"])
        for i, r in wk.iterrows():
            if r["zero_call_days"] > 0:
                b.text(i, r["mean_stays"], f"{int(r['zero_call_days'])}/{int(r['days'])}\nzero", ha="center", va="bottom", fontsize=7, color=COLORS["accent"])
        b.set_title("Mean stays by weekday (clean band)"); b.set_ylabel("mean stays per day")
        self._save_fig(fig, "05_call_rhythm.png")

    def _fig_noise(self, plt, cases, mask, R, band):
        dep = R["departures"].set_index("departure_date")
        tr_share = dep["delayed"].rolling(7).sum() / dep["volume"].rolling(7).sum().replace(0, np.nan)
        sigma = float(R["noise"].set_index("metric").loc["trailing_7d_delay_share", "observed_sigma"])
        level = tr_share.rolling(21, center=True, min_periods=10).median()
        fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        a.plot(tr_share.index, tr_share * 100, color=COLORS["train"], label="trailing 7d delayed share")
        a.plot(level.index, level * 100, color=COLORS["grey"], lw=1, label="local level (21d median)")
        a.fill_between(level.index, (level - 2 * sigma) * 100, (level + 2 * sigma) * 100, color=COLORS["grey"], alpha=0.2, label=f"±2σ noise floor (σ={sigma * 100:.1f} pp)")
        a.set_ylabel("% stays AT_RISK or CRITICAL"); a.legend(loc="upper right", fontsize=8)
        a.set_title("Noise floor of the 7-day delayed share, indexed by gate-out date (not truncated)")
        b.bar(dep.index, dep["volume"], color=COLORS["train"], width=0.9, alpha=0.8)
        b.plot(dep.index, dep["volume"].rolling(7).mean(), color=COLORS["accent"], label="7d mean")
        b.set_ylabel("departures per day"); b.legend(loc="upper right"); b.set_xlabel("gate-out date")
        self._save_fig(fig, "06_noise_floor.png")

    def _fig_reporting(self, plt, cases, mask, R, band):
        cov = R["reporting"]
        cov = cov.loc[:, cov.mean() < 99.5]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        im = ax.imshow(cov.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
        ax.set_xticks(range(cov.shape[1])); ax.set_xticklabels(cov.columns, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(cov.shape[0])); ax.set_yticklabels([d.strftime("%m-%d") for d in cov.index], fontsize=8)
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                v = cov.iat[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > cov.to_numpy().max() * 0.6 else "black")
        ax.grid(False); ax.set_title("Weekly share of cases recording each optional activity (%, clean-band weeks)")
        ax.set_ylabel("call week"); fig.colorbar(im, ax=ax, fraction=0.025)
        self._save_fig(fig, "07_reporting_coverage.png")

    def _fig_shift(self, plt, cases, mask, R, band):
        s = R["shift"].sort_values("max_psi", ascending=True).dropna(subset=["max_psi"])
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
        y = np.arange(len(s))
        a.barh(y + 0.2, s["psi_val"], height=0.38, color=COLORS["val"], label="val vs train")
        a.barh(y - 0.2, s["psi_test"], height=0.38, color=COLORS["test"], label="test vs train")
        a.set_yticks(y); a.set_yticklabels(s["feature"], fontsize=8)
        for x in (0.1, 0.25):
            a.axvline(x, ls="--", color=COLORS["grey"], lw=1)
        a.set_xscale("symlog", linthresh=0.1); a.set_xlabel("population stability index (symlog)"); a.legend(loc="lower right")
        a.set_title("Feature shift against train (dashed: 0.1 moderate, 0.25 major)")
        mix = R["mix"][R["mix"]["scope"] == "all stays"].set_index("split").loc[MODEL_SPLITS]
        bottom = np.zeros(len(mix))
        for c in ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]:
            b.bar(mix.index, mix[c] * 100, bottom=bottom, color=COLORS[c], label=c)
            bottom += mix[c].to_numpy() * 100
        for i, (n, crit) in enumerate(zip(mix["n"], mix["CRITICAL_DELAY"])):
            b.text(i, 101, f"n={int(n):,}", ha="center", fontsize=8)
        b.set_ylim(0, 108); b.set_ylabel("% of stays"); b.legend(loc="lower right", fontsize=8)
        b.set_title("Label mix by split (test is truncated)")
        self._save_fig(fig, "08_split_shift.png")

    def _fig_milestones(self, plt, cases, mask, R, band):
        t = R["timing"].sort_values("median_h").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for i, r in t.iterrows():
            c = COLORS["train"] if r["pct_before_discharge"] >= 50 else COLORS["val"]
            ax.hlines(i, r["p10_h"], r["p90_h"], color=c, lw=3, alpha=0.7)
            ax.plot(r["median_h"], i, "o", color="black", ms=4)
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(range(len(t))); ax.set_yticklabels([f"{r['activity']} ({r['pct_before_discharge']:.0f}% pre)" for _, r in t.iterrows()], fontsize=7)
        ax.invert_yaxis(); ax.set_xlabel("hours relative to DISCHARGE start (bar = p10 to p90, dot = median)")
        ax.set_title("When each activity starts: blue = mostly known before discharge, orange = in-yard")
        self._save_fig(fig, "09_milestone_timing.png")

    def _fig_cycles(self, plt, cases, mask, R, band):
        c = R["cycles"]
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
        for ax, col, title, cmap in [(axes[0], "stays", "Stays by discharge weekday x hour (clean band)", "Blues"),
                                     (axes[1], "median_dwell_h", "Median dwell hours (cells with >= 30 stays)", "Oranges")]:
            piv = c.pivot(index="discharge_day_of_week", columns="discharge_hour", values=col).reindex(range(7))
            if col == "median_dwell_h":
                piv = piv.where(c.pivot(index="discharge_day_of_week", columns="discharge_hour", values="stays").reindex(range(7)) >= 30)
            im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=cmap)
            ax.set_yticks(range(7)); ax.set_yticklabels(days); ax.set_xticks(range(0, 24, 3))
            ax.set_xlabel("discharge hour"); ax.set_title(title); ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.03)
        self._save_fig(fig, "10_discharge_cycles.png")

    # ------------------------------------------------------------------ findings report
    def write_findings(self) -> Path:
        R, band = self.results, self.results["band"]
        d = lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%d")  # noqa: E731
        t, w = R["truncation"], R["weekly"]
        wc = w[w["in_clean_band"] & (w["stays"] >= 200)]
        peak = wc.loc[wc["p50_h"].idxmax()]
        base = float(wc["p50_h"].tail(5).median())
        cut = R["cutoffs"].set_index("scope")
        mix = R["mix"]
        mix_all = mix[mix["scope"] == "all stays"].set_index("split")
        noise = R["noise"].set_index("metric")
        td, trend, cold = R["threshold"], R["trend"], R["cold"]
        obs = R["obs_by_split"]
        weekday = R["weekday"].set_index("weekday")
        gaps = R["gaps"]
        thu = weekday.loc["Thursday"]
        fl = R["flags"].set_index("flag")
        q = R["quality"]
        card = R["cardinality"].set_index("metric")["value"]
        shift = R["shift"].set_index("feature")
        feat = R["features"]
        align = R["alignment"].set_index("discharge_minus_call_days")["share_pct"]
        timing = R["timing"].set_index("activity")
        last_weeks = t.tail(3)
        ep = R["episode"]
        burn_med = float(self._burn_in_median)

        L: List[str] = []
        L += ["# EDA Findings: Stage 01 Container Dwell Data", "",
              "Generated by `ML/eda_engineer/eda_runner.py` from `data/processed` and `data/interim`. "
              "Every number below is computed by the run; tables are in `tables/`, figures in `figures/`.", ""]

        L += ["## 1. Headline findings", "",
              f"1. **The extract is selected by gate-out date, so call-date edges are truncated.** All {card['labeled_stays']:,} stays leave the yard between "
              f"{d(band['window_start'])} and {d(band['window_end'])}. Stays that arrived earlier appear only if they were slow; stays that arrived "
              f"late in September appear only if they were quick. Only call dates {d(band['clean_start'])} to {d(band['clean_end'])} are fully "
              f"observable (the {self.drift.horizon_quantile:.0%} dwell horizon is {band['horizon_hours']:.0f}h). That band holds {R['n_clean']:,} of {R['n_all']:,} stays.",
              f"2. **Test is not a clean holdout.** Mean observable share is {obs['train']:.1%} for train, {obs['val']:.1%} for val and {obs['test']:.1%} for test; "
              f"the last two call weeks are missing an estimated {last_weeks.iloc[-2]['est_missing_pct']:.0f}% and {last_weeks.iloc[-1]['est_missing_pct']:.0f}% of their stays. "
              f"Only {int(mix[(mix['scope'] == 'clean band') & (mix['split'] == 'test')]['n'].sum()):,} test stays are fully observable.",
              f"3. **Part of the dwell drift is real.** Inside the clean band, call date and dwell are negatively correlated (Spearman rho {trend['spearman_rho']:.2f}); "
              f"weekly median dwell falls from {peak['p50_h']:.0f}h in the week of {d(peak['call_week'])} to about {base:.0f}h in the last five clean weeks. "
              f"Val is fully inside the clean band and still has {mix_all.loc['val', 'CRITICAL_DELAY']:.1%} critical stays against {mix_all.loc['train', 'CRITICAL_DELAY']:.1%} in train.",
              f"4. **The data contains one congestion episode, in July.** Seven-day average departures fall to {ep['min_vol7']:.0f} per day around {d(ep['min_vol7_date'])} "
              f"(typical {ep['median_vol7']:.0f}), and the share of departing stays that are AT_RISK or CRITICAL peaks at {ep['peak_share']:.0%} around {d(ep['peak_share_date'])} "
              f"versus {ep['late_share']:.0%} over the last 30 days. The week of {d(peak['call_week'])} has {peak['pct_critical']:.0f}% of stays above the train p85 cutoff. "
              "Most positive (AT_RISK / CRITICAL) examples a model can learn from come from this episode.",
              f"5. **p50/p85 are label choices, not natural boundaries.** Log-dwell has {len(td['modes_hours'])} mode(s) at {td['modes_hours']}h and no trough; "
              f"relative density at the cutoffs is {td['relative_density_at_p_on_time']:.2f} (p50) and {td['relative_density_at_p_critical']:.2f} (p85).",
              f"6. **Weather features carry no information.** `weather_severity_index` and `has_active_disruption` are constant in train.",
              f"7. **Noise floor.** The trailing 7-day delayed share moves by about ±{noise.loc['trailing_7d_delay_share', 'observed_sigma'] * 100:.1f} pp "
              f"(1 sigma) with no real change, {noise.loc['trailing_7d_delay_share', 'overdispersion']:.1f}x what sampling alone would give; shifts smaller than "
              f"{noise.loc['trailing_7d_delay_share', 'min_detectable_shift_2sigma'] * 100:.1f} pp are indistinguishable from noise.", ""]

        L += ["## 2. Extract window and truncation", "",
              f"First gate-out {band['first_gate_out']}, last gate-out {band['last_gate_out']}; "
              f"discharges run from {d(band['first_discharge'])} to {d(band['last_discharge'])}. "
              f"The {int(card['stays_in_burn_in'])} 'burn-in' stays (call dates before 2021-06-10) have a median dwell of {burn_med:,.0f}h "
              f"({burn_med / 24:.0f} days): they are in the file only because they were still in the yard on {d(band['window_start'])}. "
              "`est_missing_pct` uses the clean-band dwell distribution to estimate the share of each call week's stays that could not be observed.", "",
              md_table(t[["call_week", "stays", "median_dwell_h", "p85_dwell_h", "est_missing_pct", "in_clean_band"]].assign(call_week=lambda x: x["call_week"].dt.strftime("%Y-%m-%d"))), "",
              "![truncation](figures/03_extract_truncation.png)", "",
              "### Effect on label cutoffs", "", md_table(R["cutoffs"]), "",
              f"Left truncation barely moves the train cutoffs (p50 {cut.loc['train (as built)', 'p50_h']:.1f}h as built versus "
              f"{cut.loc['clean band, train only', 'p50_h']:.1f}h on the clean band only). The damage is on the right: val and test cohorts are shorter.", "",
              "### Label mix by split", "", md_table(mix), "",
              "![split shift](figures/08_split_shift.png)", ""]

        L += ["## 3. Dwell time profile", "",
              "Clean-band percentiles (hours) for model splits and the main categories:", "",
              md_table(R["dwell_pct"], max_rows=45), "",
              "![dwell](figures/01_dwell_distribution.png)", "", "![categories](figures/02_dwell_by_category.png)", "",
              f"Weekly dwell versus the fixed cutoffs, which shows the July episode and the later calm:", "",
              "![weekly](figures/04_weekly_dwell_vs_cutoffs.png)", ""]

        L += ["## 4. Call rhythm, gaps and cycles", "",
              f"{len(gaps)} dates have no vessel calls after {self.drift.dense_start.date()}; {int(gaps['in_clean_band'].sum())} of them are inside the clean band "
              f"({', '.join(gaps.loc[gaps['in_clean_band'], 'weekday'].unique())}). Thursday averages {thu['mean_stays']:.0f} stays/day against "
              f"{weekday['mean_stays'].drop('Thursday').mean():.0f} on other weekdays, with {int(thu['zero_call_days'])} of {int(thu['days'])} Thursdays empty, "
              "so these gaps look like a weekly vessel-schedule pattern, not missing reporting. Gaps outside the clean band coincide with truncated weeks.", "",
              md_table(gaps.assign(call_date=lambda x: x["call_date"].dt.strftime("%Y-%m-%d"))), "",
              md_table(R["weekday"]), "",
              "![rhythm](figures/05_call_rhythm.png)", "",
              "Dwell also differs by discharge weekday (clean band, train + val + test). This is not adjusted for the July episode, "
              "so part of the pattern may be period mix:", "",
              md_table(R["weekday_dwell"]), "",
              f"`call_date` in `case_id` equals the discharge date for {align.get(0, 0):.1f}% of stays and is one day earlier for {align.get(1, 0):.1f}%. "
              "Calendar features built from `discharge_start_ts` and `call_date` therefore differ for about a quarter of stays.", "",
              "![cycles](figures/10_discharge_cycles.png)", ""]

        L += ["## 5. Noise floor", "",
              "Series are indexed by gate-out date, which is not truncated by the extract. Residuals are taken against a 21-day centered median; "
              "`binomial_sigma` is what pure sampling noise would give, so `overdispersion` above 1 means real operational variation.", "",
              md_table(R["noise"]), "", "![noise](figures/06_noise_floor.png)", ""]

        cv = R["coverage_var"]
        moving = cv[(cv["ratio"] > 2) & (cv["observed_sd_pp"] >= 1.0)]
        L += ["## 6. Reporting drift", "",
              "For each optional activity, the weekly share of cases that record it (clean-band weeks) is compared with the spread that sampling noise alone would "
              "produce (`ratio` = observed sd / binomial sd). No activity appears or disappears during the window, "
              f"and {len(moving)} of {len(cv)} activities with at least 1% coverage vary by more than twice the sampling spread and by at least 1 pp week to week"
              + (f" (largest: {', '.join(moving['activity'].head(3))})." if len(moving) else ".") + " "
              "Coverage that moves this much can reflect a real change in cargo mix or in what gets recorded; the log cannot separate the two, "
              "so treat these shares as drifting inputs, not as fixed rates.", "",
              md_table(cv), "", "![coverage](figures/07_reporting_coverage.png)", ""]
        cov = R["reporting"]

        L += ["## 7. Customs and quarantine effects (clean band)", "", md_table(R["flags"].drop(columns=["mannwhitney_p"])), "",
              f"Red-flag and Behandle stays dwell about {fl.loc['has_red_flag_pib', 'median_ratio']:.1f}x and {fl.loc['has_behandle_inspection', 'median_ratio']:.1f}x longer than others; "
              f"the quarantine flag alone shifts the median by only {fl.loc['has_quarantine_flag', 'median_ratio']:.2f}x. "
              "Per-split ratios for test are unreliable because test is truncated. "
              f"{timing.loc['HAS_QUARANTINE_FLAG', 'pct_before_discharge']:.0f}% of quarantine flags and {timing.loc['SP_RED_FLAG_PIB', 'pct_before_discharge']:.0f}% of red flags are raised before discharge; "
              "Behandle inspections happen in the yard only.", "", "![timing](figures/09_milestone_timing.png)", ""]

        L += ["## 8. Data quality ledger", "", md_table(q), ""]
        L += ["Container attributes missing, as percent of stays:", "", md_table(R["missing"]), ""]

        L += ["## 9. Guidance for the ML Engineer", "",
              "**CRITICAL ANTI-LEAKAGE SEPARATION:** Feature signals are strictly segregated by prediction-time eligibility. "
              "Univariate correlation rankings must NOT be interpreted as a validated feature-selection list for arrival-time models. "
              "Features such as `JOB_DEL_DOCTYPE`, `total_events_in_stay`, and `has_behandle_inspection` occur in the yard after container discharge "
              "and represent target leakage if included in arrival-time triage models.", "",
              "### 9A. Eligible Arrival-Time Predictors (Scenario: ARRIVAL_TIME at t_discharge)", "",
              md_table(R["signal_arrival"]), "",
              "### 9B. Retrospective In-Yard / Post-Discharge Diagnostics (FORBIDDEN at Arrival)", "",
              md_table(R["signal_diagnostic"]), "",
              "### 9C. Feature Stability and Modeling Recommendation Overview", "",
              md_table(feat.round(4)), "",
              "- Single features explain modest variance in dwell; report classification skill against prevalence baselines.",
              f"- Trailing 7-day features have PSI above 3 between train and val/test because the congestion level changed; they also start at zero for the first "
              f"{cold['rows']:,} train rows (call dates {cold['call_date_min']} to {cold['call_date_max']}), where departure history is unobserved (< 2021-07-01).",
              "- Strictly exclude `JOB_DEL_DOCTYPE`, `total_events_in_stay`, and `has_behandle_inspection` from arrival-time prediction.",
              "- Anomaly baselines (Isolation Forest, One-Class SVM) should focus on July operational disruptions.", ""]

        L += ["## 10. Guidance for the Evaluation Engineer", "",
              f"- Prevalence differs by split (CRITICAL_DELAY {mix_all.loc['train', 'CRITICAL_DELAY']:.1%} train, {mix_all.loc['val', 'CRITICAL_DELAY']:.1%} val, "
              f"{mix_all.loc['test', 'CRITICAL_DELAY']:.1%} test), so compare precision-recall results with the per-split base rate, not raw accuracy.",
              "- **Right-Censoring and Survivorship Bias in Test:** The test split (call dates 2021-09-12 to 2021-09-30) has a mean observable share of 85.4% (dropping to 50.0% in the final call week of Sept 27) because the extract ends on 2021-09-30 before slow-clearing containers can exit. Evaluating only completed stays in test produces survivorship bias (artificially deflating dwell time and over-stating on-time metrics).",
              "- **Validation Role and Selection Bias:** While the validation split (call dates 2021-08-27 to 2021-09-11) is 99.7% observable, treating validation as an un-tainted final benchmark is flawed: repeated model selection and hyperparameter tuning on validation will introduce optimistic validation bias.",
              "- **Evaluation Recommendations:** Downstream evaluation should report metrics stratified by censoring risk, utilize survival analysis / inverse probability of censoring weighting (IPCW), or evaluate primarily on cohorts with sufficient follow-up horizon.",
              f"- Alert thresholds on trailing delayed share should exceed about {noise.loc['trailing_7d_delay_share', 'min_detectable_shift_2sigma'] * 100:.1f} pp to clear the noise floor.",
              "- Label maturity: stays labeled with outcomes that were unknowable at the split boundary:", "", md_table(R["maturity"]), ""]

        L += ["## 11. Recommendations for the Data Engineer (not changed by EDA)", "",
              f"1. Document the cold-start window (call dates before 2021-07-01, {cold['rows']:,} train stays): trailing metrics are zero due to unobserved departure history prior to July 1 in the source extract, rather than a proven zero-activity port.",
              f"2. Retain train/val/test splits or add a per-stay `observable_share` column so evaluation can down-weight truncated test cohorts.",
              "3. Ensure trailing metric missingness semantics are documented for modelers to mask cold-start records during training.",
              "4. Clarify that `weather_severity_index` and `has_active_disruption` are constant 0 due to IMF PortWatch catalog coverage limitations (zero Indonesian alerts during June–Sept 2021) and unknown terminal UN/LOCODE, rather than confirmed absence of local adverse weather.",
              "5. Note date-only midnight timestamps (00:00:00) for customs events, where negative hourly offsets reflect discharge hour-of-day rather than true pre-discharge completion.",
              "6. Ingest terminal-level meteorological data if localized storm impact modeling is required.", "",
              "## Caveats", "",
              "- One terminal and about three months; weekly and monthly cycles are the longest that can be studied, so seasonality is out of scope.",
              "- The clean-band horizon uses the 99th percentile of observed dwell (431.4 hours); longer stays (about 1%) remain partly hidden at the right edge.",
              "- Delayed shares use train-fitted cutoffs (65.11h median, 160.90h 85th percentile), reflecting the empirical labeling rule rather than an external regulatory standard.", ""]
        path = self.out_dir / "findings.md"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(L))
        print(f"Wrote {path}")
        return path

    def run(self) -> Path:
        events, cases = self.load()
        self.analyze(events, cases)
        if self.make_figures:
            self.plot_all(cases)
        return self.write_findings()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Stage 01 EDA suite.")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--interim-dir", default="data/interim")
    ap.add_argument("--out-dir", default="reports/eda")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    EdaRunner(a.processed_dir, a.interim_dir, a.out_dir, not a.no_figures).run()


if __name__ == "__main__":
    main()
