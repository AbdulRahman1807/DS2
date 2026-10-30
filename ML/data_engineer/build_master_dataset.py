"""Hybrid Master Dataset Builder (real stays + external context + synthetic training stays).

Assigned Role: Data Engineer

Outputs (data/master/):
- master_real.parquet       every reconstructed real stay with milestone availability timestamps,
                            cohort / holdout flags, GROSS in tonnes, and discharge-time context
                            (PPKM/holiday calendar, PortWatch with 7-day publication lag)
- master_synthetic.parquet  synthetic stays from a classical process simulator (no deep learning)
- master_manifest.json      row counts, hashes, generation parameters and fidelity checks

Synthetic generation (process-segment recombination):
1. Pick a real *template* stay from the training period (class-weighted to a target class mix);
   keep its container attributes, vessel call and milestone pattern; jitter discharge by +/-2 h.
2. Draw the customs segment (discharge -> customs release -> delivery job) from one donor and the
   collection segment (delivery job -> truck in -> gate-out) from another donor in the same stratum
   (class x quarantine x red flag x behandle x discharge week), with lognormal noise (sigma 0.1).
3. Rebuild the event chain (customs keeps the donor's whole-day offset, as the field is date-only), drop template
   milestones that would fall after the simulated gate-out, and recompute dwell and the label.

Usage rules (enforced by load_training_rows and tests):
- Synthetic rows are for TRAINING ONLY; never evaluated, never counted in yard/call/terminal aggregates.
- A synthetic row may be used in a fold only if synth_donor_max_discharge < fold train_end, so no
  outcome from the fold's evaluation window (or later) reaches training through the generator.
- Donors are restricted to discharges before the holdout training cutoff (2021-08-20).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from ML.ml_engineer.reassessment.asof_features import _ns, calendar_state, portwatch_state
from ML.ml_engineer.reassessment.cohort import HOLDOUT_TEST, MILESTONES, assign_target, load_case_table

MASTER_DIR = REPO_ROOT / "data/master"
CASE_TABLE_PATH = REPO_ROOT / "data/interim/reassessment_case_table.parquet"
REAL_SOURCE = "mendeley_tps_2021"
SYNTH_SOURCE = "synthetic_segment_recombination_v1"
DONOR_CUTOFF = pd.Timestamp(HOLDOUT_TEST["train_end"])
CHAIN = ["CUSTOMS_DEL", "JOB_DEL", "TRUCK_IN", "TRUCK_OUT"]
FIXED_OFFSET = ["BAPLIE", "VESSEL_ATB"]
TEMPLATE_MILESTONES = [m for m in MILESTONES if m not in CHAIN + FIXED_OFFSET + ["DISCHARGE"]]
HOUR = pd.Timedelta(hours=1)


# --------------------------------------------------------------------------------------------
# Real rows
# --------------------------------------------------------------------------------------------
def gross_tonnes(gross: pd.Series) -> pd.Series:
    """18.9% of GROSS values are recorded in kilograms (> 100); the rest in tonnes."""
    return gross.where(gross <= 100, gross / 1000.0)


def discharge_context(d0: pd.Series) -> pd.DataFrame:
    t = _ns(pd.Series(pd.to_datetime(d0.to_numpy())))
    ctx = pd.concat([calendar_state(t), portwatch_state(t)], axis=1)
    return ctx.add_prefix("ctx_discharge_").set_index(d0.index)


def build_real(cases: pd.DataFrame) -> pd.DataFrame:
    real = cases.copy()
    d0 = real["discharge_start_ts"]
    real["source"] = REAL_SOURCE
    real["is_synthetic"] = False
    real["gross_t"] = gross_tonnes(real["GROSS"])
    real["gross_recorded_in_kg"] = real["GROSS"] > 100
    real["is_holdout_window"] = real["in_cohort"] & (d0 >= pd.Timestamp(HOLDOUT_TEST["eval_start"])) & (
        d0 < pd.Timestamp(HOLDOUT_TEST["eval_end"]))
    real["synth_template_case_id"] = None
    real["synth_donor_case_ids"] = None
    real["synth_donor_max_discharge"] = pd.NaT
    return pd.concat([real, discharge_context(d0)], axis=1)


# --------------------------------------------------------------------------------------------
# Synthetic rows
# --------------------------------------------------------------------------------------------
def _segments(pool: pd.DataFrame) -> pd.DataFrame:
    """Gaps (hours) of the customs and collection segments; missing milestones collapse onto exit."""
    out = pool["truck_out_end_ts"]
    c = pool["avail_CUSTOMS_DEL"].fillna(out)
    j = pool["avail_JOB_DEL"].fillna(out)
    ti = pool["avail_TRUCK_IN"].fillna(out)
    d0 = pool["discharge_start_ts"]
    return pd.DataFrame({
        "g_customs": (c - d0) / HOUR,
        "g_jobdel": (j - c) / HOUR,
        "g_truckin": (ti - j) / HOUR,
        "g_exit": (out - ti) / HOUR,
    }, index=pool.index)


def _strata(pool: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "cls": pool["target_class"],
        "q": pool["avail_HAS_QUARANTINE_FLAG"].notna(),
        "rf": pool["avail_SP_RED_FLAG_PIB"].notna(),
        "bh": pool["avail_FIRST_JOB_CUSTOMS_BEHANDLE"].notna(),
        "wk": pool["discharge_start_ts"].dt.to_period("W").astype(str),
    }, index=pool.index)


def _noisy(g: np.ndarray, rng, sigma: float) -> np.ndarray:
    """Multiplicative lognormal noise on magnitudes; preserves sign (pre-discharge customs stays negative)."""
    return g * np.exp(rng.normal(0.0, sigma, len(g)))


def synthesize(cases: pd.DataFrame, n: int = 30_000, class_mix=(1 / 3, 1 / 3, 1 / 3), noise_sigma: float = 0.10,
               jitter_hours: float = 2.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = cases[cases["in_cohort"] & (cases["discharge_start_ts"] < DONOR_CUTOFF)].copy()
    seg = _segments(pool)
    st = _strata(pool)
    fine_key = st.astype(str).agg("|".join, axis=1)
    coarse_key = st.drop(columns="wk").astype(str).agg("|".join, axis=1)
    fine_groups = {k: v.to_numpy() for k, v in pd.Series(pool.index).groupby(fine_key.to_numpy())}
    coarse_groups = {k: v.to_numpy() for k, v in pd.Series(pool.index).groupby(coarse_key.to_numpy())}

    # Template sampling weighted to the requested class mix.
    cls_share = pool["target_class"].value_counts(normalize=True)
    w = pool["target_class"].map(lambda k: class_mix[k] / cls_share[k]).to_numpy()
    templates = rng.choice(pool.index.to_numpy(), size=n, replace=True, p=w / w.sum())

    def donor(idx):
        fk, ck = fine_key[idx], coarse_key[idx]
        grp = fine_groups[fk] if len(fine_groups[fk]) >= 5 else coarse_groups[ck]
        return grp[rng.integers(len(grp))]

    donors_a = np.array([donor(i) for i in templates])
    donors_b = np.array([donor(i) for i in templates])

    tpl = pool.loc[templates].reset_index(drop=True)
    shift = pd.to_timedelta(rng.uniform(-jitter_hours, jitter_hours, n), unit="h")
    d0 = tpl["discharge_start_ts"] + shift

    g_j = _noisy(seg.loc[donors_a, "g_jobdel"].to_numpy(), rng, noise_sigma)
    g_t = np.maximum(_noisy(seg.loc[donors_b, "g_truckin"].to_numpy(), rng, noise_sigma), 0.0)
    g_x = np.maximum(_noisy(seg.loc[donors_b, "g_exit"].to_numpy(), rng, noise_sigma), 0.25)

    # Customs availability is a midnight (date + 24 h); keep the donor's whole-day offset from its
    # discharge date instead of rounding a noisy time up, which would add up to a day of delay.
    donor_a = pool.loc[donors_a]
    day_offset = ((donor_a["avail_CUSTOMS_DEL"].fillna(donor_a["truck_out_end_ts"]).dt.normalize()
                   - donor_a["discharge_start_ts"].dt.normalize()) / pd.Timedelta(days=1)).to_numpy()
    customs = d0.dt.normalize() + pd.to_timedelta(day_offset, unit="D")
    jobdel = customs + pd.to_timedelta(g_j, unit="h")
    truck_in = np.maximum(jobdel, d0 + HOUR) + pd.to_timedelta(g_t, unit="h")
    truck_out = truck_in + pd.to_timedelta(g_x, unit="h")

    syn = tpl.copy()
    syn["case_id"] = [f"SYN-{i:06d}" for i in range(n)]
    syn["container_id"] = [f"{p}SYN{i:06d}"[:11] for i, p in enumerate(tpl["carrier_prefix"].fillna("UNK"))]
    syn["discharge_start_ts"] = d0
    for m in FIXED_OFFSET:
        syn[f"avail_{m}"] = tpl[f"avail_{m}"] + shift
    syn["avail_DISCHARGE"] = tpl["avail_DISCHARGE"] + shift
    for m in TEMPLATE_MILESTONES:
        moved = tpl[f"avail_{m}"] + shift
        syn[f"avail_{m}"] = moved.where(moved <= truck_out)
    syn["avail_CUSTOMS_DEL"] = customs
    syn["avail_JOB_DEL"] = jobdel
    syn["avail_TRUCK_IN"] = truck_in
    syn["avail_UNSTACK_TO_TRUCK"] = truck_in.where(syn["avail_UNSTACK_TO_TRUCK"].notna())
    syn["avail_TRUCK_OUT"] = truck_out
    syn["truck_out_end_ts"] = truck_out
    syn["dwell_hours"] = (truck_out - d0) / HOUR
    syn["target_class"] = assign_target(syn["dwell_hours"]).to_numpy()
    syn["call_date"] = tpl["call_date"]
    syn["source"] = SYNTH_SOURCE
    syn["is_synthetic"] = True
    syn["in_cohort"] = False  # never part of the evaluation cohort
    syn["is_holdout_window"] = False
    syn["gross_t"] = gross_tonnes(tpl["GROSS"])
    syn["gross_recorded_in_kg"] = tpl["GROSS"] > 100
    syn["synth_template_case_id"] = tpl["case_id"].to_numpy()
    syn["synth_donor_case_ids"] = [f"{pool.at[a, 'case_id']}|{pool.at[b, 'case_id']}" for a, b in zip(donors_a, donors_b)]
    syn["synth_donor_max_discharge"] = np.maximum.reduce([
        tpl["discharge_start_ts"].to_numpy(),
        pool.loc[donors_a, "discharge_start_ts"].to_numpy(),
        pool.loc[donors_b, "discharge_start_ts"].to_numpy(),
    ])
    syn = pd.concat([syn, discharge_context(syn["discharge_start_ts"])], axis=1)
    return syn


# --------------------------------------------------------------------------------------------
# Fidelity checks and loading
# --------------------------------------------------------------------------------------------
def fidelity_report(real_pool: pd.DataFrame, syn: pd.DataFrame) -> Dict:
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    order_ok = (
        (syn["truck_out_end_ts"] >= syn["avail_TRUCK_IN"]) & (syn["avail_TRUCK_IN"] >= syn["discharge_start_ts"])
    ).mean()
    exact_copies = syn.merge(real_pool, on=["discharge_start_ts", "truck_out_end_ts"], how="inner").shape[0]
    return {
        "n_synthetic": int(len(syn)),
        "class_mix_real_pool": real_pool["target_class"].value_counts(normalize=True).sort_index().round(4).tolist(),
        "class_mix_synthetic": syn["target_class"].value_counts(normalize=True).sort_index().round(4).tolist(),
        "dwell_quantiles_real_pool_h": real_pool["dwell_hours"].quantile(q).round(1).tolist(),
        "dwell_quantiles_synthetic_h": syn["dwell_hours"].quantile(q).round(1).tolist(),
        "dwell_quantiles_by_class": {
            int(k): {"real": real_pool.loc[real_pool["target_class"] == k, "dwell_hours"].quantile([0.1, 0.5, 0.9]).round(1).tolist(),
                     "synthetic": syn.loc[syn["target_class"] == k, "dwell_hours"].quantile([0.1, 0.5, 0.9]).round(1).tolist()}
            for k in (0, 1, 2)
        },
        "label_kept_template_class_share": float((syn["target_class"].to_numpy()
                                                  == real_pool.set_index("case_id").loc[syn["synth_template_case_id"], "target_class"].to_numpy()).mean()),
        "event_order_valid_share": float(order_ok),
        "exact_timestamp_copies_of_real_rows": int(exact_copies),
        "max_donor_discharge": str(syn["synth_donor_max_discharge"].max()),
    }


def load_training_rows(train_end: str, include_synthetic: bool = True, master_dir: Path = MASTER_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Real rows (for features, aggregates and evaluation) and the synthetic rows a fold may train on."""
    real = pd.read_parquet(master_dir / "master_real.parquet")
    if not include_synthetic:
        return real, real.iloc[0:0]
    syn = pd.read_parquet(master_dir / "master_synthetic.parquet")
    usable = syn[pd.to_datetime(syn["synth_donor_max_discharge"]) < pd.Timestamp(train_end)]
    return real, usable


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_master_dataset(n_synthetic: int = 30_000, seed: int = 42, master_dir: Path = MASTER_DIR) -> Dict:
    master_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_parquet(CASE_TABLE_PATH) if CASE_TABLE_PATH.exists() else load_case_table(
        str(REPO_ROOT / "data/interim/cleaned_event_log.parquet"))
    real = build_real(cases)
    syn = synthesize(cases, n=n_synthetic, seed=seed)
    pool = cases[cases["in_cohort"] & (cases["discharge_start_ts"] < DONOR_CUTOFF)]

    p_real, p_syn = master_dir / "master_real.parquet", master_dir / "master_synthetic.parquet"
    real.to_parquet(p_real, index=False)
    syn.to_parquet(p_syn, index=False)
    manifest = {
        "dataset": "DS2 hybrid master dataset",
        "version": "1.0.0",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "real": {"path": _rel(p_real), "sha256": _sha(p_real), "rows": int(len(real)),
                 "cohort_rows": int(real["in_cohort"].sum()), "holdout_window_rows": int(real["is_holdout_window"].sum()),
                 "gross_in_kg_rows": int(real["gross_recorded_in_kg"].sum())},
        "synthetic": {"path": _rel(p_syn), "sha256": _sha(p_syn), "rows": int(len(syn)),
                      "method": SYNTH_SOURCE, "seed": seed, "class_mix_target": [1 / 3, 1 / 3, 1 / 3],
                      "noise_sigma": 0.10, "discharge_jitter_hours": 2.0, "donor_cutoff": str(DONOR_CUTOFF)},
        "external_context": ["data/reference/indonesia_calendar_2021.csv",
                             "data/raw/portwatch_daily_ports.parquet (7-day publication lag)"],
        "usage_rules": [
            "Synthetic rows are training-only; never evaluate on them.",
            "Never include synthetic rows in yard, vessel-call or terminal aggregates (as-of features use real rows only).",
            "Use a synthetic row in a fold only if synth_donor_max_discharge < fold train_end.",
            "Holdout-window real rows remain quarantined until explicit approval.",
        ],
        "fidelity": fidelity_report(pool, syn),
    }
    (master_dir / "master_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


if __name__ == "__main__":
    m = build_master_dataset()
    print(json.dumps({k: m[k] for k in ("real", "synthetic", "fidelity")}, indent=2, default=str))
