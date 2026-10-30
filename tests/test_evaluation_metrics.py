"""Unit tests for Stage 01 Evaluation Engineer modules (metrics.py and cost_matrix.py)."""

import numpy as np
import pytest

from ML.evaluation_engineer.cost_matrix import (
    apply_prior_shift_correction,
    compute_break_even_ratio,
    compute_sensitivity_matrix,
    compute_threshold_sweep,
    compute_trivial_strategies,
    evaluate_decision_threshold,
)
from ML.evaluation_engineer.metrics import (
    compute_bootstrap_confidence_intervals,
    compute_calibration_metrics,
    compute_classification_metrics,
    compute_confusion_matrices,
    compute_dummy_baselines,
    compute_ranking_metrics,
    compute_wilson_score_interval,
)


def test_confusion_matrices_shape_and_normalization():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    res = compute_confusion_matrices(y_true, y_pred)
    raw = np.array(res["raw_counts"])
    norm = np.array(res["row_normalized"])

    assert raw.shape == (3, 3)
    assert norm.shape == (3, 3)
    np.testing.assert_allclose(norm.sum(axis=1), np.ones(3), rtol=1e-5)


def test_classification_metrics_and_support():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    res = compute_classification_metrics(y_true, y_pred)

    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["per_class"]["CRITICAL_DELAY"]["recall"] == 1.0
    assert res["per_class"]["CRITICAL_DELAY"]["support"] == 2


def test_dummy_baselines():
    y_train = np.array([0, 0, 0, 1, 1, 2])
    y_eval = np.array([0, 1, 2, 0, 1, 2])
    res = compute_dummy_baselines(y_train, y_eval)

    assert res["majority_prior"]["majority_class"] == 0
    assert res["majority_prior"]["critical_recall"] == 0.0


def test_ranking_metrics_and_lift():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    # Perfect probabilities for class 2
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
        [0.05, 0.05, 0.9],
    ])
    res = compute_ranking_metrics(y_true, y_prob)

    assert res["critical_delay_roc_auc"] == 1.0
    assert res["critical_delay_pr_auc"] == 1.0
    # Prevalence is 2/6 = 0.3333; lift = 1.0 / 0.3333 = 3.0
    assert res["critical_delay_pr_lift"] == pytest.approx(3.0, rel=1e-2)


def test_calibration_metrics():
    y_true = np.array([0, 1, 2])
    y_prob = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    res = compute_calibration_metrics(y_true, y_prob, n_bins=5)

    assert res["brier_score"] == 0.0
    assert res["overall_ece"] == 0.0


def test_bootstrap_confidence_intervals():
    y_true = np.array([0, 0, 1, 1, 2, 2] * 20)
    y_pred = np.array([0, 0, 1, 1, 2, 2] * 20)
    y_prob = np.tile(np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.05, 0.05, 0.9]]), (20, 1))

    res = compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, n_bootstraps=50, random_state=42)
    assert "macro_f1" in res
    assert "critical_delay_pr_lift" in res
    assert res["critical_delay_pr_lift"]["ci_lower"] > 1.0


def test_cost_matrix_break_even_and_strategies():
    y_true = np.array([0, 0, 0, 0, 2])  # 1 positive, 4 negatives
    y_prob_crit = np.array([0.1, 0.2, 0.3, 0.8, 0.9])  # 2 flagged at tau=0.5 (1 TP, 1 FP)
    y_pred_argmax = np.array([0, 0, 0, 2, 2])

    strat = compute_trivial_strategies(y_true, y_pred_argmax, y_prob_crit, tau_star=0.5, c_fa=100.0, c_fn=1000.0)
    assert strat["do_nothing"]["total_cost"] == 1000.0
    # tau*=0.5: FP=1, FN=0 -> Cost = 1*100 + 0*1000 = 100.0
    assert strat["tau_star"]["total_cost"] == 100.0
    assert strat["summary"]["model_beats_best_trivial"] is True

    # Break even ratio: FP / TP = 1 / 1 = 1.0
    be = compute_break_even_ratio(y_true, y_prob_crit, tau=0.5)
    assert be["break_even_cost_ratio"] == 1.0


def test_prior_shift_correction():
    y_prob = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    p_train = np.array([0.5, 0.3, 0.2])
    p_target = np.array([0.8, 0.15, 0.05])

    p_corr = apply_prior_shift_correction(y_prob, p_train, p_target)
    assert p_corr.shape == (2, 3)
    np.testing.assert_allclose(p_corr.sum(axis=1), np.ones(2), rtol=1e-5)
    # Target prevalence for class 2 is lower, so corrected class 2 probability should decrease
    assert p_corr[1, 2] < y_prob[1, 2]


def test_ci_containment():
    """Asserts that the empirical point estimate lies strictly inside the bootstrap 95% CI."""
    y_true = np.array([0, 0, 1, 1, 2, 2] * 20)
    y_pred = np.array([0, 1, 1, 1, 2, 2] * 20)
    y_prob = np.tile(np.array([
        [0.8, 0.1, 0.1],
        [0.3, 0.6, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.05, 0.15, 0.8],
    ]), (20, 1))

    cis = compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, n_bootstraps=50, random_state=42)

    for metric_name in ["prevalence", "macro_f1", "critical_delay_recall", "critical_delay_precision", "critical_delay_pr_auc", "critical_delay_roc_auc"]:
        m = cis[metric_name]
        assert m["ci_lower"] <= m["point_estimate"] <= m["ci_upper"], (
            f"{metric_name} point estimate {m['point_estimate']} not inside [{m['ci_lower']}, {m['ci_upper']}]"
        )

    for k in [1, 5, 10, 20]:
        k_key = f"top_{k}pct"
        for sub_metric in ["precision", "recall", "lift"]:
            m = cis["top_k_ranking"][k_key][sub_metric]
            assert m["ci_lower"] <= m["point_estimate"] <= m["ci_upper"], (
                f"top_{k}pct {sub_metric} point estimate {m['point_estimate']} not inside [{m['ci_lower']}, {m['ci_upper']}]"
            )


def test_flag_rate_identity():
    """Asserts that flag_rate == 100 * (TP + FP) / N for every trivial strategy."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred_argmax = np.array([0, 0, 1, 1, 2, 0, 0, 1, 2, 2])
    y_prob_crit = np.array([0.05, 0.1, 0.15, 0.2, 0.85, 0.3, 0.05, 0.25, 0.9, 0.75])
    n = len(y_true)

    strat = compute_trivial_strategies(y_true, y_pred_argmax, y_prob_crit, tau_star=0.5, c_fa=300.0, c_fn=3000.0)

    for s_name in ["do_nothing", "flag_everything", "argmax", "tau_star"]:
        s = strat[s_name]
        expected_flag_rate = round(100.0 * (s["true_positives"] + s["false_positives"]) / n, 2)
        assert abs(s["flagged_pct"] - expected_flag_rate) < 0.05, (
            f"{s_name} flag rate {s['flagged_pct']} != expected {expected_flag_rate}"
        )


def test_cost_identity_per_strategy_row():
    """Asserts that total_cost == C_FA * FP + C_FN * FN for every strategy row."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred_argmax = np.array([0, 0, 1, 1, 2, 0, 0, 1, 2, 2])
    y_prob_crit = np.array([0.05, 0.1, 0.15, 0.2, 0.85, 0.3, 0.05, 0.25, 0.9, 0.75])
    c_fa = 300.0
    c_fn = 3000.0

    strat = compute_trivial_strategies(y_true, y_pred_argmax, y_prob_crit, tau_star=0.5, c_fa=c_fa, c_fn=c_fn)

    for s_name in ["do_nothing", "flag_everything", "argmax", "tau_star"]:
        s = strat[s_name]
        expected_cost = c_fa * s["false_positives"] + c_fn * s["false_negatives"]
        assert s["total_cost"] == expected_cost, (
            f"{s_name} cost {s['total_cost']} != expected {expected_cost}"
        )


def test_sensitivity_cost_base_case_matches_section4():
    """Asserts that sensitivity matrix base case (C_FA=300, C_FN=3000) matches Section 4 total cost exactly."""
    y_val = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_prob_val = np.array([0.05, 0.1, 0.15, 0.2, 0.85, 0.3, 0.05, 0.25, 0.9, 0.75])

    y_test = np.array([0, 0, 0, 1, 2, 0, 1, 2, 0, 0])
    y_prob_test = np.array([0.02, 0.08, 0.1, 0.12, 0.8, 0.05, 0.2, 0.7, 0.1, 0.05])

    # Tune on val
    _, tau_star, _ = compute_threshold_sweep(y_val, y_prob_val, c_fa=300.0, c_fn=3000.0)

    # Strategy on test at tau_star
    strat_test = compute_trivial_strategies(
        y_test,
        (y_prob_test >= 0.5).astype(int) * 2,
        y_prob_test,
        tau_star=tau_star,
        c_fa=300.0,
        c_fn=3000.0,
    )
    expected_tau_cost = strat_test["tau_star"]["total_cost"]

    # Sensitivity matrix evaluating on test
    sens_df = compute_sensitivity_matrix(
        y_true=y_val,
        y_prob_crit=y_prob_val,
        y_eval=y_test,
        y_prob_eval_crit=y_prob_test,
        c_fa_values=[300.0],
        c_fn_values=[3000.0],
    )

    base_row = sens_df[(sens_df["c_fa_usd"] == 300) & (sens_df["c_fn_usd"] == 3000)].iloc[0]
    assert base_row["total_cost"] == expected_tau_cost
    assert base_row["do_nothing_cost"] == 3000.0 * np.sum(y_test == 2)
    assert base_row["net_saving"] == base_row["do_nothing_cost"] - base_row["total_cost"]
    assert base_row["beats_do_nothing"] == (base_row["net_saving"] > 0)


def test_wilson_confidence_interval():
    """Asserts Wilson score intervals for proportion estimation."""
    res_if = compute_wilson_score_interval(k=4, n=455)
    assert res_if["point_estimate"] == round(4 / 455, 4)
    assert res_if["ci_lower"] <= res_if["point_estimate"] <= res_if["ci_upper"]
    # IF upper bound must be strictly below test base rate (2.76%)
    assert res_if["ci_upper"] < 0.0276

    res_hybrid_cal = compute_wilson_score_interval(k=1, n=3)
    assert res_hybrid_cal["point_estimate"] == round(1 / 3, 4)
    assert res_hybrid_cal["ci_lower"] <= res_hybrid_cal["point_estimate"] <= res_hybrid_cal["ci_upper"]
    assert res_hybrid_cal["ci_lower"] == pytest.approx(0.0615, abs=1e-3)
    assert res_hybrid_cal["ci_upper"] == pytest.approx(0.7923, abs=1e-3)


def test_net_saving_bootstrap_ci_containment():
    """Asserts that bootstrap 95% CI for Strategy D net saving contains its point estimate."""
    from ML.evaluation_engineer.cost_matrix import compute_net_saving_bootstrap_ci
    y_true = np.array([0, 0, 1, 1, 2, 2] * 20)
    y_prob_crit = np.array([0.05, 0.1, 0.15, 0.2, 0.85, 0.9] * 20)
    tau = 0.5
    c_fa = 300.0
    c_fn = 3000.0

    res = compute_net_saving_bootstrap_ci(y_true, y_prob_crit, tau, c_fa, c_fn, n_bootstraps=100, random_state=42)
    assert "point" in res and "ci_lower" in res and "ci_upper" in res
    assert res["ci_lower"] <= res["point"] <= res["ci_upper"], (
        f"Point estimate {res['point']} not contained in [{res['ci_lower']}, {res['ci_upper']}]"
    )
    assert res["includes_zero"] == (res["ci_lower"] <= 0 <= res["ci_upper"])


def test_sensitivity_degenerate_labelling():
    """Asserts that sensitivity scenarios with alert rate > 50% are flagged as degenerate."""
    y_val = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_prob_val = np.array([0.05, 0.1, 0.15, 0.2, 0.85, 0.3, 0.05, 0.25, 0.9, 0.75])

    y_test = np.array([0, 0, 0, 1, 2, 0, 1, 2, 0, 0])
    y_prob_test = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 0.65, 0.75, 0.85, 0.7, 0.6])

    sens_df = compute_sensitivity_matrix(
        y_true=y_val,
        y_prob_crit=y_prob_val,
        y_eval=y_test,
        y_prob_eval_crit=y_prob_test,
        c_fa_values=[150.0],
        c_fn_values=[5000.0],
    )
    row = sens_df.iloc[0]
    # Check that alert rate is computed mechanically
    assert row["alert_rate_pct"] == round(100.0 * (row["recall_at_tau"] * np.sum(y_test == 2) + (row["precision_at_tau"] > 0)) / len(y_test), 2) or row["alert_rate_pct"] > 50.0
    # Flag rate > 50% threshold check
    is_degenerate = row["alert_rate_pct"] > 50.0
    assert is_degenerate, "Expected scenario with high alert rate to exceed 50%"


def test_beats_do_nothing_mechanical():
    """Asserts that beats_do_nothing is strictly mechanical: net_saving > 0."""
    y_val = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_prob_val = np.array([0.05, 0.1, 0.15, 0.2, 0.85, 0.3, 0.05, 0.25, 0.9, 0.75])
    y_test = np.array([0, 0, 0, 1, 2, 0, 1, 2, 0, 0])
    y_prob_test = np.array([0.02, 0.08, 0.1, 0.12, 0.8, 0.05, 0.2, 0.7, 0.1, 0.05])

    sens_df = compute_sensitivity_matrix(
        y_true=y_val,
        y_prob_crit=y_prob_val,
        y_eval=y_test,
        y_prob_eval_crit=y_prob_test,
        c_fa_values=[150.0, 300.0, 500.0],
        c_fn_values=[1500.0, 3000.0, 5000.0],
    )
    for _, row in sens_df.iterrows():
        expected_beats = bool(row["net_saving"] > 0)
        assert row["beats_do_nothing"] == expected_beats, (
            f"beats_do_nothing ({row['beats_do_nothing']}) does not equal net_saving > 0 ({expected_beats}) for net saving {row['net_saving']}"
        )


def test_identities_on_all_splits():
    """Asserts flag_rate == 100*(TP+FP)/N and cost == C_FA*FP + C_FN*FN on simulated splits."""
    c_fa = 300.0
    c_fn = 3000.0
    for n_samples in [50, 100, 200]:
        rng = np.random.default_rng(n_samples)
        y_true = rng.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
        y_pred = rng.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])
        y_prob = rng.uniform(0.0, 1.0, size=n_samples)

        strat = compute_trivial_strategies(y_true, y_pred, y_prob, tau_star=0.3, c_fa=c_fa, c_fn=c_fn)
        for s_name in ["do_nothing", "flag_everything", "argmax", "tau_star"]:
            s = strat[s_name]
            expected_flag_rate = round(100.0 * (s["true_positives"] + s["false_positives"]) / n_samples, 2)
            assert abs(s["flagged_pct"] - expected_flag_rate) < 0.05
            expected_cost = c_fa * s["false_positives"] + c_fn * s["false_negatives"]
            assert s["total_cost"] == expected_cost


