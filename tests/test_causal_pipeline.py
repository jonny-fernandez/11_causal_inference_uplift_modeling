from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_contract import load_data
from src.propensity import CATEGORICAL_FEATURES, NUMERIC_FEATURES


PROJECT_CONFIG_PATH = Path("config/project_config.json")
METHODOLOGY_PATH = Path("config/final_methodology.json")
VARIABLE_POLICY_PATH = Path("data/reference/variable_policy.csv")

BASELINE_PATH = Path("artifacts/naive_baseline.json")
PROPENSITY_SCORES_PATH = Path("artifacts/propensity_scores.csv")
PROPENSITY_DIAGNOSTICS_PATH = Path("artifacts/propensity_diagnostics.json")
WEIGHTS_PATH = Path("artifacts/ipw_weights.csv")
WEIGHT_DIAGNOSTICS_PATH = Path("artifacts/weight_diagnostics.json")
BALANCE_PATH = Path("artifacts/balance_diagnostics.csv")
BALANCE_SUMMARY_PATH = Path("artifacts/balance_summary.json")
IPW_PATH = Path("artifacts/ate_summary.json")
AIPW_PATH = Path("artifacts/aipw_summary.json")
CROSSFIT_PATH = Path("artifacts/aipw_crossfit_summary.json")
CROSSFIT_SCORES_PATH = Path("artifacts/aipw_crossfit_scores.csv")
GROUND_TRUTH_PATH = Path("artifacts/ground_truth_evaluation.json")
SENSITIVITY_PATH = Path("artifacts/sensitivity_analysis_summary.json")
SEGMENT_EFFECTS_PATH = Path("artifacts/segment_effects.csv")
SEGMENT_SUMMARY_PATH = Path("artifacts/segment_effects_summary.json")
FINAL_EVALUATION_PATH = Path("artifacts/final_evaluation.json")
MODEL_METADATA_PATH = Path("artifacts/model_metadata.json")
REPORT_PATH = Path("outputs/causal_report.md")


def load_json(path: Path) -> dict:
    assert path.exists(), f"Expected artifact does not exist: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return load_data()


@pytest.fixture(scope="module")
def policy() -> pd.DataFrame:
    return pd.read_csv(VARIABLE_POLICY_PATH)


def test_project_config_is_project_11_and_synthetic():
    config = load_json(PROJECT_CONFIG_PATH)

    assert config["project_number"] == 11
    assert config["synthetic_data_only"] is True
    assert config["treatment"] == "training_assigned"
    assert config["outcome"] == "retained_or_promoted_180d"


def test_propensity_feature_list_matches_pre_treatment_policy(policy):
    expected = set(
        policy.loc[
            policy["policy"] == "PRE_TREATMENT_COVARIATE",
            "column",
        ]
    )

    actual = set(
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    assert actual == expected


def test_protected_columns_are_not_propensity_features():
    actual = set(
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    forbidden = {
        "person_id",
        "training_assigned",
        "retained_or_promoted_180d",
        "post_training_completion",
        "true_propensity",
        "true_individual_treatment_effect",
    }

    assert actual.isdisjoint(forbidden)


def test_post_treatment_and_truth_policies_are_locked(policy):
    mapping = policy.set_index("column")["policy"].to_dict()

    assert mapping["post_training_completion"] == "EXCLUDED_POST_TREATMENT"
    assert mapping["true_propensity"] == "EVALUATION_ONLY"
    assert mapping["true_individual_treatment_effect"] == "EVALUATION_ONLY"
    assert mapping["person_id"] == "EXCLUDED_IDENTIFIER"


def test_approved_covariates_have_no_missing_values(data):
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    assert not data[features].isna().any().any()


def test_baseline_artifact_matches_raw_data(data):
    baseline = load_json(BASELINE_PATH)

    control = data.loc[
        data["training_assigned"] == 0,
        "retained_or_promoted_180d",
    ]
    treated = data.loc[
        data["training_assigned"] == 1,
        "retained_or_promoted_180d",
    ]

    expected = float(treated.mean() - control.mean())

    assert baseline["control_count"] == len(control)
    assert baseline["treated_count"] == len(treated)
    assert baseline["naive_difference"] == pytest.approx(
        expected,
        abs=1e-12,
    )
    assert baseline["causal_interpretation_allowed"] is False


def test_baseline_confidence_interval_is_ordered():
    baseline = load_json(BASELINE_PATH)
    ci = baseline["confidence_interval"]

    assert ci["lower"] < baseline["naive_difference"] < ci["upper"]
    assert baseline["standard_error"] > 0


def test_propensity_scores_align_one_to_one_with_data(data):
    scores = pd.read_csv(PROPENSITY_SCORES_PATH)

    assert len(scores) == len(data)
    assert scores["person_id"].is_unique
    assert set(scores["person_id"]) == set(data["person_id"])


def test_propensity_scores_are_strict_probabilities():
    scores = pd.read_csv(PROPENSITY_SCORES_PATH)
    p = scores["propensity_score"].to_numpy(dtype=float)

    assert np.isfinite(p).all()
    assert (p > 0).all()
    assert (p < 1).all()


def test_propensity_overlap_is_healthy_for_locked_dataset():
    diagnostics = load_json(PROPENSITY_DIAGNOSTICS_PATH)

    assert diagnostics["empirical_overlap_region"]["valid"] is True
    assert diagnostics["extreme_propensity_counts"]["below_0_10"] == 0
    assert diagnostics["extreme_propensity_counts"]["above_0_90"] == 0


def test_ipw_weights_align_and_are_positive(data):
    weights = pd.read_csv(WEIGHTS_PATH)

    assert len(weights) == len(data)
    assert weights["person_id"].is_unique

    for column in [
        "ate_ipw",
        "ate_stabilized_ipw",
    ]:
        values = weights[column].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert (values > 0).all()


def test_stabilized_weights_are_centered_near_one():
    weights = pd.read_csv(WEIGHTS_PATH)
    mean_weight = float(
        weights["ate_stabilized_ipw"].mean()
    )

    assert 0.95 <= mean_weight <= 1.05


def test_weight_diagnostics_preserve_effective_sample_size():
    diagnostics = load_json(WEIGHT_DIAGNOSTICS_PATH)
    ess = diagnostics["effective_sample_size"]

    assert ess["weighted_total_ess_fraction"] >= 0.95
    assert diagnostics["trimming_applied"] is False
    assert diagnostics["clipping_applied"] is False
    assert diagnostics["overlap_restriction_applied"] is False


def test_balance_improves_after_weighting():
    summary = load_json(BALANCE_SUMMARY_PATH)

    assert summary["pre_weighting"]["rows_at_or_above_threshold"] >= 1
    assert summary["post_weighting"]["rows_at_or_above_threshold"] == 0
    assert (
        summary["post_weighting"]["maximum_abs_smd"]
        < summary["pre_weighting"]["maximum_abs_smd"]
    )


def test_all_post_weighting_covariates_are_below_threshold():
    balance = pd.read_csv(BALANCE_PATH)

    assert (
        balance["abs_weighted_smd"] < 0.10
    ).all()


def test_ipw_artifact_matches_direct_weighted_outcome_calculation(data):
    weights = pd.read_csv(WEIGHTS_PATH)
    summary = load_json(IPW_PATH)

    aligned = data[
        [
            "person_id",
            "training_assigned",
            "retained_or_promoted_180d",
        ]
    ].merge(
        weights[
            [
                "person_id",
                "ate_stabilized_ipw",
            ]
        ],
        on="person_id",
        validate="one_to_one",
    )

    treated = aligned["training_assigned"] == 1
    control = ~treated

    treated_mean = np.average(
        aligned.loc[
            treated,
            "retained_or_promoted_180d",
        ],
        weights=aligned.loc[
            treated,
            "ate_stabilized_ipw",
        ],
    )

    control_mean = np.average(
        aligned.loc[
            control,
            "retained_or_promoted_180d",
        ],
        weights=aligned.loc[
            control,
            "ate_stabilized_ipw",
        ],
    )

    ate = float(treated_mean - control_mean)

    assert summary["ipw_ate"] == pytest.approx(
        ate,
        abs=1e-12,
    )


def test_ipw_uncertainty_is_present_and_ordered():
    summary = load_json(IPW_PATH)
    uncertainty = summary["uncertainty"]
    ci = uncertainty["confidence_interval"]

    assert summary["uncertainty_status"] == "estimated"
    assert uncertainty["bootstrap_resamples"] == 1000
    assert ci["lower"] < summary["ipw_ate"] < ci["upper"]


def test_aipw_and_ipw_are_close():
    ipw = load_json(IPW_PATH)
    aipw = load_json(AIPW_PATH)

    assert abs(
        aipw["aipw_ate"]
        - ipw["ipw_ate"]
    ) < 0.01


def test_crossfit_and_non_crossfit_aipw_are_close():
    aipw = load_json(AIPW_PATH)
    crossfit = load_json(CROSSFIT_PATH)

    assert crossfit["cross_fitted"] is True
    assert crossfit["n_splits"] == 5
    assert abs(
        crossfit["crossfit_aipw_ate"]
        - aipw["aipw_ate"]
    ) < 0.01


def test_crossfit_scores_cover_all_rows_and_folds(data):
    scores = pd.read_csv(CROSSFIT_SCORES_PATH)

    assert len(scores) == len(data)
    assert scores["person_id"].is_unique
    assert set(scores["crossfit_fold"]) == {1, 2, 3, 4, 5}

    required = [
        "propensity_score_oof",
        "mu0_oof",
        "mu1_oof",
        "aipw_pseudo_outcome_crossfit",
    ]

    assert not scores[required].isna().any().any()


def test_ground_truth_remains_evaluation_only():
    truth = load_json(GROUND_TRUTH_PATH)

    assert truth["evaluation_only"] is True
    assert truth["ground_truth_used_for_training"] is False
    assert truth["ground_truth_used_for_estimation"] is False


def test_adjusted_confidence_intervals_cover_synthetic_true_ate():
    truth = load_json(GROUND_TRUTH_PATH)

    assert (
        truth["ate_recovery"][
            "all_adjusted_confidence_intervals_cover_true_ate"
        ]
        is True
    )


def test_sensitivity_analysis_is_tight_and_directionally_stable():
    sensitivity = load_json(SENSITIVITY_PATH)

    assert sensitivity["all_estimates_positive"] is True
    assert (
        sensitivity[
            "overall_estimate_range_percentage_points"
        ]
        <= 1.0
    )


def test_subgroup_analysis_respects_minimum_size():
    effects = pd.read_csv(SEGMENT_EFFECTS_PATH)

    assert (effects["n"] >= 100).all()
    assert not effects["small_segment_flag"].any()


def test_subgroup_truth_is_evaluation_only():
    summary = load_json(SEGMENT_SUMMARY_PATH)

    assert summary["segment_definitions_use_ground_truth"] is False
    assert summary["true_effect_used_for_estimation"] is False
    assert summary["true_effect_used_for_evaluation_only"] is True


def test_locked_methodology_rejects_individual_targeting():
    methodology = load_json(METHODOLOGY_PATH)

    hte = methodology["heterogeneous_effects"]

    assert hte["individual_cate_status"] == "rejected_for_final_targeting"
    assert hte["uplift_qini_auuc_status"] == "not_reported_as_success_metric"
    assert hte["subgroup_hte_status"] == "retained"


def test_locked_methodology_records_rejected_cate_methods():
    methodology = load_json(METHODOLOGY_PATH)

    rejected = {
        item["method"]
        for item in methodology["rejected_methods"]
    }

    assert {
        "random_forest_dr_learner",
        "regularized_dr_learner",
        "interaction_s_learner",
    }.issubset(rejected)


def test_final_evaluation_matches_locked_methodology():
    methodology = load_json(METHODOLOGY_PATH)
    final = load_json(FINAL_EVALUATION_PATH)

    assert final["methodology_locked"] is True
    assert (
        final["primary_estimator"]
        == methodology["primary_estimator"]
    )
    assert final["primary_result"]["estimate"] == pytest.approx(
        methodology["primary_effect_summary"]["estimate"],
        abs=1e-6,
    )
    assert final["ai_used_for_calculation"] is False


def test_model_metadata_integrity_check_is_clean():
    metadata = load_json(MODEL_METADATA_PATH)

    integrity = metadata["file_integrity"]

    assert integrity["missing_key_source_files"] == []
    assert integrity["missing_key_artifacts"] == []


def test_deterministic_report_exists_and_documents_ai_boundary():
    assert REPORT_PATH.exists()

    report = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "No AI system calculated or modified treatment effects" in report
    assert "Individual-level CATE/uplift targeting was rejected after multiple validation failures" in report
    assert "Deterministic causal analysis complete" in report
