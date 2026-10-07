from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


METHODOLOGY_PATH = Path("config/final_methodology.json")

NAIVE_PATH = Path("artifacts/naive_baseline.json")
PROPENSITY_PATH = Path("artifacts/propensity_diagnostics.json")
WEIGHTS_PATH = Path("artifacts/weight_diagnostics.json")
BALANCE_PATH = Path("artifacts/balance_summary.json")
IPW_PATH = Path("artifacts/ate_summary.json")
AIPW_PATH = Path("artifacts/aipw_summary.json")
CROSSFIT_PATH = Path("artifacts/aipw_crossfit_summary.json")
GROUND_TRUTH_PATH = Path("artifacts/ground_truth_evaluation.json")
FAILURE_PATH = Path("artifacts/failure_analysis_summary.json")
SEGMENT_PATH = Path("artifacts/segment_effects_summary.json")
SENSITIVITY_PATH = Path("artifacts/sensitivity_analysis_summary.json")
CATE_RF_PATH = Path("artifacts/cate_evaluation_summary.json")
CATE_RIDGE_PATH = Path("artifacts/cate_regularized_validation_summary.json")
CATE_SLEARNER_PATH = Path("artifacts/cate_slearner_validation_summary.json")

FINAL_JSON_PATH = Path("artifacts/final_evaluation.json")
FINAL_CSV_PATH = Path("artifacts/final_evaluation_metrics.csv")


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label} artifact: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_locked_methodology(
    methodology: dict,
    crossfit: dict,
) -> None:
    if methodology.get("status") != "methodology_locked":
        raise ValueError(
            "Final methodology is not marked as locked."
        )

    if methodology.get("primary_estimand") != "ATE":
        raise ValueError(
            "Locked primary estimand must be ATE."
        )

    expected_estimator = (
        "cross_fitted_augmented_inverse_probability_weighting"
    )

    if methodology.get("primary_estimator") != expected_estimator:
        raise ValueError(
            "Locked primary estimator does not match the expected "
            "cross-fitted AIPW estimator."
        )

    locked_estimate = float(
        methodology["primary_effect_summary"]["estimate"]
    )

    actual_estimate = float(
        crossfit["crossfit_aipw_ate"]
    )

    if abs(locked_estimate - actual_estimate) > 1e-6:
        raise ValueError(
            "Locked primary estimate does not match the current "
            "cross-fitted AIPW artifact."
        )


def build_final_evaluation(
    methodology: dict,
    naive: dict,
    propensity: dict,
    weights: dict,
    balance: dict,
    ipw: dict,
    aipw: dict,
    crossfit: dict,
    ground_truth: dict,
    failure: dict,
    segments: dict,
    sensitivity: dict,
    cate_rf: dict | None,
    cate_ridge: dict | None,
    cate_slearner: dict | None,
) -> dict:
    primary_ci = crossfit[
        "confidence_interval_percentage_points"
    ]

    result = {
        "project": methodology["project"],
        "evaluation_status": "complete",
        "methodology_locked": True,
        "primary_estimand": "ATE",
        "primary_estimator": methodology[
            "primary_estimator"
        ],
        "primary_result": {
            "estimate": float(
                crossfit["crossfit_aipw_ate"]
            ),
            "estimate_percentage_points": float(
                crossfit[
                    "crossfit_aipw_ate_percentage_points"
                ]
            ),
            "standard_error": float(
                crossfit["standard_error"]
            ),
            "standard_error_percentage_points": float(
                crossfit[
                    "standard_error_percentage_points"
                ]
            ),
            "confidence_level": float(
                crossfit["confidence_level"]
            ),
            "confidence_interval_percentage_points": {
                "lower": float(
                    primary_ci["lower"]
                ),
                "upper": float(
                    primary_ci["upper"]
                ),
            },
        },
        "reference_estimators": {
            "naive_difference_percentage_points": float(
                naive[
                    "naive_difference_percentage_points"
                ]
            ),
            "ipw_ate_percentage_points": float(
                ipw[
                    "ipw_ate_percentage_points"
                ]
            ),
            "aipw_ate_percentage_points": float(
                aipw[
                    "aipw_ate_percentage_points"
                ]
            ),
            "crossfit_aipw_ate_percentage_points": float(
                crossfit[
                    "crossfit_aipw_ate_percentage_points"
                ]
            ),
        },
        "diagnostics": {
            "propensity_overlap_valid": bool(
                propensity[
                    "empirical_overlap_region"
                ]["valid"]
            ),
            "propensity_minimum": float(
                propensity["overall"]["minimum"]
            ),
            "propensity_maximum": float(
                propensity["overall"]["maximum"]
            ),
            "poor_overlap_rows": int(
                failure["overlap"][
                    "poor_overlap_count"
                ]
            ),
            "extreme_weight_count": int(
                failure["weights"][
                    "extreme_weight_count"
                ]
            ),
            "effective_sample_size_fraction": float(
                weights[
                    "effective_sample_size"
                ]["weighted_total_ess_fraction"]
            ),
            "post_weighting_rows_at_or_above_abs_smd_0_10": int(
                balance[
                    "post_weighting"
                ]["rows_at_or_above_threshold"]
            ),
            "maximum_post_weighting_abs_smd": float(
                balance[
                    "post_weighting"
                ]["maximum_abs_smd"]
            ),
        },
        "synthetic_ground_truth_evaluation": {
            "evaluation_only": True,
            "used_for_training": False,
            "used_for_estimation": False,
            "true_ate_percentage_points": float(
                ground_truth[
                    "true_effect_summary"
                ]["true_ate_percentage_points"]
            ),
            "primary_estimator_signed_error_percentage_points": float(
                crossfit[
                    "crossfit_aipw_ate_percentage_points"
                ]
                - ground_truth[
                    "true_effect_summary"
                ]["true_ate_percentage_points"]
            ),
            "all_adjusted_confidence_intervals_cover_true_ate": bool(
                ground_truth[
                    "ate_recovery"
                ][
                    "all_adjusted_confidence_intervals_cover_true_ate"
                ]
            ),
        },
        "sensitivity": {
            "minimum_estimate_percentage_points": float(
                sensitivity[
                    "minimum_estimate_percentage_points"
                ]
            ),
            "maximum_estimate_percentage_points": float(
                sensitivity[
                    "maximum_estimate_percentage_points"
                ]
            ),
            "range_width_percentage_points": float(
                sensitivity[
                    "overall_estimate_range_percentage_points"
                ]
            ),
            "largest_absolute_propensity_shift_percentage_points": float(
                sensitivity[
                    "largest_absolute_propensity_sensitivity_shift_percentage_points"
                ]
            ),
            "all_estimates_positive": bool(
                sensitivity[
                    "all_estimates_positive"
                ]
            ),
        },
        "heterogeneous_effects": {
            "individual_cate_status": methodology[
                "heterogeneous_effects"
            ]["individual_cate_status"],
            "uplift_qini_auuc_status": methodology[
                "heterogeneous_effects"
            ]["uplift_qini_auuc_status"],
            "subgroup_hte_status": methodology[
                "heterogeneous_effects"
            ]["subgroup_hte_status"],
            "subgroup_rows": int(
                segments[
                    "n_segment_rows"
                ]
            ),
            "subgroup_mean_absolute_error_percentage_points": float(
                segments[
                    "overall_mean_absolute_error_percentage_points"
                ]
            ),
            "all_subgroup_cis_cover_true_effect": bool(
                segments[
                    "all_segment_cis_cover_true_effect"
                ]
            ),
        },
        "cate_attempts": {
            "random_forest_dr_learner_available": (
                cate_rf is not None
            ),
            "regularized_dr_learner_available": (
                cate_ridge is not None
            ),
            "interaction_s_learner_available": (
                cate_slearner is not None
            ),
            "final_individual_targeting_accepted": False,
        },
        "final_interpretation": (
            "Under the stated identification assumptions, the locked "
            "cross-fitted AIPW analysis estimates that training assignment "
            "increased the probability of retention or promotion within "
            "180 days by approximately "
            f"{crossfit['crossfit_aipw_ate_percentage_points']:.2f} "
            "percentage points. Observed overlap, covariate balance, "
            "weight stability, estimator agreement, and sensitivity "
            "analysis support the adjustment design, but do not prove "
            "absence of unmeasured confounding. Individual-level CATE/uplift "
            "ranking was rejected after validation failures; predefined "
            "subgroup HTE analysis was retained."
        ),
        "ai_used_for_calculation": False,
    }

    return result


def build_metrics_table(
    final: dict,
) -> pd.DataFrame:
    rows = [
        {
            "metric": "Primary cross-fitted AIPW ATE",
            "value": final[
                "primary_result"
            ]["estimate_percentage_points"],
            "unit": "percentage_points",
        },
        {
            "metric": "Primary CI lower",
            "value": final[
                "primary_result"
            ][
                "confidence_interval_percentage_points"
            ]["lower"],
            "unit": "percentage_points",
        },
        {
            "metric": "Primary CI upper",
            "value": final[
                "primary_result"
            ][
                "confidence_interval_percentage_points"
            ]["upper"],
            "unit": "percentage_points",
        },
        {
            "metric": "Naive difference",
            "value": final[
                "reference_estimators"
            ]["naive_difference_percentage_points"],
            "unit": "percentage_points",
        },
        {
            "metric": "IPW ATE",
            "value": final[
                "reference_estimators"
            ]["ipw_ate_percentage_points"],
            "unit": "percentage_points",
        },
        {
            "metric": "AIPW ATE",
            "value": final[
                "reference_estimators"
            ]["aipw_ate_percentage_points"],
            "unit": "percentage_points",
        },
        {
            "metric": "Synthetic true ATE",
            "value": final[
                "synthetic_ground_truth_evaluation"
            ]["true_ate_percentage_points"],
            "unit": "percentage_points",
        },
        {
            "metric": "Primary estimator error vs synthetic truth",
            "value": final[
                "synthetic_ground_truth_evaluation"
            ][
                "primary_estimator_signed_error_percentage_points"
            ],
            "unit": "percentage_points",
        },
        {
            "metric": "Effective sample size retained",
            "value": (
                final[
                    "diagnostics"
                ]["effective_sample_size_fraction"]
                * 100.0
            ),
            "unit": "percent",
        },
        {
            "metric": "Maximum post-weighting |SMD|",
            "value": final[
                "diagnostics"
            ]["maximum_post_weighting_abs_smd"],
            "unit": "smd",
        },
        {
            "metric": "Sensitivity range width",
            "value": final[
                "sensitivity"
            ]["range_width_percentage_points"],
            "unit": "percentage_points",
        },
        {
            "metric": "Subgroup mean absolute error",
            "value": final[
                "heterogeneous_effects"
            ][
                "subgroup_mean_absolute_error_percentage_points"
            ],
            "unit": "percentage_points",
        },
    ]

    return pd.DataFrame(rows)


def save_outputs(
    final: dict,
    metrics: pd.DataFrame,
) -> None:
    FINAL_JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FINAL_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    metrics.to_csv(
        FINAL_CSV_PATH,
        index=False,
    )


def print_summary(
    final: dict,
) -> None:
    primary = final["primary_result"]
    diagnostics = final["diagnostics"]
    sensitivity = final["sensitivity"]
    hte = final["heterogeneous_effects"]

    print("Final Deterministic Evaluation")
    print("------------------------------")

    print(
        f"Primary estimator: "
        f"{final['primary_estimator']}"
    )

    print(
        "Primary ATE: "
        f"{primary['estimate_percentage_points']:+.2f} "
        "percentage points"
    )

    print(
        f"{int(primary['confidence_level'] * 100)}% CI: "
        f"[{primary['confidence_interval_percentage_points']['lower']:+.2f}, "
        f"{primary['confidence_interval_percentage_points']['upper']:+.2f}] "
        "percentage points"
    )

    print("\nObserved-data diagnostics")
    print(
        "Poor-overlap rows: "
        f"{diagnostics['poor_overlap_rows']}"
    )
    print(
        "Extreme weights: "
        f"{diagnostics['extreme_weight_count']}"
    )
    print(
        "ESS retained: "
        f"{diagnostics['effective_sample_size_fraction']:.1%}"
    )
    print(
        "Post-weighting rows at/above |SMD| 0.10: "
        f"{diagnostics['post_weighting_rows_at_or_above_abs_smd_0_10']}"
    )
    print(
        "Maximum post-weighting |SMD|: "
        f"{diagnostics['maximum_post_weighting_abs_smd']:.6f}"
    )

    print("\nSensitivity")
    print(
        "Estimate range: "
        f"{sensitivity['minimum_estimate_percentage_points']:+.2f} "
        "to "
        f"{sensitivity['maximum_estimate_percentage_points']:+.2f} "
        "percentage points"
    )
    print(
        "Range width: "
        f"{sensitivity['range_width_percentage_points']:.2f} "
        "percentage points"
    )

    print("\nHeterogeneous effects")
    print(
        "Individual CATE: "
        f"{hte['individual_cate_status']}"
    )
    print(
        "Subgroup HTE: "
        f"{hte['subgroup_hte_status']}"
    )
    print(
        "Subgroup mean absolute error: "
        f"{hte['subgroup_mean_absolute_error_percentage_points']:.2f} "
        "percentage points"
    )

    print("\nArtifacts saved:")
    print(f"- {FINAL_JSON_PATH}")
    print(f"- {FINAL_CSV_PATH}")

    print(
        "\nNo AI was used to calculate or modify causal estimates."
    )


def main() -> None:
    methodology = load_json(
        METHODOLOGY_PATH,
        "locked methodology",
    )

    naive = load_json(
        NAIVE_PATH,
        "naive baseline",
    )

    propensity = load_json(
        PROPENSITY_PATH,
        "propensity diagnostics",
    )

    weights = load_json(
        WEIGHTS_PATH,
        "weight diagnostics",
    )

    balance = load_json(
        BALANCE_PATH,
        "balance summary",
    )

    ipw = load_json(
        IPW_PATH,
        "IPW ATE",
    )

    aipw = load_json(
        AIPW_PATH,
        "AIPW",
    )

    crossfit = load_json(
        CROSSFIT_PATH,
        "cross-fitted AIPW",
    )

    ground_truth = load_json(
        GROUND_TRUTH_PATH,
        "ground-truth evaluation",
    )

    failure = load_json(
        FAILURE_PATH,
        "failure analysis",
    )

    segments = load_json(
        SEGMENT_PATH,
        "segment effects",
    )

    sensitivity = load_json(
        SENSITIVITY_PATH,
        "sensitivity analysis",
    )

    cate_rf = load_optional_json(
        CATE_RF_PATH
    )

    cate_ridge = load_optional_json(
        CATE_RIDGE_PATH
    )

    cate_slearner = load_optional_json(
        CATE_SLEARNER_PATH
    )

    validate_locked_methodology(
        methodology=methodology,
        crossfit=crossfit,
    )

    final = build_final_evaluation(
        methodology=methodology,
        naive=naive,
        propensity=propensity,
        weights=weights,
        balance=balance,
        ipw=ipw,
        aipw=aipw,
        crossfit=crossfit,
        ground_truth=ground_truth,
        failure=failure,
        segments=segments,
        sensitivity=sensitivity,
        cate_rf=cate_rf,
        cate_ridge=cate_ridge,
        cate_slearner=cate_slearner,
    )

    metrics = build_metrics_table(
        final=final
    )

    save_outputs(
        final=final,
        metrics=metrics,
    )

    print_summary(
        final=final
    )


if __name__ == "__main__":
    main()
