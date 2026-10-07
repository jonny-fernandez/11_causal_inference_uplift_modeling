from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_contract import load_data


ID_COLUMN = "person_id"
TREATMENT_COLUMN = "training_assigned"
TRUE_PROPENSITY_COLUMN = "true_propensity"
TRUE_ITE_COLUMN = "true_individual_treatment_effect"

PROPENSITY_PATH = Path("artifacts/propensity_scores.csv")
WEIGHTS_PATH = Path("artifacts/ipw_weights.csv")
BALANCE_PATH = Path("artifacts/balance_diagnostics.csv")
GROUND_TRUTH_COMPARISON_PATH = Path(
    "artifacts/ate_ground_truth_comparison.csv"
)
CROSSFIT_SCORES_PATH = Path(
    "artifacts/aipw_crossfit_scores.csv"
)

FAILURE_ROWS_PATH = Path(
    "artifacts/failure_analysis_rows.csv"
)
FAILURE_SUMMARY_PATH = Path(
    "artifacts/failure_analysis_summary.json"
)

PROPENSITY_COLUMN = "propensity_score"
WEIGHT_COLUMN = "ate_stabilized_ipw"

POOR_OVERLAP_LOWER = 0.10
POOR_OVERLAP_UPPER = 0.90
EXTREME_WEIGHT_THRESHOLD = 2.00
RESIDUAL_BALANCE_THRESHOLD = 0.10
LARGE_PROPENSITY_ERROR_THRESHOLD = 0.05


def require_file(
    path: Path,
    label: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label} artifact: {path}"
        )


def load_artifacts() -> dict[str, pd.DataFrame]:
    require_file(
        PROPENSITY_PATH,
        "propensity score",
    )
    require_file(
        WEIGHTS_PATH,
        "weight",
    )
    require_file(
        BALANCE_PATH,
        "balance diagnostic",
    )
    require_file(
        GROUND_TRUTH_COMPARISON_PATH,
        "ground-truth comparison",
    )
    require_file(
        CROSSFIT_SCORES_PATH,
        "cross-fitted AIPW score",
    )

    return {
        "propensity": pd.read_csv(
            PROPENSITY_PATH
        ),
        "weights": pd.read_csv(
            WEIGHTS_PATH
        ),
        "balance": pd.read_csv(
            BALANCE_PATH
        ),
        "ground_truth_comparison": pd.read_csv(
            GROUND_TRUTH_COMPARISON_PATH
        ),
        "crossfit": pd.read_csv(
            CROSSFIT_SCORES_PATH
        ),
    }


def validate_source_data(
    data: pd.DataFrame,
) -> None:
    required = {
        ID_COLUMN,
        TREATMENT_COLUMN,
        TRUE_PROPENSITY_COLUMN,
        TRUE_ITE_COLUMN,
    }

    missing = sorted(
        required.difference(data.columns)
    )

    if missing:
        raise ValueError(
            "Source data is missing failure-analysis columns: "
            + ", ".join(missing)
        )

    if data[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Source data contains duplicate person IDs."
        )


def build_row_level_analysis(
    data: pd.DataFrame,
    propensity: pd.DataFrame,
    weights: pd.DataFrame,
    crossfit: pd.DataFrame,
) -> pd.DataFrame:
    required_propensity = {
        ID_COLUMN,
        PROPENSITY_COLUMN,
    }

    missing = sorted(
        required_propensity.difference(
            propensity.columns
        )
    )

    if missing:
        raise ValueError(
            "Propensity artifact missing columns: "
            + ", ".join(missing)
        )

    required_weights = {
        ID_COLUMN,
        WEIGHT_COLUMN,
    }

    missing = sorted(
        required_weights.difference(
            weights.columns
        )
    )

    if missing:
        raise ValueError(
            "Weight artifact missing columns: "
            + ", ".join(missing)
        )

    required_crossfit = {
        ID_COLUMN,
        "propensity_score_oof",
        "mu0_oof",
        "mu1_oof",
        "aipw_pseudo_outcome_crossfit",
    }

    missing = sorted(
        required_crossfit.difference(
            crossfit.columns
        )
    )

    if missing:
        raise ValueError(
            "Cross-fitted AIPW artifact missing columns: "
            + ", ".join(missing)
        )

    merged = (
        data[
            [
                ID_COLUMN,
                TREATMENT_COLUMN,
                TRUE_PROPENSITY_COLUMN,
                TRUE_ITE_COLUMN,
            ]
        ]
        .merge(
            propensity[
                [
                    ID_COLUMN,
                    PROPENSITY_COLUMN,
                ]
            ],
            on=ID_COLUMN,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            weights[
                [
                    ID_COLUMN,
                    WEIGHT_COLUMN,
                ]
            ],
            on=ID_COLUMN,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            crossfit[
                [
                    ID_COLUMN,
                    "propensity_score_oof",
                    "mu0_oof",
                    "mu1_oof",
                    "aipw_pseudo_outcome_crossfit",
                ]
            ],
            on=ID_COLUMN,
            how="inner",
            validate="one_to_one",
        )
    )

    if len(merged) != len(data):
        raise ValueError(
            "Failure-analysis artifacts do not align "
            "one-to-one with the source dataset."
        )

    merged["propensity_error"] = (
        merged[PROPENSITY_COLUMN]
        - merged[TRUE_PROPENSITY_COLUMN]
    )

    merged["abs_propensity_error"] = (
        merged["propensity_error"].abs()
    )

    merged["crossfit_cate_proxy"] = (
        merged["mu1_oof"]
        - merged["mu0_oof"]
    )

    merged["cate_proxy_error"] = (
        merged["crossfit_cate_proxy"]
        - merged[TRUE_ITE_COLUMN]
    )

    merged["abs_cate_proxy_error"] = (
        merged["cate_proxy_error"].abs()
    )

    merged["poor_overlap_flag"] = (
        (merged[PROPENSITY_COLUMN] < POOR_OVERLAP_LOWER)
        | (merged[PROPENSITY_COLUMN] > POOR_OVERLAP_UPPER)
    )

    merged["extreme_weight_flag"] = (
        merged[WEIGHT_COLUMN]
        > EXTREME_WEIGHT_THRESHOLD
    )

    merged["large_propensity_error_flag"] = (
        merged["abs_propensity_error"]
        >= LARGE_PROPENSITY_ERROR_THRESHOLD
    )

    return merged


def summarize_overlap(
    rows: pd.DataFrame,
) -> dict:
    poor_overlap = rows[
        "poor_overlap_flag"
    ]

    return {
        "lower_threshold":
            POOR_OVERLAP_LOWER,
        "upper_threshold":
            POOR_OVERLAP_UPPER,
        "poor_overlap_count":
            int(poor_overlap.sum()),
        "poor_overlap_fraction":
            float(poor_overlap.mean()),
        "minimum_propensity":
            float(
                rows[PROPENSITY_COLUMN].min()
            ),
        "maximum_propensity":
            float(
                rows[PROPENSITY_COLUMN].max()
            ),
        "minimum_oof_propensity":
            float(
                rows[
                    "propensity_score_oof"
                ].min()
            ),
        "maximum_oof_propensity":
            float(
                rows[
                    "propensity_score_oof"
                ].max()
            ),
    }


def summarize_weights(
    rows: pd.DataFrame,
) -> dict:
    weights = rows[
        WEIGHT_COLUMN
    ].astype(float)

    return {
        "extreme_weight_threshold":
            EXTREME_WEIGHT_THRESHOLD,
        "extreme_weight_count":
            int(
                rows[
                    "extreme_weight_flag"
                ].sum()
            ),
        "extreme_weight_fraction":
            float(
                rows[
                    "extreme_weight_flag"
                ].mean()
            ),
        "minimum":
            float(weights.min()),
        "median":
            float(weights.median()),
        "p95":
            float(weights.quantile(0.95)),
        "p99":
            float(weights.quantile(0.99)),
        "maximum":
            float(weights.max()),
    }


def summarize_balance(
    balance: pd.DataFrame,
) -> dict:
    required = {
        "covariate",
        "abs_unweighted_smd",
        "abs_weighted_smd",
    }

    missing = sorted(
        required.difference(
            balance.columns
        )
    )

    if missing:
        raise ValueError(
            "Balance artifact missing columns: "
            + ", ".join(missing)
        )

    residual = (
        balance[
            "abs_weighted_smd"
        ]
        >= RESIDUAL_BALANCE_THRESHOLD
    )

    worst_row = balance.loc[
        balance[
            "abs_weighted_smd"
        ].idxmax()
    ]

    return {
        "residual_balance_threshold":
            RESIDUAL_BALANCE_THRESHOLD,
        "residual_imbalance_count":
            int(residual.sum()),
        "maximum_abs_weighted_smd":
            float(
                balance[
                    "abs_weighted_smd"
                ].max()
            ),
        "mean_abs_weighted_smd":
            float(
                balance[
                    "abs_weighted_smd"
                ].mean()
            ),
        "worst_weighted_covariate":
            str(
                worst_row["covariate"]
            ),
    }


def summarize_ground_truth_recovery(
    comparison: pd.DataFrame,
) -> dict:
    required = {
        "estimator",
        "causal_adjustment",
        "estimate_percentage_points",
        "absolute_error_percentage_points",
        "ci_covers_true_ate",
    }

    missing = sorted(
        required.difference(
            comparison.columns
        )
    )

    if missing:
        raise ValueError(
            "Ground-truth comparison artifact missing columns: "
            + ", ".join(missing)
        )

    adjusted = comparison.loc[
        comparison[
            "causal_adjustment"
        ].astype(bool)
    ].copy()

    best = adjusted.loc[
        adjusted[
            "absolute_error_percentage_points"
        ].idxmin()
    ]

    worst = adjusted.loc[
        adjusted[
            "absolute_error_percentage_points"
        ].idxmax()
    ]

    return {
        "adjusted_estimator_count":
            int(len(adjusted)),
        "best_adjusted_estimator":
            str(
                best["estimator"]
            ),
        "best_adjusted_absolute_error_pp":
            float(
                best[
                    "absolute_error_percentage_points"
                ]
            ),
        "worst_adjusted_estimator":
            str(
                worst["estimator"]
            ),
        "worst_adjusted_absolute_error_pp":
            float(
                worst[
                    "absolute_error_percentage_points"
                ]
            ),
        "all_adjusted_cis_cover_true_ate":
            bool(
                adjusted[
                    "ci_covers_true_ate"
                ].dropna().all()
            ),
    }


def summarize_propensity_error(
    rows: pd.DataFrame,
) -> dict:
    return {
        "large_error_threshold":
            LARGE_PROPENSITY_ERROR_THRESHOLD,
        "large_error_count":
            int(
                rows[
                    "large_propensity_error_flag"
                ].sum()
            ),
        "large_error_fraction":
            float(
                rows[
                    "large_propensity_error_flag"
                ].mean()
            ),
        "mae":
            float(
                rows[
                    "abs_propensity_error"
                ].mean()
            ),
        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        rows[
                            "propensity_error"
                        ].to_numpy(
                            dtype=float
                        ) ** 2
                    )
                )
            ),
        "maximum_absolute_error":
            float(
                rows[
                    "abs_propensity_error"
                ].max()
            ),
    }


def summarize_heterogeneity_signal(
    rows: pd.DataFrame,
) -> dict:
    true_ite = rows[
        TRUE_ITE_COLUMN
    ].astype(float)

    cate_proxy = rows[
        "crossfit_cate_proxy"
    ].astype(float)

    correlation = float(
        np.corrcoef(
            true_ite,
            cate_proxy,
        )[0, 1]
    )

    return {
        "true_ite_minimum":
            float(true_ite.min()),
        "true_ite_maximum":
            float(true_ite.max()),
        "true_ite_std":
            float(true_ite.std(ddof=1)),
        "crossfit_outcome_model_contrast_minimum":
            float(cate_proxy.min()),
        "crossfit_outcome_model_contrast_maximum":
            float(cate_proxy.max()),
        "crossfit_outcome_model_contrast_std":
            float(
                cate_proxy.std(ddof=1)
            ),
        "proxy_vs_true_ite_correlation":
            correlation,
        "proxy_mae_vs_true_ite":
            float(
                rows[
                    "abs_cate_proxy_error"
                ].mean()
            ),
        "note": (
            "This is a diagnostic only. The out-of-fold outcome-model "
            "contrast is not yet the selected CATE/uplift estimator."
        ),
    }


def build_failure_flags(
    overlap: dict,
    weights: dict,
    balance: dict,
    ground_truth: dict,
) -> dict:
    return {
        "poor_overlap_detected":
            bool(
                overlap[
                    "poor_overlap_count"
                ] > 0
            ),
        "extreme_weights_detected":
            bool(
                weights[
                    "extreme_weight_count"
                ] > 0
            ),
        "residual_imbalance_detected":
            bool(
                balance[
                    "residual_imbalance_count"
                ] > 0
            ),
        "ground_truth_ci_failure_detected":
            bool(
                not ground_truth[
                    "all_adjusted_cis_cover_true_ate"
                ]
            ),
    }


def save_outputs(
    rows: pd.DataFrame,
    summary: dict,
) -> None:
    FAILURE_ROWS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_output_columns = [
        ID_COLUMN,
        TREATMENT_COLUMN,
        TRUE_PROPENSITY_COLUMN,
        PROPENSITY_COLUMN,
        "propensity_score_oof",
        "propensity_error",
        "abs_propensity_error",
        WEIGHT_COLUMN,
        TRUE_ITE_COLUMN,
        "crossfit_cate_proxy",
        "cate_proxy_error",
        "abs_cate_proxy_error",
        "poor_overlap_flag",
        "extreme_weight_flag",
        "large_propensity_error_flag",
    ]

    rows[
        row_output_columns
    ].sort_values(
        by=[
            "poor_overlap_flag",
            "extreme_weight_flag",
            "large_propensity_error_flag",
            "abs_propensity_error",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).to_csv(
        FAILURE_ROWS_PATH,
        index=False,
    )

    with FAILURE_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def print_summary(
    summary: dict,
) -> None:
    overlap = summary["overlap"]
    weights = summary["weights"]
    balance = summary["balance"]
    propensity = summary[
        "propensity_recovery"
    ]
    ground_truth = summary[
        "ground_truth_recovery"
    ]
    heterogeneity = summary[
        "heterogeneity_signal"
    ]
    flags = summary["failure_flags"]

    print(
        "Causal Failure / Error Analysis"
    )
    print(
        "-------------------------------"
    )

    print(
        "\nOverlap"
    )
    print(
        "Poor-overlap rows: "
        f"{overlap['poor_overlap_count']:,}"
    )
    print(
        "Propensity range: "
        f"[{overlap['minimum_propensity']:.6f}, "
        f"{overlap['maximum_propensity']:.6f}]"
    )
    print(
        "OOF propensity range: "
        f"[{overlap['minimum_oof_propensity']:.6f}, "
        f"{overlap['maximum_oof_propensity']:.6f}]"
    )

    print(
        "\nWeights"
    )
    print(
        "Extreme stabilized weights (> "
        f"{weights['extreme_weight_threshold']:.2f}): "
        f"{weights['extreme_weight_count']:,}"
    )
    print(
        f"Maximum stabilized weight: "
        f"{weights['maximum']:.6f}"
    )

    print(
        "\nResidual balance"
    )
    print(
        "Rows at/above |SMD| 0.10 after weighting: "
        f"{balance['residual_imbalance_count']}"
    )
    print(
        "Maximum post-weighting |SMD|: "
        f"{balance['maximum_abs_weighted_smd']:.6f}"
    )
    print(
        "Worst post-weighting covariate: "
        f"{balance['worst_weighted_covariate']}"
    )

    print(
        "\nPropensity recovery"
    )
    print(
        f"MAE: {propensity['mae']:.6f}"
    )
    print(
        f"RMSE: {propensity['rmse']:.6f}"
    )
    print(
        "Rows with absolute propensity error >= "
        f"{propensity['large_error_threshold']:.2f}: "
        f"{propensity['large_error_count']:,}"
    )
    print(
        "Maximum absolute propensity error: "
        f"{propensity['maximum_absolute_error']:.6f}"
    )

    print(
        "\nATE ground-truth recovery"
    )
    print(
        "Best adjusted estimator: "
        f"{ground_truth['best_adjusted_estimator']}"
    )
    print(
        "Best adjusted absolute error: "
        f"{ground_truth['best_adjusted_absolute_error_pp']:.2f} "
        "percentage points"
    )
    print(
        "Worst adjusted estimator: "
        f"{ground_truth['worst_adjusted_estimator']}"
    )
    print(
        "Worst adjusted absolute error: "
        f"{ground_truth['worst_adjusted_absolute_error_pp']:.2f} "
        "percentage points"
    )
    print(
        "All adjusted CIs cover true ATE: "
        f"{ground_truth['all_adjusted_cis_cover_true_ate']}"
    )

    print(
        "\nEarly heterogeneity diagnostic"
    )
    print(
        "True ITE range: "
        f"[{heterogeneity['true_ite_minimum']:.6f}, "
        f"{heterogeneity['true_ite_maximum']:.6f}]"
    )
    print(
        "OOF outcome-model contrast vs true ITE correlation: "
        f"{heterogeneity['proxy_vs_true_ite_correlation']:.6f}"
    )
    print(
        "OOF outcome-model contrast MAE vs true ITE: "
        f"{heterogeneity['proxy_mae_vs_true_ite']:.6f}"
    )

    print(
        "\nFailure flags"
    )
    for key, value in flags.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nArtifacts saved:"
    )
    print(
        f"- {FAILURE_ROWS_PATH}"
    )
    print(
        f"- {FAILURE_SUMMARY_PATH}"
    )

    print(
        "\nThis checkpoint diagnoses current weaknesses. "
        "It does not retune the estimators to match synthetic truth."
    )


def main() -> None:
    data = load_data()

    validate_source_data(
        data=data
    )

    artifacts = load_artifacts()

    rows = build_row_level_analysis(
        data=data,
        propensity=artifacts[
            "propensity"
        ],
        weights=artifacts[
            "weights"
        ],
        crossfit=artifacts[
            "crossfit"
        ],
    )

    overlap = summarize_overlap(
        rows=rows
    )

    weights = summarize_weights(
        rows=rows
    )

    balance = summarize_balance(
        balance=artifacts[
            "balance"
        ]
    )

    ground_truth = summarize_ground_truth_recovery(
        comparison=artifacts[
            "ground_truth_comparison"
        ]
    )

    propensity = summarize_propensity_error(
        rows=rows
    )

    heterogeneity = summarize_heterogeneity_signal(
        rows=rows
    )

    flags = build_failure_flags(
        overlap=overlap,
        weights=weights,
        balance=balance,
        ground_truth=ground_truth,
    )

    summary = {
        "overlap": overlap,
        "weights": weights,
        "balance": balance,
        "propensity_recovery": propensity,
        "ground_truth_recovery": ground_truth,
        "heterogeneity_signal": heterogeneity,
        "failure_flags": flags,
        "policy": (
            "Failure analysis is diagnostic. Synthetic truth is used "
            "only to measure recovery and expose weaknesses, not to "
            "retroactively choose an estimator because it looks best."
        ),
    }

    save_outputs(
        rows=rows,
        summary=summary,
    )

    print_summary(
        summary=summary,
    )


if __name__ == "__main__":
    main()
