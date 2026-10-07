from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_contract import load_data


ID_COLUMN = "person_id"
TRUE_PROPENSITY_COLUMN = "true_propensity"
TRUE_ITE_COLUMN = "true_individual_treatment_effect"

PROPENSITY_PATH = Path("artifacts/propensity_scores.csv")
NAIVE_PATH = Path("artifacts/naive_baseline.json")
IPW_PATH = Path("artifacts/ate_summary.json")
AIPW_PATH = Path("artifacts/aipw_summary.json")
CROSSFIT_AIPW_PATH = Path("artifacts/aipw_crossfit_summary.json")

ATE_COMPARISON_PATH = Path("artifacts/ate_ground_truth_comparison.csv")
GROUND_TRUTH_SUMMARY_PATH = Path("artifacts/ground_truth_evaluation.json")


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label} artifact: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_ground_truth_columns(data: pd.DataFrame) -> None:
    required = {
        ID_COLUMN,
        TRUE_PROPENSITY_COLUMN,
        TRUE_ITE_COLUMN,
    }

    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError(
            "Synthetic ground-truth evaluation columns are missing: "
            + ", ".join(missing)
        )

    if data[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Source data contains duplicate person IDs."
        )

    if data[
        [TRUE_PROPENSITY_COLUMN, TRUE_ITE_COLUMN]
    ].isna().any().any():
        raise ValueError(
            "Synthetic ground-truth columns contain missing values."
        )

    if (
        (data[TRUE_PROPENSITY_COLUMN] <= 0.0)
        | (data[TRUE_PROPENSITY_COLUMN] >= 1.0)
    ).any():
        raise ValueError(
            "true_propensity must be strictly between 0 and 1."
        )


def calculate_true_effect_summary(
    data: pd.DataFrame,
) -> dict:
    true_ite = data[
        TRUE_ITE_COLUMN
    ].astype(float)

    return {
        "true_ate": float(true_ite.mean()),
        "true_ate_percentage_points": float(
            true_ite.mean() * 100.0
        ),
        "true_ite_minimum": float(true_ite.min()),
        "true_ite_maximum": float(true_ite.max()),
        "true_ite_mean": float(true_ite.mean()),
        "true_ite_std": float(true_ite.std(ddof=1)),
        "true_ite_p05": float(true_ite.quantile(0.05)),
        "true_ite_median": float(true_ite.median()),
        "true_ite_p95": float(true_ite.quantile(0.95)),
    }


def extract_estimates(
    naive: dict,
    ipw: dict,
    aipw: dict,
    crossfit: dict,
) -> list[dict]:
    return [
        {
            "estimator": "naive_difference_in_means",
            "estimate": float(
                naive["naive_difference"]
            ),
            "confidence_interval": (
                naive.get("confidence_interval")
            ),
            "causal_adjustment": False,
        },
        {
            "estimator": "stabilized_ipw",
            "estimate": float(
                ipw["ipw_ate"]
            ),
            "confidence_interval": (
                ipw.get(
                    "uncertainty",
                    {},
                ).get("confidence_interval")
            ),
            "causal_adjustment": True,
        },
        {
            "estimator": "aipw",
            "estimate": float(
                aipw["aipw_ate"]
            ),
            "confidence_interval": (
                aipw.get("confidence_interval")
            ),
            "causal_adjustment": True,
        },
        {
            "estimator": "cross_fitted_aipw",
            "estimate": float(
                crossfit["crossfit_aipw_ate"]
            ),
            "confidence_interval": (
                crossfit.get("confidence_interval")
            ),
            "causal_adjustment": True,
        },
    ]


def compare_estimators_to_truth(
    estimators: list[dict],
    true_ate: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    for item in estimators:
        estimate = float(item["estimate"])
        signed_error = estimate - true_ate
        absolute_error = abs(signed_error)

        ci = item.get("confidence_interval")

        if ci is None:
            ci_lower = np.nan
            ci_upper = np.nan
            ci_covers_true_ate = np.nan
        else:
            ci_lower = float(ci["lower"])
            ci_upper = float(ci["upper"])
            ci_covers_true_ate = bool(
                ci_lower <= true_ate <= ci_upper
            )

        rows.append(
            {
                "estimator": item["estimator"],
                "causal_adjustment": bool(
                    item["causal_adjustment"]
                ),
                "estimate": estimate,
                "estimate_percentage_points": (
                    estimate * 100.0
                ),
                "true_ate": true_ate,
                "true_ate_percentage_points": (
                    true_ate * 100.0
                ),
                "signed_error": signed_error,
                "signed_error_percentage_points": (
                    signed_error * 100.0
                ),
                "absolute_error": absolute_error,
                "absolute_error_percentage_points": (
                    absolute_error * 100.0
                ),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_covers_true_ate": (
                    ci_covers_true_ate
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_propensity_recovery(
    data: pd.DataFrame,
) -> dict:
    if not PROPENSITY_PATH.exists():
        raise FileNotFoundError(
            f"Missing propensity artifact: {PROPENSITY_PATH}. "
            "Run `python -m src.propensity` first."
        )

    estimated = pd.read_csv(
        PROPENSITY_PATH
    )

    required = {
        ID_COLUMN,
        "propensity_score",
    }

    missing = sorted(
        required.difference(estimated.columns)
    )

    if missing:
        raise ValueError(
            "Propensity artifact is missing required columns: "
            + ", ".join(missing)
        )

    comparison = data[
        [
            ID_COLUMN,
            TRUE_PROPENSITY_COLUMN,
        ]
    ].merge(
        estimated[
            [
                ID_COLUMN,
                "propensity_score",
            ]
        ],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != len(data):
        raise ValueError(
            "Estimated propensity scores do not align one-to-one "
            "with the source data."
        )

    truth = comparison[
        TRUE_PROPENSITY_COLUMN
    ].to_numpy(dtype=float)

    estimate = comparison[
        "propensity_score"
    ].to_numpy(dtype=float)

    residual = estimate - truth

    correlation = float(
        np.corrcoef(
            truth,
            estimate,
        )[0, 1]
    )

    return {
        "mae": float(
            np.mean(np.abs(residual))
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        ),
        "mean_signed_error": float(
            np.mean(residual)
        ),
        "pearson_correlation": correlation,
        "true_propensity_mean": float(
            np.mean(truth)
        ),
        "estimated_propensity_mean": float(
            np.mean(estimate)
        ),
        "true_propensity_min": float(
            np.min(truth)
        ),
        "true_propensity_max": float(
            np.max(truth)
        ),
        "estimated_propensity_min": float(
            np.min(estimate)
        ),
        "estimated_propensity_max": float(
            np.max(estimate)
        ),
    }


def build_summary(
    true_effect: dict,
    comparison: pd.DataFrame,
    propensity_recovery: dict,
) -> dict:
    adjusted = comparison.loc[
        comparison["causal_adjustment"]
    ].copy()

    best_adjusted_row = adjusted.loc[
        adjusted[
            "absolute_error"
        ].idxmin()
    ]

    return {
        "evaluation_only": True,
        "ground_truth_used_for_training": False,
        "ground_truth_used_for_estimation": False,
        "true_effect_summary": true_effect,
        "propensity_recovery": propensity_recovery,
        "ate_recovery": {
            "n_estimators_compared": int(
                len(comparison)
            ),
            "n_adjusted_estimators_compared": int(
                len(adjusted)
            ),
            "best_adjusted_estimator_by_absolute_error": (
                str(
                    best_adjusted_row[
                        "estimator"
                    ]
                )
            ),
            "best_adjusted_absolute_error": float(
                best_adjusted_row[
                    "absolute_error"
                ]
            ),
            "best_adjusted_absolute_error_percentage_points": float(
                best_adjusted_row[
                    "absolute_error_percentage_points"
                ]
            ),
            "all_adjusted_confidence_intervals_cover_true_ate": bool(
                adjusted[
                    "ci_covers_true_ate"
                ].dropna().all()
            ),
        },
        "policy": (
            "Synthetic truth is used only after estimator construction "
            "to evaluate recovery. Do not retune the causal pipeline merely "
            "to minimize error against the known synthetic truth."
        ),
    }


def save_outputs(
    comparison: pd.DataFrame,
    summary: dict,
) -> None:
    ATE_COMPARISON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        ATE_COMPARISON_PATH,
        index=False,
    )

    with GROUND_TRUTH_SUMMARY_PATH.open(
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
    comparison: pd.DataFrame,
    summary: dict,
) -> None:
    truth = summary[
        "true_effect_summary"
    ]

    propensity = summary[
        "propensity_recovery"
    ]

    print(
        "Synthetic Ground-Truth Evaluation"
    )
    print(
        "---------------------------------"
    )

    print(
        "Evaluation-only fields are now being used."
    )

    print(
        "They were NOT used to fit the propensity, IPW, "
        "AIPW, or cross-fitted AIPW estimators."
    )

    print(
        "\nTrue synthetic treatment effect"
    )

    print(
        "True ATE: "
        f"{truth['true_ate']:.6f} "
        f"({truth['true_ate_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "True ITE range: "
        f"[{truth['true_ite_minimum']:.6f}, "
        f"{truth['true_ite_maximum']:.6f}]"
    )

    print(
        "True ITE standard deviation: "
        f"{truth['true_ite_std']:.6f}"
    )

    display_columns = [
        "estimator",
        "estimate_percentage_points",
        "signed_error_percentage_points",
        "absolute_error_percentage_points",
        "ci_covers_true_ate",
    ]

    display = comparison[
        display_columns
    ].copy()

    print(
        "\nATE recovery"
    )

    print(
        display.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\nPropensity-score recovery"
    )

    print(
        f"MAE:         {propensity['mae']:.6f}"
    )

    print(
        f"RMSE:        {propensity['rmse']:.6f}"
    )

    print(
        "Mean error:  "
        f"{propensity['mean_signed_error']:.6f}"
    )

    print(
        "Correlation: "
        f"{propensity['pearson_correlation']:.6f}"
    )

    print(
        "\nArtifacts saved:"
    )

    print(
        f"- {ATE_COMPARISON_PATH}"
    )

    print(
        f"- {GROUND_TRUTH_SUMMARY_PATH}"
    )

    print(
        "\nPolicy:"
    )

    print(
        summary["policy"]
    )


def main() -> None:
    data = load_data()

    validate_ground_truth_columns(
        data=data
    )

    true_effect = calculate_true_effect_summary(
        data=data
    )

    naive = load_json(
        NAIVE_PATH,
        "naive baseline",
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
        CROSSFIT_AIPW_PATH,
        "cross-fitted AIPW",
    )

    estimators = extract_estimates(
        naive=naive,
        ipw=ipw,
        aipw=aipw,
        crossfit=crossfit,
    )

    comparison = compare_estimators_to_truth(
        estimators=estimators,
        true_ate=true_effect["true_ate"],
    )

    propensity_recovery = evaluate_propensity_recovery(
        data=data
    )

    summary = build_summary(
        true_effect=true_effect,
        comparison=comparison,
        propensity_recovery=propensity_recovery,
    )

    save_outputs(
        comparison=comparison,
        summary=summary,
    )

    print_summary(
        comparison=comparison,
        summary=summary,
    )


if __name__ == "__main__":
    main()
