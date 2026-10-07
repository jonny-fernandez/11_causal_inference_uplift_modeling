from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.data_contract import load_data


ID_COLUMN = "person_id"
TRUE_ITE_COLUMN = "true_individual_treatment_effect"

CROSSFIT_SCORES_PATH = Path("artifacts/aipw_crossfit_scores.csv")

OUTPUT_PATH = Path("artifacts/segment_effects.csv")
SUMMARY_PATH = Path("artifacts/segment_effects_summary.json")

CONFIDENCE_LEVEL = 0.95
MIN_SEGMENT_N = 100

SEGMENT_COLUMNS = [
    "department",
    "job_level",
]


def load_crossfit_scores() -> pd.DataFrame:
    if not CROSSFIT_SCORES_PATH.exists():
        raise FileNotFoundError(
            f"Missing cross-fitted AIPW artifact: {CROSSFIT_SCORES_PATH}. "
            "Run `python -m src.aipw_crossfit` first."
        )

    scores = pd.read_csv(CROSSFIT_SCORES_PATH)

    required = {
        ID_COLUMN,
        "aipw_pseudo_outcome_crossfit",
    }

    missing = sorted(required.difference(scores.columns))

    if missing:
        raise ValueError(
            "Cross-fitted AIPW artifact is missing required columns: "
            + ", ".join(missing)
        )

    if scores[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Cross-fitted AIPW artifact contains duplicate person IDs."
        )

    if scores["aipw_pseudo_outcome_crossfit"].isna().any():
        raise ValueError(
            "Cross-fitted AIPW pseudo-outcomes contain missing values."
        )

    return scores


def add_predefined_segments(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a small, predeclared set of pre-treatment segments.

    Segment definitions use only observed pre-treatment covariates.
    Synthetic treatment-effect truth is not used to construct any segment.
    """

    result = data.copy()

    required = {
        ID_COLUMN,
        TRUE_ITE_COLUMN,
        "department",
        "job_level",
        "baseline_performance",
        "baseline_engagement",
    }

    missing = sorted(required.difference(result.columns))

    if missing:
        raise ValueError(
            "Source data is missing required subgroup columns: "
            + ", ".join(missing)
        )

    if result[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Source data contains duplicate person IDs."
        )

    # Rank before qcut so tied values do not produce unstable bin counts.
    # Segment labels are defined without looking at outcomes or synthetic truth.
    result["baseline_performance_tertile"] = pd.qcut(
        result["baseline_performance"].rank(method="first"),
        q=3,
        labels=["Low", "Middle", "High"],
    ).astype(str)

    result["baseline_engagement_tertile"] = pd.qcut(
        result["baseline_engagement"].rank(method="first"),
        q=3,
        labels=["Low", "Middle", "High"],
    ).astype(str)

    return result


def align_data_and_scores(
    data: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    merged = data.merge(
        scores[
            [
                ID_COLUMN,
                "aipw_pseudo_outcome_crossfit",
            ]
        ],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(data):
        raise ValueError(
            "Cross-fitted AIPW scores do not align one-to-one "
            "with the source dataset."
        )

    return merged


def estimate_segment(
    segment_family: str,
    segment_value: str,
    group: pd.DataFrame,
) -> dict:
    pseudo = group[
        "aipw_pseudo_outcome_crossfit"
    ].to_numpy(dtype=float)

    truth = group[
        TRUE_ITE_COLUMN
    ].to_numpy(dtype=float)

    n = int(len(group))

    if n <= 1:
        raise ValueError(
            f"Segment {segment_family}={segment_value} "
            "does not contain enough rows."
        )

    estimate = float(np.mean(pseudo))

    standard_error = float(
        np.std(
            pseudo,
            ddof=1,
        )
        / sqrt(n)
    )

    alpha = 1.0 - CONFIDENCE_LEVEL
    z_value = NormalDist().inv_cdf(
        1.0 - alpha / 2.0
    )

    ci_lower = (
        estimate
        - z_value * standard_error
    )

    ci_upper = (
        estimate
        + z_value * standard_error
    )

    true_effect = float(
        np.mean(truth)
    )

    error = (
        estimate
        - true_effect
    )

    return {
        "segment_family": segment_family,
        "segment_value": str(segment_value),
        "n": n,
        "small_segment_flag": bool(
            n < MIN_SEGMENT_N
        ),
        "estimated_effect": estimate,
        "estimated_effect_percentage_points": (
            estimate * 100.0
        ),
        "standard_error": standard_error,
        "standard_error_percentage_points": (
            standard_error * 100.0
        ),
        "confidence_level": CONFIDENCE_LEVEL,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_lower_percentage_points": (
            ci_lower * 100.0
        ),
        "ci_upper_percentage_points": (
            ci_upper * 100.0
        ),
        "true_effect_evaluation_only": true_effect,
        "true_effect_percentage_points": (
            true_effect * 100.0
        ),
        "signed_error": error,
        "signed_error_percentage_points": (
            error * 100.0
        ),
        "absolute_error": abs(error),
        "absolute_error_percentage_points": (
            abs(error) * 100.0
        ),
        "ci_covers_true_effect": bool(
            ci_lower <= true_effect <= ci_upper
        ),
    }


def calculate_segment_effects(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    segment_families = {
        "department": "department",
        "job_level": "job_level",
        "baseline_performance_tertile": (
            "baseline_performance_tertile"
        ),
        "baseline_engagement_tertile": (
            "baseline_engagement_tertile"
        ),
    }

    for family_name, column in segment_families.items():
        for value, group in data.groupby(
            column,
            dropna=False,
        ):
            rows.append(
                estimate_segment(
                    segment_family=family_name,
                    segment_value=str(value),
                    group=group,
                )
            )

    result = pd.DataFrame(rows)

    return result.sort_values(
        by=[
            "segment_family",
            "estimated_effect",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def family_recovery_summary(
    effects: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []

    for family, group in effects.groupby(
        "segment_family"
    ):
        if len(group) >= 3:
            pearson = float(
                np.corrcoef(
                    group["estimated_effect"],
                    group[
                        "true_effect_evaluation_only"
                    ],
                )[0, 1]
            )
        else:
            pearson = np.nan

        estimated_range = float(
            group["estimated_effect"].max()
            - group["estimated_effect"].min()
        )

        true_range = float(
            group[
                "true_effect_evaluation_only"
            ].max()
            - group[
                "true_effect_evaluation_only"
            ].min()
        )

        rows.append(
            {
                "segment_family": family,
                "groups": int(len(group)),
                "pearson_estimated_vs_true": (
                    pearson
                ),
                "mean_absolute_error": float(
                    group[
                        "absolute_error"
                    ].mean()
                ),
                "mean_absolute_error_percentage_points": float(
                    group[
                        "absolute_error_percentage_points"
                    ].mean()
                ),
                "estimated_effect_range": (
                    estimated_range
                ),
                "estimated_effect_range_percentage_points": (
                    estimated_range * 100.0
                ),
                "true_effect_range": (
                    true_range
                ),
                "true_effect_range_percentage_points": (
                    true_range * 100.0
                ),
                "all_cis_cover_true_effect": bool(
                    group[
                        "ci_covers_true_effect"
                    ].all()
                ),
            }
        )

    return rows


def build_summary(
    effects: pd.DataFrame,
) -> dict:
    family_summary = family_recovery_summary(
        effects=effects,
    )

    return {
        "method": (
            "predefined_subgroup_means_of_cross_fitted_aipw_pseudo_outcomes"
        ),
        "confidence_level": CONFIDENCE_LEVEL,
        "minimum_segment_n_policy": MIN_SEGMENT_N,
        "segment_definitions": [
            "department",
            "job_level",
            "baseline_performance_tertile",
            "baseline_engagement_tertile",
        ],
        "segment_definitions_use_ground_truth": False,
        "true_effect_used_for_estimation": False,
        "true_effect_used_for_evaluation_only": True,
        "n_segment_rows": int(
            len(effects)
        ),
        "small_segment_count": int(
            effects[
                "small_segment_flag"
            ].sum()
        ),
        "overall_mean_absolute_error": float(
            effects[
                "absolute_error"
            ].mean()
        ),
        "overall_mean_absolute_error_percentage_points": float(
            effects[
                "absolute_error_percentage_points"
            ].mean()
        ),
        "all_segment_cis_cover_true_effect": bool(
            effects[
                "ci_covers_true_effect"
            ].all()
        ),
        "family_recovery": family_summary,
        "interpretation_policy": (
            "These subgroup estimates are predeclared descriptive "
            "heterogeneous-treatment-effect analyses based on observed "
            "pre-treatment segments. They are not individual-level CATE "
            "scores and should not be interpreted as guaranteed treatment "
            "effects for a specific person."
        ),
    }


def save_outputs(
    effects: pd.DataFrame,
    summary: dict,
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    effects.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    with SUMMARY_PATH.open(
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
    effects: pd.DataFrame,
    summary: dict,
) -> None:
    print(
        "Predefined Subgroup Treatment-Effect Analysis"
    )
    print(
        "---------------------------------------------"
    )

    print(
        f"Segment rows: "
        f"{summary['n_segment_rows']}"
    )

    print(
        f"Small segments (< {MIN_SEGMENT_N} rows): "
        f"{summary['small_segment_count']}"
    )

    print(
        "Ground truth used to define segments: "
        f"{summary['segment_definitions_use_ground_truth']}"
    )

    print(
        "Ground truth used for estimation: "
        f"{summary['true_effect_used_for_estimation']}"
    )

    print(
        "\nOverall recovery"
    )

    print(
        "Mean absolute subgroup error: "
        f"{summary['overall_mean_absolute_error']:.6f} "
        f"({summary['overall_mean_absolute_error_percentage_points']:.2f} "
        "percentage points)"
    )

    print(
        "All subgroup CIs cover true subgroup effect: "
        f"{summary['all_segment_cis_cover_true_effect']}"
    )

    print(
        "\nFamily-level recovery"
    )

    family_df = pd.DataFrame(
        summary["family_recovery"]
    )

    display_columns = [
        "segment_family",
        "groups",
        "pearson_estimated_vs_true",
        "mean_absolute_error_percentage_points",
        "estimated_effect_range_percentage_points",
        "true_effect_range_percentage_points",
        "all_cis_cover_true_effect",
    ]

    print(
        family_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\nSubgroup estimates"
    )

    display = effects[
        [
            "segment_family",
            "segment_value",
            "n",
            "estimated_effect_percentage_points",
            "ci_lower_percentage_points",
            "ci_upper_percentage_points",
            "true_effect_percentage_points",
            "absolute_error_percentage_points",
            "ci_covers_true_effect",
        ]
    ]

    print(
        display.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print(
        "\nArtifacts saved:"
    )

    print(
        f"- {OUTPUT_PATH}"
    )

    print(
        f"- {SUMMARY_PATH}"
    )

    print(
        "\nInterpretation policy:"
    )

    print(
        summary[
            "interpretation_policy"
        ]
    )


def main() -> None:
    data = load_data()

    data = add_predefined_segments(
        data=data
    )

    crossfit = load_crossfit_scores()

    aligned = align_data_and_scores(
        data=data,
        scores=crossfit,
    )

    effects = calculate_segment_effects(
        data=aligned,
    )

    summary = build_summary(
        effects=effects,
    )

    save_outputs(
        effects=effects,
        summary=summary,
    )

    print_summary(
        effects=effects,
        summary=summary,
    )


if __name__ == "__main__":
    main()
