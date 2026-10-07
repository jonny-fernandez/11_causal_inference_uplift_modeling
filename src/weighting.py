from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_contract import load_data


ID_COLUMN = "person_id"
TREATMENT_COLUMN = "training_assigned"

PROPENSITY_PATH = Path("artifacts/propensity_scores.csv")
WEIGHTS_OUTPUT_PATH = Path("artifacts/ipw_weights.csv")
DIAGNOSTICS_OUTPUT_PATH = Path("artifacts/weight_diagnostics.json")

PROPENSITY_COLUMN = "propensity_score"
IPW_COLUMN = "ate_ipw"
STABILIZED_IPW_COLUMN = "ate_stabilized_ipw"


def load_propensity_scores() -> pd.DataFrame:
    """Load previously generated propensity scores."""

    if not PROPENSITY_PATH.exists():
        raise FileNotFoundError(
            f"Missing propensity artifact: {PROPENSITY_PATH}. "
            "Run `python -m src.propensity` first."
        )

    scores = pd.read_csv(PROPENSITY_PATH)

    required = {
        ID_COLUMN,
        TREATMENT_COLUMN,
        PROPENSITY_COLUMN,
    }

    missing = sorted(required.difference(scores.columns))

    if missing:
        raise ValueError(
            "Propensity-score artifact is missing required columns: "
            + ", ".join(missing)
        )

    if scores[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Propensity-score artifact contains duplicate person IDs."
        )

    if scores[PROPENSITY_COLUMN].isna().any():
        raise ValueError("Propensity scores contain missing values.")

    if (
        (scores[PROPENSITY_COLUMN] <= 0.0)
        | (scores[PROPENSITY_COLUMN] >= 1.0)
    ).any():
        raise ValueError(
            "All propensity scores must be strictly between 0 and 1."
        )

    return scores


def validate_alignment(
    data: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Verify propensity scores align with the source treatment values."""

    merged = data[
        [ID_COLUMN, TREATMENT_COLUMN]
    ].merge(
        scores,
        on=ID_COLUMN,
        how="inner",
        suffixes=("_data", "_score"),
        validate="one_to_one",
    )

    if len(merged) != len(data):
        raise ValueError(
            "Propensity-score rows do not align one-to-one with source data."
        )

    mismatch = (
        merged[f"{TREATMENT_COLUMN}_data"]
        != merged[f"{TREATMENT_COLUMN}_score"]
    )

    if mismatch.any():
        raise ValueError(
            "Treatment values in propensity_scores.csv do not match "
            "the source dataset."
        )

    return merged.rename(
        columns={
            f"{TREATMENT_COLUMN}_data": TREATMENT_COLUMN,
        }
    ).drop(
        columns=[f"{TREATMENT_COLUMN}_score"]
    )


def effective_sample_size(weights: pd.Series) -> float:
    """Calculate Kish effective sample size."""

    weight_array = weights.to_numpy(dtype=float)

    numerator = float(np.sum(weight_array) ** 2)
    denominator = float(np.sum(weight_array ** 2))

    if denominator == 0.0:
        raise ValueError(
            "Cannot calculate effective sample size with zero denominator."
        )

    return numerator / denominator


def calculate_weights(
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate unstabilized and stabilized ATE IPW weights."""

    result = aligned[
        [
            ID_COLUMN,
            TREATMENT_COLUMN,
            PROPENSITY_COLUMN,
        ]
    ].copy()

    treatment = result[TREATMENT_COLUMN].astype(int)
    propensity = result[PROPENSITY_COLUMN].astype(float)

    treatment_prevalence = float(treatment.mean())
    control_prevalence = 1.0 - treatment_prevalence

    result[IPW_COLUMN] = np.where(
        treatment == 1,
        1.0 / propensity,
        1.0 / (1.0 - propensity),
    )

    result[STABILIZED_IPW_COLUMN] = np.where(
        treatment == 1,
        treatment_prevalence / propensity,
        control_prevalence / (1.0 - propensity),
    )

    validate_weights(result)

    return result


def validate_weights(weights: pd.DataFrame) -> None:
    """Validate calculated weighting artifacts."""

    for column in [
        IPW_COLUMN,
        STABILIZED_IPW_COLUMN,
    ]:
        if weights[column].isna().any():
            raise ValueError(
                f"{column} contains missing values."
            )

        if not np.isfinite(weights[column]).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

        if (weights[column] <= 0.0).any():
            raise ValueError(
                f"{column} must contain only positive values."
            )


def summarize_weight_series(
    weights: pd.Series,
) -> dict:
    quantiles = weights.quantile(
        [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )

    return {
        "count": int(weights.shape[0]),
        "minimum": float(weights.min()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "maximum": float(weights.max()),
        "mean": float(weights.mean()),
        "std": float(weights.std()),
        "sum": float(weights.sum()),
        "effective_sample_size": float(
            effective_sample_size(weights)
        ),
    }


def create_diagnostics(
    weights: pd.DataFrame,
) -> dict:
    treatment_prevalence = float(
        weights[TREATMENT_COLUMN].mean()
    )

    control_mask = weights[TREATMENT_COLUMN] == 0
    treated_mask = weights[TREATMENT_COLUMN] == 1

    control_stabilized = weights.loc[
        control_mask,
        STABILIZED_IPW_COLUMN,
    ]

    treated_stabilized = weights.loc[
        treated_mask,
        STABILIZED_IPW_COLUMN,
    ]

    overall_stabilized = weights[
        STABILIZED_IPW_COLUMN
    ]

    overall_unstabilized = weights[
        IPW_COLUMN
    ]

    original_control_n = int(control_mask.sum())
    original_treated_n = int(treated_mask.sum())
    original_total_n = int(len(weights))

    control_ess = effective_sample_size(
        control_stabilized
    )
    treated_ess = effective_sample_size(
        treated_stabilized
    )
    overall_ess = effective_sample_size(
        overall_stabilized
    )

    return {
        "estimand": "ATE",
        "weighting_method": (
            "inverse_probability_of_treatment_weighting"
        ),
        "selected_analysis_weight": STABILIZED_IPW_COLUMN,
        "propensity_column": PROPENSITY_COLUMN,
        "treatment_prevalence": treatment_prevalence,
        "control_prevalence": 1.0 - treatment_prevalence,
        "trimming_applied": False,
        "clipping_applied": False,
        "overlap_restriction_applied": False,
        "unstabilized_ate_ipw": summarize_weight_series(
            overall_unstabilized
        ),
        "stabilized_ate_ipw": {
            "overall": summarize_weight_series(
                overall_stabilized
            ),
            "control": summarize_weight_series(
                control_stabilized
            ),
            "treated": summarize_weight_series(
                treated_stabilized
            ),
        },
        "effective_sample_size": {
            "original_total_n": original_total_n,
            "weighted_total_ess": float(
                overall_ess
            ),
            "weighted_total_ess_fraction": float(
                overall_ess / original_total_n
            ),
            "original_control_n": original_control_n,
            "weighted_control_ess": float(
                control_ess
            ),
            "weighted_control_ess_fraction": float(
                control_ess / original_control_n
            ),
            "original_treated_n": original_treated_n,
            "weighted_treated_ess": float(
                treated_ess
            ),
            "weighted_treated_ess_fraction": float(
                treated_ess / original_treated_n
            ),
        },
        "interpretation": (
            "Weights were constructed for the ATE estimand using "
            "estimated propensity scores. Stabilized weights are selected "
            "for subsequent weighted diagnostics. No trimming, clipping, "
            "or overlap restriction was applied because the preceding "
            "propensity diagnostics did not identify extreme propensity "
            "scores. Weight diagnostics and post-weighting covariate "
            "balance must still be evaluated before using the weights "
            "for treatment-effect estimation."
        ),
    }


def save_outputs(
    weights: pd.DataFrame,
    diagnostics: dict,
) -> None:
    WEIGHTS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights.to_csv(
        WEIGHTS_OUTPUT_PATH,
        index=False,
    )

    with DIAGNOSTICS_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            diagnostics,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def print_diagnostics(
    diagnostics: dict,
) -> None:
    stabilized = diagnostics[
        "stabilized_ate_ipw"
    ]

    overall = stabilized["overall"]
    control = stabilized["control"]
    treated = stabilized["treated"]

    ess = diagnostics[
        "effective_sample_size"
    ]

    print("ATE inverse-probability weighting")
    print("---------------------------------")

    print(
        f"Treatment prevalence: "
        f"{diagnostics['treatment_prevalence']:.4f}"
    )

    print(
        f"Selected analysis weight: "
        f"{diagnostics['selected_analysis_weight']}"
    )

    print("\nStabilized weight distribution")
    print(
        f"Minimum: {overall['minimum']:.6f}"
    )
    print(
        f"Median:  {overall['median']:.6f}"
    )
    print(
        f"Mean:    {overall['mean']:.6f}"
    )
    print(
        f"P95:     {overall['p95']:.6f}"
    )
    print(
        f"P99:     {overall['p99']:.6f}"
    )
    print(
        f"Maximum: {overall['maximum']:.6f}"
    )

    print("\nControl stabilized weights")
    print(
        f"Mean:    {control['mean']:.6f}"
    )
    print(
        f"Maximum: {control['maximum']:.6f}"
    )

    print("\nTreated stabilized weights")
    print(
        f"Mean:    {treated['mean']:.6f}"
    )
    print(
        f"Maximum: {treated['maximum']:.6f}"
    )

    print("\nEffective sample size")
    print(
        f"Overall: "
        f"{ess['weighted_total_ess']:.1f} / "
        f"{ess['original_total_n']:,} "
        f"({ess['weighted_total_ess_fraction']:.1%})"
    )

    print(
        f"Control: "
        f"{ess['weighted_control_ess']:.1f} / "
        f"{ess['original_control_n']:,} "
        f"({ess['weighted_control_ess_fraction']:.1%})"
    )

    print(
        f"Treated: "
        f"{ess['weighted_treated_ess']:.1f} / "
        f"{ess['original_treated_n']:,} "
        f"({ess['weighted_treated_ess_fraction']:.1%})"
    )

    print("\nAdjustment policy")
    print(
        f"Trimming applied: "
        f"{diagnostics['trimming_applied']}"
    )
    print(
        f"Clipping applied: "
        f"{diagnostics['clipping_applied']}"
    )
    print(
        "Overlap restriction applied: "
        f"{diagnostics['overlap_restriction_applied']}"
    )

    print("\nArtifacts saved:")
    print(f"- {WEIGHTS_OUTPUT_PATH}")
    print(f"- {DIAGNOSTICS_OUTPUT_PATH}")

    print(
        "\nDo not estimate the adjusted ATE yet. "
        "Post-weighting covariate balance must be checked first."
    )


def main() -> None:
    data = load_data()
    scores = load_propensity_scores()

    aligned = validate_alignment(
        data=data,
        scores=scores,
    )

    weights = calculate_weights(
        aligned
    )

    diagnostics = create_diagnostics(
        weights
    )

    save_outputs(
        weights=weights,
        diagnostics=diagnostics,
    )

    print_diagnostics(
        diagnostics
    )


if __name__ == "__main__":
    main()