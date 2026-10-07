from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_contract import load_data
from src.propensity import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_propensity_pipeline,
)


ID_COLUMN = "person_id"
TREATMENT_COLUMN = "training_assigned"
OUTCOME_COLUMN = "retained_or_promoted_180d"

WEIGHTS_PATH = Path("artifacts/ipw_weights.csv")
BASELINE_PATH = Path("artifacts/naive_baseline.json")

OUTPUT_PATH = Path("artifacts/ate_summary.json")
BOOTSTRAP_OUTPUT_PATH = Path("artifacts/ate_bootstrap_distribution.csv")

WEIGHT_COLUMN = "ate_stabilized_ipw"

CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_RANDOM_STATE = 42


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """Calculate a weighted arithmetic mean."""

    values_array = values.to_numpy(dtype=float)
    weights_array = weights.to_numpy(dtype=float)

    denominator = float(np.sum(weights_array))

    if denominator <= 0.0:
        raise ValueError(
            "Weighted mean requires a positive total weight."
        )

    return float(
        np.sum(values_array * weights_array)
        / denominator
    )


def load_weights() -> pd.DataFrame:
    """Load the previously validated ATE weighting artifact."""

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing weighting artifact: {WEIGHTS_PATH}. "
            "Run `python -m src.weighting` first."
        )

    weights = pd.read_csv(WEIGHTS_PATH)

    required_columns = {
        ID_COLUMN,
        TREATMENT_COLUMN,
        WEIGHT_COLUMN,
    }

    missing = sorted(
        required_columns.difference(weights.columns)
    )

    if missing:
        raise ValueError(
            "Weight artifact is missing required columns: "
            + ", ".join(missing)
        )

    if weights[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Weight artifact contains duplicate person IDs."
        )

    if weights[WEIGHT_COLUMN].isna().any():
        raise ValueError(
            f"{WEIGHT_COLUMN} contains missing values."
        )

    if not np.isfinite(
        weights[WEIGHT_COLUMN]
    ).all():
        raise ValueError(
            f"{WEIGHT_COLUMN} contains non-finite values."
        )

    if (
        weights[WEIGHT_COLUMN] <= 0.0
    ).any():
        raise ValueError(
            f"{WEIGHT_COLUMN} must contain "
            "only positive values."
        )

    return weights


def align_data_and_weights(
    data: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Align outcomes with the validated weighting artifact."""

    required_data_columns = {
        ID_COLUMN,
        TREATMENT_COLUMN,
        OUTCOME_COLUMN,
    }

    missing = sorted(
        required_data_columns.difference(data.columns)
    )

    if missing:
        raise ValueError(
            "Source data is missing required columns: "
            + ", ".join(missing)
        )

    merged = data[
        [
            ID_COLUMN,
            TREATMENT_COLUMN,
            OUTCOME_COLUMN,
        ]
    ].merge(
        weights[
            [
                ID_COLUMN,
                TREATMENT_COLUMN,
                WEIGHT_COLUMN,
            ]
        ],
        on=ID_COLUMN,
        how="inner",
        suffixes=("_data", "_weight"),
        validate="one_to_one",
    )

    if len(merged) != len(data):
        raise ValueError(
            "Weights do not align one-to-one "
            "with the source dataset."
        )

    treatment_mismatch = (
        merged[f"{TREATMENT_COLUMN}_data"]
        != merged[f"{TREATMENT_COLUMN}_weight"]
    )

    if treatment_mismatch.any():
        raise ValueError(
            "Treatment values in the weighting artifact "
            "do not match the source data."
        )

    merged = merged.rename(
        columns={
            f"{TREATMENT_COLUMN}_data":
                TREATMENT_COLUMN,
        }
    ).drop(
        columns=[
            f"{TREATMENT_COLUMN}_weight"
        ]
    )

    return merged


def load_naive_baseline() -> dict:
    """Load the previously computed unadjusted baseline."""

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline artifact: {BASELINE_PATH}. "
            "Run `python -m src.baseline` first."
        )

    with BASELINE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def estimate_ipw_ate(
    data: pd.DataFrame,
) -> dict:
    """Estimate the ATE using stabilized inverse-probability weights."""

    treated_mask = (
        data[TREATMENT_COLUMN] == 1
    )

    control_mask = (
        data[TREATMENT_COLUMN] == 0
    )

    if not treated_mask.any():
        raise ValueError(
            "No treated observations are available."
        )

    if not control_mask.any():
        raise ValueError(
            "No control observations are available."
        )

    treated_outcome = data.loc[
        treated_mask,
        OUTCOME_COLUMN,
    ].astype(float)

    control_outcome = data.loc[
        control_mask,
        OUTCOME_COLUMN,
    ].astype(float)

    treated_weights = data.loc[
        treated_mask,
        WEIGHT_COLUMN,
    ].astype(float)

    control_weights = data.loc[
        control_mask,
        WEIGHT_COLUMN,
    ].astype(float)

    weighted_treated_mean = weighted_mean(
        treated_outcome,
        treated_weights,
    )

    weighted_control_mean = weighted_mean(
        control_outcome,
        control_weights,
    )

    ate = (
        weighted_treated_mean
        - weighted_control_mean
    )

    return {
        "weighted_treated_outcome_mean":
            weighted_treated_mean,
        "weighted_control_outcome_mean":
            weighted_control_mean,
        "ipw_ate":
            ate,
        "ipw_ate_percentage_points":
            ate * 100.0,
    }


def estimate_bootstrap_replicate(
    sample: pd.DataFrame,
) -> float:
    """Refit the propensity model and estimate one bootstrap ATE."""

    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    treatment = (
        sample[TREATMENT_COLUMN]
        .astype(int)
        .to_numpy()
    )

    outcome = (
        sample[OUTCOME_COLUMN]
        .astype(float)
        .to_numpy()
    )

    if np.unique(treatment).size != 2:
        raise ValueError(
            "Bootstrap replicate does not contain both "
            "treatment groups."
        )

    pipeline = build_propensity_pipeline()

    pipeline.fit(
        sample[feature_columns],
        treatment,
    )

    propensity = pipeline.predict_proba(
        sample[feature_columns]
    )[:, 1]

    if not np.isfinite(propensity).all():
        raise ValueError(
            "Bootstrap propensity scores contain "
            "non-finite values."
        )

    if (
        (propensity <= 0.0)
        | (propensity >= 1.0)
    ).any():
        raise ValueError(
            "Bootstrap propensity scores must be "
            "strictly between 0 and 1."
        )

    treatment_prevalence = float(
        treatment.mean()
    )

    control_prevalence = (
        1.0 - treatment_prevalence
    )

    weights = np.where(
        treatment == 1,
        treatment_prevalence / propensity,
        control_prevalence / (1.0 - propensity),
    )

    treated_mask = treatment == 1
    control_mask = treatment == 0

    treated_weight_sum = float(
        weights[treated_mask].sum()
    )

    control_weight_sum = float(
        weights[control_mask].sum()
    )

    if (
        treated_weight_sum <= 0.0
        or control_weight_sum <= 0.0
    ):
        raise ValueError(
            "Bootstrap replicate has invalid total weights."
        )

    treated_mean = float(
        np.sum(
            weights[treated_mask]
            * outcome[treated_mask]
        )
        / treated_weight_sum
    )

    control_mean = float(
        np.sum(
            weights[control_mask]
            * outcome[control_mask]
        )
        / control_weight_sum
    )

    return treated_mean - control_mean


def bootstrap_ipw_ate(
    data: pd.DataFrame,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> pd.DataFrame:
    """Bootstrap the complete IPW estimator.

    Each bootstrap sample refits the propensity model before constructing
    stabilized ATE weights. This captures sampling variation in both the
    observed data and the estimated propensity model.
    """

    if n_resamples <= 1:
        raise ValueError(
            "Bootstrap requires at least two resamples."
        )

    rng = np.random.default_rng(
        random_state
    )

    n_rows = len(data)

    if n_rows == 0:
        raise ValueError(
            "Cannot bootstrap an empty dataset."
        )

    estimates: list[float] = []

    attempts = 0
    max_attempts = n_resamples * 5

    while (
        len(estimates) < n_resamples
        and attempts < max_attempts
    ):
        attempts += 1

        indices = rng.integers(
            low=0,
            high=n_rows,
            size=n_rows,
        )

        sample = (
            data.iloc[indices]
            .reset_index(drop=True)
        )

        try:
            estimate = estimate_bootstrap_replicate(
                sample
            )
        except ValueError:
            continue

        estimates.append(
            float(estimate)
        )

    if len(estimates) != n_resamples:
        raise RuntimeError(
            "Unable to produce the requested number of valid "
            f"bootstrap resamples. Requested {n_resamples}, "
            f"completed {len(estimates)}."
        )

    return pd.DataFrame(
        {
            "bootstrap_iteration":
                np.arange(
                    1,
                    n_resamples + 1,
                    dtype=int,
                ),
            "ipw_ate": estimates,
            "ipw_ate_percentage_points": (
                np.asarray(estimates) * 100.0
            ),
        }
    )


def summarize_bootstrap(
    bootstrap: pd.DataFrame,
) -> dict:
    """Summarize bootstrap uncertainty using a percentile interval."""

    estimates = (
        bootstrap["ipw_ate"]
        .to_numpy(dtype=float)
    )

    alpha = (
        1.0 - CONFIDENCE_LEVEL
    )

    lower = float(
        np.quantile(
            estimates,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            estimates,
            1.0 - alpha / 2.0,
        )
    )

    standard_error = float(
        np.std(
            estimates,
            ddof=1,
        )
    )

    return {
        "method": (
            "nonparametric_percentile_bootstrap_with_"
            "propensity_refitting"
        ),
        "confidence_level":
            CONFIDENCE_LEVEL,
        "bootstrap_resamples":
            int(len(estimates)),
        "bootstrap_random_state":
            BOOTSTRAP_RANDOM_STATE,
        "standard_error":
            standard_error,
        "standard_error_percentage_points":
            standard_error * 100.0,
        "confidence_interval": {
            "lower": lower,
            "upper": upper,
        },
        "confidence_interval_percentage_points": {
            "lower": lower * 100.0,
            "upper": upper * 100.0,
        },
        "bootstrap_mean":
            float(estimates.mean()),
        "bootstrap_mean_percentage_points":
            float(estimates.mean() * 100.0),
    }


def build_summary(
    estimate: dict,
    baseline: dict,
    uncertainty: dict,
    n_observations: int,
) -> dict:
    naive_difference = float(
        baseline["naive_difference"]
    )

    adjusted_ate = float(
        estimate["ipw_ate"]
    )

    adjustment_change = (
        adjusted_ate
        - naive_difference
    )

    return {
        "estimand": "ATE",
        "estimator": (
            "stabilized_inverse_probability_weighting"
        ),
        "treatment_column":
            TREATMENT_COLUMN,
        "outcome_column":
            OUTCOME_COLUMN,
        "weight_column":
            WEIGHT_COLUMN,
        "n_observations":
            n_observations,
        "weighted_treated_outcome_mean": (
            estimate[
                "weighted_treated_outcome_mean"
            ]
        ),
        "weighted_control_outcome_mean": (
            estimate[
                "weighted_control_outcome_mean"
            ]
        ),
        "ipw_ate":
            adjusted_ate,
        "ipw_ate_percentage_points":
            adjusted_ate * 100.0,
        "naive_difference":
            naive_difference,
        "naive_difference_percentage_points":
            naive_difference * 100.0,
        "adjustment_change":
            adjustment_change,
        "adjustment_change_percentage_points":
            adjustment_change * 100.0,
        "uncertainty_status":
            "estimated",
        "uncertainty":
            uncertainty,
        "causal_interpretation": (
            "Under the stated identification assumptions, "
            "the IPW estimate represents the estimated "
            "average treatment effect of training assignment "
            "on the probability of retention or promotion "
            "within 180 days."
        ),
        "limitations": [
            (
                "This estimate relies on conditional "
                "exchangeability given the approved "
                "pre-treatment covariates."
            ),
            (
                "Observed covariate balance does not prove "
                "absence of unmeasured confounding."
            ),
            (
                "The confidence interval is a nonparametric "
                "bootstrap percentile interval."
            ),
            (
                "The propensity model is refit within each "
                "bootstrap replicate."
            ),
            (
                "Synthetic ground-truth treatment-effect "
                "fields were not used to calculate this "
                "estimate or its uncertainty."
            ),
        ],
    }


def save_outputs(
    summary: dict,
    bootstrap: pd.DataFrame,
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
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

    bootstrap.to_csv(
        BOOTSTRAP_OUTPUT_PATH,
        index=False,
    )


def print_summary(
    summary: dict,
) -> None:
    uncertainty = summary[
        "uncertainty"
    ]

    interval = uncertainty[
        "confidence_interval"
    ]

    interval_pp = uncertainty[
        "confidence_interval_percentage_points"
    ]

    print(
        "IPW-adjusted Average Treatment Effect"
    )
    print(
        "-------------------------------------"
    )

    print(
        f"Estimator: "
        f"{summary['estimator']}"
    )

    print(
        f"Observations: "
        f"{summary['n_observations']:,}"
    )

    print(
        "\nWeighted outcome means"
    )

    print(
        "Control: "
        f"{summary['weighted_control_outcome_mean']:.6f} "
        f"({summary['weighted_control_outcome_mean']:.2%})"
    )

    print(
        "Treated: "
        f"{summary['weighted_treated_outcome_mean']:.6f} "
        f"({summary['weighted_treated_outcome_mean']:.2%})"
    )

    print(
        "\nAdjusted ATE"
    )

    print(
        f"{summary['ipw_ate']:.6f} "
        f"({summary['ipw_ate_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "\nNaive baseline"
    )

    print(
        f"{summary['naive_difference']:.6f} "
        f"({summary['naive_difference_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "\nChange after adjustment"
    )

    print(
        f"{summary['adjustment_change']:.6f} "
        f"({summary['adjustment_change_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "\nBootstrap uncertainty"
    )

    print(
        f"Method: "
        f"{uncertainty['method']}"
    )

    print(
        f"Resamples: "
        f"{uncertainty['bootstrap_resamples']:,}"
    )

    print(
        "Standard error: "
        f"{uncertainty['standard_error']:.6f} "
        f"({uncertainty['standard_error_percentage_points']:.2f} "
        "percentage points)"
    )

    print(
        f"{int(uncertainty['confidence_level'] * 100)}% CI: "
        f"[{interval['lower']:.6f}, "
        f"{interval['upper']:.6f}] "
        f"([{interval_pp['lower']:+.2f}, "
        f"{interval_pp['upper']:+.2f}] "
        "percentage points)"
    )

    print(
        "\nArtifacts saved:"
    )

    print(
        f"- {OUTPUT_PATH}"
    )

    print(
        f"- {BOOTSTRAP_OUTPUT_PATH}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        summary["causal_interpretation"]
    )

    print(
        "\nImportant: observed balance and overlap "
        "support the adjustment design but do not "
        "prove absence of unmeasured confounding."
    )


def main() -> None:
    data = load_data()

    weights = load_weights()

    aligned = align_data_and_weights(
        data=data,
        weights=weights,
    )

    baseline = load_naive_baseline()

    estimate = estimate_ipw_ate(
        data=aligned,
    )

    bootstrap = bootstrap_ipw_ate(
        data=data,
    )

    uncertainty = summarize_bootstrap(
        bootstrap=bootstrap,
    )

    summary = build_summary(
        estimate=estimate,
        baseline=baseline,
        uncertainty=uncertainty,
        n_observations=len(aligned),
    )

    save_outputs(
        summary=summary,
        bootstrap=bootstrap,
    )

    print_summary(
        summary=summary,
    )


if __name__ == "__main__":
    main()
