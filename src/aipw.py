from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_contract import load_data
from src.propensity import CATEGORICAL_FEATURES, NUMERIC_FEATURES


ID_COLUMN = "person_id"
TREATMENT_COLUMN = "training_assigned"
OUTCOME_COLUMN = "retained_or_promoted_180d"

PROPENSITY_PATH = Path("artifacts/propensity_scores.csv")
IPW_SUMMARY_PATH = Path("artifacts/ate_summary.json")
BASELINE_PATH = Path("artifacts/naive_baseline.json")

AIPW_SCORES_PATH = Path("artifacts/aipw_scores.csv")
AIPW_SUMMARY_PATH = Path("artifacts/aipw_summary.json")
CONTROL_MODEL_PATH = Path("artifacts/aipw_outcome_model_control.joblib")
TREATED_MODEL_PATH = Path("artifacts/aipw_outcome_model_treated.joblib")

PROPENSITY_COLUMN = "propensity_score"

CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42


def build_outcome_pipeline() -> Pipeline:
    """Create a deterministic binary-outcome model pipeline."""

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop=None,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def load_propensity_scores() -> pd.DataFrame:
    """Load the previously validated propensity-score artifact."""

    if not PROPENSITY_PATH.exists():
        raise FileNotFoundError(
            f"Missing propensity artifact: {PROPENSITY_PATH}. "
            "Run `python -m src.propensity` first."
        )

    scores = pd.read_csv(PROPENSITY_PATH)

    required_columns = {
        ID_COLUMN,
        TREATMENT_COLUMN,
        PROPENSITY_COLUMN,
    }

    missing = sorted(required_columns.difference(scores.columns))

    if missing:
        raise ValueError(
            "Propensity artifact is missing required columns: "
            + ", ".join(missing)
        )

    if scores[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Propensity artifact contains duplicate person IDs."
        )

    if scores[PROPENSITY_COLUMN].isna().any():
        raise ValueError(
            "Propensity scores contain missing values."
        )

    if (
        (scores[PROPENSITY_COLUMN] <= 0.0)
        | (scores[PROPENSITY_COLUMN] >= 1.0)
    ).any():
        raise ValueError(
            "AIPW requires propensity scores strictly between 0 and 1."
        )

    return scores


def align_data_and_propensity(
    data: pd.DataFrame,
    propensity: pd.DataFrame,
) -> pd.DataFrame:
    """Align source data with the previously generated propensity scores."""

    required_data_columns = {
        ID_COLUMN,
        TREATMENT_COLUMN,
        OUTCOME_COLUMN,
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }

    missing = sorted(required_data_columns.difference(data.columns))

    if missing:
        raise ValueError(
            "Source data is missing required AIPW columns: "
            + ", ".join(missing)
        )

    merged = data.merge(
        propensity[
            [
                ID_COLUMN,
                TREATMENT_COLUMN,
                PROPENSITY_COLUMN,
            ]
        ],
        on=ID_COLUMN,
        how="inner",
        suffixes=("_data", "_propensity"),
        validate="one_to_one",
    )

    if len(merged) != len(data):
        raise ValueError(
            "Propensity scores do not align one-to-one with source data."
        )

    treatment_mismatch = (
        merged[f"{TREATMENT_COLUMN}_data"]
        != merged[f"{TREATMENT_COLUMN}_propensity"]
    )

    if treatment_mismatch.any():
        raise ValueError(
            "Treatment values in propensity_scores.csv do not match "
            "the source dataset."
        )

    merged = merged.rename(
        columns={
            f"{TREATMENT_COLUMN}_data": TREATMENT_COLUMN,
        }
    ).drop(
        columns=[f"{TREATMENT_COLUMN}_propensity"]
    )

    return merged


def fit_outcome_models(
    data: pd.DataFrame,
) -> tuple[Pipeline, Pipeline, np.ndarray, np.ndarray]:
    """Fit treatment-specific outcome models and predict both potential outcomes."""

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    control_data = data.loc[
        data[TREATMENT_COLUMN] == 0
    ].copy()

    treated_data = data.loc[
        data[TREATMENT_COLUMN] == 1
    ].copy()

    if control_data.empty:
        raise ValueError(
            "Cannot fit the control outcome model with zero control rows."
        )

    if treated_data.empty:
        raise ValueError(
            "Cannot fit the treated outcome model with zero treated rows."
        )

    if control_data[OUTCOME_COLUMN].nunique() < 2:
        raise ValueError(
            "Control outcome data must contain both outcome classes."
        )

    if treated_data[OUTCOME_COLUMN].nunique() < 2:
        raise ValueError(
            "Treated outcome data must contain both outcome classes."
        )

    control_model = build_outcome_pipeline()
    treated_model = build_outcome_pipeline()

    control_model.fit(
        control_data[feature_columns],
        control_data[OUTCOME_COLUMN].astype(int),
    )

    treated_model.fit(
        treated_data[feature_columns],
        treated_data[OUTCOME_COLUMN].astype(int),
    )

    mu0 = control_model.predict_proba(
        data[feature_columns]
    )[:, 1]

    mu1 = treated_model.predict_proba(
        data[feature_columns]
    )[:, 1]

    if not np.isfinite(mu0).all():
        raise ValueError(
            "Control outcome-model predictions contain non-finite values."
        )

    if not np.isfinite(mu1).all():
        raise ValueError(
            "Treated outcome-model predictions contain non-finite values."
        )

    if (
        (mu0 < 0.0)
        | (mu0 > 1.0)
        | (mu1 < 0.0)
        | (mu1 > 1.0)
    ).any():
        raise ValueError(
            "Outcome-model probabilities must remain between 0 and 1."
        )

    return (
        control_model,
        treated_model,
        mu0,
        mu1,
    )


def calculate_aipw_scores(
    data: pd.DataFrame,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> pd.DataFrame:
    """Calculate row-level AIPW influence scores for the ATE."""

    treatment = (
        data[TREATMENT_COLUMN]
        .astype(int)
        .to_numpy()
    )

    outcome = (
        data[OUTCOME_COLUMN]
        .astype(float)
        .to_numpy()
    )

    propensity = (
        data[PROPENSITY_COLUMN]
        .astype(float)
        .to_numpy()
    )

    if len(mu0) != len(data) or len(mu1) != len(data):
        raise ValueError(
            "Outcome-model prediction lengths do not match the dataset."
        )

    augmentation_treated = (
        treatment
        * (outcome - mu1)
        / propensity
    )

    augmentation_control = (
        (1 - treatment)
        * (outcome - mu0)
        / (1.0 - propensity)
    )

    pseudo_outcome = (
        mu1
        - mu0
        + augmentation_treated
        - augmentation_control
    )

    if not np.isfinite(pseudo_outcome).all():
        raise ValueError(
            "AIPW pseudo-outcomes contain non-finite values."
        )

    return pd.DataFrame(
        {
            ID_COLUMN: data[ID_COLUMN].to_numpy(),
            TREATMENT_COLUMN: treatment,
            OUTCOME_COLUMN: outcome,
            PROPENSITY_COLUMN: propensity,
            "mu0": mu0,
            "mu1": mu1,
            "outcome_model_effect": mu1 - mu0,
            "aipw_pseudo_outcome": pseudo_outcome,
        }
    )


def summarize_aipw(
    scores: pd.DataFrame,
) -> dict:
    """Estimate the AIPW ATE and influence-function uncertainty."""

    pseudo_outcome = (
        scores["aipw_pseudo_outcome"]
        .to_numpy(dtype=float)
    )

    n = int(len(pseudo_outcome))

    if n <= 1:
        raise ValueError(
            "AIPW uncertainty requires at least two observations."
        )

    ate = float(
        np.mean(pseudo_outcome)
    )

    standard_error = float(
        np.std(
            pseudo_outcome,
            ddof=1,
        )
        / sqrt(n)
    )

    alpha = 1.0 - CONFIDENCE_LEVEL

    z_value = NormalDist().inv_cdf(
        1.0 - alpha / 2.0
    )

    ci_lower = (
        ate - z_value * standard_error
    )

    ci_upper = (
        ate + z_value * standard_error
    )

    return {
        "aipw_ate": ate,
        "aipw_ate_percentage_points": ate * 100.0,
        "standard_error": standard_error,
        "standard_error_percentage_points": standard_error * 100.0,
        "confidence_level": CONFIDENCE_LEVEL,
        "confidence_interval": {
            "lower": ci_lower,
            "upper": ci_upper,
        },
        "confidence_interval_percentage_points": {
            "lower": ci_lower * 100.0,
            "upper": ci_upper * 100.0,
        },
        "mean_mu0": float(
            scores["mu0"].mean()
        ),
        "mean_mu1": float(
            scores["mu1"].mean()
        ),
        "mean_outcome_model_effect": float(
            scores["outcome_model_effect"].mean()
        ),
    }


def load_json_artifact(
    path: Path,
    label: str,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label} artifact: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_summary(
    aipw: dict,
    baseline: dict,
    ipw: dict,
    n_observations: int,
) -> dict:
    naive_difference = float(
        baseline["naive_difference"]
    )

    ipw_ate = float(
        ipw["ipw_ate"]
    )

    aipw_ate = float(
        aipw["aipw_ate"]
    )

    return {
        "estimand": "ATE",
        "estimator": (
            "augmented_inverse_probability_weighting"
        ),
        "doubly_robust": True,
        "cross_fitted": False,
        "treatment_column": TREATMENT_COLUMN,
        "outcome_column": OUTCOME_COLUMN,
        "propensity_column": PROPENSITY_COLUMN,
        "outcome_model": (
            "treatment_specific_logistic_regression"
        ),
        "n_observations": n_observations,
        "aipw_ate": aipw_ate,
        "aipw_ate_percentage_points": (
            aipw_ate * 100.0
        ),
        "standard_error": (
            aipw["standard_error"]
        ),
        "standard_error_percentage_points": (
            aipw[
                "standard_error_percentage_points"
            ]
        ),
        "confidence_level": (
            aipw["confidence_level"]
        ),
        "confidence_interval": (
            aipw["confidence_interval"]
        ),
        "confidence_interval_percentage_points": (
            aipw[
                "confidence_interval_percentage_points"
            ]
        ),
        "outcome_model_component": {
            "mean_mu0": aipw["mean_mu0"],
            "mean_mu1": aipw["mean_mu1"],
            "mean_mu1_minus_mu0": (
                aipw[
                    "mean_outcome_model_effect"
                ]
            ),
        },
        "comparison": {
            "naive_difference": (
                naive_difference
            ),
            "naive_difference_percentage_points": (
                naive_difference * 100.0
            ),
            "ipw_ate": ipw_ate,
            "ipw_ate_percentage_points": (
                ipw_ate * 100.0
            ),
            "aipw_minus_ipw": (
                aipw_ate - ipw_ate
            ),
            "aipw_minus_ipw_percentage_points": (
                (aipw_ate - ipw_ate) * 100.0
            ),
            "aipw_minus_naive": (
                aipw_ate - naive_difference
            ),
            "aipw_minus_naive_percentage_points": (
                (aipw_ate - naive_difference) * 100.0
            ),
        },
        "uncertainty_method": (
            "influence_function_normal_approximation"
        ),
        "interpretation": (
            "Under the stated identification assumptions, the AIPW "
            "estimate combines the previously estimated propensity "
            "scores with treatment-specific outcome models to estimate "
            "the average treatment effect."
        ),
        "doubly_robust_interpretation": (
            "Doubly robust means the estimator can remain consistent "
            "if either the propensity model or the outcome model is "
            "correctly specified, under the required causal assumptions "
            "and regularity conditions. It does not protect against "
            "unmeasured confounding, positivity violations, incorrect "
            "treatment/outcome timing, or other design failures."
        ),
        "limitations": [
            (
                "The current AIPW implementation is not cross-fitted. "
                "Cross-fitting will be evaluated at a later checkpoint "
                "after the basic doubly robust estimator is validated."
            ),
            (
                "The influence-function confidence interval is a "
                "normal approximation."
            ),
            (
                "Observed balance and overlap do not prove absence "
                "of unmeasured confounding."
            ),
            (
                "Synthetic ground-truth propensity and treatment-effect "
                "fields are not used to fit the estimator."
            ),
        ],
    }


def save_outputs(
    scores: pd.DataFrame,
    summary: dict,
    control_model: Pipeline,
    treated_model: Pipeline,
) -> None:
    AIPW_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores.to_csv(
        AIPW_SCORES_PATH,
        index=False,
    )

    with AIPW_SUMMARY_PATH.open(
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

    joblib.dump(
        control_model,
        CONTROL_MODEL_PATH,
    )

    joblib.dump(
        treated_model,
        TREATED_MODEL_PATH,
    )


def print_summary(
    summary: dict,
) -> None:
    interval = summary[
        "confidence_interval"
    ]

    interval_pp = summary[
        "confidence_interval_percentage_points"
    ]

    comparison = summary[
        "comparison"
    ]

    print(
        "Augmented Inverse Probability Weighting"
    )
    print(
        "---------------------------------------"
    )

    print(
        f"Estimand: {summary['estimand']}"
    )

    print(
        f"Estimator: {summary['estimator']}"
    )

    print(
        f"Observations: {summary['n_observations']:,}"
    )

    print(
        f"Cross-fitted: {summary['cross_fitted']}"
    )

    print(
        "\nOutcome-model component"
    )

    print(
        "Mean predicted outcome under control: "
        f"{summary['outcome_model_component']['mean_mu0']:.6f}"
    )

    print(
        "Mean predicted outcome under treatment: "
        f"{summary['outcome_model_component']['mean_mu1']:.6f}"
    )

    print(
        "Mean outcome-model contrast: "
        f"{summary['outcome_model_component']['mean_mu1_minus_mu0']:.6f}"
    )

    print(
        "\nAIPW-adjusted ATE"
    )

    print(
        f"{summary['aipw_ate']:.6f} "
        f"({summary['aipw_ate_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "\nInfluence-function uncertainty"
    )

    print(
        "Standard error: "
        f"{summary['standard_error']:.6f} "
        f"({summary['standard_error_percentage_points']:.2f} "
        "percentage points)"
    )

    print(
        f"{int(summary['confidence_level'] * 100)}% CI: "
        f"[{interval['lower']:.6f}, "
        f"{interval['upper']:.6f}] "
        f"([{interval_pp['lower']:+.2f}, "
        f"{interval_pp['upper']:+.2f}] "
        "percentage points)"
    )

    print(
        "\nEstimator comparison"
    )

    print(
        "Naive difference: "
        f"{comparison['naive_difference']:.6f} "
        f"({comparison['naive_difference_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "IPW ATE: "
        f"{comparison['ipw_ate']:.6f} "
        f"({comparison['ipw_ate_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "AIPW - IPW: "
        f"{comparison['aipw_minus_ipw']:.6f} "
        f"({comparison['aipw_minus_ipw_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "\nArtifacts saved:"
    )

    print(
        f"- {AIPW_SCORES_PATH}"
    )

    print(
        f"- {AIPW_SUMMARY_PATH}"
    )

    print(
        f"- {CONTROL_MODEL_PATH}"
    )

    print(
        f"- {TREATED_MODEL_PATH}"
    )

    print(
        "\nDoubly robust does not mean assumption-proof."
    )

    print(
        summary["doubly_robust_interpretation"]
    )


def main() -> None:
    data = load_data()

    propensity = load_propensity_scores()

    aligned = align_data_and_propensity(
        data=data,
        propensity=propensity,
    )

    (
        control_model,
        treated_model,
        mu0,
        mu1,
    ) = fit_outcome_models(
        data=aligned,
    )

    scores = calculate_aipw_scores(
        data=aligned,
        mu0=mu0,
        mu1=mu1,
    )

    aipw = summarize_aipw(
        scores=scores,
    )

    baseline = load_json_artifact(
        path=BASELINE_PATH,
        label="naive baseline",
    )

    ipw = load_json_artifact(
        path=IPW_SUMMARY_PATH,
        label="IPW ATE",
    )

    summary = build_summary(
        aipw=aipw,
        baseline=baseline,
        ipw=ipw,
        n_observations=len(aligned),
    )

    save_outputs(
        scores=scores,
        summary=summary,
        control_model=control_model,
        treated_model=treated_model,
    )

    print_summary(
        summary=summary,
    )


if __name__ == "__main__":
    main()
