from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_contract import load_data


ID_COLUMN = "person_id"
TREATMENT_COLUMN = "training_assigned"

CATEGORICAL_FEATURES = [
    "department",
    "job_level",
]

NUMERIC_FEATURES = [
    "tenure_months",
    "baseline_performance",
    "baseline_engagement",
    "prior_training_hours_12m",
    "remote_work_ratio",
]

PROPENSITY_OUTPUT_PATH = Path("artifacts/propensity_scores.csv")
DIAGNOSTICS_OUTPUT_PATH = Path("artifacts/propensity_diagnostics.json")
MODEL_OUTPUT_PATH = Path("artifacts/propensity_model.joblib")

RANDOM_STATE = 42


def build_propensity_pipeline() -> Pipeline:
    """Create the deterministic propensity-score pipeline."""

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


def validate_input(df: pd.DataFrame) -> None:
    required_columns = {
        ID_COLUMN,
        TREATMENT_COLUMN,
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }

    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(
            "Missing required propensity-model columns: "
            + ", ".join(missing_columns)
        )

    treatment_values = set(df[TREATMENT_COLUMN].dropna().unique())

    if not treatment_values.issubset({0, 1}):
        raise ValueError(
            f"{TREATMENT_COLUMN} must contain only 0/1 values. "
            f"Found: {sorted(treatment_values)}"
        )

    if df[TREATMENT_COLUMN].isna().any():
        raise ValueError(f"{TREATMENT_COLUMN} contains missing values.")

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    if df[feature_columns].isna().any().any():
        missing = (
            df[feature_columns]
            .isna()
            .sum()
            .loc[lambda x: x > 0]
            .to_dict()
        )

        raise ValueError(
            f"Propensity-model covariates contain missing values: {missing}"
        )


def fit_propensity_model(
    df: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit the propensity model and return deterministic propensity scores."""

    validate_input(df)

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    X = df[feature_columns]
    y = df[TREATMENT_COLUMN]

    pipeline = build_propensity_pipeline()
    pipeline.fit(X, y)

    propensity_scores = pipeline.predict_proba(X)[:, 1]

    scores = pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN].values,
            TREATMENT_COLUMN: df[TREATMENT_COLUMN].values,
            "propensity_score": propensity_scores,
        }
    )

    validate_propensity_scores(scores)

    return pipeline, scores


def validate_propensity_scores(scores: pd.DataFrame) -> None:
    if scores["propensity_score"].isna().any():
        raise ValueError("Propensity scores contain missing values.")

    if not scores["propensity_score"].between(0.0, 1.0).all():
        raise ValueError("Propensity scores must be between 0 and 1.")

    if (scores["propensity_score"] <= 0.0).any():
        raise ValueError(
            "At least one propensity score is exactly zero or negative."
        )

    if (scores["propensity_score"] >= 1.0).any():
        raise ValueError(
            "At least one propensity score is exactly one or greater."
        )


def summarize_group(group: pd.Series) -> dict:
    quantiles = group.quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )

    return {
        "count": int(group.shape[0]),
        "minimum": float(group.min()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "maximum": float(group.max()),
        "mean": float(group.mean()),
        "std": float(group.std()),
    }


def create_diagnostics(scores: pd.DataFrame) -> dict:
    control_scores = scores.loc[
        scores[TREATMENT_COLUMN] == 0,
        "propensity_score",
    ]

    treated_scores = scores.loc[
        scores[TREATMENT_COLUMN] == 1,
        "propensity_score",
    ]

    overlap_lower = max(
        float(control_scores.min()),
        float(treated_scores.min()),
    )

    overlap_upper = min(
        float(control_scores.max()),
        float(treated_scores.max()),
    )

    overall_min = float(scores["propensity_score"].min())
    overall_max = float(scores["propensity_score"].max())

    below_005 = int((scores["propensity_score"] < 0.05).sum())
    above_095 = int((scores["propensity_score"] > 0.95).sum())

    below_010 = int((scores["propensity_score"] < 0.10).sum())
    above_090 = int((scores["propensity_score"] > 0.90).sum())

    return {
        "model": "LogisticRegression",
        "random_state": RANDOM_STATE,
        "treatment_column": TREATMENT_COLUMN,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "n_observations": int(scores.shape[0]),
        "overall": {
            "minimum": overall_min,
            "maximum": overall_max,
            "mean": float(scores["propensity_score"].mean()),
            "std": float(scores["propensity_score"].std()),
        },
        "control": summarize_group(control_scores),
        "treated": summarize_group(treated_scores),
        "empirical_overlap_region": {
            "lower": overlap_lower,
            "upper": overlap_upper,
            "valid": bool(overlap_lower < overlap_upper),
        },
        "extreme_propensity_counts": {
            "below_0_05": below_005,
            "above_0_95": above_095,
            "below_0_10": below_010,
            "above_0_90": above_090,
        },
        "interpretation_policy": (
            "Propensity scores describe estimated treatment-assignment "
            "probabilities conditional on approved pre-treatment covariates. "
            "Classification accuracy is not the primary objective. Overlap "
            "and covariate-balance diagnostics determine whether the scores "
            "are useful for causal adjustment."
        ),
    }


def save_outputs(
    pipeline: Pipeline,
    scores: pd.DataFrame,
    diagnostics: dict,
) -> None:
    PROPENSITY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    scores.to_csv(
        PROPENSITY_OUTPUT_PATH,
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

    joblib.dump(
        pipeline,
        MODEL_OUTPUT_PATH,
    )


def print_diagnostics(diagnostics: dict) -> None:
    overall = diagnostics["overall"]
    control = diagnostics["control"]
    treated = diagnostics["treated"]
    overlap = diagnostics["empirical_overlap_region"]
    extreme = diagnostics["extreme_propensity_counts"]

    print("Propensity-score model")
    print("----------------------")
    print("Model: Logistic Regression")
    print(f"Observations: {diagnostics['n_observations']:,}")

    print("\nOverall propensity scores")
    print(f"Minimum: {overall['minimum']:.6f}")
    print(f"Maximum: {overall['maximum']:.6f}")
    print(f"Mean:    {overall['mean']:.6f}")

    print("\nControl group")
    print(f"Count:   {control['count']:,}")
    print(f"Minimum: {control['minimum']:.6f}")
    print(f"Median:  {control['median']:.6f}")
    print(f"Mean:    {control['mean']:.6f}")
    print(f"Maximum: {control['maximum']:.6f}")

    print("\nTreated group")
    print(f"Count:   {treated['count']:,}")
    print(f"Minimum: {treated['minimum']:.6f}")
    print(f"Median:  {treated['median']:.6f}")
    print(f"Mean:    {treated['mean']:.6f}")
    print(f"Maximum: {treated['maximum']:.6f}")

    print("\nEmpirical overlap region")
    print(
        f"[{overlap['lower']:.6f}, "
        f"{overlap['upper']:.6f}]"
    )
    print(f"Overlap exists: {overlap['valid']}")

    print("\nExtreme propensity counts")
    print(f"< 0.05: {extreme['below_0_05']:,}")
    print(f"> 0.95: {extreme['above_0_95']:,}")
    print(f"< 0.10: {extreme['below_0_10']:,}")
    print(f"> 0.90: {extreme['above_0_90']:,}")

    print("\nArtifacts saved:")
    print(f"- {PROPENSITY_OUTPUT_PATH}")
    print(f"- {DIAGNOSTICS_OUTPUT_PATH}")
    print(f"- {MODEL_OUTPUT_PATH}")

    print(
        "\nDo not interpret propensity-model classification performance "
        "as evidence of causal validity."
    )


def main() -> None:
    df = load_data()

    pipeline, scores = fit_propensity_model(df)

    diagnostics = create_diagnostics(scores)

    save_outputs(
        pipeline=pipeline,
        scores=scores,
        diagnostics=diagnostics,
    )

    print_diagnostics(diagnostics)


if __name__ == "__main__":
    main()