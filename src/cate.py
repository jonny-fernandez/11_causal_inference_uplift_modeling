from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.aipw_crossfit import (
    calculate_crossfit_aipw_scores,
    generate_oof_nuisance_predictions,
)
from src.data_contract import load_data
from src.propensity import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


ID_COLUMN = "person_id"
TREATMENT_COLUMN = "training_assigned"
TRUE_ITE_COLUMN = "true_individual_treatment_effect"

DEVELOPMENT_FRACTION = 0.80
RANDOM_STATE = 42

CATE_MODEL_PATH = Path("artifacts/cate_dr_learner.joblib")
CATE_SCORES_PATH = Path("artifacts/cate_evaluation_scores.csv")
CATE_SUMMARY_PATH = Path("artifacts/cate_evaluation_summary.json")
CATE_SPLIT_PATH = Path("artifacts/cate_split_assignments.csv")


def validate_input(data: pd.DataFrame) -> None:
    required_columns = {
        ID_COLUMN,
        TREATMENT_COLUMN,
        TRUE_ITE_COLUMN,
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }

    missing = sorted(required_columns.difference(data.columns))

    if missing:
        raise ValueError(
            "Source data is missing required CATE columns: "
            + ", ".join(missing)
        )

    if data[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Source data contains duplicate person IDs."
        )

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    if data[feature_columns].isna().any().any():
        missing_counts = (
            data[feature_columns]
            .isna()
            .sum()
            .loc[lambda x: x > 0]
            .to_dict()
        )
        raise ValueError(
            "CATE covariates contain missing values: "
            f"{missing_counts}"
        )

    treatment_values = set(
        data[TREATMENT_COLUMN]
        .dropna()
        .unique()
    )

    if treatment_values != {0, 1}:
        raise ValueError(
            f"{TREATMENT_COLUMN} must contain both 0 and 1. "
            f"Found: {sorted(treatment_values)}"
        )


def make_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    development, evaluation = train_test_split(
        data,
        train_size=DEVELOPMENT_FRACTION,
        random_state=RANDOM_STATE,
        stratify=data[TREATMENT_COLUMN],
    )

    development = development.copy().reset_index(drop=True)
    evaluation = evaluation.copy().reset_index(drop=True)

    development_ids = set(development[ID_COLUMN])
    evaluation_ids = set(evaluation[ID_COLUMN])

    if development_ids.intersection(evaluation_ids):
        raise RuntimeError(
            "Development and evaluation samples overlap."
        )

    split = pd.concat(
        [
            pd.DataFrame(
                {
                    ID_COLUMN: development[ID_COLUMN].to_numpy(),
                    "cate_split": "development",
                }
            ),
            pd.DataFrame(
                {
                    ID_COLUMN: evaluation[ID_COLUMN].to_numpy(),
                    "cate_split": "evaluation",
                }
            ),
        ],
        ignore_index=True,
    )

    if split[ID_COLUMN].duplicated().any():
        raise RuntimeError(
            "CATE split assignments contain duplicate IDs."
        )

    if len(split) != len(data):
        raise RuntimeError(
            "CATE split assignments do not cover the full dataset."
        )

    return development, evaluation, split


def build_cate_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=30,
        max_features=0.80,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def create_development_pseudo_outcomes(
    development: pd.DataFrame,
) -> pd.DataFrame:
    predictions = generate_oof_nuisance_predictions(
        data=development,
    )

    scores = calculate_crossfit_aipw_scores(
        predictions=predictions,
    )

    pseudo_outcomes = scores[
        [
            ID_COLUMN,
            "aipw_pseudo_outcome_crossfit",
        ]
    ].copy()

    if pseudo_outcomes[
        "aipw_pseudo_outcome_crossfit"
    ].isna().any():
        raise ValueError(
            "Development AIPW pseudo-outcomes contain missing values."
        )

    if not np.isfinite(
        pseudo_outcomes[
            "aipw_pseudo_outcome_crossfit"
        ]
    ).all():
        raise ValueError(
            "Development AIPW pseudo-outcomes contain "
            "non-finite values."
        )

    return pseudo_outcomes


def fit_cate_model(
    development: pd.DataFrame,
    pseudo_outcomes: pd.DataFrame,
) -> Pipeline:
    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    training = development.merge(
        pseudo_outcomes,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(training) != len(development):
        raise ValueError(
            "Development pseudo-outcomes do not align "
            "one-to-one with development data."
        )

    model = build_cate_pipeline()

    model.fit(
        training[feature_columns],
        training[
            "aipw_pseudo_outcome_crossfit"
        ].astype(float),
    )

    return model


def predict_evaluation_cate(
    model: Pipeline,
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    estimated_cate = model.predict(
        evaluation[feature_columns]
    )

    if not np.isfinite(estimated_cate).all():
        raise ValueError(
            "Estimated CATE scores contain non-finite values."
        )

    scores = pd.DataFrame(
        {
            ID_COLUMN: evaluation[ID_COLUMN].to_numpy(),
            TREATMENT_COLUMN: evaluation[TREATMENT_COLUMN].to_numpy(),
            "estimated_cate": estimated_cate,
            TRUE_ITE_COLUMN: evaluation[
                TRUE_ITE_COLUMN
            ].astype(float).to_numpy(),
        }
    )

    scores["cate_error"] = (
        scores["estimated_cate"]
        - scores[TRUE_ITE_COLUMN]
    )

    scores["abs_cate_error"] = (
        scores["cate_error"].abs()
    )

    return scores


def rank_correlation(
    x: pd.Series,
    y: pd.Series,
) -> float:
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")

    return float(
        np.corrcoef(
            x_rank.to_numpy(dtype=float),
            y_rank.to_numpy(dtype=float),
        )[0, 1]
    )


def summarize_evaluation(
    scores: pd.DataFrame,
    development_n: int,
    evaluation_n: int,
) -> dict:
    truth = scores[TRUE_ITE_COLUMN].to_numpy(dtype=float)
    prediction = scores["estimated_cate"].to_numpy(dtype=float)
    error = prediction - truth

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))

    pearson = float(
        np.corrcoef(
            truth,
            prediction,
        )[0, 1]
    )

    spearman = rank_correlation(
        scores[TRUE_ITE_COLUMN],
        scores["estimated_cate"],
    )

    true_top_quartile_cutoff = float(
        np.quantile(truth, 0.75)
    )

    estimated_top_quartile_cutoff = float(
        np.quantile(prediction, 0.75)
    )

    true_top = (
        scores[TRUE_ITE_COLUMN]
        >= true_top_quartile_cutoff
    )

    estimated_top = (
        scores["estimated_cate"]
        >= estimated_top_quartile_cutoff
    )

    top_quartile_overlap = float(
        (
            true_top
            & estimated_top
        ).sum()
        / max(1, estimated_top.sum())
    )

    return {
        "method": (
            "dr_learner_random_forest_on_cross_fitted_aipw_pseudo_outcomes"
        ),
        "development_fraction": DEVELOPMENT_FRACTION,
        "random_state": RANDOM_STATE,
        "development_n": int(development_n),
        "evaluation_n": int(evaluation_n),
        "evaluation_only_ground_truth": True,
        "true_ite_used_for_training": False,
        "true_ite_used_for_model_selection": False,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "pehe": rmse,
            "mean_signed_error": bias,
            "pearson_correlation": pearson,
            "spearman_rank_correlation": spearman,
            "top_quartile_overlap": top_quartile_overlap,
        },
        "evaluation_distribution": {
            "estimated_cate_minimum": float(np.min(prediction)),
            "estimated_cate_p05": float(
                np.quantile(prediction, 0.05)
            ),
            "estimated_cate_median": float(np.median(prediction)),
            "estimated_cate_p95": float(
                np.quantile(prediction, 0.95)
            ),
            "estimated_cate_maximum": float(np.max(prediction)),
            "true_ite_minimum": float(np.min(truth)),
            "true_ite_median": float(np.median(truth)),
            "true_ite_maximum": float(np.max(truth)),
        },
        "model_configuration": {
            "regressor": "RandomForestRegressor",
            "n_estimators": 400,
            "max_depth": 6,
            "min_samples_leaf": 30,
            "max_features": 0.80,
        },
        "interpretation": (
            "The CATE model is trained on cross-fitted AIPW "
            "pseudo-outcomes from the development sample only. "
            "The untouched evaluation sample is scored without "
            "using its outcomes or synthetic individual treatment "
            "effects for fitting. Synthetic individual treatment "
            "effects are opened only after prediction to evaluate "
            "CATE recovery."
        ),
    }


def save_outputs(
    model: Pipeline,
    scores: pd.DataFrame,
    summary: dict,
    split: pd.DataFrame,
) -> None:
    CATE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, CATE_MODEL_PATH)

    scores.to_csv(
        CATE_SCORES_PATH,
        index=False,
    )

    split.to_csv(
        CATE_SPLIT_PATH,
        index=False,
    )

    with CATE_SUMMARY_PATH.open(
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
    metrics = summary["metrics"]
    distribution = summary["evaluation_distribution"]

    print("DR-Learner CATE Evaluation")
    print("--------------------------")
    print(
        f"Development rows: "
        f"{summary['development_n']:,}"
    )
    print(
        f"Untouched evaluation rows: "
        f"{summary['evaluation_n']:,}"
    )
    print(
        "True ITE used for training: "
        f"{summary['true_ite_used_for_training']}"
    )

    print("\nEvaluation metrics")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE / PEHE: {metrics['rmse']:.6f}")
    print(
        "Mean signed error: "
        f"{metrics['mean_signed_error']:.6f}"
    )
    print(
        "Pearson correlation: "
        f"{metrics['pearson_correlation']:.6f}"
    )
    print(
        "Spearman rank correlation: "
        f"{metrics['spearman_rank_correlation']:.6f}"
    )
    print(
        "Top-quartile overlap: "
        f"{metrics['top_quartile_overlap']:.2%}"
    )

    print("\nEstimated CATE distribution")
    print(
        "Minimum: "
        f"{distribution['estimated_cate_minimum']:.6f}"
    )
    print(
        "P05:     "
        f"{distribution['estimated_cate_p05']:.6f}"
    )
    print(
        "Median:  "
        f"{distribution['estimated_cate_median']:.6f}"
    )
    print(
        "P95:     "
        f"{distribution['estimated_cate_p95']:.6f}"
    )
    print(
        "Maximum: "
        f"{distribution['estimated_cate_maximum']:.6f}"
    )

    print("\nTrue ITE evaluation range")
    print(
        "Minimum: "
        f"{distribution['true_ite_minimum']:.6f}"
    )
    print(
        "Median:  "
        f"{distribution['true_ite_median']:.6f}"
    )
    print(
        "Maximum: "
        f"{distribution['true_ite_maximum']:.6f}"
    )

    print("\nArtifacts saved:")
    print(f"- {CATE_MODEL_PATH}")
    print(f"- {CATE_SCORES_PATH}")
    print(f"- {CATE_SUMMARY_PATH}")
    print(f"- {CATE_SPLIT_PATH}")

    print(
        "\nDo not interpret a high CATE score as a guaranteed "
        "individual outcome. It is an estimated conditional "
        "treatment benefit used for ranking and subgroup analysis."
    )


def main() -> None:
    data = load_data()

    validate_input(data=data)

    development, evaluation, split = make_split(
        data=data
    )

    pseudo_outcomes = create_development_pseudo_outcomes(
        development=development
    )

    model = fit_cate_model(
        development=development,
        pseudo_outcomes=pseudo_outcomes,
    )

    scores = predict_evaluation_cate(
        model=model,
        evaluation=evaluation,
    )

    summary = summarize_evaluation(
        scores=scores,
        development_n=len(development),
        evaluation_n=len(evaluation),
    )

    save_outputs(
        model=model,
        scores=scores,
        summary=summary,
        split=split,
    )

    print_summary(summary=summary)


if __name__ == "__main__":
    main()
