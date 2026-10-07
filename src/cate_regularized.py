from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

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

SPLIT_PATH = Path("artifacts/cate_split_assignments.csv")
ORIGINAL_SCORES_PATH = Path("artifacts/cate_evaluation_scores.csv")

MODEL_PATH = Path("artifacts/cate_regularized_dr_learner.joblib")
SCORES_PATH = Path("artifacts/cate_regularized_validation_scores.csv")
SUMMARY_PATH = Path("artifacts/cate_regularized_validation_summary.json")

ALPHAS = np.logspace(-2, 3, 11)


def load_locked_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the existing CATE split for a fair development comparison.

    The original 1,000-row evaluation sample has already been inspected,
    so it is treated as a validation sample from this checkpoint forward.
    It is not described as the final untouched evaluation sample.
    """

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing CATE split artifact: {SPLIT_PATH}. "
            "Run `python -m src.cate` first."
        )

    split = pd.read_csv(SPLIT_PATH)

    required = {
        ID_COLUMN,
        "cate_split",
    }

    missing = sorted(required.difference(split.columns))

    if missing:
        raise ValueError(
            "CATE split artifact is missing required columns: "
            + ", ".join(missing)
        )

    if split[ID_COLUMN].duplicated().any():
        raise ValueError(
            "CATE split artifact contains duplicate person IDs."
        )

    merged = data.merge(
        split,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(data):
        raise ValueError(
            "CATE split assignments do not cover the full source dataset."
        )

    development = (
        merged.loc[
            merged["cate_split"] == "development"
        ]
        .drop(columns=["cate_split"])
        .reset_index(drop=True)
    )

    validation = (
        merged.loc[
            merged["cate_split"] == "evaluation"
        ]
        .drop(columns=["cate_split"])
        .reset_index(drop=True)
    )

    if development.empty or validation.empty:
        raise ValueError(
            "Both development and validation samples are required."
        )

    return development, validation


def build_regularized_cate_pipeline() -> Pipeline:
    """Build a lower-variance DR-learner second stage.

    The failed random-forest second stage produced treatment-effect scores
    that were nearly uncorrelated with synthetic truth and almost twice as
    dispersed as the true effects. This replacement keeps the DR-learner
    framework but uses shrinkage and pairwise interactions to reduce variance
    while still allowing interpretable effect heterogeneity.
    """

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
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
        verbose_feature_names_out=False,
    )

    interactions = PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False,
    )

    ridge = RidgeCV(
        alphas=ALPHAS,
        cv=5,
        scoring="neg_mean_squared_error",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("interactions", interactions),
            ("ridge", ridge),
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

    result = scores[
        [
            ID_COLUMN,
            "aipw_pseudo_outcome_crossfit",
        ]
    ].copy()

    if result[
        "aipw_pseudo_outcome_crossfit"
    ].isna().any():
        raise ValueError(
            "Development pseudo-outcomes contain missing values."
        )

    if not np.isfinite(
        result["aipw_pseudo_outcome_crossfit"]
    ).all():
        raise ValueError(
            "Development pseudo-outcomes contain non-finite values."
        )

    return result


def fit_regularized_cate(
    development: pd.DataFrame,
    pseudo_outcomes: pd.DataFrame,
) -> Pipeline:
    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    training = development.merge(
        pseudo_outcomes,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(training) != len(development):
        raise ValueError(
            "Development pseudo-outcomes do not align one-to-one."
        )

    model = build_regularized_cate_pipeline()

    model.fit(
        training[feature_columns],
        training[
            "aipw_pseudo_outcome_crossfit"
        ].astype(float),
    )

    return model


def score_validation(
    model: Pipeline,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    estimate = model.predict(
        validation[feature_columns]
    )

    if not np.isfinite(estimate).all():
        raise ValueError(
            "Regularized CATE predictions contain non-finite values."
        )

    result = pd.DataFrame(
        {
            ID_COLUMN:
                validation[ID_COLUMN].to_numpy(),
            TREATMENT_COLUMN:
                validation[TREATMENT_COLUMN].to_numpy(),
            "estimated_cate_regularized":
                estimate,
            TRUE_ITE_COLUMN:
                validation[
                    TRUE_ITE_COLUMN
                ].astype(float).to_numpy(),
        }
    )

    result["cate_error"] = (
        result["estimated_cate_regularized"]
        - result[TRUE_ITE_COLUMN]
    )

    result["abs_cate_error"] = (
        result["cate_error"].abs()
    )

    return result


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


def top_bottom_decile_spread(
    scores: pd.DataFrame,
    estimate_column: str,
) -> dict:
    ranked = scores.sort_values(
        estimate_column,
        ascending=False,
    ).reset_index(drop=True)

    decile_size = max(
        1,
        len(ranked) // 10,
    )

    top_true_mean = float(
        ranked.head(decile_size)[
            TRUE_ITE_COLUMN
        ].mean()
    )

    bottom_true_mean = float(
        ranked.tail(decile_size)[
            TRUE_ITE_COLUMN
        ].mean()
    )

    return {
        "top_true_ite_mean": top_true_mean,
        "bottom_true_ite_mean": bottom_true_mean,
        "spread": (
            top_true_mean
            - bottom_true_mean
        ),
    }


def summarize_model(
    model: Pipeline,
    scores: pd.DataFrame,
) -> dict:
    truth = scores[
        TRUE_ITE_COLUMN
    ].to_numpy(dtype=float)

    prediction = scores[
        "estimated_cate_regularized"
    ].to_numpy(dtype=float)

    error = prediction - truth

    ridge = model.named_steps["ridge"]

    spread = top_bottom_decile_spread(
        scores=scores,
        estimate_column="estimated_cate_regularized",
    )

    return {
        "method": (
            "regularized_dr_learner_with_pairwise_interactions"
        ),
        "validation_rows": int(len(scores)),
        "true_ite_used_for_training": False,
        "selected_ridge_alpha": float(
            ridge.alpha_
        ),
        "metrics": {
            "mae": float(
                np.mean(np.abs(error))
            ),
            "rmse_pehe": float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),
            "mean_signed_error": float(
                np.mean(error)
            ),
            "pearson_correlation": float(
                np.corrcoef(
                    prediction,
                    truth,
                )[0, 1]
            ),
            "spearman_rank_correlation": rank_correlation(
                scores[
                    "estimated_cate_regularized"
                ],
                scores[TRUE_ITE_COLUMN],
            ),
        },
        "dispersion": {
            "estimated_mean": float(
                np.mean(prediction)
            ),
            "true_mean": float(
                np.mean(truth)
            ),
            "estimated_std": float(
                np.std(
                    prediction,
                    ddof=1,
                )
            ),
            "true_std": float(
                np.std(
                    truth,
                    ddof=1,
                )
            ),
            "estimated_to_true_std_ratio": float(
                np.std(
                    prediction,
                    ddof=1,
                )
                / np.std(
                    truth,
                    ddof=1,
                )
            ),
            "estimated_minimum": float(
                np.min(prediction)
            ),
            "estimated_maximum": float(
                np.max(prediction)
            ),
        },
        "ranking": {
            "top_estimated_decile_true_ite_mean": (
                spread["top_true_ite_mean"]
            ),
            "bottom_estimated_decile_true_ite_mean": (
                spread["bottom_true_ite_mean"]
            ),
            "top_minus_bottom_true_ite_spread": (
                spread["spread"]
            ),
        },
    }


def load_original_scores() -> pd.DataFrame:
    if not ORIGINAL_SCORES_PATH.exists():
        raise FileNotFoundError(
            f"Missing original CATE scores: {ORIGINAL_SCORES_PATH}"
        )

    original = pd.read_csv(
        ORIGINAL_SCORES_PATH
    )

    required = {
        ID_COLUMN,
        "estimated_cate",
        TRUE_ITE_COLUMN,
    }

    missing = sorted(
        required.difference(
            original.columns
        )
    )

    if missing:
        raise ValueError(
            "Original CATE scores are missing columns: "
            + ", ".join(missing)
        )

    return original


def summarize_original(
    original: pd.DataFrame,
) -> dict:
    truth = original[
        TRUE_ITE_COLUMN
    ].to_numpy(dtype=float)

    prediction = original[
        "estimated_cate"
    ].to_numpy(dtype=float)

    error = prediction - truth

    spread = top_bottom_decile_spread(
        scores=original,
        estimate_column="estimated_cate",
    )

    return {
        "method": "original_random_forest_dr_learner",
        "metrics": {
            "mae": float(
                np.mean(np.abs(error))
            ),
            "rmse_pehe": float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),
            "pearson_correlation": float(
                np.corrcoef(
                    prediction,
                    truth,
                )[0, 1]
            ),
            "spearman_rank_correlation": rank_correlation(
                original["estimated_cate"],
                original[TRUE_ITE_COLUMN],
            ),
        },
        "dispersion": {
            "estimated_std": float(
                np.std(
                    prediction,
                    ddof=1,
                )
            ),
            "true_std": float(
                np.std(
                    truth,
                    ddof=1,
                )
            ),
            "estimated_to_true_std_ratio": float(
                np.std(
                    prediction,
                    ddof=1,
                )
                / np.std(
                    truth,
                    ddof=1,
                )
            ),
        },
        "ranking": {
            "top_estimated_decile_true_ite_mean": (
                spread["top_true_ite_mean"]
            ),
            "bottom_estimated_decile_true_ite_mean": (
                spread["bottom_true_ite_mean"]
            ),
            "top_minus_bottom_true_ite_spread": (
                spread["spread"]
            ),
        },
    }


def build_comparison(
    regularized: dict,
    original: dict,
) -> dict:
    return {
        "development_status": (
            "The original 1,000-row CATE evaluation sample has been "
            "consumed for model diagnosis and is treated as validation "
            "from this checkpoint forward, not as the final untouched "
            "evaluation sample."
        ),
        "reason_for_change": (
            "The original random-forest second stage had near-zero "
            "rank correlation with true synthetic treatment effects, "
            "1.91x excess dispersion, negative effect predictions even "
            "though the synthetic true effects were positive, and only "
            "3.09% of the oracle top-minus-bottom decile spread. "
            "The corrective change therefore targets variance reduction "
            "through regularization rather than adding a new meta-learner."
        ),
        "original": original,
        "regularized": regularized,
        "change": {
            "mae": (
                regularized["metrics"]["mae"]
                - original["metrics"]["mae"]
            ),
            "rmse_pehe": (
                regularized["metrics"]["rmse_pehe"]
                - original["metrics"]["rmse_pehe"]
            ),
            "pearson_correlation": (
                regularized["metrics"]["pearson_correlation"]
                - original["metrics"]["pearson_correlation"]
            ),
            "spearman_rank_correlation": (
                regularized["metrics"]["spearman_rank_correlation"]
                - original["metrics"]["spearman_rank_correlation"]
            ),
            "dispersion_ratio": (
                regularized["dispersion"][
                    "estimated_to_true_std_ratio"
                ]
                - original["dispersion"][
                    "estimated_to_true_std_ratio"
                ]
            ),
            "top_minus_bottom_true_ite_spread": (
                regularized["ranking"][
                    "top_minus_bottom_true_ite_spread"
                ]
                - original["ranking"][
                    "top_minus_bottom_true_ite_spread"
                ]
            ),
        },
        "selection_policy": (
            "Synthetic truth is used to evaluate the documented corrective "
            "change, not to repeatedly search over many CATE algorithms. "
            "No further method should be added unless this regularized "
            "replacement also fails in a way that justifies another change."
        ),
    }


def save_outputs(
    model: Pipeline,
    scores: pd.DataFrame,
    summary: dict,
) -> None:
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    scores.to_csv(
        SCORES_PATH,
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
    summary: dict,
) -> None:
    original = summary["original"]
    regularized = summary["regularized"]
    change = summary["change"]

    print("Regularized DR-Learner CATE Validation")
    print("--------------------------------------")
    print(summary["development_status"])

    print("\nCorrective change")
    print(summary["reason_for_change"])

    print("\nOriginal random-forest DR-Learner")
    print(
        f"MAE: {original['metrics']['mae']:.6f}"
    )
    print(
        f"RMSE / PEHE: "
        f"{original['metrics']['rmse_pehe']:.6f}"
    )
    print(
        f"Pearson: "
        f"{original['metrics']['pearson_correlation']:.6f}"
    )
    print(
        f"Spearman: "
        f"{original['metrics']['spearman_rank_correlation']:.6f}"
    )
    print(
        "Dispersion ratio: "
        f"{original['dispersion']['estimated_to_true_std_ratio']:.2f}x"
    )
    print(
        "Top-bottom true ITE spread: "
        f"{original['ranking']['top_minus_bottom_true_ite_spread']:.6f}"
    )

    print("\nRegularized DR-Learner")
    print(
        f"Selected ridge alpha: "
        f"{regularized['selected_ridge_alpha']:.6f}"
    )
    print(
        f"MAE: {regularized['metrics']['mae']:.6f}"
    )
    print(
        f"RMSE / PEHE: "
        f"{regularized['metrics']['rmse_pehe']:.6f}"
    )
    print(
        f"Pearson: "
        f"{regularized['metrics']['pearson_correlation']:.6f}"
    )
    print(
        f"Spearman: "
        f"{regularized['metrics']['spearman_rank_correlation']:.6f}"
    )
    print(
        "Dispersion ratio: "
        f"{regularized['dispersion']['estimated_to_true_std_ratio']:.2f}x"
    )
    print(
        "Top-bottom true ITE spread: "
        f"{regularized['ranking']['top_minus_bottom_true_ite_spread']:.6f}"
    )
    print(
        "Estimated CATE range: "
        f"[{regularized['dispersion']['estimated_minimum']:.6f}, "
        f"{regularized['dispersion']['estimated_maximum']:.6f}]"
    )

    print("\nChange vs original")
    print(
        f"MAE change: "
        f"{change['mae']:+.6f}"
    )
    print(
        f"RMSE change: "
        f"{change['rmse_pehe']:+.6f}"
    )
    print(
        "Pearson change: "
        f"{change['pearson_correlation']:+.6f}"
    )
    print(
        "Spearman change: "
        f"{change['spearman_rank_correlation']:+.6f}"
    )
    print(
        "Dispersion-ratio change: "
        f"{change['dispersion_ratio']:+.6f}"
    )
    print(
        "Top-bottom spread change: "
        f"{change['top_minus_bottom_true_ite_spread']:+.6f}"
    )

    print("\nArtifacts saved:")
    print(f"- {MODEL_PATH}")
    print(f"- {SCORES_PATH}")
    print(f"- {SUMMARY_PATH}")

    print("\nSelection policy:")
    print(summary["selection_policy"])


def main() -> None:
    data = load_data()

    development, validation = load_locked_split(
        data=data
    )

    pseudo_outcomes = create_development_pseudo_outcomes(
        development=development
    )

    model = fit_regularized_cate(
        development=development,
        pseudo_outcomes=pseudo_outcomes,
    )

    scores = score_validation(
        model=model,
        validation=validation,
    )

    regularized = summarize_model(
        model=model,
        scores=scores,
    )

    original_scores = load_original_scores()

    original = summarize_original(
        original=original_scores
    )

    summary = build_comparison(
        regularized=regularized,
        original=original,
    )

    save_outputs(
        model=model,
        scores=scores,
        summary=summary,
    )

    print_summary(
        summary=summary,
    )


if __name__ == "__main__":
    main()
