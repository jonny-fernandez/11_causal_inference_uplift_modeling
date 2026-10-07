from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_contract import load_data
from src.propensity import CATEGORICAL_FEATURES, NUMERIC_FEATURES


ID_COLUMN = "person_id"
TREATMENT_COLUMN = "training_assigned"
OUTCOME_COLUMN = "retained_or_promoted_180d"
TRUE_ITE_COLUMN = "true_individual_treatment_effect"

ORIGINAL_SPLIT_PATH = Path("artifacts/cate_split_assignments.csv")
ORIGINAL_RF_SCORES_PATH = Path("artifacts/cate_evaluation_scores.csv")
REGULARIZED_SCORES_PATH = Path(
    "artifacts/cate_regularized_validation_scores.csv"
)

MODEL_PATH = Path("artifacts/cate_slearner_logistic.joblib")
VALIDATION_SCORES_PATH = Path(
    "artifacts/cate_slearner_validation_scores.csv"
)
SUMMARY_PATH = Path(
    "artifacts/cate_slearner_validation_summary.json"
)
LOCKED_SPLIT_PATH = Path(
    "artifacts/cate_final_evaluation_split.csv"
)

FINAL_HOLDOUT_FRACTION_OF_ORIGINAL_DEVELOPMENT = 0.25
FINAL_HOLDOUT_RANDOM_STATE = 137
MODEL_RANDOM_STATE = 42


def load_original_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the original CATE development/validation split.

    The original 1,000-row evaluation sample has already been inspected and is
    therefore treated as consumed validation. It is used only for method
    comparison, not as the final untouched evaluation sample.
    """

    if not ORIGINAL_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing split artifact: {ORIGINAL_SPLIT_PATH}. "
            "Run `python -m src.cate` first."
        )

    split = pd.read_csv(ORIGINAL_SPLIT_PATH)

    required = {
        ID_COLUMN,
        "cate_split",
    }

    missing = sorted(required.difference(split.columns))

    if missing:
        raise ValueError(
            "Original CATE split is missing required columns: "
            + ", ".join(missing)
        )

    merged = data.merge(
        split,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(data):
        raise ValueError(
            "Original CATE split does not cover the full dataset."
        )

    development = (
        merged.loc[
            merged["cate_split"] == "development"
        ]
        .drop(columns=["cate_split"])
        .reset_index(drop=True)
    )

    consumed_validation = (
        merged.loc[
            merged["cate_split"] == "evaluation"
        ]
        .drop(columns=["cate_split"])
        .reset_index(drop=True)
    )

    if development.empty or consumed_validation.empty:
        raise ValueError(
            "Both original development and validation samples are required."
        )

    return development, consumed_validation


def lock_future_final_holdout(
    development: pd.DataFrame,
    consumed_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a new untouched final evaluation sample.

    The original development sample has never been evaluated against its
    synthetic individual treatment-effect truth. We now reserve 25% of that
    original development set as the future final CATE/uplift evaluation sample.

    This checkpoint DOES NOT score or evaluate that holdout.
    """

    model_development, final_holdout = train_test_split(
        development,
        test_size=FINAL_HOLDOUT_FRACTION_OF_ORIGINAL_DEVELOPMENT,
        random_state=FINAL_HOLDOUT_RANDOM_STATE,
        stratify=development[TREATMENT_COLUMN],
    )

    model_development = (
        model_development
        .copy()
        .reset_index(drop=True)
    )

    final_holdout = (
        final_holdout
        .copy()
        .reset_index(drop=True)
    )

    if set(model_development[ID_COLUMN]).intersection(
        set(final_holdout[ID_COLUMN])
    ):
        raise RuntimeError(
            "Model-development and final-holdout IDs overlap."
        )

    if set(consumed_validation[ID_COLUMN]).intersection(
        set(final_holdout[ID_COLUMN])
    ):
        raise RuntimeError(
            "Consumed validation and final-holdout IDs overlap."
        )

    split = pd.concat(
        [
            pd.DataFrame(
                {
                    ID_COLUMN:
                        model_development[ID_COLUMN].to_numpy(),
                    "cate_role":
                        "model_development",
                }
            ),
            pd.DataFrame(
                {
                    ID_COLUMN:
                        consumed_validation[ID_COLUMN].to_numpy(),
                    "cate_role":
                        "consumed_validation",
                }
            ),
            pd.DataFrame(
                {
                    ID_COLUMN:
                        final_holdout[ID_COLUMN].to_numpy(),
                    "cate_role":
                        "future_final_holdout",
                }
            ),
        ],
        ignore_index=True,
    )

    if split[ID_COLUMN].duplicated().any():
        raise RuntimeError(
            "Locked CATE split contains duplicate IDs."
        )

    if len(split) != (
        len(model_development)
        + len(consumed_validation)
        + len(final_holdout)
    ):
        raise RuntimeError(
            "Locked split row count is inconsistent."
        )

    return (
        model_development,
        final_holdout,
        split,
    )


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
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


def augment_with_treatment_interactions(
    base_features: np.ndarray,
    treatment: np.ndarray,
) -> np.ndarray:
    treatment = np.asarray(
        treatment,
        dtype=float,
    ).reshape(-1, 1)

    if len(base_features) != len(treatment):
        raise ValueError(
            "Base features and treatment lengths do not match."
        )

    interactions = (
        base_features
        * treatment
    )

    return np.hstack(
        [
            base_features,
            treatment,
            interactions,
        ]
    )


def fit_slearner(
    development: pd.DataFrame,
) -> dict:
    """Fit a regularized S-learner with treatment interactions.

    This method is justified because both DR-learner second-stage approaches
    produced unstable/noisy heterogeneous-effect rankings. The S-learner fits
    the observed binary outcome directly and uses explicit treatment-by-
    covariate interactions to represent effect modification.
    """

    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    preprocessor = build_preprocessor()

    base_features = preprocessor.fit_transform(
        development[feature_columns]
    )

    treatment = (
        development[TREATMENT_COLUMN]
        .astype(int)
        .to_numpy()
    )

    outcome = (
        development[OUTCOME_COLUMN]
        .astype(int)
        .to_numpy()
    )

    design = augment_with_treatment_interactions(
        base_features=base_features,
        treatment=treatment,
    )

    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 10),
        cv=5,
        scoring="neg_log_loss",
        penalty="l2",
        solver="lbfgs",
        max_iter=4000,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    model.fit(
        design,
        outcome,
    )

    return {
        "preprocessor": preprocessor,
        "model": model,
        "feature_columns": feature_columns,
    }


def predict_cate(
    fitted: dict,
    data: pd.DataFrame,
) -> np.ndarray:
    preprocessor = fitted[
        "preprocessor"
    ]

    model = fitted[
        "model"
    ]

    feature_columns = fitted[
        "feature_columns"
    ]

    base_features = preprocessor.transform(
        data[feature_columns]
    )

    treatment_zero = np.zeros(
        len(data),
        dtype=int,
    )

    treatment_one = np.ones(
        len(data),
        dtype=int,
    )

    design_zero = augment_with_treatment_interactions(
        base_features=base_features,
        treatment=treatment_zero,
    )

    design_one = augment_with_treatment_interactions(
        base_features=base_features,
        treatment=treatment_one,
    )

    p0 = model.predict_proba(
        design_zero
    )[:, 1]

    p1 = model.predict_proba(
        design_one
    )[:, 1]

    estimated_cate = (
        p1 - p0
    )

    if not np.isfinite(
        estimated_cate
    ).all():
        raise ValueError(
            "S-learner CATE predictions contain non-finite values."
        )

    return estimated_cate


def rank_correlation(
    x: pd.Series,
    y: pd.Series,
) -> float:
    x_rank = x.rank(
        method="average"
    )

    y_rank = y.rank(
        method="average"
    )

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
        "top_true_ite_mean":
            top_true_mean,
        "bottom_true_ite_mean":
            bottom_true_mean,
        "spread":
            top_true_mean
            - bottom_true_mean,
    }


def score_consumed_validation(
    fitted: dict,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    estimated_cate = predict_cate(
        fitted=fitted,
        data=validation,
    )

    scores = pd.DataFrame(
        {
            ID_COLUMN:
                validation[ID_COLUMN].to_numpy(),
            TREATMENT_COLUMN:
                validation[
                    TREATMENT_COLUMN
                ].to_numpy(),
            "estimated_cate_slearner":
                estimated_cate,
            TRUE_ITE_COLUMN:
                validation[
                    TRUE_ITE_COLUMN
                ].astype(float).to_numpy(),
        }
    )

    scores["cate_error"] = (
        scores[
            "estimated_cate_slearner"
        ]
        - scores[
            TRUE_ITE_COLUMN
        ]
    )

    scores["abs_cate_error"] = (
        scores["cate_error"].abs()
    )

    return scores


def summarize_scores(
    scores: pd.DataFrame,
    estimate_column: str,
) -> dict:
    truth = scores[
        TRUE_ITE_COLUMN
    ].to_numpy(dtype=float)

    prediction = scores[
        estimate_column
    ].to_numpy(dtype=float)

    error = (
        prediction - truth
    )

    spread = top_bottom_decile_spread(
        scores=scores,
        estimate_column=estimate_column,
    )

    true_std = float(
        np.std(
            truth,
            ddof=1,
        )
    )

    estimated_std = float(
        np.std(
            prediction,
            ddof=1,
        )
    )

    return {
        "mae": float(
            np.mean(
                np.abs(error)
            )
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
        "spearman_rank_correlation": (
            rank_correlation(
                scores[estimate_column],
                scores[
                    TRUE_ITE_COLUMN
                ],
            )
        ),
        "estimated_mean": float(
            np.mean(prediction)
        ),
        "true_mean": float(
            np.mean(truth)
        ),
        "estimated_std": (
            estimated_std
        ),
        "true_std": (
            true_std
        ),
        "estimated_to_true_std_ratio": float(
            estimated_std / true_std
        ),
        "estimated_minimum": float(
            np.min(prediction)
        ),
        "estimated_maximum": float(
            np.max(prediction)
        ),
        "top_estimated_decile_true_ite_mean": (
            spread[
                "top_true_ite_mean"
            ]
        ),
        "bottom_estimated_decile_true_ite_mean": (
            spread[
                "bottom_true_ite_mean"
            ]
        ),
        "top_minus_bottom_true_ite_spread": (
            spread["spread"]
        ),
    }


def load_prior_validation_results() -> dict:
    result: dict = {}

    if ORIGINAL_RF_SCORES_PATH.exists():
        rf = pd.read_csv(
            ORIGINAL_RF_SCORES_PATH
        )

        result[
            "random_forest_dr_learner"
        ] = summarize_scores(
            scores=rf,
            estimate_column="estimated_cate",
        )

    if REGULARIZED_SCORES_PATH.exists():
        ridge = pd.read_csv(
            REGULARIZED_SCORES_PATH
        )

        result[
            "regularized_dr_learner"
        ] = summarize_scores(
            scores=ridge,
            estimate_column="estimated_cate_regularized",
        )

    return result


def build_summary(
    fitted: dict,
    scores: pd.DataFrame,
    prior: dict,
    model_development_n: int,
    consumed_validation_n: int,
    future_final_holdout_n: int,
) -> dict:
    current = summarize_scores(
        scores=scores,
        estimate_column="estimated_cate_slearner",
    )

    model = fitted[
        "model"
    ]

    selected_c = float(
        np.ravel(model.C_)[0]
    )

    return {
        "method": (
            "regularized_logistic_s_learner_with_explicit_"
            "treatment_covariate_interactions"
        ),
        "model_development_n":
            int(model_development_n),
        "consumed_validation_n":
            int(consumed_validation_n),
        "future_final_holdout_n":
            int(future_final_holdout_n),
        "future_final_holdout_scored":
            False,
        "true_ite_used_for_training":
            False,
        "selected_logistic_c":
            selected_c,
        "current_validation":
            current,
        "prior_validation":
            prior,
        "reason_for_method_change": (
            "Both DR-learner second-stage models failed to recover "
            "heterogeneous-effect ranking on the consumed validation sample. "
            "The random forest had near-zero rank correlation and excess "
            "dispersion; the regularized pseudo-outcome regression worsened "
            "both ranking and dispersion. The S-learner therefore changes "
            "the learning target rather than continuing to regress noisy "
            "AIPW pseudo-outcomes: it models the observed binary outcome "
            "directly with explicit treatment-by-covariate interactions."
        ),
        "evaluation_policy": (
            "The existing 1,000-row sample is used only as consumed "
            "validation for method comparison. A new 1,000-row final "
            "holdout was reserved from the original development sample "
            "before fitting this S-learner and is not scored in this "
            "checkpoint. If the S-learner is accepted and methodology is "
            "locked, the next checkpoint will evaluate it once on that "
            "untouched final holdout."
        ),
    }


def save_outputs(
    fitted: dict,
    scores: pd.DataFrame,
    summary: dict,
    locked_split: pd.DataFrame,
) -> None:
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        fitted,
        MODEL_PATH,
    )

    scores.to_csv(
        VALIDATION_SCORES_PATH,
        index=False,
    )

    locked_split.to_csv(
        LOCKED_SPLIT_PATH,
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
    current = summary[
        "current_validation"
    ]

    print(
        "S-Learner CATE Validation"
    )
    print(
        "-------------------------"
    )

    print(
        f"Model-development rows: "
        f"{summary['model_development_n']:,}"
    )

    print(
        f"Consumed validation rows: "
        f"{summary['consumed_validation_n']:,}"
    )

    print(
        f"Locked future final holdout rows: "
        f"{summary['future_final_holdout_n']:,}"
    )

    print(
        "Future final holdout scored: "
        f"{summary['future_final_holdout_scored']}"
    )

    print(
        "True ITE used for training: "
        f"{summary['true_ite_used_for_training']}"
    )

    print(
        f"Selected logistic C: "
        f"{summary['selected_logistic_c']:.6f}"
    )

    print(
        "\nReason for method change"
    )
    print(
        summary[
            "reason_for_method_change"
        ]
    )

    print(
        "\nS-Learner validation metrics"
    )

    print(
        f"MAE: "
        f"{current['mae']:.6f}"
    )

    print(
        f"RMSE / PEHE: "
        f"{current['rmse_pehe']:.6f}"
    )

    print(
        f"Pearson: "
        f"{current['pearson_correlation']:.6f}"
    )

    print(
        f"Spearman: "
        f"{current['spearman_rank_correlation']:.6f}"
    )

    print(
        "Dispersion ratio: "
        f"{current['estimated_to_true_std_ratio']:.2f}x"
    )

    print(
        "Top-bottom true ITE spread: "
        f"{current['top_minus_bottom_true_ite_spread']:.6f}"
    )

    print(
        "Estimated CATE range: "
        f"[{current['estimated_minimum']:.6f}, "
        f"{current['estimated_maximum']:.6f}]"
    )

    prior = summary[
        "prior_validation"
    ]

    if prior:
        print(
            "\nPrior-model comparison"
        )

        for name, metrics in prior.items():
            print(
                f"{name}: "
                f"MAE={metrics['mae']:.6f}, "
                f"RMSE={metrics['rmse_pehe']:.6f}, "
                f"Pearson={metrics['pearson_correlation']:.6f}, "
                f"Spearman={metrics['spearman_rank_correlation']:.6f}, "
                f"spread={metrics['top_minus_bottom_true_ite_spread']:.6f}"
            )

    print(
        "\nArtifacts saved:"
    )

    print(
        f"- {MODEL_PATH}"
    )

    print(
        f"- {VALIDATION_SCORES_PATH}"
    )

    print(
        f"- {SUMMARY_PATH}"
    )

    print(
        f"- {LOCKED_SPLIT_PATH}"
    )

    print(
        "\nEvaluation policy:"
    )

    print(
        summary[
            "evaluation_policy"
        ]
    )


def main() -> None:
    data = load_data()

    (
        original_development,
        consumed_validation,
    ) = load_original_split(
        data=data,
    )

    (
        model_development,
        future_final_holdout,
        locked_split,
    ) = lock_future_final_holdout(
        development=original_development,
        consumed_validation=consumed_validation,
    )

    fitted = fit_slearner(
        development=model_development,
    )

    scores = score_consumed_validation(
        fitted=fitted,
        validation=consumed_validation,
    )

    prior = load_prior_validation_results()

    summary = build_summary(
        fitted=fitted,
        scores=scores,
        prior=prior,
        model_development_n=len(
            model_development
        ),
        consumed_validation_n=len(
            consumed_validation
        ),
        future_final_holdout_n=len(
            future_final_holdout
        ),
    )

    save_outputs(
        fitted=fitted,
        scores=scores,
        summary=summary,
        locked_split=locked_split,
    )

    print_summary(
        summary=summary,
    )


if __name__ == "__main__":
    main()
