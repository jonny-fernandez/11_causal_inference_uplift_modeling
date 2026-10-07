from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.aipw import build_outcome_pipeline
from src.data_contract import load_data
from src.propensity import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_propensity_pipeline,
)


ID_COLUMN = "person_id"
TREATMENT_COLUMN = "training_assigned"
OUTCOME_COLUMN = "retained_or_promoted_180d"

NON_CROSSFIT_AIPW_PATH = Path("artifacts/aipw_summary.json")
IPW_SUMMARY_PATH = Path("artifacts/ate_summary.json")

CROSSFIT_SCORES_PATH = Path("artifacts/aipw_crossfit_scores.csv")
CROSSFIT_SUMMARY_PATH = Path("artifacts/aipw_crossfit_summary.json")

N_SPLITS = 5
RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95


def validate_input(data: pd.DataFrame) -> None:
    required_columns = {
        ID_COLUMN,
        TREATMENT_COLUMN,
        OUTCOME_COLUMN,
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }

    missing = sorted(required_columns.difference(data.columns))

    if missing:
        raise ValueError(
            "Source data is missing required cross-fitting columns: "
            + ", ".join(missing)
        )

    if data[ID_COLUMN].duplicated().any():
        raise ValueError("Source data contains duplicate person IDs.")

    treatment_values = set(
        data[TREATMENT_COLUMN].dropna().unique()
    )

    if treatment_values != {0, 1}:
        raise ValueError(
            f"{TREATMENT_COLUMN} must contain both 0 and 1. "
            f"Found: {sorted(treatment_values)}"
        )

    outcome_values = set(
        data[OUTCOME_COLUMN].dropna().unique()
    )

    if not outcome_values.issubset({0, 1}):
        raise ValueError(
            f"{OUTCOME_COLUMN} must contain only 0/1 values. "
            f"Found: {sorted(outcome_values)}"
        )

    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    if data[feature_columns].isna().any().any():
        missing_counts = (
            data[feature_columns]
            .isna()
            .sum()
            .loc[lambda x: x > 0]
            .to_dict()
        )

        raise ValueError(
            "Cross-fitting covariates contain missing values: "
            f"{missing_counts}"
        )


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


def generate_oof_nuisance_predictions(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Generate honest out-of-fold nuisance predictions.

    Each row receives:
    - propensity_score_oof from a model that did not train on that row,
    - mu0_oof from a control outcome model that did not train on that row,
    - mu1_oof from a treated outcome model that did not train on that row.
    """

    validate_input(data)

    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    n_rows = len(data)

    propensity_oof = np.full(
        n_rows,
        np.nan,
        dtype=float,
    )

    mu0_oof = np.full(
        n_rows,
        np.nan,
        dtype=float,
    )

    mu1_oof = np.full(
        n_rows,
        np.nan,
        dtype=float,
    )

    fold_assignment = np.full(
        n_rows,
        -1,
        dtype=int,
    )

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    treatment = (
        data[TREATMENT_COLUMN]
        .astype(int)
        .to_numpy()
    )

    for fold_number, (
        train_index,
        validation_index,
    ) in enumerate(
        splitter.split(
            data[feature_columns],
            treatment,
        ),
        start=1,
    ):
        train = data.iloc[train_index]
        validation = data.iloc[validation_index]

        train_treatment = (
            train[TREATMENT_COLUMN]
            .astype(int)
        )

        if train_treatment.nunique() != 2:
            raise ValueError(
                f"Fold {fold_number} training data does not contain "
                "both treatment groups."
            )

        propensity_model = build_propensity_pipeline()

        propensity_model.fit(
            train[feature_columns],
            train_treatment,
        )

        fold_propensity = propensity_model.predict_proba(
            validation[feature_columns]
        )[:, 1]

        control_train = train.loc[
            train[TREATMENT_COLUMN] == 0
        ]

        treated_train = train.loc[
            train[TREATMENT_COLUMN] == 1
        ]

        if control_train[OUTCOME_COLUMN].nunique() < 2:
            raise ValueError(
                f"Fold {fold_number} control outcome model "
                "does not contain both outcome classes."
            )

        if treated_train[OUTCOME_COLUMN].nunique() < 2:
            raise ValueError(
                f"Fold {fold_number} treated outcome model "
                "does not contain both outcome classes."
            )

        control_model = build_outcome_pipeline()
        treated_model = build_outcome_pipeline()

        control_model.fit(
            control_train[feature_columns],
            control_train[OUTCOME_COLUMN].astype(int),
        )

        treated_model.fit(
            treated_train[feature_columns],
            treated_train[OUTCOME_COLUMN].astype(int),
        )

        fold_mu0 = control_model.predict_proba(
            validation[feature_columns]
        )[:, 1]

        fold_mu1 = treated_model.predict_proba(
            validation[feature_columns]
        )[:, 1]

        propensity_oof[validation_index] = (
            fold_propensity
        )

        mu0_oof[validation_index] = (
            fold_mu0
        )

        mu1_oof[validation_index] = (
            fold_mu1
        )

        fold_assignment[validation_index] = (
            fold_number
        )

    for name, values in {
        "propensity_score_oof": propensity_oof,
        "mu0_oof": mu0_oof,
        "mu1_oof": mu1_oof,
    }.items():
        if np.isnan(values).any():
            raise RuntimeError(
                f"{name} contains missing out-of-fold predictions."
            )

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"{name} contains non-finite values."
            )

    if (
        (propensity_oof <= 0.0)
        | (propensity_oof >= 1.0)
    ).any():
        raise ValueError(
            "Out-of-fold propensity scores must be "
            "strictly between 0 and 1."
        )

    if (
        (mu0_oof < 0.0)
        | (mu0_oof > 1.0)
        | (mu1_oof < 0.0)
        | (mu1_oof > 1.0)
    ).any():
        raise ValueError(
            "Out-of-fold outcome predictions must remain "
            "between 0 and 1."
        )

    if (fold_assignment < 1).any():
        raise RuntimeError(
            "At least one row was not assigned to a cross-fitting fold."
        )

    return pd.DataFrame(
        {
            ID_COLUMN:
                data[ID_COLUMN].to_numpy(),
            TREATMENT_COLUMN:
                data[TREATMENT_COLUMN].to_numpy(),
            OUTCOME_COLUMN:
                data[OUTCOME_COLUMN].to_numpy(),
            "crossfit_fold":
                fold_assignment,
            "propensity_score_oof":
                propensity_oof,
            "mu0_oof":
                mu0_oof,
            "mu1_oof":
                mu1_oof,
        }
    )


def calculate_crossfit_aipw_scores(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    treatment = (
        predictions[TREATMENT_COLUMN]
        .astype(int)
        .to_numpy()
    )

    outcome = (
        predictions[OUTCOME_COLUMN]
        .astype(float)
        .to_numpy()
    )

    propensity = (
        predictions["propensity_score_oof"]
        .astype(float)
        .to_numpy()
    )

    mu0 = (
        predictions["mu0_oof"]
        .astype(float)
        .to_numpy()
    )

    mu1 = (
        predictions["mu1_oof"]
        .astype(float)
        .to_numpy()
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
            "Cross-fitted AIPW pseudo-outcomes contain "
            "non-finite values."
        )

    result = predictions.copy()

    result["outcome_model_effect_oof"] = (
        mu1 - mu0
    )

    result["aipw_pseudo_outcome_crossfit"] = (
        pseudo_outcome
    )

    return result


def summarize_crossfit_aipw(
    scores: pd.DataFrame,
) -> dict:
    pseudo_outcome = (
        scores["aipw_pseudo_outcome_crossfit"]
        .to_numpy(dtype=float)
    )

    n = int(len(pseudo_outcome))

    if n <= 1:
        raise ValueError(
            "Cross-fitted AIPW uncertainty requires "
            "at least two observations."
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
        ate
        - z_value * standard_error
    )

    ci_upper = (
        ate
        + z_value * standard_error
    )

    propensity = (
        scores["propensity_score_oof"]
        .to_numpy(dtype=float)
    )

    return {
        "crossfit_aipw_ate":
            ate,
        "crossfit_aipw_ate_percentage_points":
            ate * 100.0,
        "standard_error":
            standard_error,
        "standard_error_percentage_points":
            standard_error * 100.0,
        "confidence_level":
            CONFIDENCE_LEVEL,
        "confidence_interval": {
            "lower":
                ci_lower,
            "upper":
                ci_upper,
        },
        "confidence_interval_percentage_points": {
            "lower":
                ci_lower * 100.0,
            "upper":
                ci_upper * 100.0,
        },
        "mean_mu0_oof":
            float(
                scores["mu0_oof"].mean()
            ),
        "mean_mu1_oof":
            float(
                scores["mu1_oof"].mean()
            ),
        "mean_outcome_model_effect_oof":
            float(
                scores[
                    "outcome_model_effect_oof"
                ].mean()
            ),
        "oof_propensity_min":
            float(propensity.min()),
        "oof_propensity_max":
            float(propensity.max()),
        "oof_propensity_mean":
            float(propensity.mean()),
    }


def build_summary(
    crossfit: dict,
    non_crossfit: dict,
    ipw: dict,
    n_observations: int,
) -> dict:
    crossfit_ate = float(
        crossfit[
            "crossfit_aipw_ate"
        ]
    )

    non_crossfit_ate = float(
        non_crossfit[
            "aipw_ate"
        ]
    )

    ipw_ate = float(
        ipw[
            "ipw_ate"
        ]
    )

    return {
        "estimand":
            "ATE",
        "estimator":
            "cross_fitted_augmented_inverse_probability_weighting",
        "cross_fitted":
            True,
        "n_splits":
            N_SPLITS,
        "shuffle":
            True,
        "random_state":
            RANDOM_STATE,
        "confidence_level":
            CONFIDENCE_LEVEL,
        "n_observations":
            n_observations,
        "crossfit_aipw_ate":
            crossfit_ate,
        "crossfit_aipw_ate_percentage_points":
            crossfit_ate * 100.0,
        "standard_error":
            crossfit[
                "standard_error"
            ],
        "standard_error_percentage_points":
            crossfit[
                "standard_error_percentage_points"
            ],
        "confidence_interval":
            crossfit[
                "confidence_interval"
            ],
        "confidence_interval_percentage_points":
            crossfit[
                "confidence_interval_percentage_points"
            ],
        "oof_nuisance_summary": {
            "mean_mu0_oof":
                crossfit[
                    "mean_mu0_oof"
                ],
            "mean_mu1_oof":
                crossfit[
                    "mean_mu1_oof"
                ],
            "mean_mu1_minus_mu0_oof":
                crossfit[
                    "mean_outcome_model_effect_oof"
                ],
            "propensity_min":
                crossfit[
                    "oof_propensity_min"
                ],
            "propensity_max":
                crossfit[
                    "oof_propensity_max"
                ],
            "propensity_mean":
                crossfit[
                    "oof_propensity_mean"
                ],
        },
        "comparison": {
            "non_crossfit_aipw_ate":
                non_crossfit_ate,
            "non_crossfit_aipw_ate_percentage_points":
                non_crossfit_ate * 100.0,
            "crossfit_minus_non_crossfit":
                crossfit_ate
                - non_crossfit_ate,
            "crossfit_minus_non_crossfit_percentage_points":
                (
                    crossfit_ate
                    - non_crossfit_ate
                ) * 100.0,
            "ipw_ate":
                ipw_ate,
            "ipw_ate_percentage_points":
                ipw_ate * 100.0,
            "crossfit_minus_ipw":
                crossfit_ate
                - ipw_ate,
            "crossfit_minus_ipw_percentage_points":
                (
                    crossfit_ate
                    - ipw_ate
                ) * 100.0,
        },
        "uncertainty_method":
            "cross_fitted_influence_function_normal_approximation",
        "interpretation": (
            "Each row's propensity score and potential-outcome "
            "predictions were generated from nuisance models that did "
            "not train on that row. Under the stated identification "
            "assumptions, the resulting AIPW estimate targets the "
            "average treatment effect."
        ),
        "limitations": [
            (
                "Cross-fitting reduces own-observation nuisance-model "
                "overfitting but does not remove unmeasured confounding."
            ),
            (
                "Observed overlap and balance remain diagnostic evidence, "
                "not proof of causal identification."
            ),
            (
                "The confidence interval uses an influence-function "
                "normal approximation."
            ),
            (
                "Synthetic ground-truth treatment-effect fields remain "
                "excluded from estimator fitting."
            ),
        ],
    }


def save_outputs(
    scores: pd.DataFrame,
    summary: dict,
) -> None:
    CROSSFIT_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores.to_csv(
        CROSSFIT_SCORES_PATH,
        index=False,
    )

    with CROSSFIT_SUMMARY_PATH.open(
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
    interval = summary[
        "confidence_interval"
    ]

    interval_pp = summary[
        "confidence_interval_percentage_points"
    ]

    nuisance = summary[
        "oof_nuisance_summary"
    ]

    comparison = summary[
        "comparison"
    ]

    print(
        "Cross-fitted Augmented Inverse Probability Weighting"
    )
    print(
        "----------------------------------------------------"
    )

    print(
        f"Estimand: {summary['estimand']}"
    )

    print(
        f"Folds: {summary['n_splits']}"
    )

    print(
        f"Observations: {summary['n_observations']:,}"
    )

    print(
        f"Cross-fitted: {summary['cross_fitted']}"
    )

    print(
        "\nOut-of-fold nuisance predictions"
    )

    print(
        "Mean predicted outcome under control: "
        f"{nuisance['mean_mu0_oof']:.6f}"
    )

    print(
        "Mean predicted outcome under treatment: "
        f"{nuisance['mean_mu1_oof']:.6f}"
    )

    print(
        "Mean outcome-model contrast: "
        f"{nuisance['mean_mu1_minus_mu0_oof']:.6f}"
    )

    print(
        "OOF propensity range: "
        f"[{nuisance['propensity_min']:.6f}, "
        f"{nuisance['propensity_max']:.6f}]"
    )

    print(
        "\nCross-fitted AIPW ATE"
    )

    print(
        f"{summary['crossfit_aipw_ate']:.6f} "
        f"({summary['crossfit_aipw_ate_percentage_points']:+.2f} "
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
        "Non-cross-fitted AIPW: "
        f"{comparison['non_crossfit_aipw_ate']:.6f} "
        f"({comparison['non_crossfit_aipw_ate_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "Cross-fitted - non-cross-fitted AIPW: "
        f"{comparison['crossfit_minus_non_crossfit']:.6f} "
        f"({comparison['crossfit_minus_non_crossfit_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "IPW ATE: "
        f"{comparison['ipw_ate']:.6f} "
        f"({comparison['ipw_ate_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "Cross-fitted AIPW - IPW: "
        f"{comparison['crossfit_minus_ipw']:.6f} "
        f"({comparison['crossfit_minus_ipw_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "\nArtifacts saved:"
    )

    print(
        f"- {CROSSFIT_SCORES_PATH}"
    )

    print(
        f"- {CROSSFIT_SUMMARY_PATH}"
    )

    print(
        "\nCross-fitting prevents each row from training the nuisance "
        "models used to generate that row's propensity and outcome "
        "predictions. It does not prove the causal assumptions."
    )


def main() -> None:
    data = load_data()

    predictions = generate_oof_nuisance_predictions(
        data=data,
    )

    scores = calculate_crossfit_aipw_scores(
        predictions=predictions,
    )

    crossfit = summarize_crossfit_aipw(
        scores=scores,
    )

    non_crossfit = load_json_artifact(
        path=NON_CROSSFIT_AIPW_PATH,
        label="non-cross-fitted AIPW",
    )

    ipw = load_json_artifact(
        path=IPW_SUMMARY_PATH,
        label="IPW ATE",
    )

    summary = build_summary(
        crossfit=crossfit,
        non_crossfit=non_crossfit,
        ipw=ipw,
        n_observations=len(data),
    )

    save_outputs(
        scores=scores,
        summary=summary,
    )

    print_summary(
        summary=summary,
    )


if __name__ == "__main__":
    main()
