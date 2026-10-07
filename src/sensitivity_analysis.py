from __future__ import annotations

import json
from pathlib import Path

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
AIPW_SUMMARY_PATH = Path("artifacts/aipw_summary.json")
CROSSFIT_AIPW_SUMMARY_PATH = Path("artifacts/aipw_crossfit_summary.json")

OUTPUT_PATH = Path("artifacts/sensitivity_analysis.csv")
SUMMARY_PATH = Path("artifacts/sensitivity_analysis_summary.json")

RANDOM_STATE = 42


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label} artifact: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_propensity_pipeline(
    c_value: float,
) -> Pipeline:
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
        C=c_value,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    denominator = float(np.sum(weights))

    if denominator <= 0.0:
        raise ValueError(
            "Weighted mean requires positive total weight."
        )

    return float(
        np.sum(values * weights)
        / denominator
    )


def estimate_ipw_from_propensity(
    data: pd.DataFrame,
    propensity: np.ndarray,
    scenario: str,
    notes: str,
) -> dict:
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

    if len(propensity) != len(data):
        raise ValueError(
            f"{scenario}: propensity length does not match data."
        )

    if (
        (propensity <= 0.0)
        | (propensity >= 1.0)
    ).any():
        raise ValueError(
            f"{scenario}: propensity scores must be strictly between 0 and 1."
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

    treated_mean = weighted_mean(
        outcome[treated_mask],
        weights[treated_mask],
    )

    control_mean = weighted_mean(
        outcome[control_mask],
        weights[control_mask],
    )

    ate = treated_mean - control_mean

    return {
        "scenario": scenario,
        "estimator": "stabilized_ipw",
        "n": int(len(data)),
        "estimate": ate,
        "estimate_percentage_points": ate * 100.0,
        "weighted_control_mean": control_mean,
        "weighted_treated_mean": treated_mean,
        "propensity_min": float(
            np.min(propensity)
        ),
        "propensity_max": float(
            np.max(propensity)
        ),
        "notes": notes,
    }


def load_primary_propensity(
    data: pd.DataFrame,
) -> np.ndarray:
    if not PROPENSITY_PATH.exists():
        raise FileNotFoundError(
            f"Missing propensity artifact: {PROPENSITY_PATH}"
        )

    scores = pd.read_csv(
        PROPENSITY_PATH
    )

    required = {
        ID_COLUMN,
        "propensity_score",
    }

    missing = sorted(
        required.difference(
            scores.columns
        )
    )

    if missing:
        raise ValueError(
            "Propensity artifact is missing required columns: "
            + ", ".join(missing)
        )

    aligned = data[
        [ID_COLUMN]
    ].merge(
        scores[
            [
                ID_COLUMN,
                "propensity_score",
            ]
        ],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(aligned) != len(data):
        raise ValueError(
            "Propensity scores do not align one-to-one with source data."
        )

    return aligned[
        "propensity_score"
    ].to_numpy(dtype=float)


def alternate_propensity_scores(
    data: pd.DataFrame,
    c_value: float,
) -> np.ndarray:
    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    pipeline = build_propensity_pipeline(
        c_value=c_value
    )

    pipeline.fit(
        data[feature_columns],
        data[TREATMENT_COLUMN].astype(int),
    )

    return pipeline.predict_proba(
        data[feature_columns]
    )[:, 1]


def common_support_subset(
    data: pd.DataFrame,
    propensity: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    treatment = (
        data[TREATMENT_COLUMN]
        .astype(int)
        .to_numpy()
    )

    treated_scores = propensity[
        treatment == 1
    ]

    control_scores = propensity[
        treatment == 0
    ]

    lower = max(
        float(np.min(treated_scores)),
        float(np.min(control_scores)),
    )

    upper = min(
        float(np.max(treated_scores)),
        float(np.max(control_scores)),
    )

    keep = (
        (propensity >= lower)
        & (propensity <= upper)
    )

    subset = (
        data.loc[keep]
        .copy()
        .reset_index(drop=True)
    )

    subset_propensity = propensity[
        keep
    ]

    return (
        subset,
        subset_propensity,
        lower,
        upper,
    )


def add_existing_estimators(
    rows: list[dict],
) -> None:
    ipw = load_json(
        IPW_SUMMARY_PATH,
        "IPW ATE",
    )

    aipw = load_json(
        AIPW_SUMMARY_PATH,
        "AIPW",
    )

    crossfit = load_json(
        CROSSFIT_AIPW_SUMMARY_PATH,
        "cross-fitted AIPW",
    )

    rows.extend(
        [
            {
                "scenario": "existing_ipw_reference",
                "estimator": "stabilized_ipw",
                "n": int(
                    ipw["n_observations"]
                ),
                "estimate": float(
                    ipw["ipw_ate"]
                ),
                "estimate_percentage_points": float(
                    ipw[
                        "ipw_ate_percentage_points"
                    ]
                ),
                "weighted_control_mean": float(
                    ipw[
                        "weighted_control_outcome_mean"
                    ]
                ),
                "weighted_treated_mean": float(
                    ipw[
                        "weighted_treated_outcome_mean"
                    ]
                ),
                "propensity_min": np.nan,
                "propensity_max": np.nan,
                "notes": (
                    "Previously reported primary IPW estimate."
                ),
            },
            {
                "scenario": "existing_aipw_reference",
                "estimator": "aipw",
                "n": int(
                    aipw["n_observations"]
                ),
                "estimate": float(
                    aipw["aipw_ate"]
                ),
                "estimate_percentage_points": float(
                    aipw[
                        "aipw_ate_percentage_points"
                    ]
                ),
                "weighted_control_mean": np.nan,
                "weighted_treated_mean": np.nan,
                "propensity_min": np.nan,
                "propensity_max": np.nan,
                "notes": (
                    "Previously reported doubly robust estimate."
                ),
            },
            {
                "scenario": "existing_crossfit_aipw_reference",
                "estimator": "cross_fitted_aipw",
                "n": int(
                    crossfit["n_observations"]
                ),
                "estimate": float(
                    crossfit[
                        "crossfit_aipw_ate"
                    ]
                ),
                "estimate_percentage_points": float(
                    crossfit[
                        "crossfit_aipw_ate_percentage_points"
                    ]
                ),
                "weighted_control_mean": np.nan,
                "weighted_treated_mean": np.nan,
                "propensity_min": np.nan,
                "propensity_max": np.nan,
                "notes": (
                    "Previously reported honest cross-fitted AIPW estimate."
                ),
            },
        ]
    )


def build_sensitivity_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    primary_propensity = load_primary_propensity(
        data=data
    )

    rows: list[dict] = []

    rows.append(
        estimate_ipw_from_propensity(
            data=data,
            propensity=primary_propensity,
            scenario="primary_propensity_recomputed_ipw",
            notes=(
                "Recomputed stabilized IPW using the saved primary "
                "propensity scores."
            ),
        )
    )

    rows.append(
        estimate_ipw_from_propensity(
            data=data,
            propensity=np.clip(
                primary_propensity,
                0.05,
                0.95,
            ),
            scenario="propensity_clip_0.05_0.95",
            notes=(
                "Sensitivity scenario clipping propensity scores "
                "to [0.05, 0.95]."
            ),
        )
    )

    rows.append(
        estimate_ipw_from_propensity(
            data=data,
            propensity=np.clip(
                primary_propensity,
                0.10,
                0.90,
            ),
            scenario="propensity_clip_0.10_0.90",
            notes=(
                "Sensitivity scenario clipping propensity scores "
                "to [0.10, 0.90]."
            ),
        )
    )

    (
        common_support_data,
        common_support_propensity,
        overlap_lower,
        overlap_upper,
    ) = common_support_subset(
        data=data,
        propensity=primary_propensity,
    )

    rows.append(
        estimate_ipw_from_propensity(
            data=common_support_data,
            propensity=common_support_propensity,
            scenario="empirical_common_support_only",
            notes=(
                "Restricted to the empirical treated/control common "
                f"support interval [{overlap_lower:.6f}, "
                f"{overlap_upper:.6f}]."
            ),
        )
    )

    stronger_regularization = (
        alternate_propensity_scores(
            data=data,
            c_value=0.25,
        )
    )

    rows.append(
        estimate_ipw_from_propensity(
            data=data,
            propensity=stronger_regularization,
            scenario="propensity_logit_C_0.25",
            notes=(
                "Alternate propensity specification using stronger "
                "L2 regularization."
            ),
        )
    )

    weaker_regularization = (
        alternate_propensity_scores(
            data=data,
            c_value=4.0,
        )
    )

    rows.append(
        estimate_ipw_from_propensity(
            data=data,
            propensity=weaker_regularization,
            scenario="propensity_logit_C_4.0",
            notes=(
                "Alternate propensity specification using weaker "
                "L2 regularization."
            ),
        )
    )

    add_existing_estimators(
        rows=rows
    )

    table = pd.DataFrame(
        rows
    )

    primary = float(
        table.loc[
            table["scenario"]
            == "primary_propensity_recomputed_ipw",
            "estimate",
        ].iloc[0]
    )

    table[
        "difference_from_primary"
    ] = (
        table["estimate"]
        - primary
    )

    table[
        "difference_from_primary_percentage_points"
    ] = (
        table[
            "difference_from_primary"
        ]
        * 100.0
    )

    table[
        "absolute_difference_from_primary_percentage_points"
    ] = (
        table[
            "difference_from_primary_percentage_points"
        ].abs()
    )

    return table


def build_summary(
    table: pd.DataFrame,
) -> dict:
    sensitivity_only = table.loc[
        ~table["scenario"].str.contains(
            "reference"
        )
    ].copy()

    max_row = sensitivity_only.loc[
        sensitivity_only[
            "absolute_difference_from_primary_percentage_points"
        ].idxmax()
    ]

    return {
        "primary_scenario": (
            "primary_propensity_recomputed_ipw"
        ),
        "n_scenarios": int(
            len(table)
        ),
        "n_propensity_sensitivity_scenarios": int(
            len(sensitivity_only)
        ),
        "minimum_estimate_percentage_points": float(
            table[
                "estimate_percentage_points"
            ].min()
        ),
        "maximum_estimate_percentage_points": float(
            table[
                "estimate_percentage_points"
            ].max()
        ),
        "overall_estimate_range_percentage_points": float(
            table[
                "estimate_percentage_points"
            ].max()
            - table[
                "estimate_percentage_points"
            ].min()
        ),
        "largest_propensity_sensitivity_shift_scenario": str(
            max_row["scenario"]
        ),
        "largest_propensity_sensitivity_shift_percentage_points": float(
            max_row[
                "difference_from_primary_percentage_points"
            ]
        ),
        "largest_absolute_propensity_sensitivity_shift_percentage_points": float(
            max_row[
                "absolute_difference_from_primary_percentage_points"
            ]
        ),
        "all_estimates_positive": bool(
            (
                table[
                    "estimate"
                ]
                > 0.0
            ).all()
        ),
        "policy": (
            "Sensitivity scenarios are reported separately from the "
            "primary estimate. The primary estimate is not replaced "
            "because another sensitivity scenario looks more favorable."
        ),
    }


def save_outputs(
    table: pd.DataFrame,
    summary: dict,
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
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
    table: pd.DataFrame,
    summary: dict,
) -> None:
    print(
        "Causal Sensitivity Analysis"
    )
    print(
        "---------------------------"
    )

    print(
        f"Scenarios: "
        f"{summary['n_scenarios']}"
    )

    print(
        "Estimate range: "
        f"{summary['minimum_estimate_percentage_points']:+.2f} "
        "to "
        f"{summary['maximum_estimate_percentage_points']:+.2f} "
        "percentage points"
    )

    print(
        "Overall range width: "
        f"{summary['overall_estimate_range_percentage_points']:.2f} "
        "percentage points"
    )

    print(
        "Largest propensity-sensitivity shift: "
        f"{summary['largest_propensity_sensitivity_shift_scenario']} "
        f"({summary['largest_propensity_sensitivity_shift_percentage_points']:+.2f} "
        "percentage points)"
    )

    print(
        "All estimates positive: "
        f"{summary['all_estimates_positive']}"
    )

    print(
        "\nScenario comparison"
    )

    display_columns = [
        "scenario",
        "estimator",
        "n",
        "estimate_percentage_points",
        "difference_from_primary_percentage_points",
    ]

    print(
        table[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
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
        "\nPolicy:"
    )

    print(
        summary["policy"]
    )


def main() -> None:
    data = load_data()

    table = build_sensitivity_table(
        data=data
    )

    summary = build_summary(
        table=table
    )

    save_outputs(
        table=table,
        summary=summary,
    )

    print_summary(
        table=table,
        summary=summary,
    )


if __name__ == "__main__":
    main()
