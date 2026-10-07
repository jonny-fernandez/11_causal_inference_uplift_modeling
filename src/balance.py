from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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

WEIGHTS_PATH = Path("artifacts/ipw_weights.csv")
BALANCE_OUTPUT_PATH = Path("artifacts/balance_diagnostics.csv")
SUMMARY_OUTPUT_PATH = Path("artifacts/balance_summary.json")

WEIGHT_COLUMN = "ate_stabilized_ipw"
BALANCE_THRESHOLD = 0.10


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_array = values.to_numpy(dtype=float)
    weights_array = weights.to_numpy(dtype=float)
    denominator = float(np.sum(weights_array))

    if denominator <= 0.0:
        raise ValueError("Weighted mean requires a positive total weight.")

    return float(np.sum(values_array * weights_array) / denominator)


def pooled_standard_deviation(
    treated: pd.Series,
    control: pd.Series,
) -> float:
    treated_var = float(treated.var(ddof=1))
    control_var = float(control.var(ddof=1))
    pooled_variance = (treated_var + control_var) / 2.0

    if pooled_variance <= 0.0:
        return 0.0

    return float(np.sqrt(pooled_variance))


def binary_reference_standard_deviation(
    treated_proportion: float,
    control_proportion: float,
) -> float:
    pooled_variance = (
        treated_proportion * (1.0 - treated_proportion)
        + control_proportion * (1.0 - control_proportion)
    ) / 2.0

    if pooled_variance <= 0.0:
        return 0.0

    return float(np.sqrt(pooled_variance))


def safe_smd(
    treated_value: float,
    control_value: float,
    denominator: float,
) -> float:
    difference = treated_value - control_value

    if denominator == 0.0:
        if np.isclose(difference, 0.0):
            return 0.0

        return float(np.sign(difference) * np.inf)

    return float(difference / denominator)


def load_and_align_weights(data: pd.DataFrame) -> pd.DataFrame:
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

    missing = sorted(required_columns.difference(weights.columns))

    if missing:
        raise ValueError(
            "Weight artifact is missing required columns: "
            + ", ".join(missing)
        )

    if weights[ID_COLUMN].duplicated().any():
        raise ValueError("Weight artifact contains duplicate person IDs.")

    if weights[WEIGHT_COLUMN].isna().any():
        raise ValueError(f"{WEIGHT_COLUMN} contains missing values.")

    if not np.isfinite(weights[WEIGHT_COLUMN]).all():
        raise ValueError(f"{WEIGHT_COLUMN} contains non-finite values.")

    if (weights[WEIGHT_COLUMN] <= 0.0).any():
        raise ValueError(f"{WEIGHT_COLUMN} must contain only positive values.")

    merged = data.merge(
        weights[[ID_COLUMN, TREATMENT_COLUMN, WEIGHT_COLUMN]],
        on=ID_COLUMN,
        how="inner",
        suffixes=("_data", "_weight"),
        validate="one_to_one",
    )

    if len(merged) != len(data):
        raise ValueError(
            "Weight artifact does not align one-to-one with source data."
        )

    mismatch = (
        merged[f"{TREATMENT_COLUMN}_data"]
        != merged[f"{TREATMENT_COLUMN}_weight"]
    )

    if mismatch.any():
        raise ValueError(
            "Treatment values in ipw_weights.csv do not match "
            "the source dataset."
        )

    merged = merged.rename(
        columns={f"{TREATMENT_COLUMN}_data": TREATMENT_COLUMN}
    ).drop(columns=[f"{TREATMENT_COLUMN}_weight"])

    return merged


def numeric_balance_row(data: pd.DataFrame, feature: str) -> dict:
    treated_mask = data[TREATMENT_COLUMN] == 1
    control_mask = data[TREATMENT_COLUMN] == 0

    treated = data.loc[treated_mask, feature].astype(float)
    control = data.loc[control_mask, feature].astype(float)

    treated_weight = data.loc[treated_mask, WEIGHT_COLUMN].astype(float)
    control_weight = data.loc[control_mask, WEIGHT_COLUMN].astype(float)

    treated_mean = float(treated.mean())
    control_mean = float(control.mean())

    weighted_treated_mean = weighted_mean(treated, treated_weight)
    weighted_control_mean = weighted_mean(control, control_weight)

    reference_sd = pooled_standard_deviation(treated, control)

    unweighted_smd = safe_smd(
        treated_mean,
        control_mean,
        reference_sd,
    )

    weighted_smd = safe_smd(
        weighted_treated_mean,
        weighted_control_mean,
        reference_sd,
    )

    return {
        "covariate": feature,
        "feature_type": "numeric",
        "treated_mean_or_proportion": treated_mean,
        "control_mean_or_proportion": control_mean,
        "weighted_treated_mean_or_proportion": weighted_treated_mean,
        "weighted_control_mean_or_proportion": weighted_control_mean,
        "reference_sd": reference_sd,
        "unweighted_smd": unweighted_smd,
        "weighted_smd": weighted_smd,
        "abs_unweighted_smd": abs(unweighted_smd),
        "abs_weighted_smd": abs(weighted_smd),
        "unweighted_below_threshold": (
            abs(unweighted_smd) < BALANCE_THRESHOLD
        ),
        "weighted_below_threshold": (
            abs(weighted_smd) < BALANCE_THRESHOLD
        ),
        "balance_improved": (
            abs(weighted_smd) < abs(unweighted_smd)
        ),
    }


def categorical_balance_rows(
    data: pd.DataFrame,
    feature: str,
) -> list[dict]:
    rows: list[dict] = []

    treated_mask = data[TREATMENT_COLUMN] == 1
    control_mask = data[TREATMENT_COLUMN] == 0

    treated_weight = data.loc[treated_mask, WEIGHT_COLUMN].astype(float)
    control_weight = data.loc[control_mask, WEIGHT_COLUMN].astype(float)

    categories = sorted(
        data[feature].dropna().unique().tolist(),
        key=lambda value: str(value),
    )

    for category in categories:
        indicator = (data[feature] == category).astype(float)

        treated_indicator = indicator.loc[treated_mask]
        control_indicator = indicator.loc[control_mask]

        treated_proportion = float(treated_indicator.mean())
        control_proportion = float(control_indicator.mean())

        weighted_treated_proportion = weighted_mean(
            treated_indicator,
            treated_weight,
        )

        weighted_control_proportion = weighted_mean(
            control_indicator,
            control_weight,
        )

        reference_sd = binary_reference_standard_deviation(
            treated_proportion,
            control_proportion,
        )

        unweighted_smd = safe_smd(
            treated_proportion,
            control_proportion,
            reference_sd,
        )

        weighted_smd = safe_smd(
            weighted_treated_proportion,
            weighted_control_proportion,
            reference_sd,
        )

        rows.append(
            {
                "covariate": f"{feature}={category}",
                "feature_type": "categorical_level",
                "treated_mean_or_proportion": treated_proportion,
                "control_mean_or_proportion": control_proportion,
                "weighted_treated_mean_or_proportion": (
                    weighted_treated_proportion
                ),
                "weighted_control_mean_or_proportion": (
                    weighted_control_proportion
                ),
                "reference_sd": reference_sd,
                "unweighted_smd": unweighted_smd,
                "weighted_smd": weighted_smd,
                "abs_unweighted_smd": abs(unweighted_smd),
                "abs_weighted_smd": abs(weighted_smd),
                "unweighted_below_threshold": (
                    abs(unweighted_smd) < BALANCE_THRESHOLD
                ),
                "weighted_below_threshold": (
                    abs(weighted_smd) < BALANCE_THRESHOLD
                ),
                "balance_improved": (
                    abs(weighted_smd) < abs(unweighted_smd)
                ),
            }
        )

    return rows


def calculate_balance(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for feature in NUMERIC_FEATURES:
        rows.append(numeric_balance_row(data=data, feature=feature))

    for feature in CATEGORICAL_FEATURES:
        rows.extend(
            categorical_balance_rows(
                data=data,
                feature=feature,
            )
        )

    balance = pd.DataFrame(rows)

    return balance.sort_values(
        by=["abs_unweighted_smd", "covariate"],
        ascending=[False, True],
    ).reset_index(drop=True)


def create_summary(balance: pd.DataFrame) -> dict:
    pre_bad = int(
        (balance["abs_unweighted_smd"] >= BALANCE_THRESHOLD).sum()
    )

    post_bad = int(
        (balance["abs_weighted_smd"] >= BALANCE_THRESHOLD).sum()
    )

    improved_count = int(balance["balance_improved"].sum())

    return {
        "estimand": "ATE",
        "weight_column": WEIGHT_COLUMN,
        "balance_threshold": BALANCE_THRESHOLD,
        "balance_threshold_rule": "|SMD| < 0.10",
        "standardization_policy": (
            "The raw treated/control pooled standard deviation "
            "or pooled binary-proportion standard deviation is used "
            "as the common reference denominator for both pre- and "
            "post-weighting SMDs. This keeps before/after SMDs on "
            "the same scale."
        ),
        "n_balance_rows": int(len(balance)),
        "pre_weighting": {
            "rows_at_or_above_threshold": pre_bad,
            "maximum_abs_smd": float(
                balance["abs_unweighted_smd"].max()
            ),
            "mean_abs_smd": float(
                balance["abs_unweighted_smd"].mean()
            ),
        },
        "post_weighting": {
            "rows_at_or_above_threshold": post_bad,
            "maximum_abs_smd": float(
                balance["abs_weighted_smd"].max()
            ),
            "mean_abs_smd": float(
                balance["abs_weighted_smd"].mean()
            ),
        },
        "comparison": {
            "rows_improved": improved_count,
            "rows_not_improved": int(len(balance) - improved_count),
            "all_post_weighting_rows_below_threshold": bool(
                post_bad == 0
            ),
        },
        "interpretation": (
            "Post-weighting balance diagnostics evaluate observed "
            "pre-treatment covariates only. Good observed balance is "
            "necessary evidence for the weighting design but does not "
            "prove absence of unmeasured confounding."
        ),
    }


def save_outputs(
    balance: pd.DataFrame,
    summary: dict,
) -> None:
    BALANCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    balance.to_csv(BALANCE_OUTPUT_PATH, index=False)

    with SUMMARY_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def print_summary(
    balance: pd.DataFrame,
    summary: dict,
) -> None:
    pre = summary["pre_weighting"]
    post = summary["post_weighting"]
    comparison = summary["comparison"]

    print("Pre- and post-weighting covariate balance")
    print("----------------------------------------")
    print(
        f"Balance threshold: "
        f"|SMD| < {BALANCE_THRESHOLD:.2f}"
    )
    print(f"Analysis weight: {WEIGHT_COLUMN}")
    print(f"Balance rows: {summary['n_balance_rows']}")

    print("\nBefore weighting")
    print(
        "Rows at or above threshold: "
        f"{pre['rows_at_or_above_threshold']}"
    )
    print(
        f"Maximum |SMD|: "
        f"{pre['maximum_abs_smd']:.6f}"
    )
    print(
        f"Mean |SMD|: "
        f"{pre['mean_abs_smd']:.6f}"
    )

    print("\nAfter weighting")
    print(
        "Rows at or above threshold: "
        f"{post['rows_at_or_above_threshold']}"
    )
    print(
        f"Maximum |SMD|: "
        f"{post['maximum_abs_smd']:.6f}"
    )
    print(
        f"Mean |SMD|: "
        f"{post['mean_abs_smd']:.6f}"
    )

    print("\nBalance comparison")
    print(
        f"Rows improved: "
        f"{comparison['rows_improved']} / "
        f"{summary['n_balance_rows']}"
    )
    print(
        "All post-weighting rows below threshold: "
        f"{comparison['all_post_weighting_rows_below_threshold']}"
    )

    display_columns = [
        "covariate",
        "unweighted_smd",
        "weighted_smd",
        "abs_unweighted_smd",
        "abs_weighted_smd",
        "weighted_below_threshold",
    ]

    display = (
        balance[display_columns]
        .sort_values(
            by="abs_unweighted_smd",
            ascending=False,
        )
        .head(13)
    )

    print("\nBefore vs. after SMDs")
    print(
        display.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nArtifacts saved:")
    print(f"- {BALANCE_OUTPUT_PATH}")
    print(f"- {SUMMARY_OUTPUT_PATH}")

    print(
        "\nGood post-weighting balance does not prove "
        "that unmeasured confounding is absent."
    )


def main() -> None:
    data = load_data()

    aligned = load_and_align_weights(data=data)

    balance = calculate_balance(data=aligned)

    summary = create_summary(balance=balance)

    save_outputs(
        balance=balance,
        summary=summary,
    )

    print_summary(
        balance=balance,
        summary=summary,
    )


if __name__ == "__main__":
    main()
