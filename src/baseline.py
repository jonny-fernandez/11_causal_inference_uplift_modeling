from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

from src.data_contract import load_data


TREATMENT_COLUMN = "training_assigned"
OUTCOME_COLUMN = "retained_or_promoted_180d"
CONFIDENCE_LEVEL = 0.95
OUTPUT_PATH = Path("artifacts/naive_baseline.json")


def calculate_naive_baseline(df) -> dict:
    """Calculate the unadjusted treated-versus-control outcome difference.

    This is a descriptive baseline only. It is not a causal estimate because
    confounding, overlap, covariate balance, and adjustment quality have not
    yet been evaluated.
    """
    control = df.loc[df[TREATMENT_COLUMN] == 0, OUTCOME_COLUMN]
    treated = df.loc[df[TREATMENT_COLUMN] == 1, OUTCOME_COLUMN]

    n_control = int(control.shape[0])
    n_treated = int(treated.shape[0])

    control_mean = float(control.mean())
    treated_mean = float(treated.mean())
    naive_difference = treated_mean - control_mean

    # Standard error for the difference between two independent proportions.
    standard_error = sqrt(
        (treated_mean * (1.0 - treated_mean) / n_treated)
        + (control_mean * (1.0 - control_mean) / n_control)
    )

    alpha = 1.0 - CONFIDENCE_LEVEL
    z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    ci_lower = naive_difference - z_value * standard_error
    ci_upper = naive_difference + z_value * standard_error

    return {
        "analysis": "naive_difference_in_means",
        "causal_interpretation_allowed": False,
        "treatment_column": TREATMENT_COLUMN,
        "outcome_column": OUTCOME_COLUMN,
        "control_value": 0,
        "treated_value": 1,
        "control_count": n_control,
        "treated_count": n_treated,
        "control_outcome_mean": control_mean,
        "treated_outcome_mean": treated_mean,
        "naive_difference": naive_difference,
        "naive_difference_percentage_points": naive_difference * 100.0,
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
        "interpretation": (
            "Unadjusted association only. Do not interpret as a causal treatment "
            "effect until overlap, confounding, covariate balance, and adjustment "
            "diagnostics are complete."
        ),
    }


def save_artifact(summary: dict, output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")


def print_summary(summary: dict) -> None:
    ci = summary["confidence_interval"]
    ci_pp = summary["confidence_interval_percentage_points"]

    print("Naive difference-in-means baseline only")
    print(f"Control n: {summary['control_count']:,}")
    print(f"Treated n: {summary['treated_count']:,}")

    print(
        "Control outcome: "
        f"{summary['control_outcome_mean']:.6f} "
        f"({summary['control_outcome_mean']:.2%})"
    )

    print(
        "Treated outcome: "
        f"{summary['treated_outcome_mean']:.6f} "
        f"({summary['treated_outcome_mean']:.2%})"
    )

    print(
        "Naive difference: "
        f"{summary['naive_difference']:.6f} "
        f"({summary['naive_difference_percentage_points']:+.2f} percentage points)"
    )

    print(
        "Standard error: "
        f"{summary['standard_error']:.6f} "
        f"({summary['standard_error_percentage_points']:.2f} percentage points)"
    )

    print(
        f"{int(summary['confidence_level'] * 100)}% CI: "
        f"[{ci['lower']:.6f}, {ci['upper']:.6f}] "
        f"([{ci_pp['lower']:+.2f}, {ci_pp['upper']:+.2f}] percentage points)"
    )

    print(f"Artifact saved: {OUTPUT_PATH}")
    print(summary["interpretation"])


def main() -> None:
    df = load_data()
    summary = calculate_naive_baseline(df)
    save_artifact(summary)
    print_summary(summary)


if __name__ == "__main__":
    main()