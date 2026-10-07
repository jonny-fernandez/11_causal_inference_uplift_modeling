from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROPENSITY_SCORES_PATH = Path("artifacts/propensity_scores.csv")
BALANCE_PATH = Path("artifacts/balance_diagnostics.csv")
BASELINE_PATH = Path("artifacts/naive_baseline.json")
IPW_PATH = Path("artifacts/ate_summary.json")
AIPW_PATH = Path("artifacts/aipw_summary.json")
CROSSFIT_PATH = Path("artifacts/aipw_crossfit_summary.json")
GROUND_TRUTH_PATH = Path("artifacts/ground_truth_evaluation.json")
SENSITIVITY_PATH = Path("artifacts/sensitivity_analysis.csv")
SEGMENT_EFFECTS_PATH = Path("artifacts/segment_effects.csv")
CATE_RF_PATH = Path("artifacts/cate_evaluation_summary.json")
CATE_RIDGE_PATH = Path("artifacts/cate_regularized_validation_summary.json")
CATE_SLEARNER_PATH = Path("artifacts/cate_slearner_validation_summary.json")

CHART_DIR = Path("outputs/charts")
MANIFEST_PATH = CHART_DIR / "chart_manifest.csv"


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_chart_dir() -> None:
    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_figure(
    filename: str,
    description: str,
    manifest_rows: list[dict],
) -> None:
    output_path = CHART_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    manifest_rows.append(
        {
            "filename": filename,
            "path": str(output_path).replace("\\", "/"),
            "description": description,
        }
    )


def chart_propensity_overlap(
    manifest_rows: list[dict],
) -> None:
    scores = pd.read_csv(
        PROPENSITY_SCORES_PATH
    )

    control = scores.loc[
        scores["training_assigned"] == 0,
        "propensity_score",
    ]

    treated = scores.loc[
        scores["training_assigned"] == 1,
        "propensity_score",
    ]

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.hist(
        control,
        bins=25,
        alpha=0.55,
        label="Control",
    )

    plt.hist(
        treated,
        bins=25,
        alpha=0.55,
        label="Treated",
    )

    plt.xlabel(
        "Estimated propensity score"
    )

    plt.ylabel(
        "Observation count"
    )

    plt.title(
        "Propensity Score Overlap"
    )

    plt.legend()

    save_figure(
        filename="01_propensity_overlap.png",
        description=(
            "Distribution of estimated treatment propensities for "
            "control and treated observations."
        ),
        manifest_rows=manifest_rows,
    )


def chart_balance_before_after(
    manifest_rows: list[dict],
) -> None:
    balance = pd.read_csv(
        BALANCE_PATH
    ).copy()

    balance = balance.sort_values(
        "abs_unweighted_smd",
        ascending=True,
    )

    positions = np.arange(
        len(balance)
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        balance["unweighted_smd"],
        positions,
        label="Before weighting",
        marker="o",
    )

    plt.scatter(
        balance["weighted_smd"],
        positions,
        label="After weighting",
        marker="x",
    )

    plt.axvline(
        0.10,
        linestyle="--",
        linewidth=1,
    )

    plt.axvline(
        -0.10,
        linestyle="--",
        linewidth=1,
    )

    plt.axvline(
        0.0,
        linewidth=1,
    )

    plt.yticks(
        positions,
        balance["covariate"],
    )

    plt.xlabel(
        "Standardized mean difference"
    )

    plt.ylabel(
        "Covariate"
    )

    plt.title(
        "Covariate Balance Before and After Weighting"
    )

    plt.legend()

    save_figure(
        filename="02_balance_before_after.png",
        description=(
            "Love-plot style comparison of covariate standardized "
            "mean differences before and after stabilized ATE weighting."
        ),
        manifest_rows=manifest_rows,
    )


def chart_estimator_comparison(
    manifest_rows: list[dict],
) -> None:
    baseline = load_json(
        BASELINE_PATH,
        "naive baseline",
    )

    ipw = load_json(
        IPW_PATH,
        "IPW summary",
    )

    aipw = load_json(
        AIPW_PATH,
        "AIPW summary",
    )

    crossfit = load_json(
        CROSSFIT_PATH,
        "cross-fitted AIPW summary",
    )

    truth = load_json(
        GROUND_TRUTH_PATH,
        "ground-truth summary",
    )

    names = [
        "Naive",
        "IPW",
        "AIPW",
        "Cross-fitted AIPW",
    ]

    estimates = np.array(
        [
            baseline["naive_difference_percentage_points"],
            ipw["ipw_ate_percentage_points"],
            aipw["aipw_ate_percentage_points"],
            crossfit["crossfit_aipw_ate_percentage_points"],
        ],
        dtype=float,
    )

    lower = np.array(
        [
            baseline["confidence_interval_percentage_points"]["lower"],
            ipw["uncertainty"][
                "confidence_interval_percentage_points"
            ]["lower"],
            aipw["confidence_interval_percentage_points"]["lower"],
            crossfit["confidence_interval_percentage_points"]["lower"],
        ],
        dtype=float,
    )

    upper = np.array(
        [
            baseline["confidence_interval_percentage_points"]["upper"],
            ipw["uncertainty"][
                "confidence_interval_percentage_points"
            ]["upper"],
            aipw["confidence_interval_percentage_points"]["upper"],
            crossfit["confidence_interval_percentage_points"]["upper"],
        ],
        dtype=float,
    )

    yerr = np.vstack(
        [
            estimates - lower,
            upper - estimates,
        ]
    )

    positions = np.arange(
        len(names)
    )

    true_ate = float(
        truth[
            "true_effect_summary"
        ]["true_ate_percentage_points"]
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.errorbar(
        positions,
        estimates,
        yerr=yerr,
        fmt="o",
        capsize=5,
    )

    plt.axhline(
        true_ate,
        linestyle="--",
        linewidth=1,
        label="Synthetic true ATE",
    )

    plt.xticks(
        positions,
        names,
        rotation=15,
    )

    plt.ylabel(
        "Effect estimate (percentage points)"
    )

    plt.title(
        "Treatment-Effect Estimate Comparison"
    )

    plt.legend()

    save_figure(
        filename="03_estimator_comparison.png",
        description=(
            "Comparison of naive, IPW, AIPW, and cross-fitted AIPW "
            "effect estimates with confidence intervals and synthetic truth."
        ),
        manifest_rows=manifest_rows,
    )


def chart_sensitivity_analysis(
    manifest_rows: list[dict],
) -> None:
    sensitivity = pd.read_csv(
        SENSITIVITY_PATH
    ).copy()

    sensitivity = sensitivity.sort_values(
        "estimate_percentage_points",
        ascending=True,
    )

    positions = np.arange(
        len(sensitivity)
    )

    plt.figure(
        figsize=(10, 6.5)
    )

    plt.scatter(
        sensitivity["estimate_percentage_points"],
        positions,
    )

    plt.yticks(
        positions,
        sensitivity["scenario"],
    )

    plt.xlabel(
        "Estimated ATE (percentage points)"
    )

    plt.ylabel(
        "Sensitivity scenario"
    )

    plt.title(
        "Sensitivity Analysis of the Average Treatment Effect"
    )

    save_figure(
        filename="04_sensitivity_analysis.png",
        description=(
            "ATE estimates across propensity clipping, common support, "
            "alternate propensity specifications, and reference estimators."
        ),
        manifest_rows=manifest_rows,
    )


def chart_subgroup_effects(
    manifest_rows: list[dict],
) -> None:
    effects = pd.read_csv(
        SEGMENT_EFFECTS_PATH
    ).copy()

    effects["segment_label"] = (
        effects["segment_family"]
        + ": "
        + effects["segment_value"]
    )

    effects = effects.sort_values(
        "estimated_effect_percentage_points",
        ascending=True,
    )

    positions = np.arange(
        len(effects)
    )

    estimate = effects[
        "estimated_effect_percentage_points"
    ].to_numpy(dtype=float)

    lower = effects[
        "ci_lower_percentage_points"
    ].to_numpy(dtype=float)

    upper = effects[
        "ci_upper_percentage_points"
    ].to_numpy(dtype=float)

    yerr = np.vstack(
        [
            estimate - lower,
            upper - estimate,
        ]
    )

    truth = effects[
        "true_effect_percentage_points"
    ].to_numpy(dtype=float)

    plt.figure(
        figsize=(10, 8)
    )

    plt.errorbar(
        estimate,
        positions,
        xerr=yerr,
        fmt="o",
        capsize=4,
        label="Estimated subgroup effect",
    )

    plt.scatter(
        truth,
        positions,
        marker="x",
        label="Synthetic subgroup truth",
    )

    plt.yticks(
        positions,
        effects["segment_label"],
    )

    plt.xlabel(
        "Treatment effect (percentage points)"
    )

    plt.ylabel(
        "Predefined subgroup"
    )

    plt.title(
        "Predefined Subgroup Treatment Effects"
    )

    plt.legend()

    save_figure(
        filename="05_subgroup_effects.png",
        description=(
            "Predefined subgroup AIPW treatment-effect estimates with "
            "confidence intervals and evaluation-only synthetic truth."
        ),
        manifest_rows=manifest_rows,
    )


def chart_cate_validation_failures(
    manifest_rows: list[dict],
) -> None:
    rf = load_json(
        CATE_RF_PATH,
        "random-forest CATE summary",
    )

    ridge = load_json(
        CATE_RIDGE_PATH,
        "regularized DR-learner summary",
    )

    slearner = load_json(
        CATE_SLEARNER_PATH,
        "S-learner summary",
    )

    names = [
        "RF DR-Learner",
        "Regularized DR-Learner",
        "Interaction S-Learner",
    ]

    spearman = np.array(
        [
            rf["metrics"]["spearman_rank_correlation"],
            ridge["regularized"]["metrics"][
                "spearman_rank_correlation"
            ],
            slearner["current_validation"][
                "spearman_rank_correlation"
            ],
        ],
        dtype=float,
    )

    positions = np.arange(
        len(names)
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.bar(
        positions,
        spearman,
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.xticks(
        positions,
        names,
        rotation=12,
    )

    plt.ylabel(
        "Spearman correlation with synthetic true ITE"
    )

    plt.title(
        "Individual CATE Validation Failure"
    )

    save_figure(
        filename="06_cate_validation_failure.png",
        description=(
            "Validation Spearman correlations for the three attempted "
            "individual-level CATE approaches, documenting failure to "
            "recover reliable treatment-benefit ranking."
        ),
        manifest_rows=manifest_rows,
    )


def save_manifest(
    manifest_rows: list[dict],
) -> None:
    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )


def main() -> None:
    ensure_chart_dir()

    manifest_rows: list[dict] = []

    chart_propensity_overlap(
        manifest_rows=manifest_rows
    )

    chart_balance_before_after(
        manifest_rows=manifest_rows
    )

    chart_estimator_comparison(
        manifest_rows=manifest_rows
    )

    chart_sensitivity_analysis(
        manifest_rows=manifest_rows
    )

    chart_subgroup_effects(
        manifest_rows=manifest_rows
    )

    chart_cate_validation_failures(
        manifest_rows=manifest_rows
    )

    save_manifest(
        manifest_rows=manifest_rows
    )

    print(
        "Portfolio Visualizations"
    )
    print(
        "------------------------"
    )

    for row in manifest_rows:
        print(
            f"- {row['path']}"
        )

    print(
        f"- {MANIFEST_PATH}"
    )

    print(
        "\nCharts generated: "
        f"{len(manifest_rows)}"
    )

    print(
        "AI used to generate charts: False"
    )


if __name__ == "__main__":
    main()
