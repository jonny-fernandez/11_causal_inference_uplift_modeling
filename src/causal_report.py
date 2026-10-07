from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


FINAL_EVALUATION_PATH = Path("artifacts/final_evaluation.json")
METHODOLOGY_PATH = Path("config/final_methodology.json")
MODEL_METADATA_PATH = Path("artifacts/model_metadata.json")
SEGMENT_EFFECTS_PATH = Path("artifacts/segment_effects.csv")
SENSITIVITY_PATH = Path("artifacts/sensitivity_analysis.csv")

OUTPUT_PATH = Path("outputs/causal_report.md")


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    return pd.read_csv(path)


def pct(value: float) -> str:
    return f"{value:+.2f} percentage points"


def make_markdown_table(
    dataframe: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str],
    formats: dict[str, str] | None = None,
) -> str:
    formats = formats or {}

    header = "| " + " | ".join(
        labels[column]
        for column in columns
    ) + " |"

    divider = "| " + " | ".join(
        "---"
        for _ in columns
    ) + " |"

    rows: list[str] = []

    for _, row in dataframe[columns].iterrows():
        values: list[str] = []

        for column in columns:
            value = row[column]

            if pd.isna(value):
                display = ""
            elif column in formats:
                display = formats[column].format(value)
            else:
                display = str(value)

            values.append(display)

        rows.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(
        [header, divider, *rows]
    )


def build_report(
    final: dict,
    methodology: dict,
    metadata: dict,
    segment_effects: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> str:
    primary = final["primary_result"]
    diagnostics = final["diagnostics"]
    truth = final[
        "synthetic_ground_truth_evaluation"
    ]
    sensitivity_summary = final["sensitivity"]
    hte = final["heterogeneous_effects"]
    reference = final["reference_estimators"]

    ci = primary[
        "confidence_interval_percentage_points"
    ]

    dataset = metadata["dataset"]
    design = metadata["causal_design"]

    sensitivity_display = sensitivity[
        [
            "scenario",
            "estimator",
            "n",
            "estimate_percentage_points",
            "difference_from_primary_percentage_points",
        ]
    ].copy()

    sensitivity_table = make_markdown_table(
        dataframe=sensitivity_display,
        columns=[
            "scenario",
            "estimator",
            "n",
            "estimate_percentage_points",
            "difference_from_primary_percentage_points",
        ],
        labels={
            "scenario": "Scenario",
            "estimator": "Estimator",
            "n": "N",
            "estimate_percentage_points": "Estimate (pp)",
            "difference_from_primary_percentage_points": "Difference vs Primary (pp)",
        },
        formats={
            "n": "{:,.0f}",
            "estimate_percentage_points": "{:+.2f}",
            "difference_from_primary_percentage_points": "{:+.2f}",
        },
    )

    segment_display = segment_effects[
        [
            "segment_family",
            "segment_value",
            "n",
            "estimated_effect_percentage_points",
            "ci_lower_percentage_points",
            "ci_upper_percentage_points",
            "true_effect_percentage_points",
            "absolute_error_percentage_points",
        ]
    ].copy()

    segment_table = make_markdown_table(
        dataframe=segment_display,
        columns=[
            "segment_family",
            "segment_value",
            "n",
            "estimated_effect_percentage_points",
            "ci_lower_percentage_points",
            "ci_upper_percentage_points",
            "true_effect_percentage_points",
            "absolute_error_percentage_points",
        ],
        labels={
            "segment_family": "Segment Family",
            "segment_value": "Segment",
            "n": "N",
            "estimated_effect_percentage_points": "Estimated Effect (pp)",
            "ci_lower_percentage_points": "CI Lower",
            "ci_upper_percentage_points": "CI Upper",
            "true_effect_percentage_points": "Synthetic Truth (pp)",
            "absolute_error_percentage_points": "Abs. Error (pp)",
        },
        formats={
            "n": "{:,.0f}",
            "estimated_effect_percentage_points": "{:+.2f}",
            "ci_lower_percentage_points": "{:+.2f}",
            "ci_upper_percentage_points": "{:+.2f}",
            "true_effect_percentage_points": "{:+.2f}",
            "absolute_error_percentage_points": "{:.2f}",
        },
    )

    rejected_lines = "\n".join(
        f"- **{item['method']}** — {item['reason']}"
        for item in methodology["rejected_methods"]
    )

    lines = [
        "# Project 11 — Causal Inference & Uplift Modeling System",
        "",
        "## Executive Summary",
        "",
        (
            "Under the stated identification assumptions, the locked "
            f"cross-fitted AIPW analysis estimates an average treatment "
            f"effect of **{primary['estimate_percentage_points']:+.2f} "
            f"percentage points** on the probability of retention or "
            f"promotion within 180 days."
        ),
        "",
        (
            f"The 95% confidence interval is "
            f"**[{ci['lower']:+.2f}, {ci['upper']:+.2f}] percentage points**."
        ),
        "",
        (
            "Observed overlap, weighting stability, covariate balance, "
            "estimator agreement, and sensitivity analysis support the "
            "adjustment design. They do not prove that unmeasured "
            "confounding is absent."
        ),
        "",
        "Individual-level CATE/uplift targeting was rejected after multiple "
        "validation failures. Predefined subgroup heterogeneous-treatment-effect "
        "analysis was retained.",
        "",
        "## Causal Question",
        "",
        (
            "Among the synthetic population represented by the Project 11 "
            "dataset, what is the estimated average effect of assignment to "
            "the training intervention on the probability of being retained "
            "or promoted within 180 days?"
        ),
        "",
        "## Dataset and Design",
        "",
        f"- Dataset rows: **{dataset['rows']:,}**",
        f"- Dataset columns: **{dataset['columns']}**",
        f"- Identifier: `{dataset['id_column']}`",
        f"- Treatment: `{dataset['treatment_column']}`",
        f"- Outcome: `{dataset['outcome_column']}`",
        f"- Primary estimand: **{final['primary_estimand']}**",
        f"- Primary estimator: `{final['primary_estimator']}`",
        f"- Dataset SHA-256: `{dataset['sha256']}`",
        "",
        "### Eligible Pre-Treatment Covariates",
        "",
        *[
            f"- `{feature}`"
            for feature in (
                design["categorical_pre_treatment_covariates"]
                + design["numeric_pre_treatment_covariates"]
            )
        ],
        "",
        "### Excluded / Evaluation-Only Variables",
        "",
        *[
            f"- Post-treatment exclusion: `{feature}`"
            for feature in design["post_treatment_exclusions"]
        ],
        *[
            f"- Evaluation-only synthetic truth: `{feature}`"
            for feature in design["evaluation_only_ground_truth"]
        ],
        "",
        "## Identification Assumptions",
        "",
        "The causal interpretation requires:",
        "",
        "- Consistency.",
        "- Conditional exchangeability / no unmeasured confounding after adjustment.",
        "- Positivity / overlap.",
        "- SUTVA / no interference.",
        "- Correct treatment timing.",
        "- Correct outcome timing.",
        "",
        (
            "Observed balance and propensity overlap are diagnostics for the "
            "observed-data design. They do not verify conditional exchangeability."
        ),
        "",
        "## Baseline and Adjusted Effect Estimates",
        "",
        f"- Naive difference in means: **{reference['naive_difference_percentage_points']:+.2f} pp**",
        f"- Stabilized IPW ATE: **{reference['ipw_ate_percentage_points']:+.2f} pp**",
        f"- AIPW ATE: **{reference['aipw_ate_percentage_points']:+.2f} pp**",
        f"- Cross-fitted AIPW ATE: **{reference['crossfit_aipw_ate_percentage_points']:+.2f} pp**",
        "",
        (
            "The naive estimate is descriptive only. The locked final "
            "estimate is the cross-fitted AIPW result."
        ),
        "",
        "## Overlap, Weighting, and Balance Diagnostics",
        "",
        f"- Propensity overlap valid: **{diagnostics['propensity_overlap_valid']}**",
        (
            f"- Propensity score range: "
            f"**[{diagnostics['propensity_minimum']:.3f}, "
            f"{diagnostics['propensity_maximum']:.3f}]**"
        ),
        f"- Poor-overlap rows: **{diagnostics['poor_overlap_rows']}**",
        f"- Extreme stabilized weights: **{diagnostics['extreme_weight_count']}**",
        f"- Effective sample size retained: **{diagnostics['effective_sample_size_fraction']:.1%}**",
        (
            "- Post-weighting covariate rows at or above "
            f"|SMD| 0.10: **{diagnostics['post_weighting_rows_at_or_above_abs_smd_0_10']}**"
        ),
        (
            f"- Maximum post-weighting |SMD|: "
            f"**{diagnostics['maximum_post_weighting_abs_smd']:.6f}**"
        ),
        "",
        "## Synthetic Ground-Truth Evaluation",
        "",
        (
            "Synthetic truth was used only after estimator construction "
            "for evaluation. It was not used for training or causal estimation."
        ),
        "",
        f"- True synthetic ATE: **{truth['true_ate_percentage_points']:+.2f} pp**",
        (
            f"- Locked estimator signed error vs synthetic truth: "
            f"**{truth['primary_estimator_signed_error_percentage_points']:+.2f} pp**"
        ),
        (
            "- All adjusted confidence intervals covered the true ATE: "
            f"**{truth['all_adjusted_confidence_intervals_cover_true_ate']}**"
        ),
        "",
        "## Sensitivity Analysis",
        "",
        (
            f"The observed estimate range across sensitivity and reference "
            f"scenarios was **{sensitivity_summary['minimum_estimate_percentage_points']:+.2f} "
            f"to {sensitivity_summary['maximum_estimate_percentage_points']:+.2f} pp**, "
            f"a width of only **{sensitivity_summary['range_width_percentage_points']:.2f} pp**."
        ),
        "",
        sensitivity_table,
        "",
        "## Heterogeneous Treatment Effects",
        "",
        f"- Individual CATE status: **{hte['individual_cate_status']}**",
        f"- Subgroup HTE status: **{hte['subgroup_hte_status']}**",
        (
            f"- Mean absolute subgroup error against synthetic truth: "
            f"**{hte['subgroup_mean_absolute_error_percentage_points']:.2f} pp**"
        ),
        (
            "- All subgroup confidence intervals covered the synthetic "
            f"subgroup effect: **{hte['all_subgroup_cis_cover_true_effect']}**"
        ),
        "",
        "### Rejected Individual-Level CATE Methods",
        "",
        rejected_lines,
        "",
        (
            "Because reliable individual-level treatment-benefit ranking was "
            "not recovered, Qini/AUUC were not reported as successful final "
            "portfolio metrics."
        ),
        "",
        "### Predefined Subgroup Effects",
        "",
        segment_table,
        "",
        "## Reproducibility",
        "",
        f"- Python: **{metadata['runtime']['python_version']}**",
        f"- Random seed: **{metadata['randomness']['project_random_seed']}**",
        (
            f"- Missing key source files at metadata generation: "
            f"**{len(metadata['file_integrity']['missing_key_source_files'])}**"
        ),
        (
            f"- Missing key artifacts at metadata generation: "
            f"**{len(metadata['file_integrity']['missing_key_artifacts'])}**"
        ),
        "",
        "Key deterministic artifacts include:",
        "",
        "- `artifacts/final_evaluation.json`",
        "- `artifacts/final_evaluation_metrics.csv`",
        "- `artifacts/model_metadata.json`",
        "- `artifacts/segment_effects.csv`",
        "- `artifacts/sensitivity_analysis.csv`",
        "- `config/final_methodology.json`",
        "",
        "## AI Boundary",
        "",
        (
            "No AI system calculated or modified treatment effects, "
            "propensity scores, weights, confidence intervals, balance "
            "statistics, subgroup effects, or sensitivity results."
        ),
        "",
        (
            "Any later AI component may only summarize verified aggregate "
            "deterministic artifacts and must preserve the stated causal "
            "assumptions and limitations."
        ),
        "",
        "## Final Interpretation",
        "",
        final["final_interpretation"],
        "",
        "## Limitations",
        "",
        "- Conditional exchangeability cannot be proven from observed data.",
        "- Good overlap and balance do not rule out unmeasured confounding.",
        "- The data are synthetic and portfolio-safe rather than a real intervention dataset.",
        "- Individual-level CATE/uplift ranking was not validated successfully.",
        "- Subgroup HTE estimates should not be interpreted as guaranteed individual outcomes.",
        "",
        "## Status",
        "",
        "**Deterministic causal analysis complete.**",
        "",
        (
            "The next project layer, if enabled, is a controlled AI "
            "interpretation of these verified outputs. AI remains optional "
            "and must not alter deterministic results."
        ),
        "",
    ]

    return "\n".join(lines)


def save_report(report: str) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def print_summary(report: str) -> None:
    line_count = len(
        report.splitlines()
    )

    print("Deterministic Causal Report")
    print("---------------------------")
    print(
        f"Report lines: {line_count:,}"
    )
    print(
        f"Artifact saved: {OUTPUT_PATH}"
    )
    print(
        "AI used to generate report content: False"
    )
    print(
        "Report is constructed only from verified deterministic artifacts."
    )


def main() -> None:
    final = load_json(
        FINAL_EVALUATION_PATH,
        "final evaluation",
    )

    methodology = load_json(
        METHODOLOGY_PATH,
        "final methodology",
    )

    metadata = load_json(
        MODEL_METADATA_PATH,
        "model metadata",
    )

    segment_effects = load_csv(
        SEGMENT_EFFECTS_PATH,
        "segment effects",
    )

    sensitivity = load_csv(
        SENSITIVITY_PATH,
        "sensitivity analysis",
    )

    report = build_report(
        final=final,
        methodology=methodology,
        metadata=metadata,
        segment_effects=segment_effects,
        sensitivity=sensitivity,
    )

    save_report(
        report=report
    )

    print_summary(
        report=report
    )


if __name__ == "__main__":
    main()
