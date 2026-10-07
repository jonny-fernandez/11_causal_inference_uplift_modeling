from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMN = "person_id"
TRUE_ITE_COLUMN = "true_individual_treatment_effect"
ESTIMATED_CATE_COLUMN = "estimated_cate"

SCORES_PATH = Path("artifacts/cate_evaluation_scores.csv")
SUMMARY_PATH = Path("artifacts/cate_diagnostics_summary.json")
DECILES_PATH = Path("artifacts/cate_diagnostics_deciles.csv")


def load_scores() -> pd.DataFrame:
    if not SCORES_PATH.exists():
        raise FileNotFoundError(
            f"Missing CATE evaluation artifact: {SCORES_PATH}. "
            "Run `python -m src.cate` first."
        )

    scores = pd.read_csv(SCORES_PATH)

    required = {
        ID_COLUMN,
        TRUE_ITE_COLUMN,
        ESTIMATED_CATE_COLUMN,
    }

    missing = sorted(required.difference(scores.columns))

    if missing:
        raise ValueError(
            "CATE evaluation artifact is missing required columns: "
            + ", ".join(missing)
        )

    if scores[ID_COLUMN].duplicated().any():
        raise ValueError(
            "CATE evaluation scores contain duplicate person IDs."
        )

    if scores[
        [TRUE_ITE_COLUMN, ESTIMATED_CATE_COLUMN]
    ].isna().any().any():
        raise ValueError(
            "CATE evaluation scores contain missing treatment-effect values."
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


def add_estimated_deciles(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    result = scores.copy()

    result["estimated_cate_decile"] = (
        pd.qcut(
            result[ESTIMATED_CATE_COLUMN].rank(
                method="first",
                ascending=False,
            ),
            q=10,
            labels=False,
        )
        + 1
    ).astype(int)

    return result


def summarize_deciles(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    deciles = (
        scores.groupby(
            "estimated_cate_decile",
            as_index=False,
        )
        .agg(
            count=(ID_COLUMN, "size"),
            estimated_cate_mean=(
                ESTIMATED_CATE_COLUMN,
                "mean",
            ),
            estimated_cate_min=(
                ESTIMATED_CATE_COLUMN,
                "min",
            ),
            estimated_cate_max=(
                ESTIMATED_CATE_COLUMN,
                "max",
            ),
            true_ite_mean=(
                TRUE_ITE_COLUMN,
                "mean",
            ),
            true_ite_median=(
                TRUE_ITE_COLUMN,
                "median",
            ),
        )
        .sort_values(
            "estimated_cate_decile",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    deciles["calibration_error"] = (
        deciles["estimated_cate_mean"]
        - deciles["true_ite_mean"]
    )

    return deciles


def top_quartile_diagnostics(
    scores: pd.DataFrame,
) -> dict:
    true_cutoff = float(
        scores[TRUE_ITE_COLUMN].quantile(0.75)
    )

    estimated_cutoff = float(
        scores[ESTIMATED_CATE_COLUMN].quantile(0.75)
    )

    true_top = (
        scores[TRUE_ITE_COLUMN]
        >= true_cutoff
    )

    estimated_top = (
        scores[ESTIMATED_CATE_COLUMN]
        >= estimated_cutoff
    )

    intersection = int(
        (true_top & estimated_top).sum()
    )

    true_top_count = int(true_top.sum())
    estimated_top_count = int(estimated_top.sum())

    precision = (
        intersection / estimated_top_count
        if estimated_top_count
        else 0.0
    )

    recall = (
        intersection / true_top_count
        if true_top_count
        else 0.0
    )

    union = int(
        (true_top | estimated_top).sum()
    )

    jaccard = (
        intersection / union
        if union
        else 0.0
    )

    return {
        "true_top_quartile_cutoff": true_cutoff,
        "estimated_top_quartile_cutoff": estimated_cutoff,
        "true_top_group_count": true_top_count,
        "true_top_group_fraction": float(true_top.mean()),
        "estimated_top_group_count": estimated_top_count,
        "estimated_top_group_fraction": float(
            estimated_top.mean()
        ),
        "intersection_count": intersection,
        "precision": float(precision),
        "recall": float(recall),
        "jaccard": float(jaccard),
        "note": (
            "If the true treatment effect has tied/discrete values, "
            "a >= 75th-percentile cutoff can produce a true 'top quartile' "
            "larger than 25% of the evaluation sample. Precision/recall/Jaccard "
            "are therefore more informative than the earlier overlap statistic."
        ),
    }


def ranking_diagnostics(
    scores: pd.DataFrame,
    deciles: pd.DataFrame,
) -> dict:
    top_estimated = deciles.loc[
        deciles["estimated_cate_decile"] == 1,
        "true_ite_mean",
    ].iloc[0]

    bottom_estimated = deciles.loc[
        deciles["estimated_cate_decile"] == 10,
        "true_ite_mean",
    ].iloc[0]

    observed_true_spread = float(
        top_estimated - bottom_estimated
    )

    true_ranked = scores.sort_values(
        TRUE_ITE_COLUMN,
        ascending=False,
    ).reset_index(drop=True)

    decile_size = max(
        1,
        len(true_ranked) // 10,
    )

    oracle_top_mean = float(
        true_ranked.head(decile_size)[
            TRUE_ITE_COLUMN
        ].mean()
    )

    oracle_bottom_mean = float(
        true_ranked.tail(decile_size)[
            TRUE_ITE_COLUMN
        ].mean()
    )

    oracle_spread = (
        oracle_top_mean - oracle_bottom_mean
    )

    normalized_spread_recovery = (
        observed_true_spread / oracle_spread
        if not np.isclose(oracle_spread, 0.0)
        else 0.0
    )

    decile_rank = deciles[
        "estimated_cate_decile"
    ].astype(float)

    true_mean_by_decile = deciles[
        "true_ite_mean"
    ].astype(float)

    decile_monotonicity = float(
        np.corrcoef(
            decile_rank,
            true_mean_by_decile,
        )[0, 1]
    )

    return {
        "spearman_rank_correlation": rank_correlation(
            scores[ESTIMATED_CATE_COLUMN],
            scores[TRUE_ITE_COLUMN],
        ),
        "top_estimated_decile_true_ite_mean": float(
            top_estimated
        ),
        "bottom_estimated_decile_true_ite_mean": float(
            bottom_estimated
        ),
        "observed_top_minus_bottom_true_ite_spread": (
            observed_true_spread
        ),
        "oracle_top_decile_true_ite_mean": oracle_top_mean,
        "oracle_bottom_decile_true_ite_mean": oracle_bottom_mean,
        "oracle_top_minus_bottom_true_ite_spread": float(
            oracle_spread
        ),
        "normalized_spread_recovery": float(
            normalized_spread_recovery
        ),
        "decile_number_vs_true_mean_correlation": (
            decile_monotonicity
        ),
    }


def dispersion_diagnostics(
    scores: pd.DataFrame,
) -> dict:
    estimated = scores[
        ESTIMATED_CATE_COLUMN
    ].astype(float)

    truth = scores[
        TRUE_ITE_COLUMN
    ].astype(float)

    estimated_std = float(
        estimated.std(ddof=1)
    )

    true_std = float(
        truth.std(ddof=1)
    )

    dispersion_ratio = (
        estimated_std / true_std
        if not np.isclose(true_std, 0.0)
        else np.nan
    )

    return {
        "estimated_cate_mean": float(
            estimated.mean()
        ),
        "true_ite_mean": float(
            truth.mean()
        ),
        "estimated_cate_std": estimated_std,
        "true_ite_std": true_std,
        "estimated_to_true_std_ratio": float(
            dispersion_ratio
        ),
        "negative_estimated_cate_count": int(
            (estimated < 0.0).sum()
        ),
        "negative_estimated_cate_fraction": float(
            (estimated < 0.0).mean()
        ),
        "estimated_above_true_max_count": int(
            (estimated > truth.max()).sum()
        ),
        "estimated_above_true_max_fraction": float(
            (estimated > truth.max()).mean()
        ),
        "estimated_below_true_min_count": int(
            (estimated < truth.min()).sum()
        ),
        "estimated_below_true_min_fraction": float(
            (estimated < truth.min()).mean()
        ),
    }


def build_summary(
    scores: pd.DataFrame,
    deciles: pd.DataFrame,
) -> dict:
    unique_true_effects = int(
        scores[TRUE_ITE_COLUMN].nunique()
    )

    return {
        "evaluation_rows": int(len(scores)),
        "unique_true_ite_values": unique_true_effects,
        "pearson_correlation": float(
            np.corrcoef(
                scores[ESTIMATED_CATE_COLUMN],
                scores[TRUE_ITE_COLUMN],
            )[0, 1]
        ),
        "top_quartile": top_quartile_diagnostics(
            scores=scores,
        ),
        "ranking": ranking_diagnostics(
            scores=scores,
            deciles=deciles,
        ),
        "dispersion": dispersion_diagnostics(
            scores=scores,
        ),
        "decision_policy": (
            "This checkpoint diagnoses the failed first CATE model. "
            "It does not tune a replacement model against synthetic truth. "
            "Any replacement method must be justified from these failure "
            "patterns and then evaluated as a separately documented model."
        ),
    }


def save_outputs(
    deciles: pd.DataFrame,
    summary: dict,
) -> None:
    DECILES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    deciles.to_csv(
        DECILES_PATH,
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
    deciles: pd.DataFrame,
    summary: dict,
) -> None:
    top = summary["top_quartile"]
    ranking = summary["ranking"]
    dispersion = summary["dispersion"]

    print("CATE Failure Diagnostics")
    print("------------------------")
    print(
        f"Evaluation rows: "
        f"{summary['evaluation_rows']:,}"
    )
    print(
        f"Unique true ITE values: "
        f"{summary['unique_true_ite_values']}"
    )

    print("\nRank recovery")
    print(
        "Pearson correlation: "
        f"{summary['pearson_correlation']:.6f}"
    )
    print(
        "Spearman correlation: "
        f"{ranking['spearman_rank_correlation']:.6f}"
    )
    print(
        "Top estimated decile true ITE mean: "
        f"{ranking['top_estimated_decile_true_ite_mean']:.6f}"
    )
    print(
        "Bottom estimated decile true ITE mean: "
        f"{ranking['bottom_estimated_decile_true_ite_mean']:.6f}"
    )
    print(
        "Observed top-minus-bottom true ITE spread: "
        f"{ranking['observed_top_minus_bottom_true_ite_spread']:.6f}"
    )
    print(
        "Oracle top-minus-bottom true ITE spread: "
        f"{ranking['oracle_top_minus_bottom_true_ite_spread']:.6f}"
    )
    print(
        "Normalized spread recovery: "
        f"{ranking['normalized_spread_recovery']:.2%}"
    )
    print(
        "Decile number vs true-effect mean correlation: "
        f"{ranking['decile_number_vs_true_mean_correlation']:.6f}"
    )

    print("\nTop-quartile diagnostic")
    print(
        "True >= 75th-percentile group: "
        f"{top['true_top_group_count']:,} "
        f"({top['true_top_group_fraction']:.1%})"
    )
    print(
        "Estimated >= 75th-percentile group: "
        f"{top['estimated_top_group_count']:,} "
        f"({top['estimated_top_group_fraction']:.1%})"
    )
    print(
        f"Precision: {top['precision']:.2%}"
    )
    print(
        f"Recall:    {top['recall']:.2%}"
    )
    print(
        f"Jaccard:   {top['jaccard']:.2%}"
    )

    print("\nDispersion")
    print(
        "Estimated CATE std: "
        f"{dispersion['estimated_cate_std']:.6f}"
    )
    print(
        "True ITE std:       "
        f"{dispersion['true_ite_std']:.6f}"
    )
    print(
        "Estimated/true std ratio: "
        f"{dispersion['estimated_to_true_std_ratio']:.2f}x"
    )
    print(
        "Negative estimated CATE rows: "
        f"{dispersion['negative_estimated_cate_count']:,} "
        f"({dispersion['negative_estimated_cate_fraction']:.1%})"
    )
    print(
        "Estimated CATE above true max: "
        f"{dispersion['estimated_above_true_max_count']:,} "
        f"({dispersion['estimated_above_true_max_fraction']:.1%})"
    )
    print(
        "Estimated CATE below true min: "
        f"{dispersion['estimated_below_true_min_count']:,} "
        f"({dispersion['estimated_below_true_min_fraction']:.1%})"
    )

    print("\nEstimated-CATE deciles")
    display = deciles[
        [
            "estimated_cate_decile",
            "count",
            "estimated_cate_mean",
            "true_ite_mean",
            "calibration_error",
        ]
    ]
    print(
        display.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nArtifacts saved:")
    print(f"- {DECILES_PATH}")
    print(f"- {SUMMARY_PATH}")

    print("\nDecision policy:")
    print(summary["decision_policy"])


def main() -> None:
    scores = load_scores()

    scored = add_estimated_deciles(
        scores=scores,
    )

    deciles = summarize_deciles(
        scores=scored,
    )

    summary = build_summary(
        scores=scored,
        deciles=deciles,
    )

    save_outputs(
        deciles=deciles,
        summary=summary,
    )

    print_summary(
        deciles=deciles,
        summary=summary,
    )


if __name__ == "__main__":
    main()
