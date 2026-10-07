from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_EVALUATION_PATH = Path("artifacts/final_evaluation.json")
SEGMENT_SUMMARY_PATH = Path("artifacts/segment_effects_summary.json")
SENSITIVITY_SUMMARY_PATH = Path(
    "artifacts/sensitivity_analysis_summary.json"
)
METHODOLOGY_PATH = Path("config/final_methodology.json")

INPUT_PATH = Path("artifacts/ai_interpretation_input.json")
OUTPUT_JSON_PATH = Path("artifacts/ai_interpretation.json")
STATUS_PATH = Path("artifacts/ai_interpretation_status.json")
NARRATIVE_PATH = Path("outputs/ai_causal_narrative.md")


ALLOWED_FACT_IDS = {
    "primary_effect",
    "confidence_interval",
    "overlap",
    "weights",
    "balance",
    "sensitivity",
    "synthetic_truth",
    "individual_cate",
    "subgroup_hte",
    "identification_limitations",
    "ai_boundary",
}


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def bool_from_env(
    name: str,
    default: bool = False,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def build_verified_payload(
    final: dict,
    segments: dict,
    sensitivity: dict,
    methodology: dict,
) -> dict:
    primary = final["primary_result"]
    diagnostics = final["diagnostics"]
    truth = final[
        "synthetic_ground_truth_evaluation"
    ]
    hte = final["heterogeneous_effects"]

    ci = primary[
        "confidence_interval_percentage_points"
    ]

    facts = {
        "primary_effect": (
            "The locked primary estimator is cross-fitted AIPW. "
            f"The estimated ATE is "
            f"{primary['estimate_percentage_points']:+.2f} "
            "percentage points."
        ),
        "confidence_interval": (
            f"The {int(primary['confidence_level'] * 100)} percent "
            "confidence interval runs from "
            f"{ci['lower']:+.2f} to {ci['upper']:+.2f} "
            "percentage points."
        ),
        "overlap": (
            "Observed propensity overlap passed the project diagnostic. "
            f"Poor-overlap rows: {diagnostics['poor_overlap_rows']}."
        ),
        "weights": (
            f"Extreme stabilized weights: "
            f"{diagnostics['extreme_weight_count']}. "
            f"Effective sample size retained: "
            f"{diagnostics['effective_sample_size_fraction']:.1%}."
        ),
        "balance": (
            "Post-weighting covariate balance passed the working threshold. "
            f"Rows at or above absolute SMD 0.10: "
            f"{diagnostics['post_weighting_rows_at_or_above_abs_smd_0_10']}. "
            f"Maximum post-weighting absolute SMD: "
            f"{diagnostics['maximum_post_weighting_abs_smd']:.6f}."
        ),
        "sensitivity": (
            "Sensitivity estimates remained tightly clustered. "
            f"The observed range was "
            f"{sensitivity['minimum_estimate_percentage_points']:+.2f} "
            f"to {sensitivity['maximum_estimate_percentage_points']:+.2f} "
            "percentage points."
        ),
        "synthetic_truth": (
            "Synthetic ground truth was evaluation-only and was not used "
            "for training or causal estimation. "
            f"The true synthetic ATE was "
            f"{truth['true_ate_percentage_points']:+.2f} "
            "percentage points."
        ),
        "individual_cate": (
            "Individual-level CATE/uplift targeting was rejected after "
            "multiple validation failures. Qini and AUUC are not reported "
            "as successful final metrics."
        ),
        "subgroup_hte": (
            "Predefined subgroup HTE analysis was retained. "
            f"The mean absolute subgroup error against synthetic truth was "
            f"{hte['subgroup_mean_absolute_error_percentage_points']:.2f} "
            "percentage points."
        ),
        "identification_limitations": (
            "Causal interpretation depends on consistency, conditional "
            "exchangeability, positivity, no interference, and correct "
            "temporal ordering. Observed diagnostics do not prove absence "
            "of unmeasured confounding."
        ),
        "ai_boundary": (
            "AI did not calculate or modify treatment effects, propensity "
            "scores, weights, confidence intervals, balance statistics, "
            "subgroup effects, or sensitivity results."
        ),
    }

    return {
        "purpose": (
            "Controlled qualitative interpretation of verified aggregate "
            "causal-analysis outputs."
        ),
        "primary_estimand": final[
            "primary_estimand"
        ],
        "primary_estimator": final[
            "primary_estimator"
        ],
        "verified_facts": facts,
        "methodology_status": methodology[
            "status"
        ],
        "individual_targeting_accepted": False,
        "subgroup_hte_retained": (
            methodology[
                "heterogeneous_effects"
            ]["subgroup_hte_status"]
            == "retained"
        ),
        "segment_rows": int(
            segments["n_segment_rows"]
        ),
        "input_policy": (
            "Aggregate verified artifacts only. No raw row-level records "
            "are supplied to the AI layer."
        ),
    }


def all_generated_text(
    payload: dict,
) -> str:
    parts = [
        payload.get(
            "executive_interpretation",
            "",
        ),
        payload.get(
            "diagnostic_interpretation",
            "",
        ),
        payload.get(
            "heterogeneity_interpretation",
            "",
        ),
        *payload.get(
            "limitations",
            [],
        ),
    ]

    return "\n".join(
        str(part)
        for part in parts
    )


def validate_ai_output(
    payload: dict,
) -> None:
    evidence_ids = set(
        payload.get(
            "evidence_fact_ids",
            [],
        )
    )

    invalid_ids = (
        evidence_ids
        - ALLOWED_FACT_IDS
    )

    if invalid_ids:
        raise ValueError(
            "AI output referenced unsupported fact IDs: "
            + ", ".join(
                sorted(invalid_ids)
            )
        )

    if not evidence_ids:
        raise ValueError(
            "AI output must reference at least one verified fact ID."
        )

    checks = payload.get(
        "guardrail_checks",
        {},
    )

    prohibited_true_checks = {
        "calculated_new_causal_values",
        "overrode_deterministic_diagnostics",
        "claimed_unmeasured_confounding_absent",
        "recommended_individual_targeting",
        "invented_causal_mechanism",
    }

    failed_checks = [
        key
        for key in prohibited_true_checks
        if checks.get(key) is not False
    ]

    if failed_checks:
        raise ValueError(
            "AI guardrail validation failed: "
            + ", ".join(
                sorted(failed_checks)
            )
        )

    text = all_generated_text(
        payload
    )

    if re.search(r"\d", text):
        raise ValueError(
            "AI-generated qualitative prose contains digits. "
            "Numeric facts must remain deterministic and are inserted "
            "outside the AI-generated prose."
        )

    if "individual targeting" in text.lower():
        if "not" not in text.lower() and "reject" not in text.lower():
            raise ValueError(
                "AI narrative appears to recommend individual targeting."
            )


def build_markdown(
    ai_output: dict,
    final: dict,
) -> str:
    primary = final[
        "primary_result"
    ]

    ci = primary[
        "confidence_interval_percentage_points"
    ]

    limitations = "\n".join(
        f"- {item}"
        for item in ai_output[
            "limitations"
        ]
    )

    facts_used = "\n".join(
        f"- `{fact_id}`"
        for fact_id in ai_output[
            "evidence_fact_ids"
        ]
    )

    return "\n".join(
        [
            "# Controlled AI Causal Interpretation",
            "",
            "## Deterministic Result",
            "",
            (
                f"- Locked cross-fitted AIPW ATE: "
                f"**{primary['estimate_percentage_points']:+.2f} "
                "percentage points**"
            ),
            (
                f"- Confidence interval: "
                f"**[{ci['lower']:+.2f}, {ci['upper']:+.2f}] "
                "percentage points**"
            ),
            "",
            (
                "These numeric values are inserted directly from the "
                "verified deterministic final-evaluation artifact. "
                "They were not calculated by the AI model."
            ),
            "",
            "## AI Interpretation",
            "",
            ai_output[
                "executive_interpretation"
            ],
            "",
            "## Diagnostic Interpretation",
            "",
            ai_output[
                "diagnostic_interpretation"
            ],
            "",
            "## Heterogeneous-Effect Interpretation",
            "",
            ai_output[
                "heterogeneity_interpretation"
            ],
            "",
            "## Limitations",
            "",
            limitations,
            "",
            "## Verified Fact IDs Used",
            "",
            facts_used,
            "",
            "## Guardrail Status",
            "",
            "- New causal values calculated by AI: **False**",
            "- Deterministic diagnostics overridden: **False**",
            "- Absence of unmeasured confounding claimed: **False**",
            "- Individual targeting recommended: **False**",
            "- Causal mechanism invented: **False**",
            "",
            (
                "The AI layer is optional and interpretive only. "
                "The deterministic causal pipeline remains valid "
                "without this narrative."
            ),
            "",
        ]
    )


def save_status(
    status: str,
    message: str,
    model: str | None,
    response_id: str | None = None,
) -> None:
    save_json(
        STATUS_PATH,
        {
            "status": status,
            "message": message,
            "model": model,
            "response_id": response_id,
            "timestamp_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "deterministic_pipeline_affected": False,
        },
    )


def run_ai(
    verified_payload: dict,
    model: str,
) -> tuple[dict, str | None]:
    try:
        from openai import OpenAI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            "Optional AI dependencies are unavailable. "
            "Install only the missing optional dependency if this "
            "interpretation layer is enabled."
        ) from exc

    class GuardrailChecks(
        BaseModel
    ):
        calculated_new_causal_values: bool
        overrode_deterministic_diagnostics: bool
        claimed_unmeasured_confounding_absent: bool
        recommended_individual_targeting: bool
        invented_causal_mechanism: bool

    class CausalInterpretation(
        BaseModel
    ):
        executive_interpretation: str
        diagnostic_interpretation: str
        heterogeneity_interpretation: str
        limitations: list[str]
        evidence_fact_ids: list[str]
        guardrail_checks: GuardrailChecks

    client = OpenAI()

    system_prompt = """
You are the controlled interpretation layer for a causal-inference portfolio project.

Rules:
- Use only the verified aggregate facts supplied by the user message.
- Do not calculate, derive, revise, or estimate any causal quantity.
- Do not introduce any new numeric quantity.
- Do not include digits in any prose field.
- Do not spell out new numeric quantities in words.
- Do not claim that diagnostics prove absence of unmeasured confounding.
- Use conditional causal language tied to the stated identification assumptions.
- Do not invent mechanisms explaining why the treatment works.
- Do not recommend individual-level targeting because individual CATE/uplift validation failed.
- You may describe predefined subgroup heterogeneity as retained descriptive analysis.
- Keep the interpretation concise and technical.
- evidence_fact_ids must contain only IDs present in verified_facts.
- Every guardrail check must be false.
""".strip()

    user_prompt = json.dumps(
        verified_payload,
        indent=2,
        sort_keys=True,
    )

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text_format=CausalInterpretation,
    )

    parsed = response.output_parsed

    if parsed is None:
        raise RuntimeError(
            "The API returned no parsed structured output."
        )

    if hasattr(
        parsed,
        "model_dump",
    ):
        payload = parsed.model_dump()
    else:
        payload = parsed.dict()

    response_id = getattr(
        response,
        "id",
        None,
    )

    return payload, response_id


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    final = load_json(
        FINAL_EVALUATION_PATH,
        "final evaluation",
    )

    segments = load_json(
        SEGMENT_SUMMARY_PATH,
        "segment summary",
    )

    sensitivity = load_json(
        SENSITIVITY_SUMMARY_PATH,
        "sensitivity summary",
    )

    methodology = load_json(
        METHODOLOGY_PATH,
        "locked methodology",
    )

    verified_payload = build_verified_payload(
        final=final,
        segments=segments,
        sensitivity=sensitivity,
        methodology=methodology,
    )

    save_json(
        INPUT_PATH,
        verified_payload,
    )

    enabled = bool_from_env(
        "AI_EXPLANATIONS_ENABLED",
        default=False,
    )

    model = os.getenv(
        "OPENAI_MODEL",
        "",
    ).strip()

    if not enabled:
        message = (
            "AI interpretation is disabled. "
            "Set AI_EXPLANATIONS_ENABLED=true in the existing .env "
            "only when you want to run the optional AI layer."
        )

        save_status(
            status="disabled",
            message=message,
            model=model or None,
        )

        print(
            "Controlled AI Interpretation"
        )
        print(
            "----------------------------"
        )
        print(message)
        print(
            f"Verified aggregate input saved: "
            f"{INPUT_PATH}"
        )
        print(
            "Deterministic causal pipeline remains complete."
        )
        return

    if not os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip():
        message = (
            "AI interpretation was enabled, but OPENAI_API_KEY "
            "is not available in the environment."
        )

        save_status(
            status="skipped_missing_api_key",
            message=message,
            model=model or None,
        )

        print(message)
        print(
            "Deterministic causal pipeline remains complete."
        )
        return

    if not model:
        message = (
            "AI interpretation was enabled, but OPENAI_MODEL "
            "is not configured."
        )

        save_status(
            status="skipped_missing_model",
            message=message,
            model=None,
        )

        print(message)
        print(
            "Deterministic causal pipeline remains complete."
        )
        return

    try:
        ai_output, response_id = run_ai(
            verified_payload=verified_payload,
            model=model,
        )

        validate_ai_output(
            payload=ai_output,
        )

        save_json(
            OUTPUT_JSON_PATH,
            ai_output,
        )

        narrative = build_markdown(
            ai_output=ai_output,
            final=final,
        )

        NARRATIVE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        NARRATIVE_PATH.write_text(
            narrative,
            encoding="utf-8",
        )

        save_status(
            status="success",
            message=(
                "Structured AI interpretation completed and "
                "passed deterministic guardrail validation."
            ),
            model=model,
            response_id=response_id,
        )

        print(
            "Controlled AI Interpretation"
        )
        print(
            "----------------------------"
        )
        print(
            f"Model: {model}"
        )
        print(
            "Structured output validation: PASSED"
        )
        print(
            "Raw row-level records sent to AI: False"
        )
        print(
            "AI calculated causal values: False"
        )
        print(
            f"Verified input: {INPUT_PATH}"
        )
        print(
            f"Structured output: {OUTPUT_JSON_PATH}"
        )
        print(
            f"Narrative: {NARRATIVE_PATH}"
        )
        print(
            f"Status: {STATUS_PATH}"
        )

    except Exception as exc:
        message = (
            f"Optional AI interpretation failed: "
            f"{type(exc).__name__}: {exc}"
        )

        save_status(
            status="failed",
            message=message,
            model=model,
        )

        print(
            "Controlled AI Interpretation"
        )
        print(
            "----------------------------"
        )
        print(message)
        print(
            "The failure was isolated. "
            "Deterministic causal artifacts remain valid."
        )


if __name__ == "__main__":
    main()
