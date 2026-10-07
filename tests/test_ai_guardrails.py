from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.ai_interpretation import (
    ALLOWED_FACT_IDS,
    validate_ai_output,
)


AI_INPUT_PATH = Path("artifacts/ai_interpretation_input.json")
AI_OUTPUT_PATH = Path("artifacts/ai_interpretation.json")
AI_STATUS_PATH = Path("artifacts/ai_interpretation_status.json")
AI_NARRATIVE_PATH = Path("outputs/ai_causal_narrative.md")
GITIGNORE_PATH = Path(".gitignore")


def load_json(path: Path) -> dict:
    assert path.exists(), f"Expected file does not exist: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def test_gitignore_excludes_env_file():
    text = GITIGNORE_PATH.read_text(
        encoding="utf-8"
    )

    lines = {
        line.strip()
        for line in text.splitlines()
        if line.strip()
    }

    assert ".env" in lines


def test_ai_input_exists_and_is_aggregate_only():
    payload = load_json(AI_INPUT_PATH)

    assert payload["input_policy"].startswith(
        "Aggregate verified artifacts only"
    )

    keys = set(recursive_keys(payload))

    forbidden_keys = {
        "person_id",
        "firstname",
        "lastname",
        "email",
        "ssn",
        "raw_rows",
        "row_level_records",
    }

    assert keys.isdisjoint(forbidden_keys)


def test_ai_input_contains_only_expected_verified_fact_ids():
    payload = load_json(AI_INPUT_PATH)

    fact_ids = set(
        payload["verified_facts"].keys()
    )

    assert fact_ids == ALLOWED_FACT_IDS


def test_ai_status_is_isolated_from_deterministic_pipeline():
    status = load_json(AI_STATUS_PATH)

    assert status["deterministic_pipeline_affected"] is False
    assert status["status"] in {
        "disabled",
        "skipped_missing_api_key",
        "skipped_missing_model",
        "failed",
        "success",
    }


def test_saved_ai_output_passes_guardrail_validator_when_available():
    if not AI_OUTPUT_PATH.exists():
        pytest.skip(
            "Optional AI interpretation output was not generated."
        )

    payload = load_json(AI_OUTPUT_PATH)
    validate_ai_output(payload)


def test_saved_ai_output_uses_only_allowed_evidence_ids_when_available():
    if not AI_OUTPUT_PATH.exists():
        pytest.skip(
            "Optional AI interpretation output was not generated."
        )

    payload = load_json(AI_OUTPUT_PATH)

    assert set(
        payload["evidence_fact_ids"]
    ).issubset(ALLOWED_FACT_IDS)


def test_saved_ai_prose_contains_no_digits_when_available():
    if not AI_OUTPUT_PATH.exists():
        pytest.skip(
            "Optional AI interpretation output was not generated."
        )

    payload = load_json(AI_OUTPUT_PATH)

    text = "\n".join(
        [
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
    )

    assert re.search(r"\d", text) is None


def test_guardrail_validator_rejects_new_numeric_prose():
    payload = {
        "executive_interpretation": (
            "The estimated effect was 12 percent."
        ),
        "diagnostic_interpretation": (
            "Observed diagnostics were acceptable."
        ),
        "heterogeneity_interpretation": (
            "Individual targeting remains rejected."
        ),
        "limitations": [
            "Unmeasured confounding may remain."
        ],
        "evidence_fact_ids": [
            "primary_effect"
        ],
        "guardrail_checks": {
            "calculated_new_causal_values": False,
            "overrode_deterministic_diagnostics": False,
            "claimed_unmeasured_confounding_absent": False,
            "recommended_individual_targeting": False,
            "invented_causal_mechanism": False,
        },
    }

    with pytest.raises(
        ValueError,
        match="contains digits",
    ):
        validate_ai_output(payload)


def test_guardrail_validator_rejects_unknown_fact_id():
    payload = {
        "executive_interpretation": (
            "The estimated effect is positive under the stated assumptions."
        ),
        "diagnostic_interpretation": (
            "Observed diagnostics support the adjustment design."
        ),
        "heterogeneity_interpretation": (
            "Individual targeting remains rejected."
        ),
        "limitations": [
            "Unmeasured confounding may remain."
        ],
        "evidence_fact_ids": [
            "invented_fact"
        ],
        "guardrail_checks": {
            "calculated_new_causal_values": False,
            "overrode_deterministic_diagnostics": False,
            "claimed_unmeasured_confounding_absent": False,
            "recommended_individual_targeting": False,
            "invented_causal_mechanism": False,
        },
    }

    with pytest.raises(
        ValueError,
        match="unsupported fact IDs",
    ):
        validate_ai_output(payload)


def test_guardrail_validator_rejects_positive_prohibited_check():
    payload = {
        "executive_interpretation": (
            "The estimated effect is positive under the stated assumptions."
        ),
        "diagnostic_interpretation": (
            "Observed diagnostics support the adjustment design."
        ),
        "heterogeneity_interpretation": (
            "Individual targeting remains rejected."
        ),
        "limitations": [
            "Unmeasured confounding may remain."
        ],
        "evidence_fact_ids": [
            "primary_effect"
        ],
        "guardrail_checks": {
            "calculated_new_causal_values": True,
            "overrode_deterministic_diagnostics": False,
            "claimed_unmeasured_confounding_absent": False,
            "recommended_individual_targeting": False,
            "invented_causal_mechanism": False,
        },
    }

    with pytest.raises(
        ValueError,
        match="guardrail validation failed",
    ):
        validate_ai_output(payload)


def test_ai_narrative_inserts_deterministic_numbers_only_when_available():
    if not AI_NARRATIVE_PATH.exists():
        pytest.skip(
            "Optional AI narrative was not generated."
        )

    narrative = AI_NARRATIVE_PATH.read_text(
        encoding="utf-8"
    )

    assert "These numeric values are inserted directly from the verified deterministic" in narrative
    assert "New causal values calculated by AI: **False**" in narrative
    assert "Individual targeting recommended: **False**" in narrative
