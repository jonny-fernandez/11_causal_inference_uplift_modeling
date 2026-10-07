from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(".")
ARTIFACTS_DIR = Path("artifacts")
OUTPUTS_DIR = Path("outputs")
CHART_DIR = OUTPUTS_DIR / "charts"

PROJECT_CONFIG_PATH = Path("config/project_config.json")
METHODOLOGY_PATH = Path("config/final_methodology.json")
FINAL_EVALUATION_PATH = Path("artifacts/final_evaluation.json")
MODEL_METADATA_PATH = Path("artifacts/model_metadata.json")
AI_STATUS_PATH = Path("artifacts/ai_interpretation_status.json")
DETERMINISTIC_REPORT_PATH = Path("outputs/causal_report.md")
AI_NARRATIVE_PATH = Path("outputs/ai_causal_narrative.md")
GITIGNORE_PATH = Path(".gitignore")
ENV_PATH = Path(".env")

AUDIT_JSON_PATH = Path("artifacts/final_audit.json")
AUDIT_FINDINGS_PATH = Path("artifacts/final_audit_findings.csv")

EXPECTED_CHARTS = [
    "01_propensity_overlap.png",
    "02_balance_before_after.png",
    "03_estimator_comparison.png",
    "04_sensitivity_analysis.png",
    "05_subgroup_effects.png",
    "06_cate_validation_failure.png",
]

REQUIRED_PROJECT_FILES = [
    Path("src/data_contract.py"),
    Path("src/baseline.py"),
    Path("src/propensity.py"),
    Path("src/balance.py"),
    Path("src/weighting.py"),
    Path("src/ate.py"),
    Path("src/aipw.py"),
    Path("src/aipw_crossfit.py"),
    Path("src/ground_truth_evaluation.py"),
    Path("src/failure_analysis.py"),
    Path("src/cate.py"),
    Path("src/cate_diagnostics.py"),
    Path("src/cate_regularized.py"),
    Path("src/cate_slearner.py"),
    Path("src/subgroup_effects.py"),
    Path("src/sensitivity_analysis.py"),
    Path("src/final_evaluation.py"),
    Path("src/reproducibility_metadata.py"),
    Path("src/causal_report.py"),
    Path("src/ai_interpretation.py"),
    Path("src/visualizations.py"),
    Path("tests/test_causal_pipeline.py"),
    Path("tests/test_ai_guardrails.py"),
    Path("config/final_methodology.json"),
    Path("artifacts/final_evaluation.json"),
    Path("artifacts/model_metadata.json"),
    Path("outputs/causal_report.md"),
]

TEXT_SCAN_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".csv",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
}

SKIP_SCAN_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

SECRET_PATTERNS = [
    (
        "openai_api_key_literal",
        re.compile(
            r"(?i)\bOPENAI_API_KEY\s*=\s*['\"]?sk-[A-Za-z0-9_\-]{10,}"
        ),
    ),
    (
        "openai_secret_key",
        re.compile(
            r"\bsk-[A-Za-z0-9_\-]{20,}"
        ),
    ),
    (
        "generic_private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
    ),
]


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def add_finding(
    findings: list[dict[str, Any]],
    check_id: str,
    category: str,
    status: str,
    message: str,
    path: str | None = None,
) -> None:
    findings.append(
        {
            "check_id": check_id,
            "category": category,
            "status": status,
            "message": message,
            "path": path or "",
        }
    )


def check_required_files(
    findings: list[dict[str, Any]],
) -> None:
    missing = [
        path
        for path in REQUIRED_PROJECT_FILES
        if not path.exists()
    ]

    if missing:
        for path in missing:
            add_finding(
                findings,
                check_id="required_file_missing",
                category="completeness",
                status="FAIL",
                message="Required project file is missing.",
                path=str(path),
            )
    else:
        add_finding(
            findings,
            check_id="required_files_present",
            category="completeness",
            status="PASS",
            message=(
                f"All {len(REQUIRED_PROJECT_FILES)} required "
                "project files are present."
            ),
        )


def check_charts(
    findings: list[dict[str, Any]],
) -> None:
    missing = [
        CHART_DIR / filename
        for filename in EXPECTED_CHARTS
        if not (CHART_DIR / filename).exists()
    ]

    if missing:
        for path in missing:
            add_finding(
                findings,
                check_id="chart_missing",
                category="visualization",
                status="FAIL",
                message="Expected portfolio chart is missing.",
                path=str(path),
            )
    else:
        add_finding(
            findings,
            check_id="charts_complete",
            category="visualization",
            status="PASS",
            message="All six expected portfolio charts are present.",
        )


def read_gitignore_lines() -> set[str]:
    if not GITIGNORE_PATH.exists():
        return set()

    return {
        line.strip()
        for line in GITIGNORE_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
    }


def check_gitignore(
    findings: list[dict[str, Any]],
) -> None:
    if not GITIGNORE_PATH.exists():
        add_finding(
            findings,
            check_id="gitignore_missing",
            category="security",
            status="FAIL",
            message=".gitignore is missing.",
            path=str(GITIGNORE_PATH),
        )
        return

    lines = read_gitignore_lines()

    required_patterns = {
        ".env",
        "__pycache__/",
        "*.pyc",
    }

    missing = sorted(
        required_patterns.difference(lines)
    )

    if missing:
        add_finding(
            findings,
            check_id="gitignore_incomplete",
            category="security",
            status="FAIL",
            message=(
                "Missing recommended .gitignore entries: "
                + ", ".join(missing)
            ),
            path=str(GITIGNORE_PATH),
        )
    else:
        add_finding(
            findings,
            check_id="gitignore_required_entries",
            category="security",
            status="PASS",
            message=(
                ".gitignore excludes .env, Python bytecode, "
                "and __pycache__."
            ),
            path=str(GITIGNORE_PATH),
        )


def parse_env_keys_safely() -> tuple[list[str], list[int]]:
    if not ENV_PATH.exists():
        return [], []

    keys: list[str] = []
    invalid_lines: list[int] = []

    pattern = re.compile(
        r"^\s*([^#=\s]+)\s*=(.*)$"
    )

    for index, line in enumerate(
        ENV_PATH.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        if line.lstrip().startswith("#"):
            continue

        match = pattern.match(line)

        if match:
            keys.append(
                match.group(1)
            )
        else:
            invalid_lines.append(index)

    return keys, invalid_lines


def check_env_syntax(
    findings: list[dict[str, Any]],
) -> None:
    if not ENV_PATH.exists():
        add_finding(
            findings,
            check_id="env_missing",
            category="configuration",
            status="WARN",
            message=(
                ".env is not present. This is acceptable only if "
                "environment variables are supplied another way."
            ),
            path=str(ENV_PATH),
        )
        return

    keys, invalid_lines = parse_env_keys_safely()

    if invalid_lines:
        add_finding(
            findings,
            check_id="env_invalid_syntax",
            category="configuration",
            status="FAIL",
            message=(
                "Invalid .env syntax on line(s): "
                + ", ".join(
                    str(value)
                    for value in invalid_lines
                )
            ),
            path=str(ENV_PATH),
        )
    else:
        add_finding(
            findings,
            check_id="env_syntax",
            category="configuration",
            status="PASS",
            message=(
                ".env syntax is valid. Keys found: "
                + ", ".join(sorted(keys))
            ),
            path=str(ENV_PATH),
        )


def run_command(
    command: list[str],
) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    output = (
        completed.stdout
        + completed.stderr
    ).strip()

    return (
        completed.returncode,
        output,
    )


def check_git_tracking(
    findings: list[dict[str, Any]],
) -> dict:
    returncode, output = run_command(
        ["git", "rev-parse", "--is-inside-work-tree"]
    )

    if returncode != 0:
        add_finding(
            findings,
            check_id="git_repo_not_initialized",
            category="git",
            status="WARN",
            message=(
                "Current folder is not yet an initialized Git repository. "
                "This is acceptable before the final GitHub checkpoint."
            ),
        )

        return {
            "is_git_repository": False,
            "env_tracked": False,
            "git_status": None,
        }

    tracked_code, tracked_output = run_command(
        ["git", "ls-files", ".env"]
    )

    env_tracked = bool(
        tracked_output.strip()
    )

    if env_tracked:
        add_finding(
            findings,
            check_id="env_tracked_by_git",
            category="security",
            status="FAIL",
            message=(
                ".env is tracked by Git and must be removed from "
                "the index before publication."
            ),
            path=".env",
        )
    else:
        add_finding(
            findings,
            check_id="env_not_tracked",
            category="security",
            status="PASS",
            message=".env is not tracked by Git.",
            path=".env",
        )

    _, status_output = run_command(
        ["git", "status", "--short"]
    )

    return {
        "is_git_repository": True,
        "env_tracked": env_tracked,
        "git_status": status_output,
    }


def should_scan_file(
    path: Path,
) -> bool:
    if path == ENV_PATH:
        return False

    if any(
        part in SKIP_SCAN_DIRS
        for part in path.parts
    ):
        return False

    return (
        path.is_file()
        and path.suffix.lower()
        in TEXT_SCAN_SUFFIXES
    )


def scan_for_secrets(
    findings: list[dict[str, Any]],
) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []

    for path in PROJECT_ROOT.rglob("*"):
        if not should_scan_file(path):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except OSError:
            continue

        for pattern_name, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(
                    {
                        "pattern": pattern_name,
                        "path": str(path),
                    }
                )

    if hits:
        for hit in hits:
            add_finding(
                findings,
                check_id="potential_secret_detected",
                category="security",
                status="FAIL",
                message=(
                    "Potential secret-like value detected "
                    f"({hit['pattern']})."
                ),
                path=hit["path"],
            )
    else:
        add_finding(
            findings,
            check_id="secret_scan",
            category="security",
            status="PASS",
            message=(
                "No obvious API-key or private-key literals were "
                "detected outside .env."
            ),
        )

    return hits


def check_methodology_consistency(
    findings: list[dict[str, Any]],
) -> None:
    methodology = load_json(
        METHODOLOGY_PATH,
        "final methodology",
    )

    final = load_json(
        FINAL_EVALUATION_PATH,
        "final evaluation",
    )

    metadata = load_json(
        MODEL_METADATA_PATH,
        "model metadata",
    )

    method_estimator = methodology[
        "primary_estimator"
    ]

    final_estimator = final[
        "primary_estimator"
    ]

    metadata_estimator = metadata[
        "causal_design"
    ]["primary_estimator"]

    estimators_match = (
        method_estimator
        == final_estimator
        == metadata_estimator
    )

    method_effect = float(
        methodology[
            "primary_effect_summary"
        ]["estimate"]
    )

    final_effect = float(
        final[
            "primary_result"
        ]["estimate"]
    )

    metadata_effect = float(
        metadata[
            "locked_result"
        ]["estimate"]
    )

    effects_match = (
        abs(method_effect - final_effect)
        <= 1e-6
        and abs(
            final_effect - metadata_effect
        )
        <= 1e-6
    )

    if estimators_match and effects_match:
        add_finding(
            findings,
            check_id="locked_methodology_consistency",
            category="causal_integrity",
            status="PASS",
            message=(
                "Locked methodology, final evaluation, and metadata "
                "agree on the primary estimator and effect."
            ),
        )
    else:
        add_finding(
            findings,
            check_id="locked_methodology_consistency",
            category="causal_integrity",
            status="FAIL",
            message=(
                "Locked methodology, final evaluation, and metadata "
                "do not agree."
            ),
        )


def check_ai_boundary(
    findings: list[dict[str, Any]],
) -> None:
    final = load_json(
        FINAL_EVALUATION_PATH,
        "final evaluation",
    )

    if final.get(
        "ai_used_for_calculation"
    ) is not False:
        add_finding(
            findings,
            check_id="ai_boundary_final",
            category="ai_governance",
            status="FAIL",
            message=(
                "Final evaluation does not explicitly preserve "
                "the deterministic calculation boundary."
            ),
        )
    else:
        add_finding(
            findings,
            check_id="ai_boundary_final",
            category="ai_governance",
            status="PASS",
            message=(
                "Final evaluation confirms AI was not used to "
                "calculate causal results."
            ),
        )

    if AI_STATUS_PATH.exists():
        status = load_json(
            AI_STATUS_PATH,
            "AI interpretation status",
        )

        if (
            status.get(
                "deterministic_pipeline_affected"
            )
            is False
        ):
            add_finding(
                findings,
                check_id="ai_failure_isolation",
                category="ai_governance",
                status="PASS",
                message=(
                    "AI interpretation status confirms the deterministic "
                    "pipeline is isolated from AI success/failure."
                ),
            )
        else:
            add_finding(
                findings,
                check_id="ai_failure_isolation",
                category="ai_governance",
                status="FAIL",
                message=(
                    "AI interpretation status does not confirm "
                    "failure isolation."
                ),
            )
    else:
        add_finding(
            findings,
            check_id="ai_status_missing",
            category="ai_governance",
            status="WARN",
            message=(
                "AI interpretation status artifact is absent. "
                "This is acceptable only if the optional AI layer "
                "was intentionally not run."
            ),
        )


def check_reports(
    findings: list[dict[str, Any]],
) -> None:
    if not DETERMINISTIC_REPORT_PATH.exists():
        add_finding(
            findings,
            check_id="deterministic_report_missing",
            category="reporting",
            status="FAIL",
            message="Deterministic causal report is missing.",
            path=str(DETERMINISTIC_REPORT_PATH),
        )
    else:
        text = DETERMINISTIC_REPORT_PATH.read_text(
            encoding="utf-8"
        )

        required_phrases = [
            "Deterministic causal analysis complete",
            "No AI system calculated or modified treatment effects",
            "Individual-level CATE/uplift targeting was rejected after multiple validation failures",
        ]

        missing_phrases = [
            phrase
            for phrase in required_phrases
            if phrase not in text
        ]

        if missing_phrases:
            add_finding(
                findings,
                check_id="deterministic_report_content",
                category="reporting",
                status="FAIL",
                message=(
                    "Deterministic report is missing required language: "
                    + " | ".join(missing_phrases)
                ),
                path=str(DETERMINISTIC_REPORT_PATH),
            )
        else:
            add_finding(
                findings,
                check_id="deterministic_report_content",
                category="reporting",
                status="PASS",
                message=(
                    "Deterministic report preserves final status, "
                    "AI boundary, and individual-CATE limitation."
                ),
                path=str(DETERMINISTIC_REPORT_PATH),
            )

    if AI_NARRATIVE_PATH.exists():
        add_finding(
            findings,
            check_id="ai_narrative_present",
            category="reporting",
            status="PASS",
            message="Controlled AI narrative artifact is present.",
            path=str(AI_NARRATIVE_PATH),
        )
    else:
        add_finding(
            findings,
            check_id="ai_narrative_absent",
            category="reporting",
            status="WARN",
            message=(
                "Controlled AI narrative artifact is absent. "
                "The deterministic project remains complete without it."
            ),
            path=str(AI_NARRATIVE_PATH),
        )


def run_test_suite(
    findings: list[dict[str, Any]],
) -> dict:
    returncode, output = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-q",
        ]
    )

    passed_match = re.search(
        r"(\d+)\s+passed",
        output,
    )

    failed_match = re.search(
        r"(\d+)\s+failed",
        output,
    )

    passed = (
        int(passed_match.group(1))
        if passed_match
        else 0
    )

    failed = (
        int(failed_match.group(1))
        if failed_match
        else 0
    )

    if returncode == 0:
        add_finding(
            findings,
            check_id="pytest_suite",
            category="testing",
            status="PASS",
            message=(
                f"Automated test suite passed with {passed} tests."
            ),
        )
    else:
        add_finding(
            findings,
            check_id="pytest_suite",
            category="testing",
            status="FAIL",
            message=(
                f"Automated test suite failed. "
                f"Passed: {passed}; failed: {failed}."
            ),
        )

    return {
        "return_code": returncode,
        "passed": passed,
        "failed": failed,
        "output_tail": "\n".join(
            output.splitlines()[-12:]
        ),
    }


def summarize_findings(
    findings: list[dict[str, Any]],
) -> dict:
    counts = {
        "PASS": 0,
        "WARN": 0,
        "FAIL": 0,
    }

    for finding in findings:
        status = finding["status"]

        if status in counts:
            counts[status] += 1

    publication_ready = (
        counts["FAIL"] == 0
    )

    return {
        "pass_count": counts["PASS"],
        "warn_count": counts["WARN"],
        "fail_count": counts["FAIL"],
        "publication_ready_from_technical_audit": publication_ready,
    }


def save_outputs(
    audit: dict,
    findings: list[dict[str, Any]],
) -> None:
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    fieldnames = [
        "check_id",
        "category",
        "status",
        "message",
        "path",
    ]

    with AUDIT_FINDINGS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            findings
        )


def print_summary(
    audit: dict,
    findings: list[dict[str, Any]],
) -> None:
    summary = audit["summary"]

    print("Final Technical / Security Audit")
    print("--------------------------------")

    print(
        f"PASS: {summary['pass_count']}"
    )

    print(
        f"WARN: {summary['warn_count']}"
    )

    print(
        f"FAIL: {summary['fail_count']}"
    )

    print(
        "Publication-ready from technical audit: "
        f"{summary['publication_ready_from_technical_audit']}"
    )

    print("\nFindings")

    for finding in findings:
        path_suffix = (
            f" [{finding['path']}]"
            if finding["path"]
            else ""
        )

        print(
            f"- {finding['status']}: "
            f"{finding['message']}"
            f"{path_suffix}"
        )

    print("\nArtifacts saved:")
    print(f"- {AUDIT_JSON_PATH}")
    print(f"- {AUDIT_FINDINGS_PATH}")


def main() -> None:
    findings: list[dict[str, Any]] = []

    check_required_files(
        findings=findings
    )

    check_charts(
        findings=findings
    )

    check_gitignore(
        findings=findings
    )

    check_env_syntax(
        findings=findings
    )

    git_info = check_git_tracking(
        findings=findings
    )

    secret_hits = scan_for_secrets(
        findings=findings
    )

    check_methodology_consistency(
        findings=findings
    )

    check_ai_boundary(
        findings=findings
    )

    check_reports(
        findings=findings
    )

    test_info = run_test_suite(
        findings=findings
    )

    summary = summarize_findings(
        findings=findings
    )

    audit = {
        "project": "11_causal_inference_uplift_modeling",
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "summary": summary,
        "git": git_info,
        "secret_scan_hits": secret_hits,
        "test_suite": test_info,
        "next_gate": (
            "Screenshot Word handoff and user confirmation "
            "before README/GitHub publication."
        ),
    }

    save_outputs(
        audit=audit,
        findings=findings,
    )

    print_summary(
        audit=audit,
        findings=findings,
    )


if __name__ == "__main__":
    main()
