from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.data_contract import load_data
from src.propensity import CATEGORICAL_FEATURES, NUMERIC_FEATURES


PROJECT_CONFIG_PATH = Path("config/project_config.json")
METHODOLOGY_PATH = Path("config/final_methodology.json")
FINAL_EVALUATION_PATH = Path("artifacts/final_evaluation.json")

DATASET_PATH = Path(
    "data/raw/training_intervention_synthetic.csv"
)

OUTPUT_PATH = Path("artifacts/model_metadata.json")

PACKAGE_NAMES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "joblib",
    "pytest",
]

OPTIONAL_PACKAGE_NAMES = [
    "openai",
    "python-dotenv",
    "shap",
    "xgboost",
]

KEY_SOURCE_FILES = [
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
]

KEY_ARTIFACTS = [
    Path("artifacts/naive_baseline.json"),
    Path("artifacts/propensity_diagnostics.json"),
    Path("artifacts/propensity_scores.csv"),
    Path("artifacts/propensity_model.joblib"),
    Path("artifacts/ipw_weights.csv"),
    Path("artifacts/weight_diagnostics.json"),
    Path("artifacts/balance_diagnostics.csv"),
    Path("artifacts/balance_summary.json"),
    Path("artifacts/ate_summary.json"),
    Path("artifacts/ate_bootstrap_distribution.csv"),
    Path("artifacts/aipw_scores.csv"),
    Path("artifacts/aipw_summary.json"),
    Path("artifacts/aipw_crossfit_scores.csv"),
    Path("artifacts/aipw_crossfit_summary.json"),
    Path("artifacts/ground_truth_evaluation.json"),
    Path("artifacts/ate_ground_truth_comparison.csv"),
    Path("artifacts/failure_analysis_summary.json"),
    Path("artifacts/segment_effects.csv"),
    Path("artifacts/segment_effects_summary.json"),
    Path("artifacts/sensitivity_analysis.csv"),
    Path("artifacts/sensitivity_analysis_summary.json"),
    Path("artifacts/final_evaluation.json"),
    Path("artifacts/final_evaluation_metrics.csv"),
]


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def file_record(path: Path) -> dict:
    if not path.exists():
        return {
            "path": str(path).replace("\\", "/"),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }

    return {
        "path": str(path).replace("\\", "/"),
        "exists": True,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def package_version(
    package_name: str,
) -> str | None:
    try:
        return importlib.metadata.version(
            package_name
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_package_versions(
    package_names: list[str],
) -> dict:
    return {
        package: package_version(package)
        for package in package_names
    }


def validate_locked_results(
    methodology: dict,
    final_evaluation: dict,
) -> None:
    if methodology.get("status") != "methodology_locked":
        raise ValueError(
            "Methodology must be locked before metadata generation."
        )

    if not final_evaluation.get(
        "methodology_locked",
        False,
    ):
        raise ValueError(
            "Final evaluation does not report locked methodology."
        )

    methodology_estimator = methodology[
        "primary_estimator"
    ]

    evaluation_estimator = final_evaluation[
        "primary_estimator"
    ]

    if methodology_estimator != evaluation_estimator:
        raise ValueError(
            "Primary estimator mismatch between methodology "
            "and final evaluation."
        )

    locked_estimate = float(
        methodology[
            "primary_effect_summary"
        ]["estimate"]
    )

    evaluated_estimate = float(
        final_evaluation[
            "primary_result"
        ]["estimate"]
    )

    if abs(
        locked_estimate - evaluated_estimate
    ) > 1e-6:
        raise ValueError(
            "Primary estimate mismatch between methodology "
            "and final evaluation."
        )


def build_metadata(
    project_config: dict,
    methodology: dict,
    final_evaluation: dict,
) -> dict:
    data = load_data()

    source_records = [
        file_record(path)
        for path in KEY_SOURCE_FILES
    ]

    artifact_records = [
        file_record(path)
        for path in KEY_ARTIFACTS
    ]

    missing_key_sources = [
        record["path"]
        for record in source_records
        if not record["exists"]
    ]

    missing_key_artifacts = [
        record["path"]
        for record in artifact_records
        if not record["exists"]
    ]

    metadata = {
        "project": {
            "project_number": int(
                project_config["project_number"]
            ),
            "project_name": project_config[
                "project_name"
            ],
            "repository_folder": (
                "11_causal_inference_uplift_modeling"
            ),
            "synthetic_data_only": bool(
                project_config[
                    "synthetic_data_only"
                ]
            ),
        },
        "generated_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "runtime": {
            "python_version": (
                platform.python_version()
            ),
            "python_implementation": (
                platform.python_implementation()
            ),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "library_versions": {
            "required": collect_package_versions(
                PACKAGE_NAMES
            ),
            "optional": collect_package_versions(
                OPTIONAL_PACKAGE_NAMES
            ),
        },
        "dataset": {
            "path": str(DATASET_PATH).replace(
                "\\",
                "/",
            ),
            "exists": DATASET_PATH.exists(),
            "sha256": (
                sha256_file(DATASET_PATH)
                if DATASET_PATH.exists()
                else None
            ),
            "rows": int(len(data)),
            "columns": int(
                len(data.columns)
            ),
            "id_column": project_config[
                "id_column"
            ],
            "treatment_column": project_config[
                "treatment"
            ],
            "outcome_column": project_config[
                "outcome"
            ],
        },
        "randomness": {
            "project_random_seed": int(
                project_config[
                    "random_seed"
                ]
            ),
            "propensity_random_state": 42,
            "bootstrap_random_state": 42,
            "crossfit_random_state": 42,
            "cate_initial_random_state": 42,
            "cate_final_holdout_random_state": 137,
        },
        "causal_design": {
            "primary_estimand": methodology[
                "primary_estimand"
            ],
            "primary_estimator": methodology[
                "primary_estimator"
            ],
            "treatment": project_config[
                "treatment"
            ],
            "outcome": project_config[
                "outcome"
            ],
            "categorical_pre_treatment_covariates": (
                list(CATEGORICAL_FEATURES)
            ),
            "numeric_pre_treatment_covariates": (
                list(NUMERIC_FEATURES)
            ),
            "evaluation_only_ground_truth": [
                "true_propensity",
                project_config[
                    "synthetic_ground_truth_effect_column"
                ],
            ],
            "post_treatment_exclusions": [
                "post_training_completion"
            ],
        },
        "locked_result": final_evaluation[
            "primary_result"
        ],
        "diagnostic_summary": final_evaluation[
            "diagnostics"
        ],
        "sensitivity_summary": final_evaluation[
            "sensitivity"
        ],
        "heterogeneous_effects": final_evaluation[
            "heterogeneous_effects"
        ],
        "method_rejections": methodology[
            "rejected_methods"
        ],
        "ai_boundary": {
            "ai_used_for_causal_calculation": False,
            "ai_used_for_propensity_calculation": False,
            "ai_used_for_weight_calculation": False,
            "ai_used_for_confidence_interval_calculation": False,
            "ai_used_for_cate_calculation": False,
            "future_ai_role": (
                "Optional controlled interpretation "
                "of verified aggregate artifacts only."
            ),
        },
        "file_integrity": {
            "project_config": file_record(
                PROJECT_CONFIG_PATH
            ),
            "final_methodology": file_record(
                METHODOLOGY_PATH
            ),
            "final_evaluation": file_record(
                FINAL_EVALUATION_PATH
            ),
            "source_files": source_records,
            "artifact_files": artifact_records,
            "missing_key_source_files": (
                missing_key_sources
            ),
            "missing_key_artifacts": (
                missing_key_artifacts
            ),
        },
    }

    return metadata


def save_metadata(
    metadata: dict,
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def print_summary(
    metadata: dict,
) -> None:
    dataset = metadata["dataset"]
    runtime = metadata["runtime"]
    integrity = metadata[
        "file_integrity"
    ]

    print(
        "Reproducibility / Model Metadata"
    )
    print(
        "--------------------------------"
    )

    print(
        f"Project: "
        f"{metadata['project']['project_name']}"
    )

    print(
        f"Python: "
        f"{runtime['python_version']}"
    )

    print(
        f"Dataset rows: "
        f"{dataset['rows']:,}"
    )

    print(
        f"Dataset columns: "
        f"{dataset['columns']}"
    )

    print(
        f"Dataset SHA256: "
        f"{dataset['sha256']}"
    )

    print(
        f"Primary estimand: "
        f"{metadata['causal_design']['primary_estimand']}"
    )

    print(
        f"Primary estimator: "
        f"{metadata['causal_design']['primary_estimator']}"
    )

    print(
        "Primary ATE: "
        f"{metadata['locked_result']['estimate_percentage_points']:+.2f} "
        "percentage points"
    )

    print(
        "\nRequired package versions"
    )

    for package, version in metadata[
        "library_versions"
    ]["required"].items():
        print(
            f"{package}: {version}"
        )

    print(
        "\nIntegrity check"
    )

    print(
        "Missing key source files: "
        f"{len(integrity['missing_key_source_files'])}"
    )

    print(
        "Missing key artifacts: "
        f"{len(integrity['missing_key_artifacts'])}"
    )

    if integrity[
        "missing_key_source_files"
    ]:
        for path in integrity[
            "missing_key_source_files"
        ]:
            print(
                f"- missing source: {path}"
            )

    if integrity[
        "missing_key_artifacts"
    ]:
        for path in integrity[
            "missing_key_artifacts"
        ]:
            print(
                f"- missing artifact: {path}"
            )

    print(
        f"\nArtifact saved: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nAI boundary: no AI was used to calculate "
        "or modify causal results."
    )


def main() -> None:
    project_config = load_json(
        PROJECT_CONFIG_PATH,
        "project configuration",
    )

    methodology = load_json(
        METHODOLOGY_PATH,
        "final methodology",
    )

    final_evaluation = load_json(
        FINAL_EVALUATION_PATH,
        "final evaluation",
    )

    validate_locked_results(
        methodology=methodology,
        final_evaluation=final_evaluation,
    )

    metadata = build_metadata(
        project_config=project_config,
        methodology=methodology,
        final_evaluation=final_evaluation,
    )

    save_metadata(
        metadata=metadata,
    )

    print_summary(
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
