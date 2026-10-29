from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def get_workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def get_local_repo2doc_config() -> Path:
    return Path(__file__).resolve().parent / "repo2doc_local" / "config.yaml"


def _build_runtime_config(output_dir_name: str | None = None) -> str:
    config_path = get_local_repo2doc_config()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    env_output = output_dir_name or os.getenv("REPO2DOC_OUTPUT_DIR")
    env_model = os.getenv("REPO2DOC_LLM_MODEL")
    if env_output:
        config.setdefault("output", {})
        config["output"]["output_dir"] = f"./{env_output}"
    if env_model:
        config.setdefault("llm", {})
        config["llm"]["model"] = env_model

    fd, temp_path = tempfile.mkstemp(prefix="rek-repo2doc-", suffix=".yaml")
    os.close(fd)
    with open(temp_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return temp_path


def run_phase_a(repo_path: str) -> None:
    config_path = _build_runtime_config()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rek.parity_core.repo2doc_local.main",
            repo_path,
            "-c",
            config_path,
        ],
        check=True,
    )


def run_phase_b(repo_path: str) -> None:
    config_path = _build_runtime_config()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rek.parity_core.repo2doc_local.batch_mapper",
            repo_path,
            "-c",
            config_path,
        ],
        check=True,
    )


def run_merge(repo_path: str, output_filename: str) -> None:
    config_path = _build_runtime_config()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rek.parity_core.repo2doc_local.merge_kb",
            repo_path,
            "-c",
            config_path,
            "-o",
            output_filename,
        ],
        check=True,
    )
