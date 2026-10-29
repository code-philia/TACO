from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rek.parity_core.github_pr_collector import collect_pr_data
from rek.parity_core.repo2doc_bridge import run_merge, run_phase_a, run_phase_b
from rek.parity_core.trainer import DEFAULT_LOG_ROOT, run_training, setup_logger
from rek.parity_core.upgrade_kb_two_index import process_knowledge_graph

ROOT_DIR = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = ROOT_DIR.parent

DEFAULT_BASE_REPOS = os.getenv("REK_BASE_REPOS", str(Path.cwd()))
DEFAULT_TRAIN_DATA_DIR = os.getenv("REK_TRAIN_DATA_DIR", str(Path.cwd() / "data" / "train"))


class AutoPipeline:
    def __init__(
        self,
        repo_name: str,
        base_repos: str,
        train_data_dir: str,
        model_name: str,
        logger: logging.Logger,
        output_dir_name: str | None = None,
        output_dir_path: str | None = None,
        log_root: str = DEFAULT_LOG_ROOT,
        repo_path: str | None = None,
    ):
        self.repo_name = repo_name
        self.logger = logger
        self.model_name = model_name
        self.log_root = log_root or DEFAULT_LOG_ROOT

        self.folder_name = output_dir_name or (
            "repo2docAgent-output-claude" if "claude" in self.model_name.lower() else "repo2docAgent-output"
        )
        os.environ["REPO2DOC_OUTPUT_DIR"] = self.folder_name
        os.environ["REPO2DOC_LLM_MODEL"] = self.model_name
        os.environ["TRAINER_LOG_ROOT"] = self.log_root

        self.repo_path = Path(repo_path) if repo_path else Path(base_repos) / repo_name
        self.output_dir = Path(output_dir_path) if output_dir_path else self.repo_path / self.folder_name
        self.train_data_path = Path(train_data_dir) / f"{repo_name}_all_prs_train_base_commit.jsonl"

        self.kb_final = self.output_dir / "knowledge_graph_complete_detail.json"
        self.elements_path = self.output_dir / "program_elements.jsonl"
        self.kb_base = self.output_dir / "knowledge_graph.json"
        self.kb_detail = self.output_dir / "knowledge_graph_detail.json"

    def _run_cmd(self, cmd: list[str], cwd: Path | None = None, name: str = "Task") -> None:
        self.logger.info("Running %s...", name)
        try:
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
            self.logger.info("%s completed.", name)
        except subprocess.CalledProcessError as exc:
            self.logger.error("%s failed with error: %s", name, exc)
            raise

    def prepare_data(self, min_threshold: int = 500) -> None:
        self.logger.info("Preparing PR data for %s.", self.repo_name)

        swe_bench_lite_repos = {
            "django": "django",
            "scikit-learn": "scikit-learn",
            "flask": "pallets",
            "matplotlib": "matplotlib",
            "pytest": "pytest-dev",
            "sympy": "sympy",
            "sphinx": "sphinx-doc",
            "astropy": "astropy",
            "pylint": "pylint-dev",
            "xarray": "pydata",
            "requests": "psf",
            "seaborn": "mwaskom",
        }

        owner = swe_bench_lite_repos.get(self.repo_name, self.repo_name)
        github_repo = self.repo_name
        self.logger.info("Resolved GitHub path: %s/%s", owner, github_repo)

        current_count = 0
        if self.train_data_path.exists():
            with open(self.train_data_path, "r", encoding="utf-8") as handle:
                current_count = sum(1 for line in handle if line.strip())

        if current_count >= min_threshold:
            self.logger.info("PR data is sufficient (%s rows). Skipping collection.", current_count)
            return

        self.logger.info("Launching local parity PR collector entrypoint.")
        collect_pr_data(repo_name=self.repo_name, owner=owner, output_path=str(self.train_data_path))
        self.logger.info("PR Data Incremental Collection completed.")

    def check_pr_threshold(self, min_threshold: int = 9000) -> None:
        if not self.train_data_path.exists():
            raise FileNotFoundError(f"PR data file not found: {self.train_data_path}")

        with open(self.train_data_path, "r", encoding="utf-8") as handle:
            current_count = sum(1 for line in handle if line.strip())

        if current_count < min_threshold:
            raise RuntimeError(
                f"PR count threshold not met: current={current_count}, required={min_threshold}. "
                "Check tokens or network before continuing."
            )

        self.logger.info("PR threshold passed with %s rows.", current_count)

    def prepare_kb(self) -> None:
        if self.kb_final.exists():
            self.logger.info("Final knowledge graph already exists. Skipping build.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.kb_base.exists():
            self.logger.info("Initializing %s from base KB.", self.kb_detail.name)
            with open(self.kb_base, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            data["file_mappings"] = []
            with open(self.kb_detail, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
        elif self.kb_detail.exists():
            self.logger.info("%s already exists. Skipping Phase A bootstrap.", self.kb_detail.name)
        else:
            bootstrap_candidates = [
                self.repo_path / "repo2docAgent-output" / "knowledge_graph_detail.json",
                self.repo_path / "repo2docAgent-output-claude" / "knowledge_graph_detail.json",
            ]
            bootstrap_source = next((path for path in bootstrap_candidates if path.exists()), None)

            if bootstrap_source:
                self.logger.info("Bootstrapping from existing skeleton: %s", bootstrap_source)
                with open(bootstrap_source, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                data["file_mappings"] = []
                with open(self.kb_detail, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, ensure_ascii=False)
                requirements_src = bootstrap_source.parent / "requirements.md"
                if requirements_src.exists():
                    shutil.copy2(requirements_src, self.output_dir / "requirements.md")
            else:
                self.logger.info("No bootstrap KB found. Running Phase A agent discovery.")
                run_phase_a(str(self.repo_path))
                self.logger.info("Phase A: Agent Discovery completed.")

        run_phase_b(str(self.repo_path))
        self.logger.info("Phase B: Batch Mapping completed.")
        run_merge(str(self.repo_path), self.kb_final.name)
        self.logger.info("Phase C: KB Merging completed.")

    def upgrade_to_two_index(self) -> None:
        self.logger.info("Upgrading the knowledge graph to the two-index layout.")
        input_kb = self.output_dir / "knowledge_graph_complete_detail.json"
        output_kb = self.output_dir / "knowledge_graph_two_index.json"

        if not input_kb.exists():
            raise FileNotFoundError(f"Complete knowledge graph not found: {input_kb}")

        process_knowledge_graph(
            kb_path=str(input_kb),
            repo_root=str(self.repo_path),
            output_path=str(output_kb),
        )
        self.kb_final = output_kb
        self.logger.info("Two-index knowledge graph generated: %s", output_kb)

    def start_training_pipeline(self, epochs: int, batch_size: int, workers: int) -> dict[str, Any]:
        self.logger.info("Starting training pipeline.")
        return run_training(
            repo_path=str(self.repo_path),
            kb_path=str(self.kb_final),
            elements_path=str(self.elements_path),
            train_data_path=str(self.train_data_path),
            model_name=self.model_name,
            epochs=epochs,
            batch_size=batch_size,
            workers=workers,
            log_root=self.log_root,
        )

    def run_pipeline(
        self,
        epochs: int,
        batch_size: int,
        workers: int,
        build_kb_only: bool = False,
        assume_pr_ready: bool = False,
        train_only: bool = False,
    ) -> dict[str, Any]:
        train_result: dict[str, Any] | None = None

        if train_only:
            if not self.kb_final.exists():
                raise FileNotFoundError(f"Knowledge base not found for train-only mode: {self.kb_final}")
            if not self.elements_path.exists():
                self.logger.warning(
                    "Program elements are missing in train-only mode. Training continues with the configured path: %s",
                    self.elements_path,
                )
            train_result = self.start_training_pipeline(epochs, batch_size, workers)
            return self._result_payload(train_result)

        if not assume_pr_ready:
            self.prepare_data(min_threshold=500)
        self.prepare_kb()
        self.upgrade_to_two_index()
        if not assume_pr_ready:
            self.check_pr_threshold(min_threshold=500)
        if not build_kb_only:
            train_result = self.start_training_pipeline(epochs, batch_size, workers)
        return self._result_payload(train_result)

    def _result_payload(self, train_result: dict[str, Any] | None = None) -> dict[str, Any]:
        artifacts = [
            str(path)
            for path in [
                self.output_dir,
                self.kb_base,
                self.kb_detail,
                self.output_dir / "knowledge_graph_complete_detail.json",
                self.output_dir / "knowledge_graph_two_index.json",
                self.elements_path,
                self.train_data_path,
            ]
            if Path(path).exists()
        ]
        return {
            "repo_name": self.repo_name,
            "repo_path": str(self.repo_path),
            "output_dir": str(self.output_dir),
            "kb_final": str(self.kb_final),
            "elements_path": str(self.elements_path),
            "train_data_path": str(self.train_data_path),
            "artifacts": artifacts,
            "train_result": train_result or {},
        }


def run_auto_pipeline(
    repo_name: str,
    repo_path: str,
    base_repos: str,
    train_data_dir: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    workers: int,
    output_dir_name: str | None = None,
    output_dir_path: str | None = None,
    log_root: str | None = None,
    build_kb_only: bool = False,
    assume_pr_ready: bool = False,
    train_only: bool = False,
) -> dict[str, Any]:
    logger, log_file = setup_logger(log_root)
    pipeline = AutoPipeline(
        repo_name=repo_name,
        base_repos=base_repos,
        train_data_dir=train_data_dir,
        model_name=model_name,
        logger=logger,
        output_dir_name=output_dir_name,
        output_dir_path=output_dir_path,
        log_root=log_root or DEFAULT_LOG_ROOT,
        repo_path=repo_path,
    )
    result = pipeline.run_pipeline(
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        build_kb_only=build_kb_only,
        assume_pr_ready=assume_pr_ready,
        train_only=train_only,
    )
    result["log_file"] = log_file
    return result
