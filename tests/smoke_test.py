#!/usr/bin/env python3
"""Self-contained smoke tests for the paper artifact.

The tests create tiny temporary repositories and KB files. They avoid external
LLM/API calls and do not require private datasets.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(cmd: list[str], cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    if check and result.returncode != 0:
        print("$ " + " ".join(cmd))
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"command failed with exit code {result.returncode}")
    return result


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_tiny_repo(root: Path) -> tuple[Path, str, str, str]:
    repo = root / "tiny_repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "calc.py").write_text(
        "\n".join(
            [
                "class Calculator:",
                "    def normalize(self, value):",
                "        return value.strip().lower()",
                "",
                "def add(left, right):",
                "    return left + right",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run(["git", "init"], cwd=repo)
    run(["git", "config", "user.email", "smoke@example.com"], cwd=repo)
    run(["git", "config", "user.name", "Smoke Test"], cwd=repo)
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-m", "initial"], cwd=repo)
    base_commit = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()

    (package / "calc.py").write_text(
        "\n".join(
            [
                "class Calculator:",
                "    def normalize(self, value):",
                "        if value is None:",
                "            return ''",
                "        return value.strip().casefold()",
                "",
                "def add(left, right):",
                "    return left + right",
                "",
            ]
        ),
        encoding="utf-8",
    )
    diff = run(["git", "diff"], cwd=repo).stdout
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-m", "handle none normalization"], cwd=repo)
    head_commit = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    return repo, base_commit, head_commit, diff


def make_base_kb(path: Path) -> None:
    write_json(
        path,
        {
            "l1_categories": [{"id": "L1_Core", "name": "Core"}],
            "l2_features": [
                {
                    "id": "L2_Normalization",
                    "name": "Normalization",
                    "parent_l1_id": "L1_Core",
                    "description": "String normalization behavior",
                }
            ],
            "file_mappings": [
                {
                    "file_path": "pkg/calc.py",
                    "primary_feature_id": "L2_Normalization",
                    "secondary_feature_ids": [],
                    "role": "Calculator normalization utilities",
                    "rationale": "Handles user text normalization and arithmetic helpers.",
                    "rationale_keywords": ["calculator", "normalize", "string"],
                }
            ],
        },
    )


def test_cli_and_scanner(tmp: Path, repo: Path) -> tuple[Path, Path]:
    scan_out = tmp / "scan_out"
    run(
        [
            PYTHON,
            "-m",
            "rek.cli",
            "build",
            "run",
            "--repo-path",
            str(repo),
            "--stage",
            "scan",
            "--output",
            str(scan_out),
        ]
    )
    elements = scan_out / "program_elements.jsonl"
    assert elements.exists(), "scan did not write program_elements.jsonl"
    assert "pkg.calc.Calculator.normalize" in elements.read_text(encoding="utf-8")

    base_kb = tmp / "knowledge_graph_complete_detail.json"
    make_base_kb(base_kb)
    two_index = tmp / "knowledge_graph_two_index.json"
    from rek.parity_core.upgrade_kb_two_index import process_knowledge_graph

    result = process_knowledge_graph(str(base_kb), str(repo), str(two_index))
    assert result["total_functions_extracted"] >= 2
    return elements, two_index


def test_graph_db_and_retrieval(tmp: Path, repo: Path, elements: Path, kb: Path) -> None:
    from rek.config import REKConfig
    from rek.models import KBQueryRequest, KBRetrieveRequest, KBSearchRequest
    from rek.parity_core.graph_db import GraphDatabase
    from rek.services.kb_service import KBService

    graph_db = GraphDatabase(str(kb), str(elements))
    assert graph_db.update_node_knowledge(
        "pkg/calc.py",
        "Calculator.normalize",
        ["none", "casefold", "normalization"],
        "Handles None input and casefold-based normalization.",
    )

    config = REKConfig()
    config.paths.repo_path = str(repo)
    config.paths.elements_path = str(elements)
    config.retrieval.mode = "graph"
    service = KBService(str(kb), config)
    search = service.search(KBSearchRequest(kb_path=str(kb), symbol="normalize"))
    assert search.matches, "symbol search returned no matches"
    query = service.query(KBQueryRequest(kb_path=str(kb), question="None normalization casefold"))
    assert query.candidates, "graph query returned no candidates"

    config.retrieval.mode = "hybrid"
    retrieve = service.retrieve(
        KBRetrieveRequest(kb_path=str(kb), issue="None normalization casefold", repo_path=str(repo))
    )
    assert retrieve.candidates, "hybrid retrieve returned no candidates"


def test_pr_gold_and_distiller_parse(repo: Path, base_commit: str, patch: str) -> None:
    from rek.parity_core.git_time_travel import ASTGroundTruthExtractor
    from rek.parity_core.pr_distiller import GitTimeMachine, PRKnowledgeDistiller

    files, modules, funcs = ASTGroundTruthExtractor(str(repo)).extract_3level_gt(patch, base_commit)
    assert "pkg/calc.py" in files
    assert any(item.endswith("Calculator.normalize") for item in funcs)

    mapping = GitTimeMachine(str(repo)).build_time_mapping(
        patch,
        base_commit,
        expected_files=["pkg/calc.py"],
    )
    assert "pkg/calc.py" in mapping
    assert "Calculator.normalize" in mapping["pkg/calc.py"]["historical_functions_modified"]

    # Avoid external LLM calls: exercise the parser/normalizer directly.
    parser = object.__new__(PRKnowledgeDistiller)
    parsed = parser._parse_distilled_response(
        """
        ```json
        [{"file":"pkg/calc.py","function":"Calculator.normalize","keywords":"None, casefold","rationale":"Handles None before normalization."}]
        ```
        """
    )
    assert parsed and parsed[0]["keywords"] == ["None", "casefold"]


def test_prompt_optimization_scripts(tmp: Path, repo: Path, base_commit: str, patch: str, kb: Path, elements: Path) -> None:
    train_data_dir = tmp / "train_data"
    base_repos = tmp / "repos"
    repo_copy = base_repos / "tiny"
    base_repos.mkdir(parents=True)
    shutil.copytree(repo, repo_copy)

    sample = {
        "repo": "tiny",
        "instance_id": "tiny-1",
        "pr_number": 1,
        "title": "Handle None normalization",
        "problem_statement": "Calculator.normalize should handle None and use casefold.",
        "patch": patch,
        "base_commit": base_commit,
        "merged_at": "2024-01-01T00:00:00Z",
        "files_preview": ["pkg/calc.py"],
    }
    write_jsonl(train_data_dir / "tiny_all_prs_train_base_commit.jsonl", [sample, sample, sample])

    splits = tmp / "splits"
    run(
        [
            PYTHON,
            "experiments/prompt_optimization/scripts/build_historical_splits.py",
            "--train-data-dir",
            str(train_data_dir),
            "--base-repos",
            str(base_repos),
            "--output-dir",
            str(splits),
            "--repos",
            "tiny",
            "dummy1",
            "dummy2",
            "--heldout-repos",
            "dummy2",
            "--min-history",
            "0",
            "--dev-ratio",
            "0.5",
        ]
    )
    assert (splits / "train.jsonl").exists()

    kb_root = tmp / "kb_root" / "kb_repos" / "tiny" / "repo2docAgent-output"
    kb_root.mkdir(parents=True)
    (kb_root / "knowledge_graph_two_index.json").write_text(kb.read_text(encoding="utf-8"), encoding="utf-8")
    (kb_root / "program_elements.jsonl").write_text(elements.read_text(encoding="utf-8"), encoding="utf-8")

    retrieval = tmp / "retrieval.jsonl"
    run(
        [
            PYTHON,
            "experiments/prompt_optimization/scripts/run_lightvector_retrieval.py",
            "--manifest",
            str(splits / "dev.jsonl"),
            "--repo-root",
            str(tmp / "kb_root"),
            "--output-dir-name",
            "repo2docAgent-output",
            "--kb-name",
            "knowledge_graph_two_index.json",
            "--output",
            str(retrieval),
            "--top-k",
            "5",
        ]
    )
    rows = [json.loads(line) for line in retrieval.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["status"] == "ok"

    summary = tmp / "summary.json"
    details = tmp / "details.jsonl"
    run(
        [
            PYTHON,
            "experiments/prompt_optimization/scripts/score_retrieval.py",
            "--results",
            str(retrieval),
            "--output-summary",
            str(summary),
            "--output-details",
            str(details),
            "--top-k",
            "5",
        ]
    )
    assert json.loads(summary.read_text(encoding="utf-8"))["valid_rows"] >= 1

    safe_prompt = tmp / "safe_prompt.txt"
    safe_prompt.write_text("Distill generic PR knowledge into JSON.\n", encoding="utf-8")
    leakage = tmp / "leakage.json"
    run(
        [
            PYTHON,
            "experiments/prompt_optimization/scripts/check_prompt_leakage.py",
            "--prompt",
            str(safe_prompt),
            "--manifest",
            str(splits / "dev.jsonl"),
            "--output",
            str(leakage),
            "--allow-generic-repo-names",
        ]
    )
    assert json.loads(leakage.read_text(encoding="utf-8"))["passed"]

    report = tmp / "report.md"
    run(
        [
            PYTHON,
            "experiments/prompt_optimization/scripts/make_iteration_report.py",
            "--summary",
            str(summary),
            "--details",
            str(details),
            "--prompt",
            str(safe_prompt),
            "--output",
            str(report),
            "--iteration-name",
            "smoke",
        ]
    )
    assert "Prompt Optimization Report" in report.read_text(encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rek-smoke-") as tmp_name:
        tmp = Path(tmp_name)
        repo, base_commit, _head_commit, patch = make_tiny_repo(tmp)
        elements, kb = test_cli_and_scanner(tmp, repo)
        test_graph_db_and_retrieval(tmp, repo, elements, kb)
        test_pr_gold_and_distiller_parse(repo, base_commit, patch)
        test_prompt_optimization_scripts(tmp, repo, base_commit, patch, kb, elements)
    print("SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
