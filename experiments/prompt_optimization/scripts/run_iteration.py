#!/usr/bin/env python3
"""Run one MVP prompt-optimization evaluation iteration end to end.

This orchestrates:
1. optional historical split generation
2. prompt leakage check
3. LightVector retrieval for selected splits
4. retrieval scoring
5. markdown iteration report generation

It does not distill a new KB yet; it evaluates an existing KB produced by a
candidate prompt. This keeps the iteration runner usable for current KBs and
future prompt candidates.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
PYTHON = sys.executable

DEFAULT_REPOS = [
    "astropy",
    "django",
    "matplotlib",
    "seaborn",
    "flask",
    "requests",
    "xarray",
    "pylint",
    "pytest",
    "scikit-learn",
    "sphinx",
    "sympy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one prompt optimization evaluation iteration.")
    parser.add_argument("--iteration-name", required=True)
    parser.add_argument("--prompt", type=Path, default=EXPERIMENT_ROOT / "seeds" / "current_six_step_prompt.txt")
    parser.add_argument("--work-dir", type=Path, default=EXPERIMENT_ROOT / "results")
    parser.add_argument("--splits-dir", type=Path, default=EXPERIMENT_ROOT / "data_splits")
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument("--eval-splits", nargs="+", default=["dev"], choices=["train", "dev", "heldout_transfer"])

    parser.add_argument("--train-data-dir", type=Path, default=Path(os.environ.get("REK_TRAIN_DATA_DIR", "YOUR_TRAIN_DATA_DIR_HERE")))
    parser.add_argument("--base-repos", type=Path, default=Path(os.environ.get("REK_BASE_REPOS", "YOUR_REPO_CHECKOUTS_DIR_HERE")))
    parser.add_argument("--output-dir-name", default="repo2docAgent-output-gpt-5-4-mini")
    parser.add_argument("--kb-name", default="knowledge_graph_two_index_epoch_1.json")
    parser.add_argument("--elements-name", default="program_elements.jsonl")

    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    parser.add_argument("--heldout-repos", nargs="+", default=[])
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--max-per-repo", type=int, default=0)
    parser.add_argument("--min-history", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repo-filter", help="optional single repo for retrieval smoke tests")
    parser.add_argument("--allow-leakage", action="store_true")
    parser.add_argument("--skip-leakage-check", action="store_true")
    return parser.parse_args()


def run(cmd: List[str], log_file: Path | None = None, allow_failure: bool = False) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, env=env)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(str(part) for part in cmd) + "\n")
            if result.stdout:
                handle.write(result.stdout)
            if result.stderr:
                handle.write("\n[stderr]\n" + result.stderr)
            handle.write("\n---\n")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0 and not allow_failure:
        raise SystemExit(result.returncode)
    return result


def split_manifest(splits_dir: Path, split: str) -> Path:
    return splits_dir / f"{split}.jsonl"


def main() -> int:
    args = parse_args()
    iteration_dir = args.work_dir / args.iteration_name
    iteration_dir.mkdir(parents=True, exist_ok=True)
    log_file = iteration_dir / "iteration.log"

    if args.rebuild_splits or not split_manifest(args.splits_dir, "dev").exists():
        build_cmd = [
            PYTHON,
            str(SCRIPT_DIR / "build_historical_splits.py"),
            "--train-data-dir",
            str(args.train_data_dir),
            "--base-repos",
            str(args.base_repos),
            "--output-dir",
            str(args.splits_dir),
            "--dev-ratio",
            str(args.dev_ratio),
            "--max-per-repo",
            str(args.max_per_repo),
            "--min-history",
            str(args.min_history),
            "--repos",
            *args.repos,
        ]
        if args.heldout_repos:
            build_cmd.extend(["--heldout-repos", *args.heldout_repos])
        run(build_cmd, log_file=log_file)

    leakage_report = iteration_dir / "leakage_report.json"
    if not args.skip_leakage_check:
        leakage_cmd = [
            PYTHON,
            str(SCRIPT_DIR / "check_prompt_leakage.py"),
            "--prompt",
            str(args.prompt),
            "--output",
            str(leakage_report),
        ]
        for split in ("train", "dev"):
            manifest = split_manifest(args.splits_dir, split)
            if manifest.exists():
                leakage_cmd.extend(["--manifest", str(manifest)])
        leakage_result = run(leakage_cmd, log_file=log_file, allow_failure=args.allow_leakage)
        if leakage_result.returncode != 0 and not args.allow_leakage:
            print(f"Leakage check failed. See {leakage_report}", file=sys.stderr)
            return leakage_result.returncode

    for split in args.eval_splits:
        manifest = split_manifest(args.splits_dir, split)
        if not manifest.exists():
            print(f"Skipping missing split manifest: {manifest}")
            continue
        retrieval_path = iteration_dir / f"{split}_retrieval.jsonl"
        summary_path = iteration_dir / f"{split}_score_summary.json"
        details_path = iteration_dir / f"{split}_score_details.jsonl"
        report_path = iteration_dir / f"{split}_report.md"

        retrieval_cmd = [
            PYTHON,
            str(SCRIPT_DIR / "run_lightvector_retrieval.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(retrieval_path),
            "--base-repos",
            str(args.base_repos),
            "--output-dir-name",
            args.output_dir_name,
            "--kb-name",
            args.kb_name,
            "--elements-name",
            args.elements_name,
            "--top-k",
            str(args.top_k),
        ]
        if args.repo_filter:
            retrieval_cmd.extend(["--repo", args.repo_filter])
        run(retrieval_cmd, log_file=log_file)

        score_cmd = [
            PYTHON,
            str(SCRIPT_DIR / "score_retrieval.py"),
            "--results",
            str(retrieval_path),
            "--output-summary",
            str(summary_path),
            "--output-details",
            str(details_path),
            "--top-k",
            str(args.top_k),
        ]
        run(score_cmd, log_file=log_file)

        report_cmd = [
            PYTHON,
            str(SCRIPT_DIR / "make_iteration_report.py"),
            "--summary",
            str(summary_path),
            "--details",
            str(details_path),
            "--prompt",
            str(args.prompt),
            "--output",
            str(report_path),
            "--iteration-name",
            f"{args.iteration_name}_{split}",
        ]
        run(report_cmd, log_file=log_file)

    print(f"Iteration outputs written to {iteration_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
