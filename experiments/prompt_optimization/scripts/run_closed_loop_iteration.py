#!/usr/bin/env python3
"""Run one closed-loop prompt-training iteration inside the experiment folder."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill train PRs, evaluate retrieval, and generate next prompt.")
    parser.add_argument("--iteration-name", required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "experiments/prompt_optimization/data_splits/sample_100")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "experiments/prompt_optimization/results/closed_loop")
    parser.add_argument("--prompts-dir", type=Path, default=REPO_ROOT / "experiments/prompt_optimization/prompts")
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-patch-chars", type=int, default=7000)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--history-summary", type=Path)
    parser.add_argument("--skip-distill", action="store_true")
    parser.add_argument("--retrieval-cpu", action="store_true", help="hide CUDA devices for retrieval subprocesses")
    parser.add_argument("--failure-feedback-examples", type=int, default=0, help="examples per split to pass to the prompt optimizer")
    parser.add_argument("--max-prompt-chars", type=int, default=4200)
    parser.add_argument("--skip-optimize", action="store_true")
    return parser.parse_args()


def run(cmd: List[str], log_file: Path, env: dict[str, str] | None = None) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    printable = "$ " + " ".join(cmd)
    print(printable, flush=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(printable + "\n")
        handle.flush()
        result = subprocess.run(cmd, text=True, capture_output=True, env=env)
        if result.stdout:
            print(result.stdout, end="")
            handle.write(result.stdout)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
            handle.write("\n[stderr]\n" + result.stderr)
        handle.write("\n---\n")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def score_split(args: argparse.Namespace, iteration_dir: Path, split: str, log_file: Path) -> None:
    retrieval_path = iteration_dir / f"{split}_retrieval.jsonl"
    summary_path = iteration_dir / f"{split}_score_summary.json"
    details_path = iteration_dir / f"{split}_score_details.jsonl"
    report_path = iteration_dir / f"{split}_report.md"
    retrieval_env = os.environ.copy()
    if args.retrieval_cpu:
        retrieval_env["CUDA_VISIBLE_DEVICES"] = ""
    run(
        [
            PYTHON,
            str(SCRIPT_DIR / "run_lightvector_retrieval.py"),
            "--manifest",
            str(args.splits_dir / f"{split}.jsonl"),
            "--output",
            str(retrieval_path),
            "--repo-root",
            str(iteration_dir),
            "--output-dir-name",
            "repo2docAgent-output",
            "--kb-name",
            "knowledge_graph_two_index.json",
            "--top-k",
            str(args.top_k),
        ],
        log_file,
        env=retrieval_env,
    )
    run(
        [
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
        ],
        log_file,
    )
    run(
        [
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
        ],
        log_file,
    )


def build_failure_feedback(args: argparse.Namespace, iteration_dir: Path, split: str, log_file: Path) -> Path | None:
    if args.failure_feedback_examples <= 0:
        return None
    output_path = iteration_dir / f"{split}_failure_feedback.json"
    run(
        [
            PYTHON,
            str(SCRIPT_DIR / "build_failure_feedback.py"),
            "--manifest",
            str(args.splits_dir / f"{split}.jsonl"),
            "--score-details",
            str(iteration_dir / f"{split}_score_details.jsonl"),
            "--retrieval",
            str(iteration_dir / f"{split}_retrieval.jsonl"),
            "--distilled-results",
            str(iteration_dir / "distilled_results.jsonl"),
            "--split-name",
            split,
            "--max-examples",
            str(args.failure_feedback_examples),
            "--output",
            str(output_path),
        ],
        log_file,
    )
    return output_path


def main() -> int:
    args = parse_args()
    iteration_dir = args.work_dir / args.iteration_name
    iteration_dir.mkdir(parents=True, exist_ok=True)
    log_file = iteration_dir / "closed_loop.log"

    if not args.skip_distill:
        run(
            [
                PYTHON,
                str(SCRIPT_DIR / "distill_manifest_to_kb.py"),
                "--manifest",
                str(args.splits_dir / "train.jsonl"),
                "--prompt",
                str(args.prompt),
                "--output-dir",
                str(iteration_dir),
                "--model",
                args.model,
                "--base-url",
                args.base_url,
                "--workers",
                str(args.workers),
                "--batch-size",
                str(args.batch_size),
                "--max-patch-chars",
                str(args.max_patch_chars),
                "--request-timeout",
                str(args.request_timeout),
            ],
            log_file,
        )

    for split in ("train", "dev"):
        score_split(args, iteration_dir, split, log_file)

    feedback_paths = [
        path
        for split in ("train", "dev")
        for path in [build_failure_feedback(args, iteration_dir, split, log_file)]
        if path is not None
    ]

    if args.skip_optimize:
        print(f"Closed-loop evaluation outputs written to {iteration_dir}")
        return 0

    next_prompt = args.prompts_dir / f"{args.iteration_name}_next_prompt.txt"
    next_metadata = args.prompts_dir / f"{args.iteration_name}_next_prompt.metadata.json"
    optimize_cmd = [
        PYTHON,
        str(SCRIPT_DIR / "optimize_prompt_from_sample_scores.py"),
        "--current-prompt",
        str(args.prompt),
        "--train-details",
        str(iteration_dir / "train_score_details.jsonl"),
        "--dev-details",
        str(iteration_dir / "dev_score_details.jsonl"),
        "--distill-summary",
        str(iteration_dir / "distill_summary.json"),
        "--output-prompt",
        str(next_prompt),
        "--output-metadata",
        str(next_metadata),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--max-prompt-chars",
        str(args.max_prompt_chars),
    ]
    if args.history_summary:
        optimize_cmd.extend(["--history-summary", str(args.history_summary)])
    for feedback_path in feedback_paths:
        optimize_cmd.extend(["--failure-feedback", str(feedback_path)])
    run(optimize_cmd, log_file)
    run(
        [
            PYTHON,
            str(SCRIPT_DIR / "check_prompt_leakage.py"),
            "--prompt",
            str(next_prompt),
            "--manifest",
            str(args.splits_dir / "train.jsonl"),
            "--manifest",
            str(args.splits_dir / "dev.jsonl"),
            "--output",
            str(iteration_dir / "next_prompt_leakage.json"),
        ],
        log_file,
    )
    print(f"Closed-loop outputs written to {iteration_dir}")
    print(f"Next prompt: {next_prompt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
