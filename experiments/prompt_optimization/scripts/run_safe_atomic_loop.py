#!/usr/bin/env python3
"""Run a small safe atomic prompt-edit loop.

Teacher sees train detailed feedback and dev aggregate metrics only. Each epoch
creates one atomic edit, evaluates one variant, and accepts it only if dev gate
improves without parse regression.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safe atomic prompt edit loop.")
    parser.add_argument("--loop-name", required=True)
    parser.add_argument("--initial-prompt", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "experiments/prompt_optimization/results/safe_atomic_loops")
    parser.add_argument("--prompts-dir", type=Path, default=REPO_ROOT / "experiments/prompt_optimization/prompts/safe_atomic_loops")
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-patch-chars", type=int, default=800)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--failure-feedback-examples", type=int, default=8)
    parser.add_argument("--success-feedback-examples", type=int, default=8)
    parser.add_argument("--min-utility-delta", type=float, default=0.005)
    parser.add_argument("--parse-regression-limit", type=int, default=2)
    parser.add_argument("--function-hit-bonus", type=float, default=0.20)
    parser.add_argument("--parse-empty-penalty", type=float, default=0.25)
    parser.add_argument("--edit-focus-note", default="")
    parser.add_argument("--candidate-edits", type=int, default=1)
    parser.add_argument("--edit-temperature", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now().strftime('%F %T')}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd: List[str], log_file: Path, env: Dict[str, str]) -> None:
    printable = "$ " + " ".join(cmd)
    log(log_file, printable)
    with log_file.open("a", encoding="utf-8") as handle:
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


def run_for_status(cmd: List[str], log_file: Path, env: Dict[str, str]) -> int:
    printable = "$ " + " ".join(cmd)
    log(log_file, printable)
    with log_file.open("a", encoding="utf-8") as handle:
        result = subprocess.run(cmd, text=True, capture_output=True, env=env)
        if result.stdout:
            print(result.stdout, end="")
            handle.write(result.stdout)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
            handle.write("\n[stderr]\n" + result.stderr)
        handle.write("\n---\n")
    return result.returncode


def summarize_run(run_dir: Path, prompt: Path, args: argparse.Namespace) -> Dict[str, Any]:
    distill = load_json(run_dir / "distill_summary.json")
    train = load_json(run_dir / "train_score_summary.json")
    dev = load_json(run_dir / "dev_score_summary.json")
    rows = int(distill.get("rows") or 0)
    parse_empty = int((distill.get("status_counts") or {}).get("parse_empty") or 0)
    parse_rate = parse_empty / rows if rows else 0.0
    dev_overall = dev.get("overall") or {}
    utility = (
        float(dev_overall.get("avg_best_score") or 0.0)
        + args.function_hit_bonus * float(dev_overall.get("function_hit_rate") or 0.0)
        - args.parse_empty_penalty * parse_rate
    )
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "prompt": str(prompt),
        "rows": rows,
        "parse_empty": parse_empty,
        "parse_empty_rate": parse_rate,
        "knowledge_items_total": (distill.get("kb_summary") or {}).get("knowledge_items_total"),
        "train": train.get("overall") or {},
        "dev": dev_overall,
        "utility": utility,
    }


def evaluate(args: argparse.Namespace, run_name: str, prompt: Path, runs_dir: Path, prompts_dir: Path, log_file: Path, env: Dict[str, str]) -> Path:
    run(
        [
            PYTHON,
            str(SCRIPT_DIR / "run_closed_loop_iteration.py"),
            "--iteration-name", run_name,
            "--prompt", str(prompt),
            "--splits-dir", str(args.splits_dir),
            "--work-dir", str(runs_dir),
            "--prompts-dir", str(prompts_dir),
            "--model", args.model,
            "--base-url", args.base_url,
            "--workers", str(args.workers),
            "--batch-size", str(args.batch_size),
            "--max-patch-chars", str(args.max_patch_chars),
            "--request-timeout", str(args.request_timeout),
            "--top-k", str(args.top_k),
            "--retrieval-cpu",
            "--failure-feedback-examples", str(args.failure_feedback_examples),
            "--max-prompt-chars", "4200",
            "--skip-optimize",
        ],
        log_file,
        env,
    )
    return runs_dir / run_name


def ensure_train_feedback(args: argparse.Namespace, run_dir: Path, log_file: Path, env: Dict[str, str]) -> None:
    run(
        [
            PYTHON, str(SCRIPT_DIR / "build_success_feedback.py"),
            "--manifest", str(args.splits_dir / "train.jsonl"),
            "--score-details", str(run_dir / "train_score_details.jsonl"),
            "--retrieval", str(run_dir / "train_retrieval.jsonl"),
            "--distilled-results", str(run_dir / "distilled_results.jsonl"),
            "--split-name", "train",
            "--max-examples", str(args.success_feedback_examples),
            "--output", str(run_dir / "train_success_feedback.json"),
        ],
        log_file,
        env,
    )
    run(
        [
            PYTHON, str(SCRIPT_DIR / "summarize_knowledge_quality.py"),
            "--distilled-results", str(run_dir / "distilled_results.jsonl"),
            "--output", str(run_dir / "knowledge_quality_summary.json"),
        ],
        log_file,
        env,
    )


def generate_edit(args: argparse.Namespace, champion: Dict[str, Any], edit_path: Path, metadata_path: Path, log_file: Path, env: Dict[str, str], epoch: int) -> None:
    run_dir = Path(champion["run_dir"])
    focus = args.edit_focus_note or (
        "Propose one tiny train-derived edit. Dev details are hidden; use dev aggregate only as a generalization warning. "
        "Do not chase function hit by narrowing too much; preserve file/module and parse stability."
    )
    run(
        [
            PYTHON, str(SCRIPT_DIR / "generate_atomic_prompt_edits.py"),
            "--champion-prompt", champion["prompt"],
            "--train-run-dir", str(run_dir),
            "--dev-summary", str(run_dir / "dev_score_summary.json"),
            "--output-edits", str(edit_path),
            "--output-metadata", str(metadata_path),
            "--model", args.model,
            "--base-url", args.base_url,
            "--train-success-feedback", str(run_dir / "train_success_feedback.json"),
            "--train-failure-feedback", str(run_dir / "train_failure_feedback.json"),
            "--train-quality-summary", str(run_dir / "knowledge_quality_summary.json"),
            "--max-edits", str(args.candidate_edits),
            "--temperature", str(args.edit_temperature),
            "--focus-note", f"{focus} Epoch {epoch}.",
        ],
        log_file,
        env,
    )


def write_markdown(path: Path, state: Dict[str, Any]) -> None:
    lines = [
        f"# Safe Atomic Loop: {state.get('loop_name')}",
        "",
        f"- Updated: `{state.get('updated_at')}`",
        f"- Champion: `{(state.get('champion') or {}).get('run_name')}`",
        "",
        "| Epoch | Run | Decision | Utility | Dev Avg | MRR | File | Module | Function | Parse | Items |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in state.get("epochs", []):
        record = item.get("record") or {}
        dev = record.get("dev") or {}
        lines.append(
            f"| {item.get('epoch')} | `{record.get('run_name')}` | {item.get('decision')} | "
            f"{record.get('utility', 0):.4f} | {dev.get('avg_best_score', 0):.4f} | {dev.get('mrr', 0):.4f} | "
            f"{dev.get('file_hit_rate', 0):.2f} | {dev.get('module_hit_rate', 0):.2f} | {dev.get('function_hit_rate', 0):.2f} | "
            f"{record.get('parse_empty')} | {record.get('knowledge_items_total')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    loop_dir = args.work_dir / args.loop_name
    runs_dir = loop_dir / "runs"
    prompts_dir = args.prompts_dir / args.loop_name
    state_path = loop_dir / "loop_state.json"
    summary_path = loop_dir / "loop_summary.md"
    log_file = loop_dir / "loop_driver.log"
    loop_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    if args.resume and state_path.exists():
        state = load_json(state_path)
        start_epoch = len(state.get("epochs", [])) + 1
    else:
        state = {"loop_name": args.loop_name, "started_at": utc_now(), "updated_at": utc_now(), "epochs": [], "champion": None}
        start_epoch = 0

    if state.get("champion") is None:
        run_name = f"{args.loop_name}_epoch_00_seed"
        log(log_file, "Evaluating seed champion")
        run_dir = evaluate(args, run_name, args.initial_prompt, runs_dir, prompts_dir, log_file, env)
        ensure_train_feedback(args, run_dir, log_file, env)
        record = summarize_run(run_dir, args.initial_prompt, args)
        state["champion"] = record
        state["epochs"].append({"epoch": 0, "decision": "accept_seed", "record": record})
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        write_markdown(summary_path, state)
        start_epoch = 1

    for epoch in range(max(1, start_epoch), args.epochs + 1):
        champion = state["champion"]
        edit_path = prompts_dir / f"{args.loop_name}_epoch_{epoch:02d}_edit.json"
        edit_meta = prompts_dir / f"{args.loop_name}_epoch_{epoch:02d}_edit.metadata.json"
        generate_edit(args, champion, edit_path, edit_meta, log_file, env, epoch)

        candidate_records: List[Dict[str, Any]] = []
        for candidate_index in range(args.candidate_edits):
            suffix = f"epoch_{epoch:02d}_c{candidate_index:02d}"
            variant_prompt = prompts_dir / f"{args.loop_name}_{suffix}_variant.txt"
            variant_meta = prompts_dir / f"{args.loop_name}_{suffix}_variant.metadata.json"
            apply_code = run_for_status(
                [
                    PYTHON, str(SCRIPT_DIR / "apply_atomic_prompt_edit.py"),
                    "--prompt", champion["prompt"],
                    "--edits", str(edit_path),
                    "--edit-index", str(candidate_index),
                    "--output", str(variant_prompt),
                    "--metadata", str(variant_meta),
                ],
                log_file,
                env,
            )
            if apply_code != 0:
                log(log_file, f"SKIP epoch={epoch} candidate={candidate_index} apply_failed")
                continue

            leakage_path = variant_prompt.with_suffix(".leakage.json")
            leakage_code = run_for_status(
                [
                    PYTHON, str(SCRIPT_DIR / "check_prompt_leakage.py"),
                    "--prompt", str(variant_prompt),
                    "--manifest", str(args.splits_dir / "train.jsonl"),
                    "--manifest", str(args.splits_dir / "dev.jsonl"),
                    "--output", str(leakage_path),
                ],
                log_file,
                env,
            )
            if leakage_code != 0:
                leakage_report = load_json(leakage_path) if leakage_path.exists() else {"returncode": leakage_code}
                candidate_records.append(
                    {
                        "run_name": f"{args.loop_name}_{suffix}_candidate",
                        "run_dir": None,
                        "prompt": str(variant_prompt),
                        "rows": 0,
                        "parse_empty": None,
                        "parse_empty_rate": None,
                        "knowledge_items_total": None,
                        "train": {},
                        "dev": {},
                        "utility": float("-inf"),
                        "leakage": leakage_report,
                        "candidate_index": candidate_index,
                    }
                )
                log(log_file, f"SKIP epoch={epoch} candidate={candidate_index} leakage_failed")
                continue

            run_name = f"{args.loop_name}_{suffix}_candidate"
            run_dir = evaluate(args, run_name, variant_prompt, runs_dir, prompts_dir, log_file, env)
            ensure_train_feedback(args, run_dir, log_file, env)
            record = summarize_run(run_dir, variant_prompt, args)
            record["candidate_index"] = candidate_index
            candidate_records.append(record)

        if not candidate_records:
            record = {
                "run_name": f"{args.loop_name}_epoch_{epoch:02d}_no_valid_candidate",
                "run_dir": None,
                "prompt": champion["prompt"],
                "rows": 0,
                "parse_empty": None,
                "parse_empty_rate": None,
                "knowledge_items_total": None,
                "train": {},
                "dev": {},
                "utility": float("-inf"),
            }
            state["epochs"].append({"epoch": epoch, "decision": "reject_no_valid_candidate", "record": record, "edit": str(edit_path), "candidates": []})
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            write_markdown(summary_path, state)
            log(log_file, f"DONE epoch={epoch} decision=reject_no_valid_candidate")
            continue

        valid_records = [row for row in candidate_records if row.get("run_dir")]
        if not valid_records:
            record = max(candidate_records, key=lambda row: float(row.get("utility") or float("-inf")))
            state["epochs"].append({"epoch": epoch, "decision": "reject_no_evaluated_candidate", "record": record, "edit": str(edit_path), "candidates": candidate_records})
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            write_markdown(summary_path, state)
            log(log_file, f"DONE epoch={epoch} decision=reject_no_evaluated_candidate")
            continue

        record = max(valid_records, key=lambda row: float(row.get("utility") or float("-inf")))
        champion = state["champion"]
        parse_ok = record["parse_empty"] <= int(champion["parse_empty"]) + args.parse_regression_limit
        utility_ok = record["utility"] >= float(champion["utility"]) + args.min_utility_delta
        dev_ok = float((record["dev"] or {}).get("avg_best_score") or 0.0) >= float((champion["dev"] or {}).get("avg_best_score") or 0.0) - 0.005
        if parse_ok and utility_ok and dev_ok:
            decision = "accept_candidate"
            state["champion"] = record
        else:
            decision = "reject_candidate"
        state["epochs"].append({"epoch": epoch, "decision": decision, "record": record, "edit": str(edit_path), "candidates": candidate_records})
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        write_markdown(summary_path, state)
        log(log_file, f"DONE epoch={epoch} decision={decision} utility={record['utility']:.4f}")

    log(log_file, "ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
