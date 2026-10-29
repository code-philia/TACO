#!/usr/bin/env python3
"""Run a multi-iteration champion/challenger prompt optimization loop."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run champion/challenger prompt optimization for N iterations.")
    parser.add_argument("--loop-name", required=True)
    parser.add_argument("--initial-prompt", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "experiments/prompt_optimization/results/champion_loops")
    parser.add_argument("--prompts-dir", type=Path, default=REPO_ROOT / "experiments/prompt_optimization/prompts/champion_loops")
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-patch-chars", type=int, default=800)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--failure-feedback-examples", type=int, default=12)
    parser.add_argument("--max-prompt-chars", type=int, default=4200)
    parser.add_argument("--min-utility-delta", type=float, default=0.005)
    parser.add_argument("--reject-cursor", choices=["generated", "champion"], default="generated", help="which prompt to evaluate after rejecting a challenger")
    parser.add_argument("--parse-empty-penalty", type=float, default=0.25)
    parser.add_argument("--function-hit-bonus", type=float, default=0.20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%F %T')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd: List[str], log_file: Path, env: Optional[Dict[str, str]] = None) -> None:
    printable = "$ " + " ".join(cmd)
    append_log(log_file, printable)
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


def record_from_run(run_dir: Path, prompt: Path, prompts_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    distill = load_json(run_dir / "distill_summary.json")
    train = load_json(run_dir / "train_score_summary.json")
    dev = load_json(run_dir / "dev_score_summary.json")
    leak_path = run_dir / "next_prompt_leakage.json"
    leak = load_json(leak_path) if leak_path.exists() else {}
    dev_overall = dev.get("overall") or {}
    train_overall = train.get("overall") or {}
    rows = int(distill.get("rows") or 0)
    parse_empty = int((distill.get("status_counts") or {}).get("parse_empty") or 0)
    parse_empty_rate = parse_empty / rows if rows else 0.0
    utility = (
        float(dev_overall.get("avg_best_score") or 0.0)
        + args.function_hit_bonus * float(dev_overall.get("function_hit_rate") or 0.0)
        - args.parse_empty_penalty * parse_empty_rate
    )
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "prompt": str(prompt),
        "rows": rows,
        "ok": int((distill.get("status_counts") or {}).get("ok") or 0),
        "parse_empty": parse_empty,
        "parse_empty_rate": parse_empty_rate,
        "knowledge_items_total": (distill.get("kb_summary") or {}).get("knowledge_items_total"),
        "distill_tokens": (distill.get("usage") or {}).get("total_tokens"),
        "train": train_overall,
        "dev": dev_overall,
        "utility": utility,
        "next_prompt": str(prompts_dir / f"{run_dir.name}_next_prompt.txt"),
        "next_prompt_leakage_passed": bool(leak.get("passed", False)),
    }


def write_markdown(path: Path, state: Dict[str, Any]) -> None:
    champion = state.get("champion") or {}
    lines = [
        f"# Champion Loop: {state.get('loop_name')}",
        "",
        f"- Started: `{state.get('started_at')}`",
        f"- Updated: `{state.get('updated_at')}`",
        f"- Splits: `{state.get('splits_dir')}`",
        f"- Champion run: `{champion.get('run_name')}`",
        f"- Champion prompt: `{champion.get('prompt')}`",
        f"- Champion utility: `{champion.get('utility')}`",
        "",
        "| Iter | Run | Decision | Utility | Dev Avg | MRR | File | Module | Function | Parse Empty | Items | Prompt |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in state.get("iterations", []):
        record = item.get("record") or {}
        dev = record.get("dev") or {}
        lines.append(
            f"| {item.get('iteration')} | `{record.get('run_name')}` | {item.get('decision')} | "
            f"{record.get('utility', 0):.4f} | {dev.get('avg_best_score', 0):.4f} | {dev.get('mrr', 0):.4f} | "
            f"{dev.get('file_hit_rate', 0):.2f} | {dev.get('module_hit_rate', 0):.2f} | {dev.get('function_hit_rate', 0):.2f} | "
            f"{record.get('parse_empty')} | {record.get('knowledge_items_total')} | `{record.get('prompt')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    loop_dir = args.work_dir / args.loop_name
    loop_runs_dir = loop_dir / "runs"
    prompts_dir = args.prompts_dir / args.loop_name
    state_path = loop_dir / "loop_state.json"
    summary_path = loop_dir / "loop_summary.md"
    log_file = loop_dir / "loop_driver.log"
    loop_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and state_path.exists():
        state = load_json(state_path)
        prompt_cursor = Path(state.get("prompt_cursor") or args.initial_prompt)
        champion = state.get("champion")
        start_iteration = len(state.get("iterations", [])) + 1
        append_log(log_file, f"Resuming from iteration {start_iteration}; cursor={prompt_cursor}")
    else:
        state = {
            "loop_name": args.loop_name,
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "splits_dir": str(args.splits_dir),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "iterations": [],
            "champion": None,
            "prompt_cursor": str(args.initial_prompt),
        }
        prompt_cursor = args.initial_prompt
        champion = None
        start_iteration = 1
        write_json(state_path, state)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    for iteration in range(start_iteration, args.iterations + 1):
        run_name = f"{args.loop_name}_iter_{iteration:02d}"
        append_log(log_file, f"START iteration={iteration} prompt={prompt_cursor}")
        run(
            [
                PYTHON,
                str(SCRIPT_DIR / "run_closed_loop_iteration.py"),
                "--iteration-name",
                run_name,
                "--prompt",
                str(prompt_cursor),
                "--splits-dir",
                str(args.splits_dir),
                "--work-dir",
                str(loop_runs_dir),
                "--prompts-dir",
                str(prompts_dir),
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
                "--top-k",
                str(args.top_k),
                "--retrieval-cpu",
                "--failure-feedback-examples",
                str(args.failure_feedback_examples),
                "--max-prompt-chars",
                str(args.max_prompt_chars),
            ],
            log_file,
            env=env,
        )
        run_dir = loop_runs_dir / run_name
        record = record_from_run(run_dir, prompt_cursor, prompts_dir, args)
        next_prompt = prompts_dir / f"{run_name}_next_prompt.txt"

        if champion is None:
            decision = "accept_initial"
            champion = record
            prompt_cursor = next_prompt
        elif record["utility"] >= float(champion.get("utility") or 0.0) + args.min_utility_delta:
            decision = "accept_challenger"
            champion = record
            prompt_cursor = next_prompt
        else:
            decision = f"reject_challenger_continue_{args.reject_cursor}"
            if args.reject_cursor == "champion":
                prompt_cursor = Path(champion["prompt"])
            else:
                prompt_cursor = next_prompt

        state["iterations"].append(
            {
                "iteration": iteration,
                "started_prompt": str(record["prompt"]),
                "generated_next_prompt": str(next_prompt),
                "decision": decision,
                "record": record,
            }
        )
        state["champion"] = champion
        state["prompt_cursor"] = str(prompt_cursor)
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        write_markdown(summary_path, state)
        append_log(log_file, f"DONE iteration={iteration} decision={decision} utility={record['utility']:.4f} next_cursor={prompt_cursor}")

    append_log(log_file, "ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
