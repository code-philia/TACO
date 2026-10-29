#!/usr/bin/env python3
"""Summarize closed-loop prompt-training runs and select the current champion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize closed-loop prompt optimization runs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("experiments/prompt_optimization/results/closed_loop"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--min-dev-delta", type=float, default=0.01)
    parser.add_argument("--parse-empty-penalty", type=float, default=0.25)
    parser.add_argument("--function-hit-bonus", type=float, default=0.20)
    parser.add_argument("--min-rows", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_record(run_dir: Path, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    distill = load_json(run_dir / "distill_summary.json")
    train = load_json(run_dir / "train_score_summary.json")
    dev = load_json(run_dir / "dev_score_summary.json")
    leak = load_json(run_dir / "next_prompt_leakage.json")
    if not distill or not train or not dev:
        return None

    dev_overall = dev.get("overall", {})
    train_overall = train.get("overall", {})
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
        "prompt": distill.get("prompt"),
        "rows": rows,
        "ok": int((distill.get("status_counts") or {}).get("ok") or 0),
        "parse_empty": parse_empty,
        "parse_empty_rate": parse_empty_rate,
        "knowledge_items_total": (distill.get("kb_summary") or {}).get("knowledge_items_total"),
        "distill_tokens": (distill.get("usage") or {}).get("total_tokens"),
        "train": train_overall,
        "dev": dev_overall,
        "next_prompt_leakage_passed": None if leak is None else bool(leak.get("passed")),
        "utility": utility,
    }


def select_champion(records: List[Dict[str, Any]], min_dev_delta: float) -> Dict[str, Any]:
    if not records:
        raise ValueError("No complete closed-loop runs found.")
    champion = records[0]
    for candidate in records[1:]:
        champion_dev = float((champion.get("dev") or {}).get("avg_best_score") or 0.0)
        candidate_dev = float((candidate.get("dev") or {}).get("avg_best_score") or 0.0)
        candidate_better_dev = candidate_dev >= champion_dev + min_dev_delta
        roughly_equal_dev = abs(candidate_dev - champion_dev) < min_dev_delta
        if candidate_better_dev or (roughly_equal_dev and candidate["utility"] > champion["utility"]):
            champion = candidate
    return champion


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, records: List[Dict[str, Any]], champion: Dict[str, Any]) -> None:
    lines = [
        "# Closed-Loop Prompt Training Summary",
        "",
        f"- Champion run: `{champion['run_name']}`",
        f"- Champion prompt: `{champion['prompt']}`",
        "",
        "| Run | Parse Empty | Items | Utility | Dev Avg | Dev MRR | Dev File | Dev Module | Dev Function | Prompt |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        dev = record.get("dev") or {}
        lines.append(
            f"| `{record['run_name']}` | {record['parse_empty']} | {record['knowledge_items_total']} | "
            f"{record['utility']:.4f} | {dev.get('avg_best_score', 0):.4f} | {dev.get('mrr', 0):.4f} | "
            f"{dev.get('file_hit_rate', 0):.2f} | {dev.get('module_hit_rate', 0):.2f} | "
            f"{dev.get('function_hit_rate', 0):.2f} | `{record['prompt']}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    records = []
    for run_dir in sorted(path for path in args.runs_dir.iterdir() if path.is_dir()):
        record = run_record(run_dir, args)
        if record and record.get("rows", 0) >= args.min_rows:
            records.append(record)
    champion = select_champion(records, args.min_dev_delta)
    summary = {"champion": champion, "runs": records}
    if args.output_json:
        write_json(args.output_json, summary)
    if args.output_md:
        write_markdown(args.output_md, records, champion)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
