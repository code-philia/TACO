#!/usr/bin/env python3
"""Optimize a prompt from per-sample before/after retrieval scores."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import load_jsonl, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate next prompt from per-sample retrieval feedback.")
    parser.add_argument("--current-prompt", type=Path, required=True)
    parser.add_argument("--train-details", type=Path, required=True)
    parser.add_argument("--dev-details", type=Path, required=True)
    parser.add_argument("--distill-summary", type=Path, required=True)
    parser.add_argument("--history-summary", type=Path)
    parser.add_argument("--failure-feedback", type=Path, action="append", default=[])
    parser.add_argument("--output-prompt", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-samples-per-split", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-prompt-chars", type=int, default=4200)
    parser.add_argument("--proposal-note", default="")
    return parser.parse_args()


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:text|markdown|md)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    return match.group(1).strip() if match else stripped


def abstract_rows(path: Path, limit: int) -> Dict[str, Any]:
    rows = load_jsonl(path)
    sorted_rows = sorted(rows, key=lambda row: (row.get("best_score", 0), row.get("mrr", 0)))
    sample_signals: List[Dict[str, Any]] = []
    for row in sorted_rows[:limit]:
        sample_signals.append(
            {
                "best_level": row.get("best_level"),
                "best_score": row.get("best_score"),
                "first_hit_rank": row.get("first_hit_rank"),
                "first_hit_level": row.get("first_hit_level"),
                "mrr": row.get("mrr"),
                "hit_file": row.get("hit_file"),
                "hit_module": row.get("hit_module"),
                "hit_function": row.get("hit_function"),
                "rank_file": row.get("rank_file"),
                "rank_module": row.get("rank_module"),
                "rank_function": row.get("rank_function"),
            }
        )
    return {
        "rows": len(rows),
        "score_distribution": {
            "miss": sum(1 for row in rows if row.get("best_level") == "miss"),
            "file": sum(1 for row in rows if row.get("best_level") == "file"),
            "module": sum(1 for row in rows if row.get("best_level") == "module"),
            "function": sum(1 for row in rows if row.get("best_level") == "function"),
        },
        "hard_sample_signals": sample_signals,
    }


def build_messages(args: argparse.Namespace) -> List[Dict[str, str]]:
    current_prompt = args.current_prompt.read_text(encoding="utf-8")
    distill_summary = json.loads(args.distill_summary.read_text(encoding="utf-8"))
    train_feedback = abstract_rows(args.train_details, args.max_samples_per_split)
    dev_feedback = abstract_rows(args.dev_details, args.max_samples_per_split)
    public_summary = {
        "rows": distill_summary.get("rows"),
        "status_counts": distill_summary.get("status_counts"),
        "zero_item_rows": distill_summary.get("zero_item_rows"),
        "batch_size": distill_summary.get("batch_size"),
        "workers": distill_summary.get("workers"),
        "usage": distill_summary.get("usage"),
        "knowledge_items_total": (distill_summary.get("kb_summary") or {}).get("knowledge_items_total"),
    }
    history_summary = None
    if args.history_summary and args.history_summary.exists():
        raw_history = json.loads(args.history_summary.read_text(encoding="utf-8"))
        champion = raw_history.get("champion") or {}
        history_summary = {
            "champion": {
                "run_name": champion.get("run_name"),
                "prompt": champion.get("prompt"),
                "utility": champion.get("utility"),
                "parse_empty": champion.get("parse_empty"),
                "knowledge_items_total": champion.get("knowledge_items_total"),
                "dev": champion.get("dev"),
            },
            "runs": [
                {
                    "run_name": run.get("run_name"),
                    "prompt": run.get("prompt"),
                    "utility": run.get("utility"),
                    "parse_empty": run.get("parse_empty"),
                    "knowledge_items_total": run.get("knowledge_items_total"),
                    "dev": run.get("dev"),
                }
                for run in raw_history.get("runs", [])
            ],
        }
    failure_feedback = []
    for feedback_path in args.failure_feedback:
        if feedback_path.exists():
            failure_feedback.append(json.loads(feedback_path.read_text(encoding="utf-8")))
    system = (
        "You optimize a generic LLM prompt for PR knowledge distillation. "
        "Use only abstract aggregate and per-sample score patterns. Do not output chain-of-thought. "
        "Make a small, testable prompt revision rather than accumulating rules."
    )
    user = f"""
We are training a prompt for historical pull-request knowledge distillation.

Goal:
- Improve downstream retrieval from the generated knowledge base.
- Prefer higher granularity: function hit > module hit > file hit > miss.
- Reduce parse_empty and zero-knowledge outputs.
- Keep output compatible with raw JSON list objects containing file, function, keywords, rationale.
- Prefer conservative edits that preserve known-good behavior. Do not rewrite the whole prompt unless the feedback strongly justifies it.
- Avoid prompt bloat: if later iterations became longer but dev scores or parse_empty got worse, simplify rather than add more rules.
- Keep the revised prompt under {args.max_prompt_chars} characters. If the current prompt is already long, replace or remove weaker clauses instead of appending.
- Make at most 3 substantive changes: one for parse_empty/JSON robustness, one for retrieval granularity, and one for noise filtering or keyword quality.
- If prior runs show that a newly added clause lowered dev utility, remove or soften that clause rather than reinforcing it.

Strict anti-leakage constraints:
- Do not include repository names, file paths, function names, class names, issue ids, PR ids, sample ids, or concrete APIs from the training/dev data.
- Do not add sample-specific hints.
- Only add generalizable instructions about noise filtering, code-evidence grounding, granularity, keyword quality, JSON robustness, and conservative inference.
- The failure examples below are diagnostic evidence only. You may infer general failure modes from them, but you must not copy any sample-specific identifiers into the prompt.

Current prompt:
{current_prompt}

Historical run summary, if available:
{json.dumps(history_summary, ensure_ascii=False, indent=2) if history_summary else "null"}

Distillation run summary:
{json.dumps(public_summary, ensure_ascii=False, indent=2)}

Train per-sample retrieval feedback, identifiers removed:
{json.dumps(train_feedback, ensure_ascii=False, indent=2)}

Dev per-sample retrieval feedback, identifiers removed:
{json.dumps(dev_feedback, ensure_ascii=False, indent=2)}

Detailed low-performing examples for diagnosis:
{json.dumps(failure_feedback, ensure_ascii=False, indent=2) if failure_feedback else "[]"}

Proposal note for this candidate:
{args.proposal_note or "none"}

Return only the revised prompt text. Do not wrap it in Markdown fences.
The revised prompt should be complete and directly usable, but concise.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Missing API key in ${args.api_key_env}", file=sys.stderr)
        return 2
    messages = build_messages(args)
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    started_at = datetime.now(timezone.utc).isoformat()
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
    )
    prompt_text = strip_markdown_fence(response.choices[0].message.content or "")
    if len(prompt_text) > args.max_prompt_chars:
        print(f"Warning: generated prompt has {len(prompt_text)} chars, above requested max {args.max_prompt_chars}", file=sys.stderr)
    args.output_prompt.parent.mkdir(parents=True, exist_ok=True)
    args.output_prompt.write_text(prompt_text.strip() + "\n", encoding="utf-8")
    write_json(
        args.output_metadata,
        {
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "base_url": args.base_url,
            "current_prompt": str(args.current_prompt),
            "train_details": str(args.train_details),
            "dev_details": str(args.dev_details),
            "distill_summary": str(args.distill_summary),
            "output_prompt": str(args.output_prompt),
            "usage": response.usage.model_dump() if response.usage else None,
            "prompt_chars": len(prompt_text),
            "max_prompt_chars": args.max_prompt_chars,
            "proposal_note": args.proposal_note,
        },
    )
    print(f"Wrote next prompt to {args.output_prompt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
