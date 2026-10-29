#!/usr/bin/env python3
"""Ask an LLM to propose the next generic distillation prompt candidate."""

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
    parser = argparse.ArgumentParser(description="Generate a prompt candidate from historical-PR retrieval feedback.")
    parser.add_argument("--current-prompt", type=Path, required=True)
    parser.add_argument("--iteration-report", type=Path, required=True)
    parser.add_argument("--score-details", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--output-prompt", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-4-mini-2026-03-17"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-error-examples", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:text|markdown|md)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    return match.group(1).strip() if match else stripped


def summarize_details(path: Path, max_examples: int) -> List[Dict[str, Any]]:
    rows = load_jsonl(path)
    hard_rows = [row for row in rows if row.get("best_level") in {"miss", "file"}]
    examples = []
    for row in hard_rows[:max_examples]:
        examples.append(
            {
                "best_level": row.get("best_level"),
                "best_score": row.get("best_score"),
                "first_hit_rank": row.get("first_hit_rank"),
                "hit_file": row.get("hit_file"),
                "hit_module": row.get("hit_module"),
                "hit_function": row.get("hit_function"),
            }
        )
    return examples


def manifest_stats(path: Path) -> Dict[str, Any]:
    rows = load_jsonl(path)
    repos: Dict[str, int] = {}
    gold_counts = {"files": 0, "modules": 0, "functions": 0}
    for row in rows:
        repo = str(row.get("repo") or "unknown")
        repos[repo] = repos.get(repo, 0) + 1
        gold = row.get("gold") or {}
        for key in gold_counts:
            if gold.get(key):
                gold_counts[key] += 1
    return {"rows": len(rows), "repos": repos, "rows_with_gold": gold_counts}


def build_messages(args: argparse.Namespace) -> List[Dict[str, str]]:
    current_prompt = args.current_prompt.read_text(encoding="utf-8")
    report = args.iteration_report.read_text(encoding="utf-8")
    error_examples = summarize_details(args.score_details, args.max_error_examples)
    train_stats = manifest_stats(args.train_manifest)
    dev_stats = manifest_stats(args.dev_manifest)

    system = (
        "You are optimizing a generic prompt for distilling reusable software-maintenance knowledge "
        "from historical pull requests. Do not include hidden chain-of-thought. Return only the revised prompt text."
    )
    user = f"""
Optimize the current prompt for a historical-PR knowledge distillation pipeline.

Hard constraints:
- Use only generalizable workflow instructions.
- Do not mention repository names, file paths, function names, issue ids, sample ids, or specific APIs from the data.
- Do not optimize for SWE-bench tasks directly; this is only historical PR prompt optimization.
- The prompt must still require a raw JSON list output with file, function, keywords, and rationale.
- Improve robustness against noisy PR text, shallow summaries, and missing code-change evidence.
- Keep the prompt concise enough for repeated batch use.

Current prompt:
{current_prompt}

Train manifest aggregate stats:
{json.dumps(train_stats, ensure_ascii=False, indent=2)}

Dev manifest aggregate stats:
{json.dumps(dev_stats, ensure_ascii=False, indent=2)}

Retrieval evaluation report:
{report[:12000]}

Abstracted hard-example signals, with all identifiers removed:
{json.dumps(error_examples, ensure_ascii=False, indent=2)}

Return only the improved prompt text. Do not wrap it in Markdown fences.
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
    args.output_prompt.parent.mkdir(parents=True, exist_ok=True)
    args.output_prompt.write_text(prompt_text.strip() + "\n", encoding="utf-8")

    metadata = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "current_prompt": str(args.current_prompt),
        "iteration_report": str(args.iteration_report),
        "score_details": str(args.score_details),
        "output_prompt": str(args.output_prompt),
        "usage": response.usage.model_dump() if response.usage else None,
    }
    write_json(args.output_metadata, metadata)
    print(f"Wrote candidate prompt to {args.output_prompt}")
    print(f"Wrote metadata to {args.output_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
