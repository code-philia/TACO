#!/usr/bin/env python3
"""Generate general experience cards from one distillation/evaluation epoch."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prompt experience cards from epoch feedback.")
    parser.add_argument("--seed-prompt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-cards", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-cards", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--existing-cards", type=Path, action="append", default=[])
    parser.add_argument("--success-feedback", type=Path, action="append", default=[])
    parser.add_argument("--quality-summary", type=Path)
    parser.add_argument("--focus-note", default="")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_feedback(path: Path, max_examples: int = 5) -> Any:
    if not path.exists():
        return None
    data = load_json(path)
    examples = []
    for ex in data.get("examples", [])[:max_examples]:
        examples.append(
            {
                "score": ex.get("score"),
                "diagnostic_hints": ex.get("diagnostic_hints"),
                "query_excerpt": ex.get("query_excerpt"),
                "gold_counts": (ex.get("gold") or {}).get("counts"),
                "distilled_status": (ex.get("distilled_knowledge") or {}).get("status"),
                "distilled_item_count": (ex.get("distilled_knowledge") or {}).get("item_count"),
                "top_retrieval_previews": [item.get("preview") for item in (ex.get("top_retrieval") or [])[:3]],
            }
        )
    return {"split": data.get("split"), "examples": examples}


def strip_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def normalize_cards(raw: Any, max_cards: int) -> list[dict[str, str]]:
    if isinstance(raw, dict):
        raw = raw.get("cards") or raw.get("experience_cards") or []
    cards = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        instruction = str(item.get("instruction") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        if title and instruction:
            cards.append({"title": title, "instruction": instruction, "rationale": rationale})
    return cards[:max_cards]


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in ${args.api_key_env}")
    seed_prompt = args.seed_prompt.read_text(encoding="utf-8")
    distill = load_json(args.run_dir / "distill_summary.json")
    train = load_json(args.run_dir / "train_score_summary.json")
    dev = load_json(args.run_dir / "dev_score_summary.json")
    train_fail = compact_feedback(args.run_dir / "train_failure_feedback.json")
    dev_fail = compact_feedback(args.run_dir / "dev_failure_feedback.json")
    existing_cards = []
    for cards_path in args.existing_cards:
        if cards_path.exists():
            existing_cards.extend(load_json(cards_path))
    success_feedback = []
    for feedback_path in args.success_feedback:
        if feedback_path.exists():
            success_feedback.append(load_json(feedback_path))
    quality_summary = load_json(args.quality_summary) if args.quality_summary and args.quality_summary.exists() else None
    system = (
        "You induce general prompt experience cards for PR knowledge distillation. "
        "Do not output chain-of-thought. Do not copy sample-specific repository names, file paths, function names, issue ids, or APIs."
    )
    user = f"""
We start from a minimal seed prompt and want to learn a few general experience cards from epoch-0 feedback.

Seed prompt:
{seed_prompt}

Epoch metrics:
{json.dumps({
    "distill": {
        "rows": distill.get("rows"),
        "status_counts": distill.get("status_counts"),
        "zero_item_rows": distill.get("zero_item_rows"),
        "knowledge_items_total": (distill.get("kb_summary") or {}).get("knowledge_items_total"),
    },
    "train": train.get("overall"),
    "dev": dev.get("overall"),
}, ensure_ascii=False, indent=2)}

Existing accepted experience cards; do not duplicate them:
{json.dumps(existing_cards, ensure_ascii=False, indent=2)}

Knowledge quality proxy summary:
{json.dumps(quality_summary, ensure_ascii=False, indent=2) if quality_summary else "null"}

High-performing examples showing what the retriever can use:
{json.dumps(success_feedback, ensure_ascii=False, indent=2) if success_feedback else "[]"}

Focus note for this round:
{args.focus_note or "none"}

Low-performing examples, identifiers may appear only as diagnostic evidence and must not be copied:
{json.dumps({"train": train_fail, "dev": dev_fail}, ensure_ascii=False, indent=2)}

Generate at most {args.max_cards} new general experience cards. Each card should summarize one reusable tactic that the extraction model can apply, such as PR noise filtering, preserving behavior-relevant signals, aligning PR text to patch evidence, choosing code scope, or writing retrieval keywords. Use train success/failure patterns to propose tactics, but prefer cards that are likely to improve dev transfer. Prefer cards that address the focus note and observed weak metrics. Do not repeat existing accepted cards.

Return only JSON:
[
  {{"title": "...", "instruction": "one concise general instruction", "rationale": "why this should improve retrieval"}}
]
""".strip()
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    started = datetime.now(timezone.utc).isoformat()
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=args.temperature,
    )
    content = strip_fence(response.choices[0].message.content or "")
    try:
        cards = normalize_cards(json.loads(content), args.max_cards)
    except json.JSONDecodeError:
        cards = []
    if not cards:
        raise SystemExit(f"No valid cards generated. Raw response:\n{content}")
    args.output_cards.parent.mkdir(parents=True, exist_ok=True)
    args.output_cards.write_text(json.dumps(cards, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_metadata.write_text(
        json.dumps(
            {
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "base_url": args.base_url,
                "run_dir": str(args.run_dir),
                "output_cards": str(args.output_cards),
                "usage": response.usage.model_dump() if response.usage else None,
                "raw_response": content,
                "existing_cards": [str(path) for path in args.existing_cards],
                "success_feedback": [str(path) for path in args.success_feedback],
                "quality_summary": str(args.quality_summary) if args.quality_summary else None,
                "focus_note": args.focus_note,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(cards, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
