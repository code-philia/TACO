#!/usr/bin/env python3
"""Generate experience cards using train details and dev aggregate metrics only."""

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
    parser = argparse.ArgumentParser(description="Generate safe prompt experience cards.")
    parser.add_argument("--seed-prompt", type=Path, required=True)
    parser.add_argument("--train-run-dir", type=Path, required=True)
    parser.add_argument("--dev-summary", type=Path, required=True)
    parser.add_argument("--output-cards", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--existing-cards", type=Path, action="append", default=[])
    parser.add_argument("--train-success-feedback", type=Path, action="append", default=[])
    parser.add_argument("--train-failure-feedback", type=Path, action="append", default=[])
    parser.add_argument("--train-quality-summary", type=Path)
    parser.add_argument("--focus-note", default="")
    parser.add_argument("--max-cards", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.15)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def normalize(raw: Any, max_cards: int) -> list[dict[str, str]]:
    if isinstance(raw, dict):
        raw = raw.get("cards") or raw.get("experience_cards") or []
    cards = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        instruction = str(item.get("instruction") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        expected_effect = str(item.get("expected_effect") or "").strip()
        if title and instruction:
            card = {"title": title, "instruction": instruction, "rationale": rationale}
            if expected_effect:
                card["expected_effect"] = expected_effect
            cards.append(card)
    return cards[:max_cards]


def load_many(paths: list[Path]) -> list[Any]:
    return [load_json(path) for path in paths if path.exists()]


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in ${args.api_key_env}")
    seed_prompt = args.seed_prompt.read_text(encoding="utf-8")
    train_distill = load_json(args.train_run_dir / "distill_summary.json")
    train_summary = load_json(args.train_run_dir / "train_score_summary.json")
    dev_summary = load_json(args.dev_summary)
    existing_cards = []
    for cards_path in args.existing_cards:
        if cards_path.exists():
            existing_cards.extend(load_json(cards_path))
    train_success = load_many(args.train_success_feedback)
    train_failure = load_many(args.train_failure_feedback)
    train_quality = load_json(args.train_quality_summary) if args.train_quality_summary and args.train_quality_summary.exists() else None
    public_metrics = {
        "train_fit": {
            "distill": {
                "rows": train_distill.get("rows"),
                "status_counts": train_distill.get("status_counts"),
                "zero_item_rows": train_distill.get("zero_item_rows"),
                "knowledge_items_total": (train_distill.get("kb_summary") or {}).get("knowledge_items_total"),
            },
            "retrieval": train_summary.get("overall"),
            "quality": train_quality,
        },
        "dev_gate_aggregate_only": {
            "retrieval": dev_summary.get("overall"),
            "by_repo": dev_summary.get("by_repo"),
        },
    }
    system = (
        "You induce general prompt experience cards for PR knowledge distillation. "
        "Use detailed training feedback to propose general tactics, and use development metrics only as aggregate validation signal. "
        "Do not output chain-of-thought. Do not copy sample-specific repository names, file paths, function names, issue ids, or APIs."
    )
    user = f"""
We train a prompt from a minimal seed prompt. Detailed examples are from TRAIN only. DEV information below is aggregate metrics only and must be used only to judge generalization risk, not to infer sample-specific rules.

Seed prompt:
{seed_prompt}

Existing accepted experience cards; do not duplicate them:
{json.dumps(existing_cards, ensure_ascii=False, indent=2)}

Train fitting and dev aggregate metrics:
{json.dumps(public_metrics, ensure_ascii=False, indent=2)}

TRAIN high-performing examples:
{json.dumps(train_success, ensure_ascii=False, indent=2)}

TRAIN low-performing examples:
{json.dumps(train_failure, ensure_ascii=False, indent=2)}

Focus note:
{args.focus_note or "none"}

Generate at most {args.max_cards} new general card. It must be broad enough to transfer beyond train. Prefer one small, testable tactic. If train and dev signals conflict, avoid narrow train-specific rules. Do not include dev sample details because none are provided.

Return only JSON:
[
  {{"title": "...", "instruction": "one concise general instruction", "rationale": "why this should improve retrieval", "expected_effect": "which metric should improve"}}
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
        cards = normalize(json.loads(content), args.max_cards)
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
                "seed_prompt": str(args.seed_prompt),
                "train_run_dir": str(args.train_run_dir),
                "dev_summary": str(args.dev_summary),
                "existing_cards": [str(path) for path in args.existing_cards],
                "train_success_feedback": [str(path) for path in args.train_success_feedback],
                "train_failure_feedback": [str(path) for path in args.train_failure_feedback],
                "train_quality_summary": str(args.train_quality_summary) if args.train_quality_summary else None,
                "focus_note": args.focus_note,
                "usage": response.usage.model_dump() if response.usage else None,
                "raw_response": content,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(cards, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
