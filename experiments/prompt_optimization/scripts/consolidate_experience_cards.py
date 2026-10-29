#!/usr/bin/env python3
"""Consolidate multiple experience-card files into a compact non-duplicative set."""

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
    parser = argparse.ArgumentParser(description="Consolidate experience cards.")
    parser.add_argument("--cards", type=Path, action="append", required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-cards", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-cards", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.15)
    return parser.parse_args()


def strip_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def normalize(raw: Any, max_cards: int) -> list[dict[str, str]]:
    if isinstance(raw, dict):
        raw = raw.get("cards") or raw.get("experience_cards") or []
    output = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        instruction = str(item.get("instruction") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        if title and instruction:
            output.append({"title": title, "instruction": instruction, "rationale": rationale})
    return output[:max_cards]


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in ${args.api_key_env}")
    cards = []
    for path in args.cards:
        cards.extend(json.loads(path.read_text(encoding="utf-8")))
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    system = "You consolidate general PR knowledge-distillation experience cards. Do not output chain-of-thought."
    user = f"""
We have several experience cards learned across epochs. Later cards often duplicate earlier symbol-scope advice and can reduce dev score when accumulated.

Metrics by epoch:
{json.dumps(metrics, ensure_ascii=False, indent=2)}

Candidate cards:
{json.dumps(cards, ensure_ascii=False, indent=2)}

Consolidate them into at most {args.max_cards} non-duplicative cards.
Keep cards that explain general tactics likely to improve dev retrieval from a minimal seed prompt.
Balance file/module coverage with function-level scope.
Avoid overly narrow or repetitive rules.
Do not include repository names, file paths, function names, APIs, sample ids, or concrete issue ids.

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
        compact = normalize(json.loads(content), args.max_cards)
    except json.JSONDecodeError:
        compact = []
    if not compact:
        raise SystemExit(f"No valid cards generated. Raw response:\n{content}")
    args.output_cards.parent.mkdir(parents=True, exist_ok=True)
    args.output_cards.write_text(json.dumps(compact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_metadata.write_text(
        json.dumps(
            {
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "base_url": args.base_url,
                "input_cards": [str(path) for path in args.cards],
                "metrics_json": str(args.metrics_json),
                "usage": response.usage.model_dump() if response.usage else None,
                "raw_response": content,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
