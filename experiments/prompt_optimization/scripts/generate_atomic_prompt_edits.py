#!/usr/bin/env python3
"""Generate small atomic prompt edits from train feedback and dev aggregate metrics."""

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
    parser = argparse.ArgumentParser(description="Generate atomic prompt edits.")
    parser.add_argument("--champion-prompt", type=Path, required=True)
    parser.add_argument("--train-run-dir", type=Path, required=True)
    parser.add_argument("--dev-summary", type=Path, required=True)
    parser.add_argument("--output-edits", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--train-success-feedback", type=Path, action="append", default=[])
    parser.add_argument("--train-failure-feedback", type=Path, action="append", default=[])
    parser.add_argument("--train-quality-summary", type=Path)
    parser.add_argument("--max-edits", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--focus-note", default="")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_many(paths: list[Path]) -> list[Any]:
    return [load_json(path) for path in paths if path.exists()]


def strip_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def load_json_lenient(content: str) -> Any:
    cleaned = re.sub(r",\s*([}\]])", r"\1", content.strip())
    return json.loads(cleaned)


def normalize(raw: Any, max_edits: int) -> list[dict[str, str]]:
    if isinstance(raw, dict):
        raw = raw.get("edits") or raw.get("prompt_edits") or []
    edits = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        edit_type = str(item.get("edit_type") or "").strip()
        target = str(item.get("target") or "").strip()
        text = str(item.get("text") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        expected_effect = str(item.get("expected_effect") or "").strip()
        risk = str(item.get("risk") or "").strip()
        if edit_type in {"append_bullet", "replace_text", "remove_text", "replace_process"} and text:
            edits.append(
                {
                    "edit_type": edit_type,
                    "target": target,
                    "text": text,
                    "rationale": rationale,
                    "expected_effect": expected_effect,
                    "risk": risk,
                }
            )
    return edits[:max_edits]


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in ${args.api_key_env}")
    champion_prompt = args.champion_prompt.read_text(encoding="utf-8")
    train_distill = load_json(args.train_run_dir / "distill_summary.json")
    train_summary = load_json(args.train_run_dir / "train_score_summary.json")
    dev_summary = load_json(args.dev_summary)
    train_quality = load_json(args.train_quality_summary) if args.train_quality_summary and args.train_quality_summary.exists() else None
    public_metrics = {
        "train": {
            "distill": {
                "rows": train_distill.get("rows"),
                "status_counts": train_distill.get("status_counts"),
                "zero_item_rows": train_distill.get("zero_item_rows"),
                "knowledge_items_total": (train_distill.get("kb_summary") or {}).get("knowledge_items_total"),
            },
            "retrieval": train_summary.get("overall"),
            "quality": train_quality,
        },
        "dev_aggregate_only": {
            "retrieval": dev_summary.get("overall"),
            "by_repo": dev_summary.get("by_repo"),
        },
    }
    train_success = load_many(args.train_success_feedback)
    train_failure = load_many(args.train_failure_feedback)
    system = (
        "You propose small atomic edits to a PR knowledge-distillation prompt. "
        "Use detailed TRAIN feedback only for examples. DEV is aggregate metrics only. "
        "Do not output chain-of-thought. Do not copy repo names, file paths, function names, APIs, issue ids, or sample ids into edits."
    )
    user = f"""
Current champion prompt:
{champion_prompt}

Metrics:
{json.dumps(public_metrics, ensure_ascii=False, indent=2)}

TRAIN high-performing examples:
{json.dumps(train_success, ensure_ascii=False, indent=2)}

TRAIN low-performing examples:
{json.dumps(train_failure, ensure_ascii=False, indent=2)}

Focus note:
{args.focus_note or "none"}

Generate up to {args.max_edits} independent atomic edits. Each edit will be tested separately against the champion; do not make edits that depend on each other.
Allowed edit_type values:
- append_bullet: add one concise bullet under Learned general experience.
- replace_process: replace the existing compact numbered internal reasoning process block. Use this when the focus note asks for a CoT/process-style prompt; revise the current process in place instead of appending extra advice. The process must be followed silently and the final output must remain JSON-only.
- replace_text: replace one short existing sentence or bullet with a better one.
- remove_text: remove one short existing sentence or bullet that likely hurts transfer.

Constraints:
- Each edit must be tiny and testable.
- Prefer edits that can improve dev aggregate metrics without increasing parse_empty.
- Do not add sample-specific terms.
- Avoid broad rewrites or adding a full new card. When using replace_process, output the complete replacement process block content, not an additional note to append after the current prompt.

Return only JSON:
[
  {{"edit_type": "replace_process", "target": "Internal reasoning process", "text": "1. ...\n2. ...\n3. ...", "rationale": "...", "expected_effect": "...", "risk": "..."}}
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
    parse_error = None
    try:
        edits = normalize(load_json_lenient(content), args.max_edits)
    except json.JSONDecodeError as exc:
        parse_error = str(exc)
        edits = []
    args.output_edits.parent.mkdir(parents=True, exist_ok=True)
    args.output_edits.write_text(json.dumps(edits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_metadata.write_text(
        json.dumps(
            {
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "base_url": args.base_url,
                "champion_prompt": str(args.champion_prompt),
                "train_run_dir": str(args.train_run_dir),
                "dev_summary": str(args.dev_summary),
                "usage": response.usage.model_dump() if response.usage else None,
                "raw_response": content,
                "parse_error": parse_error,
                "valid_edits": len(edits),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(edits, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
