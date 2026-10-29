#!/usr/bin/env python3
"""Build detailed low-performing sample feedback for prompt optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join low-score rows with query, gold, retrieval, and distillation details.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--score-details", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distilled-results", type=Path)
    parser.add_argument("--split-name", default="split")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--max-query-chars", type=int, default=900)
    parser.add_argument("--max-gold-items", type=int, default=12)
    parser.add_argument("--max-retrieval-items", type=int, default=6)
    parser.add_argument("--max-distilled-items", type=int, default=8)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(row: Dict[str, Any]) -> str:
    return str(row.get("instance_id") or "")


def compact_gold(gold: Dict[str, Any], limit: int) -> Dict[str, Any]:
    return {
        "files": (gold.get("files") or [])[:limit],
        "modules": (gold.get("modules") or [])[:limit],
        "functions": (gold.get("functions") or [])[:limit],
        "counts": {
            "files": len(gold.get("files") or []),
            "modules": len(gold.get("modules") or []),
            "functions": len(gold.get("functions") or []),
        },
    }


def compact_retrieval(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    compact = []
    for item in items[:limit]:
        compact.append(
            {
                "file": item.get("file"),
                "element": item.get("element"),
                "score": item.get("score"),
                "preview": item.get("preview"),
            }
        )
    return compact


def compact_distilled(row: Optional[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    if not row:
        return {"status": "not_available", "items": []}
    return {
        "status": row.get("status"),
        "item_count": len(row.get("items") or []),
        "items": [
            {
                "file": item.get("file"),
                "function": item.get("function"),
                "keywords": item.get("keywords"),
                "rationale": item.get("rationale"),
            }
            for item in (row.get("items") or [])[:limit]
        ],
    }


def diagnose(score_row: Dict[str, Any], distilled: Dict[str, Any], retrieval: List[Dict[str, Any]]) -> List[str]:
    hints = []
    if distilled.get("status") == "parse_empty" or distilled.get("item_count") == 0:
        hints.append("distillation produced no usable knowledge")
    if score_row.get("best_level") == "miss":
        hints.append("retrieval missed all gold locations")
    elif score_row.get("best_level") == "file":
        hints.append("retrieval only reached file-level granularity")
    if not score_row.get("hit_function"):
        hints.append("function-level signal was missing or ranked too low")
    if retrieval:
        top_preview = " ".join(str(item.get("preview") or "") for item in retrieval[:3]).lower()
        if len(top_preview.split()) < 25:
            hints.append("top retrieved knowledge previews are sparse")
    return hints


def main() -> int:
    args = parse_args()
    manifest = {key(row): row for row in load_jsonl(args.manifest)}
    retrieval = {key(row): row for row in load_jsonl(args.retrieval)}
    distilled = {key(row): row for row in load_jsonl(args.distilled_results)} if args.distilled_results else {}
    details = load_jsonl(args.score_details)
    details.sort(key=lambda row: (row.get("best_score", 0), row.get("mrr", 0), row.get("first_hit_rank") or 999))

    examples = []
    for score_row in details[: args.max_examples]:
        instance_id = key(score_row)
        manifest_row = manifest.get(instance_id, {})
        retrieval_row = retrieval.get(instance_id, {})
        distilled_row = distilled.get(instance_id)
        compact_distilled_row = compact_distilled(distilled_row, args.max_distilled_items)
        compact_retrieval_rows = compact_retrieval(retrieval_row.get("retrieval") or [], args.max_retrieval_items)
        examples.append(
            {
                "sample_ref": f"{args.split_name}_{len(examples) + 1}",
                "score": {
                    "best_level": score_row.get("best_level"),
                    "best_score": score_row.get("best_score"),
                    "mrr": score_row.get("mrr"),
                    "first_hit_rank": score_row.get("first_hit_rank"),
                    "first_hit_level": score_row.get("first_hit_level"),
                    "hit_file": score_row.get("hit_file"),
                    "hit_module": score_row.get("hit_module"),
                    "hit_function": score_row.get("hit_function"),
                },
                "query_excerpt": str(manifest_row.get("query") or retrieval_row.get("query") or "")[: args.max_query_chars],
                "gold": compact_gold(manifest_row.get("gold") or retrieval_row.get("gold") or {}, args.max_gold_items),
                "distilled_knowledge": compact_distilled_row,
                "top_retrieval": compact_retrieval_rows,
                "diagnostic_hints": diagnose(score_row, compact_distilled_row, compact_retrieval_rows),
            }
        )

    output = {
        "split": args.split_name,
        "selection_policy": "lowest best_score, then lowest mrr",
        "anti_leakage_instruction": "These examples may contain sample-specific identifiers for diagnosis only. The optimized prompt must not copy repository names, file paths, function names, issue ids, PR ids, or concrete APIs from them.",
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote failure feedback to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
