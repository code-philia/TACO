#!/usr/bin/env python3
"""Build compact high-performing sample feedback for prompt optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join high-score rows with query, gold, retrieval, and distillation details.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--score-details", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distilled-results", type=Path)
    parser.add_argument("--split-name", default="split")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--max-query-chars", type=int, default=700)
    parser.add_argument("--max-retrieval-items", type=int, default=4)
    parser.add_argument("--max-distilled-items", type=int, default=5)
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


def compact_distilled(row: Optional[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    if not row:
        return {"status": "not_available", "items": []}
    return {
        "status": row.get("status"),
        "item_count": len(row.get("items") or []),
        "items": [
            {
                "function_scope_type": "file" if item.get("function") == "FILE_SUMMARY" else "symbol",
                "keywords": item.get("keywords"),
                "rationale": item.get("rationale"),
            }
            for item in (row.get("items") or [])[:limit]
        ],
    }


def compact_retrieval(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "element_scope_type": "file" if item.get("element") == "Entire File" else "symbol",
            "score": item.get("score"),
            "preview": item.get("preview"),
        }
        for index, item in enumerate(items[:limit])
    ]


def main() -> int:
    args = parse_args()
    manifest = {key(row): row for row in load_jsonl(args.manifest)}
    retrieval = {key(row): row for row in load_jsonl(args.retrieval)}
    distilled = {key(row): row for row in load_jsonl(args.distilled_results)} if args.distilled_results else {}
    details = load_jsonl(args.score_details)
    details.sort(key=lambda row: (row.get("best_score", 0), row.get("mrr", 0)), reverse=True)

    examples = []
    for score_row in details:
        if score_row.get("best_level") == "miss":
            continue
        instance_id = key(score_row)
        manifest_row = manifest.get(instance_id, {})
        retrieval_row = retrieval.get(instance_id, {})
        distilled_row = distilled.get(instance_id)
        examples.append(
            {
                "sample_ref": f"{args.split_name}_success_{len(examples) + 1}",
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
                "gold_counts": {
                    "files": len((manifest_row.get("gold") or retrieval_row.get("gold") or {}).get("files") or []),
                    "modules": len((manifest_row.get("gold") or retrieval_row.get("gold") or {}).get("modules") or []),
                    "functions": len((manifest_row.get("gold") or retrieval_row.get("gold") or {}).get("functions") or []),
                },
                "distilled_knowledge": compact_distilled(distilled_row, args.max_distilled_items),
                "top_retrieval": compact_retrieval(retrieval_row.get("retrieval") or [], args.max_retrieval_items),
                "success_signal": "knowledge phrasing and keywords were sufficient for retrieval at this granularity",
            }
        )
        if len(examples) >= args.max_examples:
            break

    output = {
        "split": args.split_name,
        "selection_policy": "highest best_score, then highest mrr; identifiers are diagnostic only",
        "anti_leakage_instruction": "Do not copy repository names, file paths, function names, APIs, issue ids, PR ids, or sample ids from examples into optimized prompts.",
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote success feedback to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
