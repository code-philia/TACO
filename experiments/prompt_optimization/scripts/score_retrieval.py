#!/usr/bin/env python3
"""Score retrieval results against historical PR file/module/function gold."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import load_jsonl, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score retrieval JSONL against gold labels.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-details", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--function-score", type=float, default=1.0)
    parser.add_argument("--module-score", type=float, default=0.7)
    parser.add_argument("--file-score", type=float, default=0.4)
    return parser.parse_args()


def normalize_function(file_path: str, element: str) -> Optional[str]:
    if not file_path or not element or element == "Entire File":
        return None
    clean = element.split("(", 1)[0].strip()
    if not clean:
        return None
    return f"{file_path}::{clean}"


def normalize_module(file_path: str, element: str) -> Optional[str]:
    if not file_path:
        return None
    if not element or element == "Entire File":
        return f"{file_path}::Global"
    clean = element.split("(", 1)[0].strip()
    if "." in clean:
        module_name = clean.rsplit(".", 1)[0]
        return f"{file_path}::{module_name}"
    return f"{file_path}::Global"


def hit_level(candidate: Dict[str, Any], gold: Dict[str, Any]) -> str:
    file_path = candidate.get("file", "")
    element = candidate.get("element", "")
    gold_files = set(gold.get("files") or [])
    gold_modules = set(gold.get("modules") or [])
    gold_functions = set(gold.get("functions") or [])

    predicted_function = normalize_function(file_path, element)
    predicted_module = normalize_module(file_path, element)
    if predicted_function and predicted_function in gold_functions:
        return "function"
    if predicted_module and predicted_module in gold_modules:
        return "module"
    if file_path in gold_files:
        return "file"
    return "miss"


def level_score(level: str, args: argparse.Namespace) -> float:
    return {
        "function": args.function_score,
        "module": args.module_score,
        "file": args.file_score,
        "miss": 0.0,
    }[level]


def score_one(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    retrieval = (row.get("retrieval") or [])[: args.top_k]
    best_level = "miss"
    best_score = 0.0
    first_hit_rank: Optional[int] = None
    first_hit_level = "miss"
    level_rank = {"function": None, "module": None, "file": None}

    for rank, candidate in enumerate(retrieval, 1):
        level = hit_level(candidate, row.get("gold", {}))
        score = level_score(level, args)
        if level != "miss" and first_hit_rank is None:
            first_hit_rank = rank
            first_hit_level = level
        if level in level_rank and level_rank[level] is None:
            level_rank[level] = rank
        if score > best_score:
            best_score = score
            best_level = level

    mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0
    return {
        "repo": row.get("repo"),
        "instance_id": row.get("instance_id"),
        "split": row.get("split"),
        "status": row.get("status", "ok"),
        "best_level": best_level,
        "best_score": best_score,
        "first_hit_rank": first_hit_rank,
        "first_hit_level": first_hit_level,
        "mrr": mrr,
        "hit_file": level_rank["file"] is not None or level_rank["module"] is not None or level_rank["function"] is not None,
        "hit_module": level_rank["module"] is not None or level_rank["function"] is not None,
        "hit_function": level_rank["function"] is not None,
        "rank_file": level_rank["file"],
        "rank_module": level_rank["module"],
        "rank_function": level_rank["function"],
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in details if row.get("status") == "ok"]
    by_repo: Dict[str, List[Dict[str, Any]]] = {}
    for row in valid:
        by_repo.setdefault(row.get("repo") or "unknown", []).append(row)

    def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "count": len(rows),
            "avg_best_score": mean([row["best_score"] for row in rows]),
            "mrr": mean([row["mrr"] for row in rows]),
            "file_hit_rate": mean([1.0 if row["hit_file"] else 0.0 for row in rows]),
            "module_hit_rate": mean([1.0 if row["hit_module"] else 0.0 for row in rows]),
            "function_hit_rate": mean([1.0 if row["hit_function"] else 0.0 for row in rows]),
        }

    return {
        "total_rows": len(details),
        "valid_rows": len(valid),
        "status_counts": {status: sum(1 for row in details if row.get("status") == status) for status in sorted({row.get("status") for row in details})},
        "overall": aggregate(valid),
        "by_repo": {repo: aggregate(rows) for repo, rows in sorted(by_repo.items())},
    }


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.results)
    details = [score_one(row, args) for row in rows]
    summary = summarize(details)
    write_json(args.output_summary, summary)
    if args.output_details:
        write_jsonl(args.output_details, details)
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
