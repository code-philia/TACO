#!/usr/bin/env python3
"""Summarize structural quality of distilled knowledge items."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize distilled knowledge quality proxies.")
    parser.add_argument("--distilled-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.distilled_results)
    total_items = 0
    symbol_items = 0
    file_items = 0
    keyword_counts = []
    rationale_lengths = []
    repo_counts: Dict[str, Counter] = defaultdict(Counter)
    generic_keywords = Counter()
    for row in rows:
        repo = str(row.get("repo") or "unknown")
        items = row.get("items") or []
        repo_counts[repo]["rows"] += 1
        if not items:
            repo_counts[repo]["zero_rows"] += 1
        for item in items:
            total_items += 1
            repo_counts[repo]["items"] += 1
            function_name = item.get("function") or ""
            if function_name == "FILE_SUMMARY":
                file_items += 1
                repo_counts[repo]["file_items"] += 1
            else:
                symbol_items += 1
                repo_counts[repo]["symbol_items"] += 1
            keywords = item.get("keywords") or []
            keyword_counts.append(len(keywords))
            rationale = str(item.get("rationale") or "")
            rationale_lengths.append(len(rationale.split()))
            for keyword in keywords:
                k = str(keyword).strip().lower()
                if k in {"bug", "fix", "update", "change", "test", "refactor", "cleanup", "support", "feature"}:
                    generic_keywords[k] += 1
    rows_count = len(rows)
    output = {
        "rows": rows_count,
        "total_items": total_items,
        "items_per_row": total_items / rows_count if rows_count else 0,
        "symbol_items": symbol_items,
        "file_items": file_items,
        "symbol_item_rate": symbol_items / total_items if total_items else 0,
        "file_item_rate": file_items / total_items if total_items else 0,
        "avg_keywords_per_item": sum(keyword_counts) / len(keyword_counts) if keyword_counts else 0,
        "avg_rationale_words": sum(rationale_lengths) / len(rationale_lengths) if rationale_lengths else 0,
        "generic_keyword_counts": dict(generic_keywords.most_common(20)),
        "by_repo": {repo: dict(counter) for repo, counter in sorted(repo_counts.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
