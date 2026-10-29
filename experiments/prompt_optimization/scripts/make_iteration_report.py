#!/usr/bin/env python3
"""Create a compact Markdown report for one prompt-optimization iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an iteration report from retrieval scoring outputs.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration-name", default="iteration")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path or not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    details = load_jsonl(args.details) if args.details else []
    misses = [row for row in details if row.get("best_level") == "miss"][:20]
    file_only = [row for row in details if row.get("best_level") == "file"][:20]

    lines = [f"# Prompt Optimization Report: {args.iteration_name}", ""]
    if args.prompt:
        lines.append(f"- Prompt: `{args.prompt}`")
    overall = summary.get("overall", {})
    lines.extend([
        f"- Valid rows: `{summary.get('valid_rows', 0)}` / `{summary.get('total_rows', 0)}`",
        f"- Avg best score: `{overall.get('avg_best_score', 0):.4f}`",
        f"- MRR: `{overall.get('mrr', 0):.4f}`",
        f"- File hit rate: `{overall.get('file_hit_rate', 0):.4f}`",
        f"- Module hit rate: `{overall.get('module_hit_rate', 0):.4f}`",
        f"- Function hit rate: `{overall.get('function_hit_rate', 0):.4f}`",
        "",
        "## By Repo",
        "",
        "| Repo | Count | Avg Score | MRR | File Hit | Module Hit | Function Hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for repo, metrics in summary.get("by_repo", {}).items():
        lines.append(
            f"| `{repo}` | {metrics.get('count', 0)} | {metrics.get('avg_best_score', 0):.4f} | "
            f"{metrics.get('mrr', 0):.4f} | {metrics.get('file_hit_rate', 0):.4f} | "
            f"{metrics.get('module_hit_rate', 0):.4f} | {metrics.get('function_hit_rate', 0):.4f} |"
        )
    lines.extend(["", "## Error Samples", "", "### Misses", ""])
    for row in misses:
        lines.append(f"- `{row.get('repo')}` `{row.get('instance_id')}`: no file/module/function hit")
    lines.extend(["", "### File-Only Hits", ""])
    for row in file_only:
        lines.append(f"- `{row.get('repo')}` `{row.get('instance_id')}`: file-level only")
    lines.extend([
        "",
        "## Prompt Update Guidance",
        "",
        "Use only aggregate failure patterns from this report. Do not copy repo names, file paths, function names, issue ids, or sample-specific APIs into the next prompt.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
