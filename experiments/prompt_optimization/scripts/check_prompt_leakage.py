#!/usr/bin/env python3
"""Reject prompt candidates that contain sample-specific leakage terms."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import DEFAULT_REPOS, load_jsonl, write_json  # noqa: E402

GENERIC_FUNCTION_TERMS = {
    "add",
    "align",
    "build",
    "check",
    "clean",
    "close",
    "copy",
    "create",
    "delete",
    "density",
    "diff",
    "dump",
    "end",
    "find",
    "format",
    "get",
    "handle",
    "init",
    "load",
    "make",
    "main",
    "map",
    "open",
    "parse",
    "patch",
    "pop",
    "post",
    "print",
    "process",
    "put",
    "read",
    "remove",
    "render",
    "request",
    "requests",
    "reset",
    "run",
    "save",
    "send",
    "set",
    "setup",
    "show",
    "start",
    "stop",
    "test",
    "update",
    "validate",
    "write",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check candidate prompt for repo/sample-specific leakage.")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    parser.add_argument("--allow-generic-repo-names", action="store_true")
    return parser.parse_args()


def collect_forbidden_terms(manifests: List[Path], repos: List[str], allow_generic_repo_names: bool) -> Dict[str, Set[str]]:
    terms: Dict[str, Set[str]] = {
        "repo": set(),
        "file_path": set(),
        "function": set(),
        "instance_id": set(),
        "issue_or_pr_id": set(),
    }
    if not allow_generic_repo_names:
        terms["repo"].update(repos)
    for manifest in manifests:
        if not manifest.exists():
            continue
        for row in load_jsonl(manifest):
            if row.get("instance_id"):
                terms["instance_id"].add(str(row["instance_id"]))
            if row.get("pr_number"):
                terms["issue_or_pr_id"].add(str(row["pr_number"]))
            for file_path in (row.get("gold", {}).get("files") or []):
                terms["file_path"].add(str(file_path))
            for function in (row.get("gold", {}).get("functions") or []):
                function_name = str(function).split("::", 1)[-1]
                leaf_name = function_name.rsplit(".", 1)[-1].lower()
                if leaf_name not in GENERIC_FUNCTION_TERMS:
                    terms["function"].add(function_name)
    return terms


def find_hits(prompt_text: str, terms: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    hits: Dict[str, List[str]] = {}
    lowered = prompt_text.lower()
    for category, values in terms.items():
        category_hits = []
        for value in values:
            if not value or len(value) < 3:
                continue
            if category == "repo" and value.lower() == "requests":
                repo_lowered = re.sub(r"\bpull\s+requests\b", "", lowered)
                pattern = r"(?<![\w/.-])" + re.escape(value.lower()) + r"(?![\w/.-])"
                matched = re.search(pattern, repo_lowered) is not None
            elif category in {"repo", "issue_or_pr_id", "function"}:
                pattern = r"(?<![\w/.-])" + re.escape(value.lower()) + r"(?![\w/.-])"
                matched = re.search(pattern, lowered) is not None
            else:
                matched = value.lower() in lowered
            if matched:
                category_hits.append(value)
        if category_hits:
            hits[category] = sorted(category_hits)[:100]
    return hits


def main() -> int:
    args = parse_args()
    prompt_text = args.prompt.read_text(encoding="utf-8", errors="ignore")
    forbidden = collect_forbidden_terms(args.manifest, args.repos, args.allow_generic_repo_names)
    hits = find_hits(prompt_text, forbidden)
    report = {
        "prompt": str(args.prompt),
        "passed": not bool(hits),
        "hits": hits,
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
