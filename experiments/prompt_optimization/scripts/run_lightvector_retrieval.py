#!/usr/bin/env python3
"""Run existing LightVectorStore retrieval for historical PR query manifests."""

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import load_jsonl, write_jsonl  # noqa: E402
from rek.parity_core.light_vector import LightVectorStore  # noqa: E402
from rek.parity_core.graph_db import GraphDatabase  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightVectorStore retrieval over a query manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-repos", type=Path, default=Path(os.environ.get("REK_BASE_REPOS", "YOUR_REPO_CHECKOUTS_DIR_HERE")))
    parser.add_argument("--repo-root", type=Path, help="optional experiment-local root containing kb_repos/<repo>/<output-dir-name>")
    parser.add_argument("--output-dir-name", default="repo2docAgent-output-gpt-5-4-mini")
    parser.add_argument("--kb-name", default="knowledge_graph_two_index_epoch_1.json")
    parser.add_argument("--elements-name", default="program_elements.jsonl")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repo", help="optional single-repo filter")
    return parser.parse_args()


def kb_paths(base_repos: Path, repo: str, output_dir_name: str, kb_name: str, elements_name: str) -> tuple[Path, Path]:
    repo_output = base_repos / repo / output_dir_name
    return repo_output / kb_name, repo_output / elements_name


def load_store(cache: Dict[str, LightVectorStore], args: argparse.Namespace, repo: str) -> Optional[LightVectorStore]:
    if repo in cache:
        return cache[repo]
    base_repos = args.repo_root / "kb_repos" if args.repo_root else args.base_repos
    kb_path, elements_path = kb_paths(base_repos, repo, args.output_dir_name, args.kb_name, args.elements_name)
    if not kb_path.exists():
        return None
    graph_db = GraphDatabase(str(kb_path), str(elements_path))
    cache[repo] = LightVectorStore(graph_db)
    return cache[repo]


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.manifest)
    if args.repo:
        rows = [row for row in rows if row.get("repo") == args.repo]

    cache: Dict[str, LightVectorStore] = {}
    outputs: List[Dict[str, Any]] = []
    for row in rows:
        repo = row.get("repo")
        query = row.get("query", "")
        output = {
            "repo": repo,
            "instance_id": row.get("instance_id"),
            "split": row.get("split"),
            "query": query,
            "gold": row.get("gold", {}),
            "retrieval": [],
            "status": "ok",
        }
        store = load_store(cache, args, repo)
        if store is None:
            output["status"] = "missing_kb_or_elements"
            outputs.append(output)
            continue
        try:
            output["retrieval"] = store.search(query, top_k=args.top_k)
        except Exception as exc:  # noqa: BLE001
            output["status"] = "retrieval_error"
            output["error"] = str(exc)
        outputs.append(output)

    write_jsonl(args.output, outputs)
    print(f"Wrote retrieval results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
