#!/usr/bin/env python3
"""Build historical-PR train/dev/held-out manifests with patch-derived gold."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_REPOS,
    extract_patch_gold,
    load_jsonl,
    production_files_from_sample,
    query_text,
    repo_path_from_base,
    repo_train_data_path,
    sample_sort_key,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build historical PR prompt-optimization splits.")
    parser.add_argument("--train-data-dir", type=Path, default=Path(os.environ.get("REK_TRAIN_DATA_DIR", "YOUR_TRAIN_DATA_DIR_HERE")))
    parser.add_argument("--base-repos", type=Path, default=Path(os.environ.get("REK_BASE_REPOS", "YOUR_REPO_CHECKOUTS_DIR_HERE")))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/prompt_optimization/data_splits"))
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    parser.add_argument("--heldout-repos", nargs="+", default=[])
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--max-per-repo", type=int, default=0, help="0 means no cap after filtering")
    parser.add_argument("--min-history", type=int, default=50, help="minimum earlier PRs required for a target query")
    return parser.parse_args()


def build_record(repo: str, sample: Dict[str, Any], gold) -> Dict[str, Any]:
    return {
        "repo": repo,
        "instance_id": sample.get("instance_id"),
        "pr_number": sample.get("pr_number"),
        "title": sample.get("title", ""),
        "query": query_text(sample),
        "patch": sample.get("patch", ""),
        "base_commit": sample.get("base_commit", "HEAD"),
        "head_commit": sample.get("head_commit"),
        "url": sample.get("url"),
        "merged_at": sample.get("merged_at"),
        "closed_at": sample.get("closed_at"),
        "updated_at": sample.get("updated_at"),
        "gold": {
            "files": gold.files,
            "modules": gold.modules,
            "functions": gold.functions,
        },
        "source": {
            "problem_statement_chars": len(sample.get("problem_statement") or ""),
            "patch_chars": len(sample.get("patch") or ""),
            "files_preview": sample.get("files_preview", []),
        },
    }


def main() -> int:
    args = parse_args()
    heldout = set(args.heldout_repos)
    optimization_repos = [repo for repo in args.repos if repo not in heldout]
    if len(args.repos) < 3:
        raise SystemExit("Need at least three repos to build optimization and held-out pools.")
    if not heldout:
        heldout = set(args.repos[-2:])
        optimization_repos = args.repos[:-2]

    split_summary: Dict[str, Any] = {
        "optimization_pool_repos": optimization_repos,
        "heldout_transfer_repos": sorted(heldout),
        "train_dev_share_repo_pool": True,
        "use_swebench_for_training": False,
        "repos": {},
    }
    train_rows: List[Dict[str, Any]] = []
    dev_rows: List[Dict[str, Any]] = []
    heldout_rows: List[Dict[str, Any]] = []

    for repo in args.repos:
        data_path = repo_train_data_path(args.train_data_dir, repo)
        repo_path = repo_path_from_base(args.base_repos, repo)
        if not data_path.exists():
            split_summary["repos"][repo] = {"status": "missing_data", "path": str(data_path)}
            continue
        if not repo_path.exists():
            split_summary["repos"][repo] = {"status": "missing_repo", "path": str(repo_path)}
            continue

        raw_samples = sorted(load_jsonl(data_path), key=sample_sort_key)
        candidate_rows: List[Dict[str, Any]] = []
        skipped = {"no_patch": 0, "no_query": 0, "no_production_files": 0, "no_gold": 0, "not_enough_history": 0}

        for index, sample in enumerate(raw_samples):
            if index < args.min_history:
                skipped["not_enough_history"] += 1
                continue
            if not sample.get("patch"):
                skipped["no_patch"] += 1
                continue
            if not query_text(sample):
                skipped["no_query"] += 1
                continue
            if not production_files_from_sample(sample):
                skipped["no_production_files"] += 1
                continue
            gold = extract_patch_gold(repo_path, sample.get("patch", ""), sample.get("base_commit", "HEAD"))
            if not gold.files:
                skipped["no_gold"] += 1
                continue
            record = build_record(repo, sample, gold)
            record["history"] = {"earlier_pr_count": index, "target_index_by_time": index}
            candidate_rows.append(record)
            if args.max_per_repo > 0 and len(candidate_rows) >= args.max_per_repo:
                break

        if repo in heldout:
            for row in candidate_rows:
                row["split"] = "heldout_transfer"
            heldout_rows.extend(candidate_rows)
            repo_summary = {"split": "heldout_transfer", "rows": len(candidate_rows), "skipped": skipped}
        else:
            dev_count = max(1, int(round(len(candidate_rows) * args.dev_ratio))) if candidate_rows else 0
            split_index = max(0, len(candidate_rows) - dev_count)
            repo_train = candidate_rows[:split_index]
            repo_dev = candidate_rows[split_index:]
            for row in repo_train:
                row["split"] = "train"
            for row in repo_dev:
                row["split"] = "dev"
            train_rows.extend(repo_train)
            dev_rows.extend(repo_dev)
            repo_summary = {
                "split": "optimization_pool",
                "train_rows": len(repo_train),
                "dev_rows": len(repo_dev),
                "skipped": skipped,
            }
        split_summary["repos"][repo] = repo_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "dev.jsonl", dev_rows)
    write_jsonl(args.output_dir / "heldout_transfer.jsonl", heldout_rows)
    split_summary["totals"] = {
        "train": len(train_rows),
        "dev": len(dev_rows),
        "heldout_transfer": len(heldout_rows),
    }
    write_json(args.output_dir / "split_summary.json", split_summary)
    print(f"Wrote splits to {args.output_dir}")
    print(split_summary["totals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
