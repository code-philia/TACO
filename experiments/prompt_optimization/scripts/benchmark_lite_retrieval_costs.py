#!/usr/bin/env python3
"""Benchmark SWE-bench Lite retrieval cost across prebuilt model KBs.

This script intentionally does not modify existing retrieval logic. It reuses
GraphDatabase + LightVectorStore, samples Lite rows, and writes per-sample plus
aggregate metrics for retrieval latency, prompt tokens, and estimated token cost.
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rek.parity_core.light_vector import LightVectorStore  # noqa: E402
from rek.parity_core.graph_db import GraphDatabase  # noqa: E402


@dataclass(frozen=True)
class ModelConfig:
    model: str
    output_dir_name: str
    input_usd_per_million: float
    output_usd_per_million: float


MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig(
        model="gpt-4o-2024-11-20",
        output_dir_name="repo2docAgent-output",
        input_usd_per_million=2.50,
        output_usd_per_million=10.00,
    ),
    ModelConfig(
        model="gpt-5.4-mini-2026-03-17",
        output_dir_name="repo2docAgent-output-gpt-5-4-mini",
        input_usd_per_million=0.75,
        output_usd_per_million=4.50,
    ),
    ModelConfig(
        model="claude-sonnet-4-6",
        output_dir_name="repo2docAgent-output-claude",
        input_usd_per_million=3.00,
        output_usd_per_million=15.00,
    ),
]

REPO_NAME_MAP = {
    "astropy/astropy": "astropy",
    "django/django": "django",
    "matplotlib/matplotlib": "matplotlib",
    "mwaskom/seaborn": "seaborn",
    "pallets/flask": "flask",
    "psf/requests": "requests",
    "pydata/xarray": "xarray",
    "pylint-dev/pylint": "pylint",
    "pytest-dev/pytest": "pytest",
    "scikit-learn/scikit-learn": "scikit-learn",
    "sphinx-doc/sphinx": "sphinx",
    "sympy/sympy": "sympy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("REK_SWEBENCH_LITE_TEST", "YOUR_EXTERNAL_BENCHMARK_PATH_HERE")))
    parser.add_argument("--base-repos", type=Path, default=Path(os.environ.get("REK_BASE_REPOS", "YOUR_REPO_CHECKOUTS_DIR_HERE")))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/prompt_optimization/results/lite_retrieval_cost_benchmark"))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--kb-name", default="knowledge_graph_two_index_epoch_1.json")
    parser.add_argument("--elements-name", default="program_elements.jsonl")
    parser.add_argument("--max-query-chars", type=int, default=12000)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_repo(repo: str) -> str:
    return REPO_NAME_MAP.get(repo, repo.rsplit("/", 1)[-1])


def build_query(row: Dict[str, Any], max_chars: int) -> str:
    parts = [row.get("problem_statement") or ""]
    hints = row.get("hints_text") or ""
    if hints:
        parts.append("\nHints:\n" + hints)
    query = "\n\n".join(part for part in parts if part.strip()).strip()
    return query[:max_chars]


def patch_files(patch: str) -> List[str]:
    files = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            bits = line.split()
            if len(bits) >= 4 and bits[2].startswith("a/"):
                files.append(bits[2][2:])
    return sorted(set(files))


def token_count(text: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, int(len(text.split()) * 1.33))


def render_retrieval_context(repo: str, query: str, retrieval: List[Dict[str, Any]]) -> str:
    payload = {
        "repo": repo,
        "issue": query,
        "retrieved_candidates": [
            {
                "rank": idx + 1,
                "file": item.get("file"),
                "element": item.get("element"),
                "score": item.get("score"),
                "preview": item.get("preview"),
            }
            for idx, item in enumerate(retrieval)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def kb_paths(base_repos: Path, repo: str, output_dir_name: str, kb_name: str, elements_name: str) -> tuple[Path, Path]:
    repo_output = base_repos / repo / output_dir_name
    return repo_output / kb_name, repo_output / elements_name


def load_store(
    cache: Dict[tuple[str, str], LightVectorStore],
    base_repos: Path,
    repo: str,
    config: ModelConfig,
    kb_name: str,
    elements_name: str,
) -> tuple[Optional[LightVectorStore], float, Optional[str]]:
    key = (config.output_dir_name, repo)
    if key in cache:
        return cache[key], 0.0, None
    kb_path, elements_path = kb_paths(base_repos, repo, config.output_dir_name, kb_name, elements_name)
    if not kb_path.exists():
        return None, 0.0, f"missing_kb:{kb_path}"
    started = time.perf_counter()
    graph_db = GraphDatabase(str(kb_path), str(elements_path))
    store = LightVectorStore(graph_db)
    load_seconds = time.perf_counter() - started
    cache[key] = store
    return store, load_seconds, None


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.dataset)
    usable = [row for row in rows if normalize_repo(row.get("repo", "")) and build_query(row, args.max_query_chars)]
    if len(usable) < args.sample_size:
        raise SystemExit(f"Only {len(usable)} usable rows in {args.dataset}; requested {args.sample_size}")

    rng = random.Random(args.seed)
    sample = rng.sample(usable, args.sample_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "sample_50.jsonl"
    write_jsonl(sample_path, sample)

    stores: Dict[tuple[str, str], LightVectorStore] = {}
    details: List[Dict[str, Any]] = []
    load_seconds_by_model: Dict[str, float] = {config.model: 0.0 for config in MODEL_CONFIGS}

    for config in MODEL_CONFIGS:
        for row in sample:
            repo = normalize_repo(row.get("repo", ""))
            query = build_query(row, args.max_query_chars)
            store, load_seconds, load_error = load_store(
                stores,
                args.base_repos,
                repo,
                config,
                args.kb_name,
                args.elements_name,
            )
            load_seconds_by_model[config.model] += load_seconds

            record: Dict[str, Any] = {
                "model": config.model,
                "kb_output_dir": config.output_dir_name,
                "repo": repo,
                "source_repo": row.get("repo"),
                "instance_id": row.get("instance_id"),
                "gold_files": patch_files(row.get("patch") or ""),
                "query_chars": len(query),
                "query_tokens": token_count(query),
                "top_k": args.top_k,
                "status": "ok",
            }
            if store is None:
                record["status"] = "missing_kb"
                record["error"] = load_error
                details.append(record)
                continue

            started = time.perf_counter()
            try:
                retrieval = store.search(query, top_k=args.top_k)
            except Exception as exc:  # noqa: BLE001
                record["status"] = "retrieval_error"
                record["error"] = str(exc)
                details.append(record)
                continue
            retrieval_seconds = time.perf_counter() - started

            context = render_retrieval_context(repo, query, retrieval)
            prompt_tokens = token_count(context)
            input_cost = prompt_tokens * config.input_usd_per_million / 1_000_000
            record.update(
                {
                    "retrieval_seconds": retrieval_seconds,
                    "retrieval_count": len(retrieval),
                    "retrieval_context_tokens": prompt_tokens - record["query_tokens"],
                    "prompt_input_tokens": prompt_tokens,
                    "estimated_input_cost_usd": input_cost,
                    "estimated_input_cost_cny": input_cost * 7.2,
                    "retrieval": retrieval,
                }
            )
            details.append(record)

    summary_rows: List[Dict[str, Any]] = []
    for config in MODEL_CONFIGS:
        model_rows = [row for row in details if row["model"] == config.model and row["status"] == "ok"]
        summary_rows.append(
            {
                "model": config.model,
                "kb_output_dir": config.output_dir_name,
                "ok_samples": len(model_rows),
                "requested_samples": args.sample_size,
                "top_k": args.top_k,
                "input_usd_per_million": config.input_usd_per_million,
                "output_usd_per_million": config.output_usd_per_million,
                "load_seconds_total": load_seconds_by_model[config.model],
                "retrieval_seconds_mean": summarize([row["retrieval_seconds"] for row in model_rows])["mean"],
                "retrieval_seconds_median": summarize([row["retrieval_seconds"] for row in model_rows])["median"],
                "query_tokens_mean": summarize([row["query_tokens"] for row in model_rows])["mean"],
                "retrieval_context_tokens_mean": summarize([row["retrieval_context_tokens"] for row in model_rows])["mean"],
                "prompt_input_tokens_mean": summarize([row["prompt_input_tokens"] for row in model_rows])["mean"],
                "estimated_input_cost_usd_total": sum(row["estimated_input_cost_usd"] for row in model_rows),
                "estimated_input_cost_usd_mean": summarize([row["estimated_input_cost_usd"] for row in model_rows])["mean"],
                "estimated_input_cost_cny_total": sum(row["estimated_input_cost_cny"] for row in model_rows),
                "estimated_input_cost_cny_mean": summarize([row["estimated_input_cost_cny"] for row in model_rows])["mean"],
            }
        )

    details_path = args.output_dir / "details.jsonl"
    summary_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "summary.csv"
    write_jsonl(details_path, details)
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote sample to {sample_path}")
    print(f"Wrote details to {details_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote CSV summary to {csv_path}")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
