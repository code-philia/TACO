#!/usr/bin/env python3
"""Distill historical PR manifests with a prompt and build local experiment KBs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import load_jsonl, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a historical PR manifest into per-repo KB JSON files.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-dir-name", default="repo2docAgent-output")
    parser.add_argument("--kb-name", default="knowledge_graph_two_index.json")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-query-chars", type=int, default=2500)
    parser.add_argument("--max-patch-chars", type=int, default=7000)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def chunks(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def extract_json_candidates(content: str) -> List[str]:
    candidates: List[str] = []
    if not content:
        return candidates
    candidates.extend(match.strip() for match in re.findall(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE))
    stripped = content.strip()
    if stripped:
        candidates.append(stripped)
    first = content.find("[")
    last = content.rfind("]")
    if first != -1 and last != -1 and first < last:
        candidates.append(content[first : last + 1].strip())
    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate.strip())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def normalize_items(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("items") or raw.get("results") or [raw]
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("file") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        function_name = str(item.get("function") or "FILE_SUMMARY").strip() or "FILE_SUMMARY"
        keywords = item.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [part.strip() for part in re.split(r"[,;|]", keywords) if part.strip()]
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        if file_path and rationale:
            normalized.append(
                {
                    "file": file_path,
                    "function": function_name,
                    "keywords": keywords[:12],
                    "rationale": rationale,
                }
            )
    return normalized


def parse_response(content: str) -> List[Dict[str, Any]]:
    for candidate in extract_json_candidates(content):
        try:
            return normalize_items(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return []


def build_user_prompt(row: Dict[str, Any], max_query_chars: int, max_patch_chars: int) -> str:
    gold = row.get("gold") or {}
    target_context = {
        "target_files": gold.get("files") or [],
        "target_modules": gold.get("modules") or [],
        "target_functions": gold.get("functions") or [],
    }
    return f"""
# Historical PR Context

Target context is provided only to map historical changes to reusable file/function knowledge. Do not copy it blindly if unsupported by the patch.

## Target Context
{json.dumps(target_context, ensure_ascii=False, indent=2)}

## PR Text
{str(row.get('query') or '')[:max_query_chars]}

## Patch
{str(row.get('patch') or '')[:max_patch_chars]}
""".strip()


def distill_one(
    row: Dict[str, Any],
    prompt_text: str,
    args: argparse.Namespace,
    api_key: str,
) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.request_timeout, max_retries=0)
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": build_user_prompt(row, args.max_query_chars, args.max_patch_chars)},
    ]
    started = time.time()
    output: Dict[str, Any] = {
        "repo": row.get("repo"),
        "instance_id": row.get("instance_id"),
        "split": row.get("split"),
        "status": "ok",
        "items": [],
        "usage": None,
        "latency_seconds": None,
    }
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
        )
        content = response.choices[0].message.content or ""
        output["items"] = parse_response(content)
        output["raw_response"] = content
        output["usage"] = response.usage.model_dump() if response.usage else None
        if not output["items"]:
            output["status"] = "parse_empty"
    except Exception as exc:  # noqa: BLE001
        output["status"] = "llm_error"
        output["error"] = str(exc)
    output["latency_seconds"] = round(time.time() - started, 3)
    return output


def append_item(kb_by_repo: Dict[str, Dict[str, Any]], repo: str, item: Dict[str, Any]) -> None:
    kb = kb_by_repo.setdefault(repo, {"l1_categories": [], "l2_features": [], "file_mappings": []})
    file_nodes = {node["file_path"]: node for node in kb["file_mappings"]}
    file_path = item["file"]
    file_node = file_nodes.get(file_path)
    if file_node is None:
        file_node = {
            "file_path": file_path,
            "primary_feature_id": None,
            "secondary_feature_ids": [],
            "role": "Historical_PR_Distilled_File",
            "rationale": "",
            "rationale_keywords": [],
            "functions": {},
        }
        kb["file_mappings"].append(file_node)

    keywords = item.get("keywords") or []
    rationale = item.get("rationale") or ""
    function_name = item.get("function") or "FILE_SUMMARY"
    if function_name == "FILE_SUMMARY":
        if rationale and rationale not in file_node["rationale"]:
            file_node["rationale"] = (file_node["rationale"] + "\n" + rationale).strip()
        file_node["rationale_keywords"] = sorted(set(file_node.get("rationale_keywords", [])).union(keywords))
        return

    functions = file_node.setdefault("functions", {})
    func_node = functions.setdefault(function_name, {"rationale": "", "rationale_keywords": []})
    if rationale and rationale not in func_node["rationale"]:
        func_node["rationale"] = (func_node["rationale"] + "\n" + rationale).strip()
    func_node["rationale_keywords"] = sorted(set(func_node.get("rationale_keywords", [])).union(keywords))


def write_kbs(output_dir: Path, output_dir_name: str, kb_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    kb_by_repo: Dict[str, Dict[str, Any]] = {}
    counts_by_repo: Dict[str, int] = {}
    for row in rows:
        repo = str(row.get("repo") or "unknown")
        for item in row.get("items") or []:
            append_item(kb_by_repo, repo, item)
            counts_by_repo[repo] = counts_by_repo.get(repo, 0) + 1

    for repo, kb in kb_by_repo.items():
        repo_dir = output_dir / "kb_repos" / repo / output_dir_name
        repo_dir.mkdir(parents=True, exist_ok=True)
        write_json(repo_dir / kb_name, kb)
        (repo_dir / "program_elements.jsonl").write_text("", encoding="utf-8")

    return {
        "repos": sorted(kb_by_repo),
        "knowledge_items_by_repo": counts_by_repo,
        "knowledge_items_total": sum(counts_by_repo.values()),
    }


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Missing API key in ${args.api_key_env}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = args.prompt.read_text(encoding="utf-8")
    manifest_rows = load_jsonl(args.manifest)

    all_results: List[Dict[str, Any]] = []
    batch_summaries: List[Dict[str, Any]] = []
    for batch_index, batch in enumerate(chunks(manifest_rows, args.batch_size), 1):
        batch_started = datetime.now(timezone.utc).isoformat()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(distill_one, row, prompt_text, args, api_key) for row in batch]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                all_results.append(result)
                print(
                    f"batch={batch_index} repo={result.get('repo')} id={result.get('instance_id')} "
                    f"status={result.get('status')} items={len(result.get('items') or [])}",
                    flush=True,
                )
        batch_summaries.append(
            {
                "batch_index": batch_index,
                "started_at": batch_started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "rows": len(batch),
                "ok": sum(1 for row in all_results[-len(batch) :] if row.get("status") == "ok"),
                "parse_empty": sum(1 for row in all_results[-len(batch) :] if row.get("status") == "parse_empty"),
                "llm_error": sum(1 for row in all_results[-len(batch) :] if row.get("status") == "llm_error"),
            }
        )

    write_jsonl(args.output_dir / "distilled_results.jsonl", all_results)
    kb_summary = write_kbs(args.output_dir, args.output_dir_name, args.kb_name, all_results)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in all_results:
        usage = row.get("usage") or {}
        for key in total_usage:
            total_usage[key] += int(usage.get(key) or 0)
    summary = {
        "manifest": str(args.manifest),
        "prompt": str(args.prompt),
        "model": args.model,
        "base_url": args.base_url,
        "rows": len(manifest_rows),
        "status_counts": {status: sum(1 for row in all_results if row.get("status") == status) for status in sorted({row.get("status") for row in all_results})},
        "zero_item_rows": sum(1 for row in all_results if not row.get("items")),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "batches": batch_summaries,
        "usage": total_usage,
        "kb_summary": kb_summary,
    }
    write_json(args.output_dir / "distill_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
