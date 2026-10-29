"""
Local copy adapted from `Repo_Reqirement_Knowledge/support/Repo2TutorialAgent/merge_kb.py`.

This helper merges a base knowledge graph with batch mapping results to produce
`knowledge_graph_complete_detail.json` inside the CLI workspace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def merge_knowledge_base(
    output_dir: str,
    output_filename: str = "knowledge_graph_complete_detail.json",
) -> dict[str, Any]:
    root = Path(output_dir)
    base_kb_path = root / "knowledge_graph_detail.json"
    batch_res_path = root / "batch_mapping_results_detail.json"

    if not base_kb_path.exists():
        fallback_base = root / "knowledge_graph.json"
        if fallback_base.exists():
            base_kb_path = fallback_base
        else:
            raise FileNotFoundError(
                f"No base knowledge graph found in {root}. "
                "Expected `knowledge_graph_detail.json` or `knowledge_graph.json`."
            )

    with open(base_kb_path, "r", encoding="utf-8") as handle:
        kb = json.load(handle)

    batch_mappings: list[dict[str, Any]] = []
    if batch_res_path.exists():
        with open(batch_res_path, "r", encoding="utf-8") as handle:
            try:
                batch_mappings = json.load(handle)
            except json.JSONDecodeError:
                batch_mappings = []

    kb.setdefault("file_mappings", [])
    existing_files = {item["file_path"] for item in kb.get("file_mappings", []) if item.get("file_path")}
    initial_count = len(existing_files)
    added_count = 0

    for mapping in batch_mappings:
        file_path = mapping.get("file_path")
        if file_path and file_path not in existing_files:
            kb["file_mappings"].append(mapping)
            existing_files.add(file_path)
            added_count += 1

    final_output_path = root / output_filename
    with open(final_output_path, "w", encoding="utf-8") as handle:
        json.dump(kb, handle, indent=2, ensure_ascii=False)

    return {
        "base_path": str(base_kb_path),
        "batch_path": str(batch_res_path) if batch_res_path.exists() else None,
        "output_path": str(final_output_path),
        "base_mappings": initial_count,
        "added_mappings": added_count,
        "total_mappings": len(kb.get("file_mappings", [])),
    }
