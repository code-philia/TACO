from __future__ import annotations

import argparse
import json
from pathlib import Path

from rek.parity_core.repo2doc_local.config_loader import Config


def merge_knowledge_base(
    repo_path: str,
    config_path: str | None = None,
    output_filename: str = "knowledge_graph_complete_detail.json",
) -> None:
    repo = Path(repo_path).resolve()
    config = Config.load(config_path)
    output_dir = repo / config.output.output_dir
    base_kb_path = output_dir / "knowledge_graph_detail.json"
    batch_res_path = output_dir / "batch_mapping_results_detail.json"
    if not base_kb_path.exists():
        base_kb_path = output_dir / "knowledge_graph.json"

    if not base_kb_path.exists():
        print("[Error] No base Knowledge Graph found. Please run Phase A first.")
        return

    print(f"[1/4] Loading Base KB from {base_kb_path.name}...")

    with open(base_kb_path, "r", encoding="utf-8") as handle:
        kb = json.load(handle)

    batch_mappings: list[dict] = []
    if batch_res_path.exists():
        print(f"[2/4] Loading Batch Results from {batch_res_path.name}...")
        with open(batch_res_path, "r", encoding="utf-8") as handle:
            try:
                batch_mappings = json.load(handle)
            except json.JSONDecodeError:
                print("[Warning] Batch results file is empty or invalid JSON.")
    else:
        print("[Warning] No batch results found. Merging skipped.")

    existing_files = {mapping["file_path"] for mapping in kb.get("file_mappings", [])}
    initial_count = len(existing_files)
    added_count = 0

    print("[3/4] Merging and Deduplicating...")
    for mapping in batch_mappings:
        file_path = mapping.get("file_path")
        if file_path and file_path not in existing_files:
            kb["file_mappings"].append(mapping)
            existing_files.add(file_path)
            added_count += 1

    final_output_path = output_dir / output_filename
    print(f"[4/4] Saving Complete Knowledge Graph to {final_output_path.name}...")
    with open(final_output_path, "w", encoding="utf-8") as handle:
        json.dump(kb, handle, indent=2, ensure_ascii=False)

    print("-" * 50)
    print("✅ MERGE SUCCESS")
    print(f"Base Mappings:  {initial_count}")
    print(f"Added from Batch: {added_count}")
    print(f"Total Files Mapped: {len(kb['file_mappings'])}")
    print(f"Final Path: {final_output_path}")
    print("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Phase A and Phase B results into a final Knowledge Graph.")
    parser.add_argument("repo_path", help="Path to the repository analyzed")
    parser.add_argument("-c", "--config", default=None, help="Path to config file")
    parser.add_argument("-o", "--output", default="knowledge_graph_complete_detail.json", help="Filename for the merged json")
    args = parser.parse_args()
    merge_knowledge_base(args.repo_path, args.config, args.output)


if __name__ == "__main__":
    main()
