"""
Local copy adapted from `Repo_Reqirement_Knowledge/src/tools/graph_db.py`.

This copy exists so the CLI project can keep a stable core dependency without
modifying or importing the original repository at runtime.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class GraphDatabase:
    def __init__(self, kb_json_path: str, elements_jsonl_path: str):
        self.kb_path = Path(kb_json_path)
        self.elements_path = Path(elements_jsonl_path)
        self.kb_data = self._load_kb_json()
        self.elements_data = self._load_elements_jsonl()
        self.explicit_element_mappings: dict[str, set[str]] = defaultdict(set)
        self._build_indexes()
        self._init_vector_db()

    def _init_vector_db(self) -> None:
        self.vector_client = None
        self.file_collection = None
        self.function_collection = None

    def _load_kb_json(self) -> dict[str, Any]:
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge Graph JSON not found: {self.kb_path}")
        with open(self.kb_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return {"file_mappings": data, "l1_categories": [], "l2_features": []}
        return data

    def _load_elements_jsonl(self) -> dict[str, dict[str, Any]]:
        elements: dict[str, dict[str, Any]] = {}
        if not str(self.elements_path) or str(self.elements_path) == ".":
            return elements
        if not self.elements_path.exists():
            return elements
        if not self.elements_path.is_file():
            return elements
        with open(self.elements_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                element_id = item.get("element_id")
                if element_id:
                    elements[element_id] = item
        return elements

    def _build_indexes(self) -> None:
        self.l1_map = {item["id"]: item for item in self.kb_data.get("l1_categories", [])}
        for l1_node in self.l1_map.values():
            l1_node.setdefault("cross_linked_l2s", [])

        self.l2_map = {item["id"]: item for item in self.kb_data.get("l2_features", [])}

        self.file_map: dict[str, dict[str, Any]] = {}
        for item in self.kb_data.get("file_mappings", []):
            item.setdefault("functions", {})
            item.setdefault("rationale_keywords", [])
            self.file_map[item["file_path"]] = item

        self.l1_to_l2: dict[str, list[str]] = defaultdict(list)
        for l2_node in self.kb_data.get("l2_features", []):
            parent_l1_id = l2_node.get("parent_l1_id")
            if parent_l1_id:
                self.l1_to_l2[parent_l1_id].append(l2_node["id"])

        self.l2_to_files: dict[str, set[str]] = defaultdict(set)
        for file_mapping in self.kb_data.get("file_mappings", []):
            file_path = file_mapping["file_path"]
            primary_feature_id = file_mapping.get("primary_feature_id")
            if primary_feature_id:
                self.l2_to_files[primary_feature_id].add(file_path)
            for secondary_feature_id in file_mapping.get("secondary_feature_ids", []):
                self.l2_to_files[secondary_feature_id].add(file_path)

        self.file_to_elements: dict[str, list[str]] = defaultdict(list)
        for element_id, node in self.elements_data.items():
            file_path = node.get("file_path")
            if file_path:
                self.file_to_elements[file_path].append(element_id)

        for file_path in self.file_to_elements:
            self.file_to_elements[file_path].sort(
                key=lambda element_id: self.elements_data[element_id]
                .get("location", {})
                .get("start_line", 0)
            )

    def get_file_ancestry(self, file_path: str) -> dict[str, str]:
        file_node = self.file_map.get(file_path)
        if not file_node:
            return {"status": "Unlinked", "l2_id": "None", "l1_id": "None"}

        l2_id = file_node.get("primary_feature_id")
        if not l2_id or l2_id not in self.l2_map:
            return {"status": "Linked to Invalid L2", "l2_id": str(l2_id), "l1_id": "None"}

        l2_node = self.l2_map[l2_id]
        l1_id = l2_node.get("parent_l1_id")
        return {
            "status": "Linked",
            "l2_id": l2_id,
            "l2_desc": l2_node.get("description", ""),
            "l1_id": l1_id or "None",
        }

    def get_root_categories(self) -> list[dict[str, Any]]:
        return list(self.l1_map.values())

    def get_l2_by_l1(self, l1_id: str) -> list[dict[str, Any]]:
        l2_ids = self.l1_to_l2.get(l1_id, [])
        cross_links = self.l1_map.get(l1_id, {}).get("cross_linked_l2s", [])
        all_l2_ids = list(set(l2_ids + cross_links))
        return [self.l2_map[item_id] for item_id in all_l2_ids if item_id in self.l2_map]

    def get_files_by_l2(self, l2_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for file_path in self.l2_to_files.get(l2_id, []):
            node = self.file_map.get(file_path)
            if not node:
                continue
            results.append(
                {
                    "file_path": file_path,
                    "role": node.get("role", "Unknown"),
                    "rationale": (node.get("rationale") or "")[:500] + "...",
                }
            )
        return results

    def get_functions_by_file(self, file_path: str) -> dict[str, Any]:
        node = self.file_map.get(file_path)
        if not node:
            return {}
        return node.get("functions", {})

    def get_elements_by_file(self, file_path: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for element_id in self.file_to_elements.get(file_path, []):
            node = self.elements_data[element_id]
            code_preview = "\n".join((node.get("code_content", "") or "").split("\n")[:15])
            results.append(
                {
                    "id": element_id,
                    "type": node.get("type", "element"),
                    "name": node.get("name"),
                    "line": node.get("location", {}).get("start_line"),
                    "code_preview": code_preview,
                }
            )
        return results

    def get_element_content(self, element_id: str) -> str:
        node = self.elements_data.get(element_id)
        return node.get("code_content", "") if node else "Element not found."

    def set_storage_path(self, new_path: str) -> None:
        new_kb_path = Path(new_path)
        new_kb_path.parent.mkdir(parents=True, exist_ok=True)
        self.kb_path = new_kb_path

    def save_state(self) -> None:
        self.kb_data["file_mappings"] = list(self.file_map.values())
        self.kb_data["l1_categories"] = list(self.l1_map.values())
        self.kb_data["l2_features"] = list(self.l2_map.values())
        try:
            with open(self.kb_path, "w", encoding="utf-8") as handle:
                json.dump(self.kb_data, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save KB state to %s: %s", self.kb_path, exc)

    def update_node_knowledge(
        self,
        file_path: str,
        function_name: str,
        new_keywords: list[str],
        new_rationale: str,
    ) -> bool:
        """
        Append lightweight knowledge into the copied two-index structure.
        """
        if file_path not in self.file_map:
            self.file_map[file_path] = {
                "file_path": file_path,
                "primary_feature_id": None,
                "secondary_feature_ids": [],
                "role": "Recovered_Modern_File",
                "rationale": "[AUTO-RECOVERED] File added during training injection.",
                "rationale_keywords": [],
                "functions": {},
            }

        file_node = self.file_map[file_path]
        file_node.setdefault("rationale_keywords", [])
        file_node.setdefault("functions", {})

        if function_name in {"FILE_SUMMARY", ""}:
            old_rationale = file_node.get("rationale", "").strip()
            if new_rationale and new_rationale not in old_rationale:
                file_node["rationale"] = f"{old_rationale}\n{new_rationale}".strip()
            keyword_set = set(file_node.get("rationale_keywords", []))
            keyword_set.update(new_keywords)
            file_node["rationale_keywords"] = sorted(keyword_set)
        else:
            function_dict = file_node["functions"]
            clean_function_name = function_name.split(".")[-1]
            target_key = next(
                (
                    key
                    for key in function_dict.keys()
                    if key == function_name or key.split(".")[-1] == clean_function_name
                ),
                function_name,
            )
            if target_key not in function_dict:
                function_dict[target_key] = {"rationale": "", "rationale_keywords": []}
            function_node = function_dict[target_key]
            function_node.setdefault("rationale_keywords", [])
            old_rationale = function_node.get("rationale", "").strip()
            if new_rationale and new_rationale not in old_rationale:
                function_node["rationale"] = f"{old_rationale}\n{new_rationale}".strip()
            keyword_set = set(function_node.get("rationale_keywords", []))
            keyword_set.update(new_keywords)
            function_node["rationale_keywords"] = sorted(keyword_set)

        self.save_state()
        return True
