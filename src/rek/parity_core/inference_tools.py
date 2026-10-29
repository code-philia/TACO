"""
Local copy adapted from `Repo_Reqirement_Knowledge/src/tools/inference_tools.py`.

This module preserves the original Track A tool surface so the copied workflow
can drive an agent with the same tool names and signatures.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover
    def tool(func):  # type: ignore
        return func


logger = logging.getLogger("InferenceTools")

_GRAPH_DB = None
_REPO_PATH = None
_BASE_COMMIT = "HEAD"


def set_inference_db(db) -> None:
    global _GRAPH_DB
    _GRAPH_DB = db


def build_inference_tools(repo_path: str, base_commit: str) -> list[Any]:
    global _REPO_PATH, _BASE_COMMIT
    _REPO_PATH = repo_path
    _BASE_COMMIT = base_commit
    return [
        get_root_categories,
        get_l2_features,
        get_files_in_feature,
        get_functions_in_file,
        search_by_tags,
    ]


@tool
def get_root_categories() -> list[dict[str, str]]:
    """Return all L1 categories from the knowledge graph."""
    logger.info("Tool get_root_categories called")
    if not _GRAPH_DB:
        return []
    return _GRAPH_DB.get_root_categories()


@tool
def get_l2_features(l1_id: str) -> list[dict[str, str]]:
    """Return all L2 features under a given L1 category."""
    logger.info("Tool get_l2_features called for %s", l1_id)
    if not _GRAPH_DB:
        return []
    return _GRAPH_DB.get_l2_by_l1(l1_id)


@tool
def get_files_in_feature(l2_id: str) -> list[dict[str, Any]]:
    """Return files linked to a given L2 feature."""
    logger.info("Tool get_files_in_feature called for %s", l2_id)
    if not _GRAPH_DB:
        return []
    return _GRAPH_DB.get_files_by_l2(l2_id)


@tool
def get_functions_in_file(file_path: str) -> str:
    """
    Return the function-level view for a specific file.
    """
    logger.info("Tool get_functions_in_file called for %s", file_path)
    if not _GRAPH_DB:
        return "Error: DB not initialized"
    functions = _GRAPH_DB.get_functions_by_file(file_path)
    if not functions:
        return f"No parseable functions found in '{file_path}'."

    output = [f"=== Functions in {file_path} ==="]
    for function_name, info in functions.items():
        function_type = info.get("type", "function")
        rationale = info.get("rationale", "No description.")
        output.append(f"[{function_type.upper()}] {function_name}")
        output.append(str(rationale))
        output.append("-" * 40)
    return "\n".join(output)


@tool
def search_by_tags(keywords: str) -> str:
    """
    Search the graph directly with technical keywords and return the top matches.
    """
    logger.info("Tool search_by_tags called with %s", keywords)
    if not _GRAPH_DB:
        return "Error: DB not initialized"

    search_terms = [item.strip().lower() for item in keywords.split(",") if item.strip()]
    if not search_terms:
        return "Please provide valid keywords."

    results = []
    for file_path, node in _GRAPH_DB.file_map.items():
        file_keywords = [item.lower() for item in node.get("rationale_keywords", [])]
        file_rationale = node.get("rationale", "").lower()
        match_score = sum(1 for term in search_terms if term in file_keywords or term in file_rationale)
        if match_score > 0:
            results.append({"file": file_path, "function": "FILE_SUMMARY", "score": match_score})

        for function_name, function_node in node.get("functions", {}).items():
            function_keywords = [item.lower() for item in function_node.get("rationale_keywords", [])]
            function_rationale = function_node.get("rationale", "").lower()
            function_score = sum(
                1
                for term in search_terms
                if term in function_keywords or term in function_rationale or term in function_name.lower()
            )
            if function_score > 0:
                results.append({"file": file_path, "function": function_name, "score": function_score})

    if not results:
        return f"No nodes found matching keywords: {keywords}. Try exploring L1 categories instead."

    results.sort(key=lambda item: item["score"], reverse=True)
    top_results = results[:5]
    output = [f"=== Top Search Results for '{keywords}' ==="]
    for result in top_results:
        output.append(f"File: {result['file']}")
        output.append(f"   Function: {result['function']} (Match Score: {result['score']})")
        output.append("-" * 30)
    return "\n".join(output)
