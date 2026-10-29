"""
Local copy adapted from `Repo_Reqirement_Knowledge/src/tools/mutation_tools.py`.
"""

from __future__ import annotations

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover
    def tool(func):  # type: ignore
        return func


_GRAPH_DB = None
_REPO_PATH = None


def set_repo_path(path: str) -> None:
    global _REPO_PATH
    _REPO_PATH = path


def set_mutation_db(db) -> None:
    global _GRAPH_DB
    _GRAPH_DB = db


@tool
def inject_dual_index_knowledge(
    file_path: str,
    function_name: str,
    keywords: list[str],
    rationale: str,
) -> str:
    """
    Inject distilled knowledge into a file or function node.
    """
    if not _GRAPH_DB:
        return "Error: DB not initialized"

    junk_phrases = ["no changes", "not modified", "no functional changes", "no direct changes"]
    if any(phrase in rationale.lower() for phrase in junk_phrases) and not keywords:
        return f"Skipped updating '{file_path}': detected empty rationale with no keywords."

    try:
        success = _GRAPH_DB.update_node_knowledge(file_path, function_name, keywords, rationale)
    except Exception as exc:
        return f"System Error: {exc}"

    if success:
        return f"Successfully injected {len(keywords)} tags into {file_path}::{function_name}."
    return f"Failed to update {file_path}."
