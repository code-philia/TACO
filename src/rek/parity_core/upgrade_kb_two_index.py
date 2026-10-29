"""
Local copy adapted from `Repo_Reqirement_Knowledge/support/upgrade_kb_two_index.py`.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class FunctionASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions_info: dict[str, dict[str, object]] = {}
        self.current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = previous_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node)
        self.generic_visit(node)

    def _get_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        arguments: list[str] = []
        for arg in node.args.args:
            arguments.append(arg.arg)
        if node.args.vararg:
            arguments.append(f"*{node.args.vararg.arg}")
        for arg in node.args.kwonlyargs:
            arguments.append(arg.arg)
        if node.args.kwarg:
            arguments.append(f"**{node.args.kwarg.arg}")
        return f"{node.name}({', '.join(arguments)})"

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        function_name = node.name
        signature = self._get_signature(node)
        initial_keywords = [function_name]

        if self.current_class:
            full_name = f"{self.current_class}.{function_name}"
            function_type = "method"
            prefix_desc = f"Method of class '{self.current_class}'."
            initial_keywords.append(self.current_class)
        else:
            full_name = function_name
            function_type = "function"
            prefix_desc = "Global function."

        docstring = ast.get_docstring(node)
        if docstring:
            rationale = f"Signature: {signature}\nDescription: {docstring.strip()}"
        else:
            rationale = (
                f"Signature: {signature}\n"
                f"Description: [No docstring provided] {prefix_desc} Named '{function_name}'."
            )

        self.functions_info[full_name] = {
            "type": function_type,
            "rationale": rationale,
            "rationale_keywords": initial_keywords,
        }


def process_knowledge_graph(kb_path: str, repo_root: str, output_path: str) -> dict[str, int | str]:
    kb_file = Path(kb_path)
    if not kb_file.exists():
        raise FileNotFoundError(f"Knowledge graph file not found: {kb_path}")

    with open(kb_file, "r", encoding="utf-8") as handle:
        kb_data = json.load(handle)

    file_list = kb_data.get("file_mappings", kb_data)
    updated_files_count = 0
    total_functions_extracted = 0

    for file_node in file_list:
        file_path = file_node.get("file_path")
        if not file_path:
            continue

        file_node.setdefault("rationale_keywords", [])
        file_stem = Path(file_path).stem
        if file_stem not in file_node["rationale_keywords"]:
            file_node["rationale_keywords"].append(file_stem)

        full_source_path = Path(repo_root) / file_path
        if not full_source_path.exists() or not full_source_path.is_file():
            file_node["functions"] = {}
            continue

        try:
            with open(full_source_path, "r", encoding="utf-8") as source_file:
                source_code = source_file.read()

            tree = ast.parse(source_code, filename=str(full_source_path))
            visitor = FunctionASTVisitor()
            visitor.visit(tree)

            file_node["functions"] = visitor.functions_info
            updated_files_count += 1
            total_functions_extracted += len(visitor.functions_info)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", file_path, exc)
            file_node["functions"] = {}

    with open(output_path, "w", encoding="utf-8") as handle:
        if isinstance(kb_data, dict):
            kb_data["file_mappings"] = file_list
            json.dump(kb_data, handle, indent=2, ensure_ascii=False)
        else:
            json.dump(file_list, handle, indent=2, ensure_ascii=False)

    return {
        "updated_files_count": updated_files_count,
        "total_functions_extracted": total_functions_extracted,
        "output_path": output_path,
    }
