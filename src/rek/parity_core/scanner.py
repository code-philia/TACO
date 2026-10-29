"""
Local copy adapted from `Repo_Reqirement_Knowledge/src/common/scanner.py`.

This copy is self-contained and does not depend on the original repository.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rek.parity_core.ast_scan_utils import max_lineno


def iter_py_files(repo_dir: str, exclude_dirs: list[str] | None = None):
    excluded = set(exclude_dirs or [".git", "__pycache__", "venv", "build", "dist", "tests", "test"])
    root = Path(repo_dir)
    for path in root.rglob("*.py"):
        if any(part in excluded for part in path.parts):
            continue
        if path.is_file():
            yield path


def relpath_posix(path: Path, repo_dir: str) -> str:
    return path.relative_to(Path(repo_dir)).as_posix()


def module_path_from_rel_py(rel_path: str) -> str:
    path = rel_path.replace("/", ".")
    if path.endswith(".py"):
        path = path[:-3]
    if path.endswith(".__init__"):
        path = path[: -len(".__init__")]
    return path


def _get_code_snippet(source_lines: list[str], start_line: int, end_line: int) -> str:
    if start_line < 1:
        start_line = 1
    subset = source_lines[start_line - 1 : end_line]
    return "".join(subset)


def scan_code_content(source_code: str, rel_path: str, module_path: str | None = None) -> list[dict[str, Any]]:
    program_elements: list[dict[str, Any]] = []
    if module_path is None:
        module_path = module_path_from_rel_py(rel_path)

    try:
        source_lines = source_code.splitlines(keepends=True)
        tree = ast.parse(source_code)
    except Exception:
        return []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            program_elements.append(
                _extract_node(node, "function", module_path, rel_path, source_lines)
            )
        elif isinstance(node, ast.AsyncFunctionDef):
            program_elements.append(
                _extract_node(node, "function", module_path, rel_path, source_lines)
            )
        elif isinstance(node, ast.ClassDef):
            program_elements.append(
                _extract_node(node, "class", module_path, rel_path, source_lines)
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    program_elements.append(
                        _extract_node(
                            item,
                            "method",
                            module_path,
                            rel_path,
                            source_lines,
                            parent_class=node.name,
                        )
                    )

    return program_elements


def scan_repo(repo_dir: str, exclude_dirs: list[str] | None = None) -> list[dict[str, Any]]:
    program_elements: list[dict[str, Any]] = []
    for abs_py in iter_py_files(repo_dir, exclude_dirs=exclude_dirs):
        rel_path = relpath_posix(abs_py, repo_dir)
        module_path = module_path_from_rel_py(rel_path)
        try:
            with open(abs_py, "r", encoding="utf-8") as handle:
                source_code = handle.read()
        except Exception:
            continue
        program_elements.extend(scan_code_content(source_code, rel_path, module_path))
    return program_elements


def _extract_node(
    node: ast.AST,
    elem_type: str,
    module_path: str,
    rel_path: str,
    source_lines: list[str],
    parent_class: str | None = None,
) -> dict[str, Any]:
    name = node.name  # type: ignore[attr-defined]
    start_line = getattr(node, "lineno", 0)
    end_line = max_lineno(node)

    if parent_class:
        qualified_name = f"{module_path}.{parent_class}.{name}"
    else:
        qualified_name = f"{module_path}.{name}"

    return {
        "element_id": qualified_name,
        "type": elem_type,
        "name": name,
        "qualified_name": qualified_name,
        "file_path": rel_path,
        "parent_class": parent_class,
        "location": {"start_line": start_line, "end_line": end_line},
        "docstring": ast.get_docstring(node) or "",
        "code_content": _get_code_snippet(source_lines, start_line, end_line),
    }


def save_elements(elements: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for item in elements:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_statistics(elements: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_counter = Counter()
    for element in elements:
        file_counter[element["file_path"]] += 1

    with open(output_path, "w", encoding="utf-8") as handle:
        summary = {
            "type": "summary",
            "total_files": len(file_counter),
            "total_elements": len(elements),
            "scanned_files_list": sorted(file_counter.keys()),
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        for file_path, count in sorted(file_counter.items()):
            record = {
                "type": "file_detail",
                "file_path": file_path,
                "element_count": count,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
