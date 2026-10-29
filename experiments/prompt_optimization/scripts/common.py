#!/usr/bin/env python3
"""Shared helpers for historical-PR prompt optimization experiments."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PRODUCTION_EXTENSIONS = (".py", ".pyx", ".pxd", ".tp")
DEFAULT_REPOS = [
    "astropy",
    "django",
    "matplotlib",
    "seaborn",
    "flask",
    "requests",
    "xarray",
    "pylint",
    "pytest",
    "scikit-learn",
    "sphinx",
    "sympy",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def sample_sort_key(sample: Dict[str, Any]) -> Tuple[datetime, str]:
    timestamp = (
        parse_datetime(sample.get("merged_at"))
        or parse_datetime(sample.get("closed_at"))
        or parse_datetime(sample.get("updated_at"))
        or datetime.min.replace(tzinfo=timezone.utc)
    )
    return timestamp, str(sample.get("instance_id") or sample.get("pr_number") or "")


def is_production_file(path: str) -> bool:
    normalized = path.strip()
    lowered = normalized.lower()
    if not normalized.endswith(PRODUCTION_EXTENSIONS):
        return False
    noise_parts = ("/test", "tests/", "/tests/", "conftest", "test_")
    return not any(part in lowered for part in noise_parts)


def patch_files(patch_text: str) -> List[str]:
    files: Set[str] = set()
    for match in re.finditer(r"^(?:--- a/|\+\+\+ b/)(.*)$", patch_text or "", re.MULTILINE):
        file_path = match.group(1).strip()
        if file_path and file_path != "dev/null":
            files.add(file_path)
    return sorted(files)


def production_files_from_sample(sample: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for field in ("files_preview", "gold_files_path", "files", "resolved"):
        value = sample.get(field)
        if isinstance(value, list):
            candidates.extend(str(item) for item in value)
    if not candidates:
        candidates.extend(patch_files(sample.get("patch", "")))
    return sorted({path for path in candidates if is_production_file(path)})


def modified_old_lines_by_file(patch_text: str) -> Dict[str, Set[int]]:
    file_lines: Dict[str, Set[int]] = {}
    current_file: Optional[str] = None
    for line in (patch_text or "").splitlines():
        if line.startswith("--- a/"):
            current_file = line[6:].strip()
            if current_file != "/dev/null":
                file_lines.setdefault(current_file, set())
        elif line.startswith("@@") and current_file and current_file != "/dev/null":
            match = re.match(r"@@ -(\d+)(?:,(\d+))?", line)
            if not match:
                continue
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) else 1
            for line_no in range(start, start + max(count, 1)):
                file_lines.setdefault(current_file, set()).add(line_no)
    return file_lines


def git_show(repo_path: Path, base_commit: str, file_path: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "show", f"{base_commit}:{file_path}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        return None
    return result.stdout


@dataclass
class PatchGold:
    files: List[str]
    modules: List[str]
    functions: List[str]


def extract_patch_gold(repo_path: Path, patch_text: str, base_commit: str) -> PatchGold:
    file_lines = modified_old_lines_by_file(patch_text)
    gold_files: Set[str] = set()
    gold_modules: Set[str] = set()
    gold_functions: Set[str] = set()

    for file_path, modified_lines in file_lines.items():
        if not is_production_file(file_path):
            continue
        gold_files.add(file_path)
        source = git_show(repo_path, base_commit, file_path)
        if not source:
            gold_modules.add(f"{file_path}::Global")
            continue
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            gold_modules.add(f"{file_path}::Global")
            continue

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.current_class: Optional[str] = None

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                previous = self.current_class
                self.current_class = node.name
                end_line = getattr(node, "end_lineno", None) or node.lineno
                if any(node.lineno <= line_no <= end_line for line_no in modified_lines):
                    gold_modules.add(f"{file_path}::{node.name}")
                self.generic_visit(node)
                self.current_class = previous

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                end_line = getattr(node, "end_lineno", None) or node.lineno
                if any(node.lineno <= line_no <= end_line for line_no in modified_lines):
                    module_name = self.current_class if self.current_class else "Global"
                    gold_modules.add(f"{file_path}::{module_name}")
                    if self.current_class:
                        gold_functions.add(f"{file_path}::{self.current_class}.{node.name}")
                    else:
                        gold_functions.add(f"{file_path}::{node.name}")
                self.generic_visit(node)

            visit_AsyncFunctionDef = visit_FunctionDef

        Visitor().visit(tree)
        if not any(module.startswith(f"{file_path}::") for module in gold_modules):
            gold_modules.add(f"{file_path}::Global")

    return PatchGold(sorted(gold_files), sorted(gold_modules), sorted(gold_functions))


def query_text(sample: Dict[str, Any], max_chars: int = 4000) -> str:
    parts = []
    title = str(sample.get("title") or "").strip()
    problem = str(sample.get("problem_statement") or "").strip()
    if title:
        parts.append(title)
    if problem and problem != title:
        parts.append(problem)
    text = "\n\n".join(parts).strip()
    return text[:max_chars]


def repo_train_data_path(train_data_dir: Path, repo: str) -> Path:
    return train_data_dir / f"{repo}_all_prs_train_base_commit.jsonl"


def repo_path_from_base(base_repos: Path, repo: str) -> Path:
    return base_repos / repo
