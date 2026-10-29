from __future__ import annotations

import ast
import difflib
import logging
import os
import re
import subprocess

logger = logging.getLogger("GitTimeTravel")


class BackwardTimeMachine:
    """Map modern files predicted at HEAD back to a historical base commit."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def get_modern_elements(self, modern_file: str) -> list[str]:
        full_path = os.path.join(self.repo_path, modern_file)
        if not os.path.exists(full_path):
            return []
        elements: list[str] = []
        try:
            with open(full_path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=modern_file)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    elements.append(node.name)
        except Exception:
            return []
        return elements

    def grep_element_in_history(self, element_name: str, base_commit: str) -> list[str]:
        try:
            pattern = rf"^[ \t]*(def|class)\s+{element_name}\b"
            command = ["git", "grep", "-E", "-l", pattern, base_commit]
            result = subprocess.run(command, cwd=self.repo_path, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout:
                files = result.stdout.strip().splitlines()
                return [item.split(":", 1)[-1] for item in files if "test" not in item and item.endswith(".py")]
        except Exception:
            return []
        return []

    def trace_prediction_backward(
        self,
        modern_file: str,
        base_commit: str,
        predicted_element: str = "",
    ) -> str:
        if not base_commit or base_commit == "HEAD":
            return modern_file

        check_command = ["git", "ls-tree", "-r", "--name-only", base_commit, modern_file]
        check_result = subprocess.run(check_command, cwd=self.repo_path, capture_output=True, text=True)
        if modern_file in check_result.stdout.splitlines():
            return modern_file

        try:
            diff_command = ["git", "diff", "--name-status", "-M", base_commit, "HEAD"]
            diff_result = subprocess.run(diff_command, cwd=self.repo_path, capture_output=True, text=True)
            for line in diff_result.stdout.splitlines():
                parts = line.split("\t")
                if parts[0].startswith("R") and len(parts) >= 3 and parts[2] == modern_file:
                    logger.info("[BackwardTrace] rename: %s -> %s", modern_file, parts[1])
                    return parts[1]
        except Exception as exc:
            logger.debug("Git diff failed: %s", exc)

        if predicted_element:
            clean_element = predicted_element.split(".")[0].split("(")[0].strip()
            historic_files = self.grep_element_in_history(clean_element, base_commit)
            if historic_files:
                historic_files.sort(
                    key=lambda item: difflib.SequenceMatcher(None, modern_file, item).ratio(),
                    reverse=True,
                )
                return historic_files[0]

        modern_elements = self.get_modern_elements(modern_file)
        if modern_elements:
            from collections import Counter

            file_votes: Counter[str] = Counter()
            for element in modern_elements:
                hits = self.grep_element_in_history(element, base_commit)
                for hit in hits:
                    file_votes[hit] += 1
            if file_votes:
                best_file, votes = file_votes.most_common(1)[0]
                if votes >= 2 or (len(modern_elements) <= 2 and votes >= 1):
                    return best_file

        logger.warning("[BackwardTrace] fallback to exact path for %s", modern_file)
        return modern_file


class ASTGroundTruthExtractor:
    """Extract file, module, and function ground truth from a patch and base commit."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def _get_modified_lines(self, patch_text: str) -> dict[str, set[int]]:
        file_lines: dict[str, set[int]] = {}
        current_file: str | None = None
        for line in patch_text.splitlines():
            if line.startswith("--- a/"):
                current_file = line[6:]
                if current_file != "/dev/null":
                    file_lines[current_file] = set()
            elif line.startswith("@@") and current_file and current_file != "/dev/null":
                match = re.match(r"@@ -(\d+)(?:,(\d+))?", line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    for line_no in range(start, start + count):
                        file_lines[current_file].add(line_no)
        return file_lines

    def extract_3level_gt(
        self,
        patch_text: str,
        base_commit: str,
    ) -> tuple[list[str], list[str], list[str]]:
        if not patch_text:
            return [], [], []

        file_mod_lines = self._get_modified_lines(patch_text)
        gt_files: set[str] = set()
        gt_modules: set[str] = set()
        gt_functions: set[str] = set()

        for file_path, modified_lines in file_mod_lines.items():
            if "test" in file_path or not file_path.endswith((".py", ".pyx")):
                continue
            gt_files.add(file_path)

            command = ["git", "show", f"{base_commit}:{file_path}"]
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                errors="replace",
            )
            if result.returncode != 0:
                continue

            try:
                tree = ast.parse(result.stdout, filename=file_path)
            except SyntaxError:
                continue

            class ModificationVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.current_class: str | None = None

                def visit_ClassDef(self, node: ast.ClassDef) -> None:
                    previous_class = self.current_class
                    self.current_class = node.name
                    end_line = getattr(node, "end_lineno", node.lineno)
                    if any(node.lineno <= line <= end_line for line in modified_lines):
                        gt_modules.add(f"{file_path}::{node.name}")
                    self.generic_visit(node)
                    self.current_class = previous_class

                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    end_line = getattr(node, "end_lineno", node.lineno)
                    if any(node.lineno <= line <= end_line for line in modified_lines):
                        module_name = self.current_class if self.current_class else "Global"
                        gt_modules.add(f"{file_path}::{module_name}")
                        if self.current_class:
                            gt_functions.add(f"{file_path}::{self.current_class}.{node.name}")
                        else:
                            gt_functions.add(f"{file_path}::{node.name}")
                    self.generic_visit(node)

            ModificationVisitor().visit(tree)

            if file_path in gt_files and not any(item.startswith(f"{file_path}::") for item in gt_modules):
                gt_modules.add(f"{file_path}::Global")

        return list(gt_files), list(gt_modules), list(gt_functions)


def parse_swebench_gt(
    edit_functions: list[str],
    added_functions: list[str],
) -> tuple[list[str], list[str], list[str]]:
    gt_files: set[str] = set()
    gt_modules: set[str] = set()
    gt_functions: set[str] = set()

    for item in set(edit_functions + added_functions):
        if ":" not in item:
            continue
        file_path, element = item.split(":", 1)
        gt_files.add(file_path)
        if "." in element:
            module_name, _ = element.rsplit(".", 1)
            gt_modules.add(f"{file_path}::{module_name}")
            gt_functions.add(f"{file_path}::{element}")
        else:
            gt_modules.add(f"{file_path}::Global")
            gt_functions.add(f"{file_path}::{element}")

    return list(gt_files), list(gt_modules), list(gt_functions)
