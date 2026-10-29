from __future__ import annotations

import ast
import difflib
import json
import logging
import os
import re
import subprocess
from datetime import datetime
from typing import Any

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    HumanMessage = SystemMessage = object  # type: ignore
    ChatOpenAI = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover
    ChatAnthropic = None

logger = logging.getLogger("PRDistiller")
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(stream_handler)


class ASTGroundTruthExtractor:
    """Extract file, module, and function modifications from a patch."""

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

    def extract_3level_gt(self, patch_text: str, base_commit: str) -> tuple[list[str], list[str], list[str]]:
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
            result = subprocess.run(command, cwd=self.repo_path, capture_output=True, text=True, errors="replace")
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


class GitTimeMachine:
    """Map historical files and functions forward into the current repository state."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def extract_old_elements_from_patch(self, patch_text: str) -> list[dict[str, str]]:
        if not patch_text:
            return []
        elements: dict[str, dict[str, str]] = {}
        current_file = ""
        for line in patch_text.splitlines():
            if line.startswith("--- a/"):
                current_file = line[6:]
            elif line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("@@"):
                match = re.search(r"@@.*@@\s*(?:def |class )(\w+)", line)
                if match and current_file.endswith(".py") and "test" not in current_file:
                    function_name = match.group(1)
                    key = f"{current_file}::{function_name}"
                    elements[key] = {"old_file": current_file, "old_function": function_name}
        return list(elements.values())

    def trace_file_to_modern(self, old_file: str, base_commit: str) -> str:
        if not base_commit or base_commit == "HEAD":
            return old_file if os.path.exists(os.path.join(self.repo_path, old_file)) else "DELETED"
        try:
            command = ["git", "diff", "--name-status", "-M", base_commit, "HEAD"]
            result = subprocess.run(command, cwd=self.repo_path, capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.split("\t")
                    if parts[0].startswith("R") and len(parts) >= 3 and parts[1] == old_file:
                        return parts[2]
                    if parts[0].startswith("D") and len(parts) >= 2 and parts[1] == old_file:
                        return "DELETED"
            if os.path.exists(os.path.join(self.repo_path, old_file)):
                return old_file
        except Exception:
            return "DELETED"
        return "DELETED"

    def trace_by_death_commit(self, old_file: str) -> str | None:
        death_command = ["git", "log", "-1", "--diff-filter=D", "--format=%H", "--all", "--", old_file]
        death_result = subprocess.run(death_command, cwd=self.repo_path, capture_output=True, text=True)
        if death_result.returncode != 0 or not death_result.stdout.strip():
            return None

        death_commit = death_result.stdout.strip()
        add_command = ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", death_commit]
        add_result = subprocess.run(add_command, cwd=self.repo_path, capture_output=True, text=True)
        if add_result.returncode != 0:
            return None

        added_files: list[str] = []
        for line in add_result.stdout.splitlines():
            parts = line.split("\t")
            if parts[0].startswith("A") and len(parts) >= 2:
                added_files.append(parts[1])

        if not added_files:
            return None

        old_basename = os.path.basename(old_file).strip("_")
        best_match: str | None = None
        highest_ratio = 0.0
        for new_file in added_files:
            new_basename = os.path.basename(new_file).strip("_")
            if old_basename == new_basename:
                best_match = new_file
                break
            ratio = difflib.SequenceMatcher(None, old_basename, new_basename).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = new_file

        if best_match and highest_ratio > 0.8 or (best_match and os.path.basename(best_match).strip("_") == old_basename):
            if os.path.exists(os.path.join(self.repo_path, best_match)):
                return best_match
            return self.trace_file_to_modern(best_match, death_commit)
        return None

    def grep_element_in_head(self, element_name: str) -> list[str]:
        try:
            pattern = rf"^[ \t]*(def|class)\s+{element_name}\b"
            command = ["git", "grep", "-E", "-l", pattern, "HEAD"]
            result = subprocess.run(command, cwd=self.repo_path, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout:
                files = result.stdout.strip().splitlines()
                return [item.split(":", 1)[-1] for item in files if "test" not in item and item.endswith(".py")]
        except Exception:
            return []
        return []

    def get_historical_elements(self, base_commit: str, old_file: str) -> list[str]:
        try:
            command = ["git", "show", f"{base_commit}:{old_file}"]
            result = subprocess.run(command, cwd=self.repo_path, capture_output=True, text=True)
            if result.returncode != 0:
                return []
            tree = ast.parse(result.stdout, filename=old_file)
            return [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        except Exception as exc:
            logger.debug("Failed to parse historical AST for %s: %s", old_file, exc)
            return []

    def advanced_file_migration_search(self, base_commit: str, old_file: str) -> str | None:
        siblings = self.get_historical_elements(base_commit, old_file)
        if not siblings:
            return None
        from collections import Counter

        file_votes: Counter[str] = Counter()
        for sibling in siblings:
            hits = self.grep_element_in_head(sibling)
            for hit in hits:
                file_votes[hit] += 1

        if file_votes:
            best_file, votes = file_votes.most_common(1)[0]
            if votes >= 2 or (len(siblings) <= 2 and votes >= 1):
                return best_file
        return None

    def get_modern_functions_in_file(self, modern_file: str) -> list[str]:
        if modern_file == "DELETED":
            return []
        full_path = os.path.join(self.repo_path, modern_file)
        if not os.path.exists(full_path):
            return []
        modern_functions: list[str] = []
        try:
            with open(full_path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=modern_file)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    modern_functions.append(node.name)
        except Exception:
            return []
        return modern_functions

    def build_time_mapping(
        self,
        patch_text: str,
        base_commit: str,
        expected_files: list[str] | None = None,
    ) -> dict[str, Any]:
        extractor = ASTGroundTruthExtractor(self.repo_path)
        gt_files, _, gt_functions = extractor.extract_3level_gt(patch_text, base_commit)

        if expected_files:
            for item in expected_files:
                if item not in gt_files:
                    gt_files.append(item)

        if not gt_functions and not gt_files:
            return {}

        mapping_context: dict[str, Any] = {}
        for old_file in gt_files:
            modern_file = self.trace_file_to_modern(old_file, base_commit)
            funcs_in_file = [item.split("::")[1] for item in gt_functions if item.startswith(f"{old_file}::")]

            if modern_file == "DELETED":
                reborn_file = self.trace_by_death_commit(old_file)
                if reborn_file:
                    modern_file = reborn_file

            if modern_file == "DELETED" and funcs_in_file:
                first_function = funcs_in_file[0].split(".")[-1]
                fallback_files = self.grep_element_in_head(first_function)
                if fallback_files:
                    fallback_files.sort(
                        key=lambda item: difflib.SequenceMatcher(None, old_file, item).ratio(),
                        reverse=True,
                    )
                    modern_file = fallback_files[0]

            if modern_file == "DELETED":
                migrated_file = self.advanced_file_migration_search(base_commit, old_file)
                if migrated_file:
                    modern_file = migrated_file

            if modern_file == "DELETED":
                mapping_context.setdefault("DELETED_FILES", [])
                for old_function in funcs_in_file:
                    mapping_context["DELETED_FILES"].append(
                        {
                            "old_file": old_file,
                            "old_function": old_function,
                            "status": "DEAD - Needs historical redirect edge",
                        }
                    )
                continue

            modern_functions = self.get_modern_functions_in_file(modern_file)
            if modern_file not in mapping_context:
                mapping_context[modern_file] = {
                    "old_file_origins": set(),
                    "historical_functions_modified": set(),
                    "available_modern_functions": modern_functions,
                }

            mapping_context[modern_file]["old_file_origins"].add(old_file)
            for old_function in funcs_in_file:
                mapping_context[modern_file]["historical_functions_modified"].add(old_function)

        for file_path in mapping_context:
            if file_path == "DELETED_FILES":
                continue
            mapping_context[file_path]["historical_functions_modified"] = list(
                mapping_context[file_path]["historical_functions_modified"]
            )
            mapping_context[file_path]["old_file_origins"] = list(mapping_context[file_path]["old_file_origins"])

        return mapping_context


class PRKnowledgeDistiller:
    """Distill searchable knowledge from an issue, patch, and time mapping."""

    def __init__(self, model_name: str = "gpt-4o"):
        self._init_llm(model_name)

    def _init_llm(self, model_name: str) -> None:
        clean_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
        openai_base = os.getenv("OPENAI_API_BASE")

        if ChatOpenAI is None:
            raise ImportError("PR distillation requires langchain_openai and langchain_core.")

        if "claude" in clean_model_name.lower():
            target_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            target_key = os.getenv("OPENAI_API_KEY")

        self.llm = ChatOpenAI(
            model=clean_model_name,
            temperature=0.0,
            base_url=openai_base,
            api_key=target_key,
        )

    def _build_messages(self, system_prompt: str, human_prompt: str, retry: bool = False) -> list[Any]:
        if not retry:
            return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

        retry_prompt = (
            human_prompt
            + "\n\n# FORMAT RETRY\n"
            + "Your previous answer could not be parsed.\n"
            + "Return RAW JSON ARRAY ONLY.\n"
            + "Do NOT use markdown fences.\n"
            + "Do NOT add explanations.\n"
            + "The first character must be '[' and the last character must be ']'."
        )
        return [SystemMessage(content=system_prompt), HumanMessage(content=retry_prompt)]

    def _save_debug_response(
        self,
        content: str,
        system_prompt: str,
        human_prompt: str,
        attempt: int,
    ) -> str | None:
        log_root = os.getenv("TRAINER_LOG_ROOT")
        if not log_root:
            return None

        debug_dir = os.path.join(log_root, "distiller_raw")
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(debug_dir, f"distiller_attempt_{attempt + 1}_{timestamp}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# SYSTEM PROMPT\n")
            handle.write(system_prompt)
            handle.write("\n\n# HUMAN PROMPT\n")
            handle.write(human_prompt)
            handle.write("\n\n# RAW RESPONSE\n")
            handle.write(content or "")
        return path

    def _extract_json_candidates(self, content: str) -> list[str]:
        candidates: list[str] = []
        if not content:
            return candidates

        fence_matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        candidates.extend([item.strip() for item in fence_matches if item.strip()])

        stripped = content.strip()
        if stripped:
            candidates.append(stripped)

        array_match = re.search(r"\[\s*{.*}\s*\]", content, re.DOTALL)
        if array_match:
            candidates.append(array_match.group(0).strip())

        first_bracket = content.find("[")
        last_bracket = content.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
            candidates.append(content[first_bracket : last_bracket + 1].strip())

        seen: set[str] = set()
        unique_candidates: list[str] = []
        for item in candidates:
            if item not in seen:
                seen.add(item)
                unique_candidates.append(item)
        return unique_candidates

    def _sanitize_candidate(self, candidate: str) -> str:
        cleaned = candidate.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return cleaned.strip()

    def _normalize_distilled_data(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, dict):
            if isinstance(data.get("items"), list):
                data = data["items"]
            elif isinstance(data.get("results"), list):
                data = data["results"]
            else:
                data = [data]

        if not isinstance(data, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            file_path = item.get("file")
            function_name = item.get("function", "FILE_SUMMARY")
            rationale = item.get("rationale", "")
            keywords = item.get("keywords", [])

            if isinstance(keywords, str):
                keywords = [part.strip() for part in re.split(r"[,;|]", keywords) if part.strip()]
            elif not isinstance(keywords, list):
                keywords = []

            keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
            if not file_path or not rationale:
                continue

            normalized.append(
                {
                    "file": str(file_path).strip(),
                    "function": str(function_name).strip() if function_name else "FILE_SUMMARY",
                    "keywords": keywords,
                    "rationale": str(rationale).strip(),
                }
            )
        return normalized

    def _parse_distilled_response(self, content: str) -> list[dict[str, Any]]:
        for candidate in self._extract_json_candidates(content):
            sanitized = self._sanitize_candidate(candidate)
            try:
                parsed = json.loads(sanitized)
            except json.JSONDecodeError:
                continue
            normalized = self._normalize_distilled_data(parsed)
            if normalized:
                return normalized
        return []

    def distill(self, issue_text: str, patch_text: str, time_mapping: dict[str, Any]) -> list[dict[str, str]]:
        if not time_mapping:
            return []

        mapping_text = json.dumps(time_mapping, indent=2)
        target_files = [key for key in time_mapping.keys() if key != "DELETED_FILES"]
        for dead_file in time_mapping.get("DELETED_FILES", []):
            target_files.append(dead_file["old_file"])

        system_prompt = f"""
You are a Senior Software Architect distilling tacit knowledge from a historical GitHub Issue and its Git Patch.
Your output will be used to inject searchable metadata into a Code Knowledge Graph.

# YOUR MISSION
I will provide you with a "Time Mapping Context". It contains:
1. Modern Files: Historical functions modified + CURRENTLY AVAILABLE functions.
2. DELETED_FILES: Functions that were removed.

# STRICT RULES
1. Keywords Extraction (CRITICAL): Extract 3-8 specific technical terms, API names, algorithm concepts, or error types. Do NOT use generic tags like 'bug_fix' or 'update'.
2. File-Level: Summarize the overall responsibility for each file. Set `function` to "FILE_SUMMARY".
3. Function-Level: Map the capability to a SPECIFIC modern function from the available list.
4. COMPLETENESS: You MUST output at least one JSON object for EACH of the target files: {json.dumps(target_files)}.

# OUTPUT FORMAT
Return ONLY a RAW JSON list.
Do NOT wrap it in markdown fences.
Do NOT add any explanation before or after the JSON.
The first character must be '[' and the last character must be ']'.
Example:
[
  {{
    "file": "path/to/modern_or_deleted_file.py",
    "function": "FILE_SUMMARY",
    "keywords": ["API_Name", "Concept", "ErrorType"],
    "rationale": "Overall architectural responsibility and changes..."
  }},
  {{
    "file": "path/to/modern_or_deleted_file.py",
    "function": "specific_function_name",
    "keywords": ["Specific_Var", "Logic_Concept"],
    "rationale": "Specific logic fix and capability..."
  }}
]
"""

        human_prompt = f"""
# TIME MAPPING CONTEXT
{mapping_text}

# ISSUE DESCRIPTION
{issue_text[:2500]}

# GIT PATCH (DIFF)
{patch_text[:5000]}
"""

        for attempt in range(2):
            try:
                logger.info("🧪 [Distiller] LLM is analyzing the PR history...")
                response = self.llm.invoke(self._build_messages(system_prompt, human_prompt, retry=attempt > 0))
                distilled_data = self._parse_distilled_response(response.content)
                if distilled_data:
                    return distilled_data
                debug_path = self._save_debug_response(response.content, system_prompt, human_prompt, attempt)
                if debug_path:
                    logger.error("❌ [Distiller] Raw response saved to %s", debug_path)
                logger.error("❌ [Distiller] Failed to parse JSON.")
            except Exception as exc:
                logger.error("❌ [Distiller] Distillation failed: %s", exc)
        return []
