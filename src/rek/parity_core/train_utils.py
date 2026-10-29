"""
Local copy adapted from `Repo_Reqirement_Knowledge/src/evolution_pipeline/trainer.py`.

This copy keeps the reusable dataset helpers while replacing the original
LLM-heavy training loop with lightweight utilities.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable


def load_jsonl(path: str) -> list[dict]:
    data: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data.append(json.loads(line))
    return data


def make_batches(data: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for index in range(0, len(data), batch_size):
        yield data[index : index + batch_size]


def filter_ground_truth(files: list[str]) -> list[str]:
    if not isinstance(files, list):
        return []
    allowed_exts = (".py", ".pyx", ".pxd", ".tp")
    return [
        file_path
        for file_path in files
        if "test" not in file_path and "conftest" not in file_path and file_path.endswith(allowed_exts)
    ]


def get_robust_ground_truth(sample: dict) -> list[str]:
    raw_gt: list[str] = []
    for field in ["files_preview", "gold_files_path", "files", "resolved"]:
        value = sample.get(field)
        if isinstance(value, list) and value:
            raw_gt = value
            break

    if not raw_gt and "patch" in sample:
        patch_text = sample.get("patch", "")
        matches = re.findall(r"^(?:--- a/|\+\+\+ b/)(.*)$", patch_text, re.MULTILINE)
        raw_gt = list({match.strip() for match in matches if match.strip() != "dev/null"})

    return raw_gt


def extract_issue_keywords(issue: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z_]\w+\b", (issue or "").lower())
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "when",
        "where",
        "after",
        "before",
        "during",
        "fails",
        "error",
        "issue",
    }
    unique_tokens = []
    seen = set()
    for token in tokens:
        if len(token) < 3 or token in stop_words or token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)
    return unique_tokens[:12]
