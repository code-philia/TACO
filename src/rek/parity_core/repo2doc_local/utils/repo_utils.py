"""
Repository utility helpers.

Provide helper functions for repository analysis.
"""

import os
from pathlib import Path
from collections import Counter


# Mapping from file extensions to programming languages
EXTENSION_LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript (React)",
    ".ts": "TypeScript",
    ".tsx": "TypeScript (React)",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".cpp": "C++",
    ".c": "C",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".vue": "Vue",
    ".svelte": "Svelte",
}


def get_file_language(file_path: str) -> str:
    """
    Get the programming language from a file extension
    
    Args:
        file_path: File path
    
    Returns:
        Programming language name
    """
    ext = Path(file_path).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, "Unknown")


def get_repo_language(repo_path: str) -> dict[str, int]:
    """
    Analyze the primary languages in the repository
    
    Args:
        repo_path: Repository path
    
    Returns:
        Dictionary of languages and file counts
    """
    language_counter = Counter()
    
    for root, _, files in os.walk(repo_path):
        # Skip common excluded directories
        if any(excluded in root for excluded in [
            ".git", "node_modules", "__pycache__", ".venv", "venv"
        ]):
            continue
        
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in EXTENSION_LANGUAGE_MAP:
                language = EXTENSION_LANGUAGE_MAP[ext]
                language_counter[language] += 1
    
    return dict(language_counter.most_common())
