"""
File operation tools.

Provide file content access and directory tree generation helpers.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
import pathspec


def _get_gitignore_spec(repo_path: str) -> Optional[pathspec.PathSpec]:
    """Get the .gitignore rules"""
    gitignore_path = Path(repo_path) / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
            patterns = f.read().splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    return None


def _should_exclude(path: str, repo_path: str, exclude_patterns: list[str]) -> bool:
    """Check whether this path should be excluded"""
    rel_path = os.path.relpath(path, repo_path)
    
    # Default excluded directories
    default_excludes = [
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        "dist", "build", ".idea", ".vscode"
    ]
    
    for part in Path(rel_path).parts:
        if part in default_excludes:
            return True
    
    # Check custom exclude rules
    if exclude_patterns:
        spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude_patterns)
        if spec.match_file(rel_path):
            return True
    
    return False


@tool
def get_file_content(file_path: str, repo_path: str) -> str:
    """
    Get the content of a specific file.
    
    Args:
        file_path: File path relative to the repository root
        repo_path: Repository root path
    
    Returns:
        File content, or an error message when the file is missing or unreadable
    """
    try:
        full_path = Path(repo_path) / file_path
        
        if not full_path.exists():
            return f"Error: file does not exist: {file_path}"
        
        if not full_path.is_file():
            return f"Error: path is not a file: {file_path}"
        
        # Check the file size.
        file_size = full_path.stat().st_size
        if file_size > 100 * 1024:  # 100KB
            return f"Error: file is too large ({file_size} bytes), exceeds the 100KB limit"
        
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        return f"=== File: {file_path} ===\n{content}"
    
    except Exception as e:
        return f"Error: failed to read file: {e}"


@tool
def get_directory_tree(
    repo_path: str, 
    max_depth: int = 3,
    include_files: bool = True
) -> str:
    """
    Get the directory tree of the repository.
    
    Args:
        repo_path: Repository root path
        max_depth: Maximum depth, default is 3
        include_files: Whether to include files, default is True
    
    Returns:
        Tree representation of the directory structure
    """
    try:
        repo = Path(repo_path)
        if not repo.exists():
            return f"Error: directory does not exist: {repo_path}"
        
        lines = [repo.name + "/"]
        
        def add_tree(path: Path, prefix: str, depth: int):
            if depth > max_depth:
                return
            
            try:
                entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            except PermissionError:
                return
            
            # Filter entries
            filtered_entries = []
            for entry in entries:
                if _should_exclude(str(entry), repo_path, []):
                    continue
                if entry.is_file() and not include_files:
                    continue
                filtered_entries.append(entry)
            
            for i, entry in enumerate(filtered_entries):
                is_last = i == len(filtered_entries) - 1
                connector = "└── " if is_last else "├── "
                
                if entry.is_dir():
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    extension = "    " if is_last else "│   "
                    add_tree(entry, prefix + extension, depth + 1)
                else:
                    lines.append(f"{prefix}{connector}{entry.name}")
        
        add_tree(repo, "", 1)
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"Error: failed to generate directory tree: {e}"


@tool
def list_files_by_extension(
    repo_path: str, 
    extension: str,
    max_files: int = 50
) -> str:
    """
    List repository files by extension.
    
    Args:
        repo_path: Repository root path
        extension: File extension, such as ".py" or ".js"
        max_files: Maximum number of files to return, default is 50
    
    Returns:
        List of files matching the extension
    """
    try:
        repo = Path(repo_path)
        if not repo.exists():
            return f"Error: directory does not exist: {repo_path}"
        
        # Ensure the extension starts with a dot
        if not extension.startswith("."):
            extension = "." + extension
        
        files = []
        for root, _, filenames in os.walk(repo):
            if _should_exclude(root, repo_path, []):
                continue
            
            for filename in filenames:
                if filename.endswith(extension):
                    rel_path = os.path.relpath(os.path.join(root, filename), repo_path)
                    files.append(rel_path)
                    
                    if len(files) >= max_files:
                        break
            
            if len(files) >= max_files:
                break
        
        if not files:
            return f"No files found with extension {extension}"
        
        result = f"=== Files with extension {extension} (total {len(files)})===\n"
        result += "\n".join(files)
        
        if len(files) >= max_files:
            result += f"\n\n(Reached max file count {max_files}, there may be more files)"
        
        return result
    
    except Exception as e:
        return f"Error: failed to list files: {e}"
