"""
Code search tools.

Provide global code search and import search helpers.
"""

import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


def _should_exclude(path: str, repo_path: str) -> bool:
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
    
    return False


@tool
def search_code(
    query: str, 
    repo_path: str,
    file_pattern: Optional[str] = None,
    max_results: int = 20
) -> str:
    """
    Search code containing the given text.
    
    Args:
        query: Search query
        repo_path: Repository root path
        file_pattern: Optional file pattern, such as "*.py"
        max_results: Maximum number of results, default is 20
    
    Returns:
        Search results with file path, line number, and matched content
    """
    try:
        repo = Path(repo_path)
        if not repo.exists():
            return f"Error: directory does not exist: {repo_path}"
        
        results = []
        
        for root, _, filenames in os.walk(repo):
            if _should_exclude(root, repo_path):
                continue
            
            for filename in filenames:
                # Check the file pattern
                if file_pattern:
                    if not Path(filename).match(file_pattern.lstrip("*")):
                        continue
                
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, repo_path)
                
                # Skip binary files
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue
                
                # Search
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if query.lower() in line.lower():
                        results.append({
                            "file": rel_path,
                            "line": i + 1,
                            "content": line.strip()[:100]
                        })
                        
                        if len(results) >= max_results:
                            break
                
                if len(results) >= max_results:
                    break
            
            if len(results) >= max_results:
                break
        
        if not results:
            return f"No code containing \"{query}\" was found"
        
        output = f"=== Search results for \"{query}\" (total {len(results)}) ===\n\n"
        
        for r in results:
            output += f"{r['file']}:{r['line']}\n"
            output += f"  {r['content']}\n\n"
        
        if len(results) >= max_results:
            output += f"\nReached max result count {max_results}, there may be more matches)"
        
        return output
    
    except Exception as e:
        return f"Error: search failed: {e}"


@tool
def search_imports(
    module_name: str, 
    repo_path: str,
    max_results: int = 20
) -> str:
    """
    Search for files importing the given module.
    
    Args:
        module_name: Module name
        repo_path: Repository root path
        max_results: Maximum number of results, default is 20
    
    Returns:
        List of files importing the module
    """
    try:
        repo = Path(repo_path)
        if not repo.exists():
            return f"Error: directory does not exist: {repo_path}"
        
        results = []
        
        # Python import patterns
        py_patterns = [
            rf"import\s+{re.escape(module_name)}",
            rf"from\s+{re.escape(module_name)}\s+import",
            rf"from\s+\S+\s+import\s+.*{re.escape(module_name)}",
        ]
        
        # JavaScript/TypeScript import patterns
        js_patterns = [
            rf"import\s+.*from\s+['\"].*{re.escape(module_name)}.*['\"]",
            rf"require\s*\(\s*['\"].*{re.escape(module_name)}.*['\"]\s*\)",
        ]
        
        for root, _, filenames in os.walk(repo):
            if _should_exclude(root, repo_path):
                continue
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, repo_path)
                
                # Decide which pattern set to use
                ext = Path(filename).suffix
                if ext == ".py":
                    patterns = py_patterns
                elif ext in [".js", ".jsx", ".ts", ".tsx"]:
                    patterns = js_patterns
                else:
                    continue
                
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue
                
                # Search import statements.
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        results.append({
                            "file": rel_path,
                            "imports": matches[:3]  # Limit matches per file
                        })
                        break
                
                if len(results) >= max_results:
                    break
            
            if len(results) >= max_results:
                break
        
        if not results:
            return f"No import of \"{module_name}\""
        
        output = f"=== Imports of \"{module_name}\" (total {len(results)})===\n\n"
        
        for r in results:
            output += f"{r['file']}\n"
            for imp in r['imports']:
                output += f"  {imp[:80]}\n"
            output += "\n"
        
        if len(results) >= max_results:
            output += f"\nReached max result count {max_results}, there may be more matches)"
        
        return output
    
    except Exception as e:
        return f"Error: import search failed: {e}"
