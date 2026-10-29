"""
Code analysis tools.

Provide structural analysis helpers for functions, classes, and file outlines.
"""

import ast
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


def _parse_python_file(file_path: Path) -> Optional[ast.Module]:
    """Parse a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return ast.parse(content)
    except (SyntaxError, Exception):
        return None


def _get_docstring(node) -> str:
    """Get the docstring for a node."""
    docstring = ast.get_docstring(node)
    return docstring if docstring else ""


def _get_function_signature(node: ast.FunctionDef) -> str:
    """Get the function signature."""
    args = []
    
    # Positional parameters.
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        args.append(arg_str)
    
    # *args
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    
    # **kwargs
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    
    # Return annotation.
    returns = ""
    if node.returns:
        returns = f" -> {ast.unparse(node.returns)}"
    
    return f"def {node.name}({', '.join(args)}){returns}"


@tool
def get_file_outline(file_path: str, repo_path: str) -> str:
    """
    Get the code outline of a file, including classes and functions.
    
    Args:
        file_path: File path relative to the repository root
        repo_path: Repository root path
    
    Returns:
        File outline
    """
    try:
        full_path = Path(repo_path) / file_path
        
        if not full_path.exists():
            return f"Error: file does not exist: {file_path}"
        
        if not full_path.is_file():
            return f"Error: path is not a file: {file_path}"
        
        # Only Python files use AST parsing for now.
        if full_path.suffix not in [".py"]:
            # Use regex matching for non-Python files.
            return _get_outline_regex(full_path, file_path)
        
        tree = _parse_python_file(full_path)
        if tree is None:
            return f"Error: failed to parse file: {file_path}"
        
        lines = [f"=== File outline: {file_path} ===\n"]
        
        # Extract imports.
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        if imports:
            lines.append("## Imports:")
            for imp in imports[:10]:  # Limit displayed imports.
                lines.append(f"  - {imp}")
            if len(imports) > 10:
                lines.append(f"  - ... remaining {len(imports) - 10} imports")
            lines.append("")
        
        # Extract classes.
        classes = [node for node in ast.iter_child_nodes(tree) 
                   if isinstance(node, ast.ClassDef)]
        
        if classes:
            lines.append("## Classes:")
            for cls in classes:
                docstring = _get_docstring(cls)
                doc_preview = docstring[:100] + "..." if len(docstring) > 100 else docstring
                lines.append(f"  class {cls.name}")
                if doc_preview:
                    lines.append(f"    \"\"\"{doc_preview}\"\"\"")
                
                # Class methods.
                methods = [node for node in ast.iter_child_nodes(cls) 
                          if isinstance(node, ast.FunctionDef)]
                for method in methods[:5]:  # Limit displayed methods.
                    lines.append(f"    - {_get_function_signature(method)}")
                if len(methods) > 5:
                    lines.append(f"    - ... remaining {len(methods) - 5} methods")
                lines.append("")
        
        # Extract top-level functions.
        functions = [node for node in ast.iter_child_nodes(tree) 
                     if isinstance(node, ast.FunctionDef)]
        
        if functions:
            lines.append("## Functions:")
            for func in functions:
                docstring = _get_docstring(func)
                doc_preview = docstring[:80] + "..." if len(docstring) > 80 else docstring
                lines.append(f"  {_get_function_signature(func)}")
                if doc_preview:
                    lines.append(f"    \"\"\"{doc_preview}\"\"\"")
            lines.append("")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"Error: failed to get file outline: {e}"


def _get_outline_regex(full_path: Path, file_path: str) -> str:
    """Get an outline for non-Python files with regex heuristics."""
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        lines = [f"=== File outline: {file_path} ===\n"]
        
        # JavaScript and TypeScript functions and classes.
        if full_path.suffix in [".js", ".jsx", ".ts", ".tsx"]:
            # Classes.
            class_pattern = r"(?:export\s+)?class\s+(\w+)"
            classes = re.findall(class_pattern, content)
            if classes:
                lines.append("## Classes:")
                for cls in classes[:10]:
                    lines.append(f"  - {cls}")
                lines.append("")
            
            # Functions.
            func_pattern = r"(?:export\s+)?(?:async\s+)?function\s+(\w+)"
            functions = re.findall(func_pattern, content)
            if functions:
                lines.append("## Functions:")
                for func in functions[:15]:
                    lines.append(f"  - {func}")
                lines.append("")
            
            # Arrow functions and exported constants.
            arrow_pattern = r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\("
            arrows = re.findall(arrow_pattern, content)
            if arrows:
                lines.append("## Arrow functions/constants:")
                for arrow in arrows[:10]:
                    lines.append(f"  - {arrow}")
                lines.append("")
        
        # Java classes and methods.
        elif full_path.suffix in [".java"]:
            class_pattern = r"(?:public\s+)?class\s+(\w+)"
            classes = re.findall(class_pattern, content)
            if classes:
                lines.append("## Classes:")
                for cls in classes:
                    lines.append(f"  - {cls}")
                lines.append("")
            
            method_pattern = r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+)\s+(\w+)\s*\("
            methods = re.findall(method_pattern, content)
            if methods:
                lines.append("## Methods:")
                for method in methods[:20]:
                    if method not in ["if", "for", "while", "switch"]:
                        lines.append(f"  - {method}")
                lines.append("")
        
        else:
            lines.append("Outline analysis is not supported for this file type yet.")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"Error: failed to get file outline: {e}"


@tool
def get_function_info(
    file_path: str, 
    function_name: str, 
    repo_path: str
) -> str:
    """
    Get detailed information for a specific function.
    
    Args:
        file_path: File path relative to the repository root
        function_name: Function name
        repo_path: Repository root path
    
    Returns:
        Detailed function information
    """
    try:
        full_path = Path(repo_path) / file_path
        
        if not full_path.exists():
            return f"Error: file does not exist: {file_path}"
        
        if full_path.suffix != ".py":
            # Use regex matching for non-Python files.
            return _get_function_regex(full_path, function_name, file_path)
        
        tree = _parse_python_file(full_path)
        if tree is None:
            return f"Error: failed to parse file: {file_path}"
        
        # Search functions, including class methods.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                
                # Read the function source.
                start = node.lineno - 1
                end = node.end_lineno
                source = "".join(lines[start:end])
                
                result = f"=== Function: {function_name} ===\n"
                result += f"File: {file_path}\n"
                result += f"Lines: {node.lineno}-{node.end_lineno}\n"
                result += f"Signature: {_get_function_signature(node)}\n\n"
                
                docstring = _get_docstring(node)
                if docstring:
                    result += f"Docstring:\n{docstring}\n\n"
                
                result += f"Source code:\n```python\n{source}\n```"
                
                return result
        
        return f"Error: function {function_name} was not found in file {file_path}"
    
    except Exception as e:
        return f"Error: failed to get function information: {e}"


def _get_function_regex(full_path: Path, function_name: str, file_path: str) -> str:
    """Get function information with regex heuristics."""
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.splitlines()
        
        # Simple function lookup.
        pattern = rf"(?:export\s+)?(?:async\s+)?function\s+{function_name}"
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                # Capture a local source window around the match.
                start = max(0, i)
                end = min(len(lines), i + 30)
                source = "\n".join(lines[start:end])
                
                result = f"=== Function: {function_name} ===\n"
                result += f"File: {file_path}\n"
                result += f"Lines: {i + 1}\n\n"
                result += f"Source code:\n```\n{source}\n```"
                
                return result
        
        return f"Error: function {function_name} was not found in file {file_path}"
    
    except Exception as e:
        return f"Error: failed to get function information: {e}"


@tool
def get_class_info(
    file_path: str, 
    class_name: str, 
    repo_path: str
) -> str:
    """
    Get detailed information for a specific class.
    
    Args:
        file_path: File path relative to the repository root
        class_name: Class name
        repo_path: Repository root path
    
    Returns:
        Detailed class information
    """
    try:
        full_path = Path(repo_path) / file_path
        
        if not full_path.exists():
            return f"Error: file does not exist: {file_path}"
        
        if full_path.suffix != ".py":
            return f"Error: class analysis is not supported for non-Python files yet"
        
        tree = _parse_python_file(full_path)
        if tree is None:
            return f"Error: failed to parse file: {file_path}"
        
        # Search classes.
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                result = f"=== Class: {class_name} ===\n"
                result += f"File: {file_path}\n"
                result += f"Lines: {node.lineno}-{node.end_lineno}\n"
                
                # Base classes.
                bases = [ast.unparse(base) for base in node.bases]
                if bases:
                    result += f"Bases: {', '.join(bases)}\n"
                
                result += "\n"
                
                # Docstring.
                docstring = _get_docstring(node)
                if docstring:
                    result += f"Docstring:\n{docstring}\n\n"
                
                # Method list.
                methods = [n for n in ast.iter_child_nodes(node) 
                          if isinstance(n, ast.FunctionDef)]
                
                if methods:
                    result += "Methods:\n"
                    for method in methods:
                        sig = _get_function_signature(method)
                        doc = _get_docstring(method)
                        doc_preview = doc[:50] + "..." if len(doc) > 50 else doc
                        result += f"  - {sig}\n"
                        if doc_preview:
                            result += f"      \"{doc_preview}\"\n"
                
                # Class attributes.
                attrs = []
                for n in ast.iter_child_nodes(node):
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                        attrs.append(n.target.id)
                    elif isinstance(n, ast.Assign):
                        for target in n.targets:
                            if isinstance(target, ast.Name):
                                attrs.append(target.id)
                
                if attrs:
                    result += f"\nAttributes: {', '.join(attrs)}\n"
                
                return result
        
        return f"Error: class {class_name} was not found in file {file_path}"
    
    except Exception as e:
        return f"Error: failed to get class information: {e}"
