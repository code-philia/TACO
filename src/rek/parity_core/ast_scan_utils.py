"""
Local copy adapted from `Repo_Reqirement_Knowledge/src/common/ast_scan_utils.py`.

This copy provides only the helpers needed by the CLI-side scanner.
"""

from __future__ import annotations

import ast


def max_lineno(node: ast.AST) -> int:
    """
    Estimate the maximum line number within an AST subtree.
    """
    max_line = getattr(node, "lineno", 0) or 0
    for child in ast.walk(node):
        line = getattr(child, "lineno", 0) or 0
        if line > max_line:
            max_line = line
    return max_line
