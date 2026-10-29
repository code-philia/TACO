"""
Local registry adapted from `Repo_Reqirement_Knowledge/src/tools/__init__.py`.
"""

from rek.parity_core.inference_tools import (
    build_inference_tools,
    get_files_in_feature,
    get_functions_in_file,
    get_l2_features,
    get_root_categories,
    search_by_tags,
    set_inference_db,
)
from rek.parity_core.mutation_tools import (
    inject_dual_index_knowledge,
    set_mutation_db,
    set_repo_path,
)

MUTATION_TOOLS = [inject_dual_index_knowledge]

__all__ = [
    "MUTATION_TOOLS",
    "build_inference_tools",
    "get_files_in_feature",
    "get_functions_in_file",
    "get_l2_features",
    "get_root_categories",
    "inject_dual_index_knowledge",
    "search_by_tags",
    "set_inference_db",
    "set_mutation_db",
    "set_repo_path",
]
