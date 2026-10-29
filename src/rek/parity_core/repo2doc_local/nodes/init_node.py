"""
Initialization node

Collect high-level repository information, including README, directory tree, and config files.
"""

import logging
from pathlib import Path

from rek.parity_core.repo2doc_local.state import AgentState
from rek.parity_core.repo2doc_local.config_loader import Config
from rek.parity_core.repo2doc_local.tools.file_tools import get_directory_tree


logger = logging.getLogger(__name__)


# Common project config files
PROJECT_CONFIG_FILES = [
    "README.md", "readme.md", "README.rst", "README.txt",
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "package-lock.json",
    "pom.xml", "build.gradle", "build.gradle.kts",
    "Cargo.toml",
    "go.mod", "go.sum",
    "Gemfile",
    "composer.json",
    "Makefile", "CMakeLists.txt",
    ".env.example", ".env.sample",
    "docker-compose.yml", "Dockerfile",
    "requirements.txt", "Pipfile",
]

# Documentation directories
DOC_DIRS = ["docs", "doc", "documentation", "wiki"]


def init_node(state: AgentState, config: Config) -> AgentState:
    """
    Initialization node.
    
    1. Read the README file
    2. Build the directory tree
    3. Collect project config files
    4. Assemble the high-level summary
    
    Args:
        state: Current state.
        config: Config object.

    Returns:
        Updated state.
    """
    logger.info("Initializing and collecting repository information...")
    
    repo_path = state["repo_path"]
    repo = Path(repo_path)
    
    if not repo.exists():
        state["status"] = "error"
        state["error"] = f"Repository path does not exist: {repo_path}"
        return state
    
    # 1. Read the README file.
    readme_content = _read_readme(repo)
    state["readme_content"] = readme_content
    logger.info(f"README content length: {len(readme_content)} characters")
    
    # 2. Build the directory tree.
    directory_tree = get_directory_tree.invoke({
        "repo_path": repo_path,
        "max_depth": 3,
        "include_files": True
    })
    state["directory_tree"] = directory_tree
    logger.info("Directory tree generation completed")
    
    # 3. Collect project config files
    project_files = _collect_project_files(repo)
    state["project_files"] = project_files
    logger.info(f"Found {len(project_files)} project config files")
    
    # 4. Read key config file contents
    config_contents = _read_config_files(repo, project_files)
    
    # 5. Assemble the high-level summary
    high_level_info = _build_high_level_info(
        readme_content, 
        directory_tree, 
        project_files,
        config_contents
    )
    state["high_level_info"] = high_level_info
    
    state["status"] = "initialized"
    logger.info("Initialization completed")
    
    return state


def _read_readme(repo: Path) -> str:
    """Read the README file"""
    readme_patterns = ["README.md", "readme.md", "README.rst", "README.txt", "README"]
    
    for pattern in readme_patterns:
        readme_path = repo / pattern
        if readme_path.exists():
            try:
                with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # Limit the preview length.
                if len(content) > 10000:
                    content = content[:10000] + "\n\n... (content truncated)"
                return content
            except Exception as e:
                logger.warning(f"Failed to read README: {e}")
    
    return "(README file not found)"


def _collect_project_files(repo: Path) -> list[str]:
    """Collect project config files"""
    found_files = []
    
    for filename in PROJECT_CONFIG_FILES:
        file_path = repo / filename
        if file_path.exists():
            found_files.append(filename)
    
    # Check documentation directories.
    for doc_dir in DOC_DIRS:
        doc_path = repo / doc_dir
        if doc_path.exists() and doc_path.is_dir():
            found_files.append(f"{doc_dir}/")
    
    return found_files


def _read_config_files(repo: Path, project_files: list[str]) -> dict[str, str]:
    """Read key config file contents"""
    config_contents = {}
    
    # Only read important config files
    important_files = [
        "pyproject.toml", "package.json", "pom.xml", 
        "build.gradle", "Cargo.toml", "go.mod"
    ]
    
    for filename in project_files:
        if filename in important_files:
            file_path = repo / filename
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # Limit the preview length.
                if len(content) > 3000:
                    content = content[:3000] + "\n... (content truncated)"
                config_contents[filename] = content
            except Exception:
                pass
    
    return config_contents


def _build_high_level_info(
    readme: str, 
    tree: str, 
    files: list[str],
    config_contents: dict[str, str]
) -> str:
    """Build the high-level repository summary"""
    info = []
    
    info.append("# Repository Overview\n")
    
    # README
    info.append("## README\n")
    if readme and readme != "(README file not found)":
        # Keep only the front portion of the README.
        readme_preview = readme[:3000]
        if len(readme) > 3000:
            readme_preview += "\n\n... (README content truncated)"
        info.append(readme_preview)
    else:
        info.append("(README file not found)")
    info.append("\n")
    
    # Directory tree
    info.append("## Directory Tree\n")
    info.append("```")
    info.append(tree)
    info.append("```\n")
    
    # Project config files
    info.append("## Project Config Files\n")
    for f in files:
        info.append(f"- {f}")
    info.append("\n")
    
    # Key config contents
    if config_contents:
        info.append("## Key Config Contents\n")
        for filename, content in config_contents.items():
            info.append(f"### {filename}\n")
            info.append(f"```")
            info.append(content)
            info.append("```\n")
    
    return "\n".join(info)
