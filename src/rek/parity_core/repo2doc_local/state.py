"""
Repo2Doc Agent state management module

Define the state structures used by the LangGraph agent workflow.
"""

from typing import TypedDict, Optional, Annotated
from dataclasses import dataclass, field
from operator import add


@dataclass
class ToolCall:
    """Tool call record."""
    tool_name: str
    arguments: dict
    result: str = ""
    success: bool = True


@dataclass
class ExplorationRecord:
    """Exploration record"""
    iteration: int
    action: str
    findings: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class AgentState(TypedDict):
    """
    Repo2Doc Agent workflow state
    
    This is the core state object for the LangGraph agent workflow,
    passed and updated between workflow nodes.
    """
    # Input arguments
    repo_path: str
    config_path: Optional[str]
    
    # Information collected during initialization
    readme_content: str                         # README content
    directory_tree: str                         # Directory tree
    project_files: list[str]                    # Project config file list
    high_level_info: str                        # High-level summary
    
    # Agent exploration stage
    messages: Annotated[list, add]              # Message history used by the agent
    exploration_history: list[ExplorationRecord]  # Exploration history
    current_tool_results: str
    
    # Document generation stage
    current_document: str                       # Current document
    document_versions: list[str]                # Document version history
    
    # Completeness check
    is_complete: bool                           # Whether the document is complete
    missing_parts: list[str]                    # Missing parts
    confidence_score: float                     # Confidence score (0-1)
    
    # Loop control
    iteration_count: int
    max_iterations: int
    
    # Status information
    status: str                                 # Current status
    error: Optional[str]
    
    # LLM usage statistics
    llm_usage: dict                             # LLM token usage


def create_initial_state(
    repo_path: str, 
    config_path: Optional[str] = None,
    max_iterations: int = 10
) -> AgentState:
    """
    Create the initial state
    
    Args:
        repo_path: Repository path
        config_path: Optional config path.
        max_iterations: Maximum iteration count.
    
    Returns:
        Initialized state object
    """
    return AgentState(
        repo_path=repo_path,
        config_path=config_path,
        readme_content="",
        directory_tree="",
        project_files=[],
        high_level_info="",
        messages=[],
        exploration_history=[],
        current_tool_results="",
        current_document="",
        document_versions=[],
        is_complete=False,
        missing_parts=[],
        confidence_score=0.0,
        iteration_count=0,
        max_iterations=max_iterations,
        status="initialized",
        error=None,
        llm_usage={
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "calls": []
        },
    )
