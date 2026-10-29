"""
LangGraph agent workflow definition

Define the full Repo2Doc Agent workflow.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from rek.parity_core.repo2doc_local.state import AgentState, create_initial_state
from rek.parity_core.repo2doc_local.config_loader import Config, setup_logging
from rek.parity_core.repo2doc_local.nodes.init_node import init_node
from rek.parity_core.repo2doc_local.nodes.doc_node import generate_doc_node
from rek.parity_core.repo2doc_local.nodes.check_node import check_completeness_node
from rek.parity_core.repo2doc_local.nodes.tool_node import tool_execution_node
from rek.parity_core.repo2doc_local.nodes.save_node import save_output_node


logger = logging.getLogger(__name__)


class Repo2DocAgentWorkflow:
    """
    Repo2Doc Agent LangGraph workflow
    
    Workflow:
    1. init_node: Initialize and collect high-level information
    2. doc_node: Generate or update the document
    3. check_node: Check completeness
    4. tool_node: Execute tool calls when more exploration is needed
    5. Loop back to doc_node until the document is complete
    """
    
    def __init__(self, config: Config):
        """
        Initialize the workflow
        
        Args:
            config: Config object
        """
        self.config = config
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow graph
        
        Returns:
            Compiled workflow graph
        """
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("init", self._wrap_node(init_node))
        workflow.add_node("generate_doc", self._wrap_node(generate_doc_node))
        workflow.add_node("check_completeness", self._wrap_node(check_completeness_node))
        workflow.add_node("execute_tools", self._wrap_node(tool_execution_node))
        workflow.add_node("save_output", self._wrap_node(save_output_node))
        
        # Set the entry point
        workflow.set_entry_point("init")
        
        # Add edges
        # init -> generate_doc
        workflow.add_conditional_edges(
            "init",
            self._check_error,
            {
                "continue": "generate_doc",
                "error": END,
            }
        )
        
        # generate_doc -> check_completeness
        workflow.add_conditional_edges(
            "generate_doc",
            self._check_error,
            {
                "continue": "check_completeness",
                "error": END,
            }
        )
        
        # check_completeness -> save_output (if complete) or execute_tools (if not)
        workflow.add_conditional_edges(
            "check_completeness",
            self._route_after_check,
            {
                "complete": "save_output",
                "explore": "execute_tools",
                "error": END,
            }
        )
        
        # execute_tools -> generate_doc (loop)
        workflow.add_conditional_edges(
            "execute_tools",
            self._check_error,
            {
                "continue": "generate_doc",
                "error": END,
            }
        )
        
        # save_output -> END
        workflow.add_edge("save_output", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _wrap_node(self, node_func):
        """
        Wrap a node function and inject the config
        
        Args:
            node_func: Node function
        
        Returns:
            Wrapped function
        """
        def wrapped(state: AgentState) -> AgentState:
            return node_func(state, self.config)
        return wrapped
    
    def _check_error(self, state: AgentState) -> Literal["continue", "error"]:
        """Check whether the workflow has entered an error state."""
        if state.get("status") == "error":
            logger.error(f"Workflow error: {state.get('error')}")
            return "error"
        return "continue"
    
    def _route_after_check(self, state: AgentState) -> Literal["complete", "explore", "error"]:
        """Route after the completeness check step."""
        if state.get("status") == "error":
            return "error"
        
        if state.get("is_complete", False):
            return "complete"
        
        return "explore"
    
    def run(self, repo_path: str, config_path: str = None) -> AgentState:
        """
        Run the workflow
        
        Args:
            repo_path: Repository path
            config_path: Optional config file path.
        
        Returns:
            Final state.
        """
        logger.info(f"Start processing repository: {repo_path}")
        
        # Create the initial state
        initial_state = create_initial_state(
            repo_path, 
            config_path,
            max_iterations=self.config.agent.max_iterations
        )
        
        # Run the workflow with a recursion limit that supports repeated iterations.
        recursion_limit = self.config.agent.max_iterations * 10 + 20
        final_state = self.graph.invoke(
            initial_state,
            {"recursion_limit": recursion_limit}
        )
        
        # Log the result summary.
        if final_state.get("status") == "completed":
            logger.info("Workflow completed")
            logger.info(f"   Iteration count: {final_state.get('iteration_count', 0)}")
            logger.info(f"   Document length: {len(final_state.get('current_document', ''))} characters")
            logger.info(f"   Confidence: {final_state.get('confidence_score', 0):.2f}")
        else:
            logger.error(f"Workflow failed: {final_state.get('error')}")
        
        return final_state


def create_workflow(config_path: str = None) -> Repo2DocAgentWorkflow:
    """
    Create a workflow instance
    
    Args:
        config_path: Optional config file path.
    
    Returns:
        Workflow instance
    """
    # Load config
    config = Config.load(config_path)
    
    # Set up logging
    setup_logging(config.logging)
    
    # Create the workflow
    return Repo2DocAgentWorkflow(config)


def run_workflow(repo_path: str, config_path: str = None) -> AgentState:
    """
    Convenience helper that creates and runs the workflow.
    
    Args:
        repo_path: Repository path
        config_path: Optional config file path.
    
    Returns:
        Final state.
    """
    workflow = create_workflow(config_path)
    return workflow.run(repo_path, config_path)
