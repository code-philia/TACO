"""
Completeness check node

Evaluate document completeness and decide whether more exploration is needed.
"""

import logging
import json
import re
from typing import Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None # Guard against missing optional dependency
from rek.parity_core.repo2doc_local.state import AgentState
from rek.parity_core.repo2doc_local.config_loader import Config
from rek.parity_core.repo2doc_local.prompts.agent_prompts import CHECK_COMPLETENESS_PROMPT
import os

logger = logging.getLogger(__name__)


def check_completeness_node(state: AgentState, config: Config) -> AgentState:
    """
    Completeness check node.

    1. Analyze document quality with the LLM
    2. Identify missing or uncertain parts
    3. Decide whether more exploration is needed
    
    Args:
        state: Current status
        config: Config object
    
    Returns:
        Updated state
    """
    logger.info("Checking document completeness...")
    
    # Stop once the maximum iteration count is reached.
    if state["iteration_count"] >= state["max_iterations"]:
        logger.info(f"Reached max iterations {state['max_iterations']}, stopping exploration")
        state["is_complete"] = True
        state["status"] = "completed"
        return state
    
    # Create the LLM client.
    llm = _create_llm(config)
    
    # Initialize or retrieve LLM usage stats.
    llm_usage = state.get("llm_usage", {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "calls": []
    })
    
    # Evaluate completeness.
    is_complete, confidence, missing_parts, suggested_tools, usage = _evaluate_completeness(
        llm,
        state["current_document"],
        state["high_level_info"],
        state["iteration_count"],
        config
    )
    
    # Record token usage.
    if usage:
        llm_usage["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
        llm_usage["total_completion_tokens"] += usage.get("completion_tokens", 0)
        llm_usage["total_tokens"] += usage.get("total_tokens", 0)
        llm_usage["calls"].append({
            "iteration": state["iteration_count"],
            "type": "check_completeness",
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        })
    
    # Use the configured confidence threshold.
    confidence_threshold = config.agent.confidence_threshold
    
    # Mark the document as complete if it reaches the threshold.
    should_complete = is_complete or (
        confidence >= confidence_threshold
    )
    
    # Update the workflow state.
    state["is_complete"] = should_complete
    state["confidence_score"] = confidence
    state["missing_parts"] = missing_parts
    state["llm_usage"] = llm_usage
    
    if should_complete:
        state["status"] = "completed"
        logger.info(f"Document is complete, confidence: {confidence:.2f}")
    else:
        state["status"] = "needs_exploration"
        # Store suggested tool calls for the next step.
        state["current_tool_results"] = json.dumps(suggested_tools, ensure_ascii=False)
        logger.info(f"Document is incomplete and needs more exploration. Missing parts: {missing_parts}")
    
    return state


def _create_llm(config):
    """
    Select the correct client based on the configured model type.
    """
    # 1. Read configuration.
    llm_config = config.llm
    model_name = llm_config.model
    
    # Normalize the model name by removing the proxy prefix.
    clean_model_name = model_name
    if "/" in model_name:
        clean_model_name = model_name.split("/", 1)[1]

    # 2. Resolve environment variables.
    anthropic_base = os.getenv("ANTHROPIC_API_BASE") or llm_config.base_url
    openai_base = os.getenv("OPENAI_API_BASE") or llm_config.base_url
    
    api_key = llm_config.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    logger.info(f"🔧 Initializing LLM: {clean_model_name}")

    # 3. Route by model family.
    if "claude" in clean_model_name.lower():
        if ChatAnthropic is None:
            raise ImportError("Please install `langchain-anthropic`.")
        
        logger.info(f"   -> Using ChatAnthropic (Base: {anthropic_base})")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=clean_model_name,
            temperature=0.1,
            max_retries=3,
            default_request_timeout=120.0, 
            base_url=anthropic_base,
            api_key=api_key,
            streaming=True
        )
    else:
        # GPT, DeepSeek, and Qwen models use the OpenAI-compatible interface.
        target_base = openai_base
        if not target_base: 
            target_base = os.getenv("OPENAI_BASE_URL", "")

        logger.info(f"   -> Using ChatOpenAI (Base: {target_base})")
        return ChatOpenAI(
            model=clean_model_name,
            temperature=0.1,
            max_retries=3,
            request_timeout=120,
            base_url=target_base,
            api_key=api_key,
            max_tokens=4096
        )

def _extract_usage(response) -> dict:
    """Extract token usage from a model response."""
    usage = {}
    if hasattr(response, 'response_metadata') and response.response_metadata:
        token_usage = response.response_metadata.get('token_usage', {})
        usage = {
            "prompt_tokens": token_usage.get('prompt_tokens', 0),
            "completion_tokens": token_usage.get('completion_tokens', 0),
            "total_tokens": token_usage.get('total_tokens', 0)
        }
    return usage


def _evaluate_completeness(
    llm: ChatOpenAI,
    document: str,
    high_level_info: str,
    iteration: int,
    config: Config
) -> Tuple[bool, float, list[str], list[dict], dict]:
    """
    Evaluate document completeness.
    
    Returns:
        (is_complete, confidence_score, missing_parts, suggested_tools, usage)
    """
    messages = [
        SystemMessage(content="You are a requirements document quality evaluator. Return the evaluation result as strict JSON only."),
        HumanMessage(content=CHECK_COMPLETENESS_PROMPT.format(
            document=document,
            high_level_info=high_level_info,
            iteration=iteration,
            max_iterations=config.agent.max_iterations
        ))
    ]
    
    response = llm.invoke(messages)
    usage = _extract_usage(response)
    
    # Parse the model response.
    try:
        # Try to extract a JSON object from the response.
        content = response.content
        
        # Look for a JSON block.
        json_match = re.search(r'\{[\s\S]*\}', content)
        
        if json_match:
            result = json.loads(json_match.group())
        else:
            # Fall back to a conservative default when no JSON is found.
            logger.warning("Unable to parse JSON from the LLM response, using defaults.")
            result = {
                "is_complete": iteration >= 3,
                "confidence_score": 0.7,
                "missing_parts": [],
                "suggested_tools": []
            }
        
        return (
            result.get("is_complete", False),
            result.get("confidence_score", 0.5),
            result.get("missing_parts", []),
            result.get("suggested_tools", []),
            usage
        )
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse evaluation result: {e}")
        # Return a conservative fallback estimate.
        return (False, 0.5, ["Unable to evaluate completeness"], [], usage)
