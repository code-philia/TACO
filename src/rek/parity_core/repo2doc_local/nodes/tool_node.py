"""
Tool execution node

Parse and execute the tool calls selected by the agent.
"""

import logging
import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None # Guard against missing optional dependency
from rek.parity_core.repo2doc_local.state import AgentState, ToolCall, ExplorationRecord
from rek.parity_core.repo2doc_local.config_loader import Config
from rek.parity_core.repo2doc_local.tools.file_tools import get_file_content, list_files_by_extension
from rek.parity_core.repo2doc_local.tools.code_tools import get_file_outline, get_function_info, get_class_info
from rek.parity_core.repo2doc_local.tools.search_tools import search_code, search_imports
from rek.parity_core.repo2doc_local.prompts.agent_prompts import TOOL_SELECTION_PROMPT
import os

logger = logging.getLogger(__name__)


# Tool registry
TOOL_REGISTRY = {
    "get_file_content": get_file_content,
    "get_file_outline": get_file_outline,
    "get_function_info": get_function_info,
    "get_class_info": get_class_info,
    "search_code": search_code,
    "search_imports": search_imports,
    "list_files_by_extension": list_files_by_extension,
}


def tool_execution_node(state: AgentState, config: Config) -> AgentState:
    """
    Tool execution node.

    1. Read the suggested tool calls from the previous step or ask the LLM
    2. Execute the tool calls
    3. Collect results and update the state
    
    Args:
        state: Current status
        config: Config object
    
    Returns:
        Updated state
    """
    logger.info("Starting tool execution...")
    
    repo_path = state["repo_path"]
    
    # Try to load suggested tool calls from the current state.
    suggested_tools = []
    if state["current_tool_results"]:
        try:
            suggested_tools = json.loads(state["current_tool_results"])
            if isinstance(suggested_tools, list):
                pass
            elif isinstance(suggested_tools, dict) and "tool_calls" in suggested_tools:
                suggested_tools = suggested_tools["tool_calls"]
            else:
                suggested_tools = []
        except json.JSONDecodeError:
            suggested_tools = []
    
    # If no tool was suggested, ask the LLM to choose one.
    if not suggested_tools:
        suggested_tools = _select_tools_with_llm(state, config)
    
    # Limit the number of tool calls per iteration.
    max_tools = config.agent.max_tool_calls_per_iteration
    suggested_tools = suggested_tools[:max_tools]
    
    # Execute the selected tools.
    tool_calls = []
    results = []
    
    for tool_spec in suggested_tools:
        tool_name = tool_spec.get("tool", "")
        args = tool_spec.get("args", {})
        reason = tool_spec.get("reason", "")
        
        if tool_name not in TOOL_REGISTRY:
            logger.warning(f"Unknown tool: {tool_name}")
            continue
        
        logger.info(f"Executing tool: {tool_name}({args})")
        
        try:
            # Inject the repository path into the tool arguments.
            args["repo_path"] = repo_path
            
            tool_fn = TOOL_REGISTRY[tool_name]
            result = tool_fn.invoke(args)
            
            tool_calls.append(ToolCall(
                tool_name=tool_name,
                arguments=args,
                result=str(result)[:2000],
                success=True
            ))
            
            results.append(f"### Tool: {tool_name}\n**Arguments**: {args}\n**Reason**: {reason}\n**Result**:\n{result}\n")
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            tool_calls.append(ToolCall(
                tool_name=tool_name,
                arguments=args,
                result=str(e),
                success=False
            ))
            results.append(f"### Tool: {tool_name}\n**Error**: {e}\n")
    
    # Record this exploration step.
    exploration = ExplorationRecord(
        iteration=state["iteration_count"],
        action=f"Executed {len(tool_calls)} tool calls",
        findings=f"Collected {len([t for t in tool_calls if t.success])} successful results",
        tool_calls=tool_calls
    )
    state["exploration_history"].append(exploration)
    
    # Update the workflow state.
    state["current_tool_results"] = "\n\n".join(results)
    state["status"] = "tools_executed"
    
    logger.info(f"Tool execution completed, succeeded {len([t for t in tool_calls if t.success])}/{len(tool_calls)}")
    
    return state


def _select_tools_with_llm(state: AgentState, config: Config) -> list[dict]:
    """Select tools with the LLM when none were suggested."""
    llm = _create_llm(config)
    
    missing_str = "\n".join(f"- {part}" for part in state["missing_parts"]) if state["missing_parts"] else "No specific missing part; further exploration is needed."
    
    messages = [
        SystemMessage(content="You are a code analysis assistant. Select appropriate tools to gather more information. Return strict JSON only."),
        HumanMessage(content=TOOL_SELECTION_PROMPT.format(
            document=state["current_document"][:2000],
            missing_parts=missing_str,
            directory_tree=state["directory_tree"],
            max_tools=config.agent.max_tool_calls_per_iteration
        ))
    ]
    
    try:
        response = llm.invoke(messages)
        
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        
        if json_match:
            result = json.loads(json_match.group())
            return result.get("tool_calls", [])
    
    except Exception as e:
        logger.error(f"LLM tool selection failed: {e}")
    
    # Return a simple fallback exploration plan.
    return [
        {
            "tool": "list_files_by_extension",
            "args": {"extension": ".py"},
            "reason": "Explore Python source files"
        }
    ]


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
