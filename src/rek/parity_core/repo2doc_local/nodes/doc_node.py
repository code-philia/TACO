"""
Document generation node

Use the LLM to generate or update the requirements document.
"""

import logging
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None # Guard against missing optional dependency
from rek.parity_core.repo2doc_local.state import AgentState
from rek.parity_core.repo2doc_local.config_loader import Config
from rek.parity_core.repo2doc_local.utils.mermaid_fixer import fix_mermaid_syntax
from rek.parity_core.repo2doc_local.prompts.agent_prompts import (
    SYSTEM_PROMPT,
    INITIAL_DOC_PROMPT,
    UPDATE_DOC_PROMPT,
)
import os
import re

logger = logging.getLogger(__name__)

def _robust_generate(llm, messages, max_loops=5):
    """
    Continuation generation helper that prevents JSON restarts.
    """
    full_content = ""
    current_messages = list(messages)  # Copy the message history.
    
    usage_stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    for i in range(max_loops):
        logger.info(f"🔄 Generation Loop {i+1}/{max_loops}...")
        
        # 1. Call the model.
        response = llm.invoke(current_messages)
        content_chunk = response.content
        
        # 2. Accumulate token usage.
        if response.response_metadata and "token_usage" in response.response_metadata:
            u = response.response_metadata["token_usage"]
            usage_stats["prompt_tokens"] += u.get("prompt_tokens", 0)
            usage_stats["completion_tokens"] += u.get("completion_tokens", 0)
            usage_stats["total_tokens"] += u.get("total_tokens", 0)

        # 3. Strip repeated Markdown fences from continuation chunks.
        if i > 0:
            # Remove a repeated ```json fence if it appears.
            if content_chunk.strip().startswith("```"):
                clean_chunk = re.sub(r"^```json\s*", "", content_chunk.strip())
                # If the output restarts the JSON object, keep only the tail.
                if clean_chunk.lstrip().startswith('{"l1'):
                    logger.warning("⚠️ The model appears to be restarting the JSON output.")
                full_content += clean_chunk
            else:
                full_content += content_chunk
        else:
            full_content += content_chunk

        # 4. Detect a complete output.
        stripped_full = full_content.strip()
        if stripped_full.endswith("```") or stripped_full.endswith("}"):
            logger.info("✅ Output appears complete.")
            break
        
        if i == max_loops - 1:
            logger.warning("⚠️ Reached max loops, output might still be truncated.")
            break
            
        # 5. Build a strong continuation prompt.
        logger.warning(f"⚠️ Output truncated. Preparing continuation prompt...")
        
        # Store the partial AI output in the message history.
        current_messages.append(AIMessage(content=content_chunk))
        
        # Use the last 50 characters to anchor the continuation.
        last_chars = content_chunk[-50:].replace("\n", " ")
        
        # Build an anti-restart continuation prompt.
        continuation_prompt = (
            f"⛔️ STOP! You were cut off due to the token limit. "
            f"You ended with: \"...{last_chars}\"\n\n"
            f"👉 IMMEDIATE ACTION: Continue generating the JSON **EXACTLY** from that character.\n"
            f"1. **DO NOT** restart the JSON from the beginning (e.g. do not output 'l1_categories' again).\n"
            f"2. **DO NOT** use markdown code blocks (```json) for this continuation part.\n"
            f"3. **ONLY** output the remaining string and the closing braces.\n"
            f"4. If you were inside a string description, finish that sentence first."
        )
        
        current_messages.append(HumanMessage(content=continuation_prompt))

    return full_content, usage_stats

def generate_doc_node(state, config):
    logger.info(f"Generating/updating document (iteration {state['iteration_count'] + 1})...")
    
    llm = _create_llm(config)
    
    # Prepare the prompt.
    if state["iteration_count"] == 0:
        prompt_content = INITIAL_DOC_PROMPT.format(high_level_info=state["high_level_info"])
        call_type = "initial_doc"
    else:
        prompt_content = UPDATE_DOC_PROMPT.format(
            tool_results=state["current_tool_results"],
            current_document=state["current_document"]
        )
        call_type = "update_doc"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt_content)
    ]

    # Use the continuation helper.
    new_document, usage = _robust_generate(llm, messages)

    # Update the workflow state.
    llm_usage = state.get("llm_usage", {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0, "calls": []})
    
    llm_usage["total_prompt_tokens"] += usage["prompt_tokens"]
    llm_usage["total_completion_tokens"] += usage["completion_tokens"]
    llm_usage["total_tokens"] += usage["total_tokens"]
    llm_usage["calls"].append({
        "iteration": state["iteration_count"] + 1,
        "type": call_type,
        **usage
    })
    
    state["current_document"] = new_document
    state["document_versions"].append(new_document)
    state["iteration_count"] += 1
    state["current_tool_results"] = ""
    state["llm_usage"] = llm_usage
    state["status"] = "doc_generated"
    
    # Save an intermediate snapshot.
    try:
        from pathlib import Path
        repo_path = Path(state["repo_path"])
        output_dir = repo_path / config.output.output_dir / "intermediate"
        output_dir.mkdir(parents=True, exist_ok=True)
        current_ver = len(state["document_versions"])
        snap_path = output_dir / f"version_{current_ver}_snapshot.md"
        with open(snap_path, "w", encoding="utf-8") as f:
            f.write(new_document)
    except Exception:
        pass
    
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


def _generate_initial_doc(
    llm: ChatOpenAI, 
    high_level_info: str,
    config: Config
) -> Tuple[str, dict]:
    """Generate the initial document."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=INITIAL_DOC_PROMPT.format(
            high_level_info=high_level_info
        ))
    ]
    
    response = llm.invoke(messages)
    usage = _extract_usage(response)
    
    # Automatically fix Mermaid syntax.
    fixed_content = fix_mermaid_syntax(response.content)
    
    return fixed_content, usage


def _update_document(
    llm: ChatOpenAI,
    current_document: str,
    tool_results: str,
    missing_parts: list[str],
    config: Config
) -> Tuple[str, dict]:
    """Update the current document."""
    missing_str = "\n".join(f"- {part}" for part in missing_parts) if missing_parts else "None"
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=UPDATE_DOC_PROMPT.format(
            current_document=current_document,
            tool_results=tool_results,
            missing_parts=missing_str
        ))
    ]
    
    response = llm.invoke(messages)
    usage = _extract_usage(response)
    
    # Automatically fix Mermaid syntax.
    fixed_content = fix_mermaid_syntax(response.content)
    
    return fixed_content, usage
