"""
Save output node

Save the final document and report.
"""

import logging
from datetime import datetime
from pathlib import Path

from rek.parity_core.repo2doc_local.state import AgentState
from rek.parity_core.repo2doc_local.config_loader import Config
import json
import re

logger = logging.getLogger(__name__)


def extract_json_from_markdown(content: str):
    """
    Robust JSON extraction for several edge cases:
    1. A standard Markdown code block.
    2. A code block with a missing closing fence.
    3. A raw JSON object without Markdown fencing.
    """
    # Strategy A: strict Markdown matching.
    # `re.DOTALL` lets `.` match newline characters.
    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)

    # Strategy B: opening fence exists but the closing fence is missing.
    match_lazy = re.search(r"```json\s*(\{.*)$", content, re.DOTALL | re.IGNORECASE)
    if match_lazy:
        # Strip trailing noise and keep everything through the last brace.
        candidate = match_lazy.group(1).strip()
        last_brace = candidate.rfind('}')
        if last_brace != -1:
            return candidate[:last_brace+1]
        return candidate

    # Strategy C: heuristic fallback.
    # Look for the last brace-wrapped block containing "l1_categories".
    if '"l1_categories"' in content:
        # Find the last closing brace.
        end_idx = content.rfind('}')
        if end_idx != -1:
            # Find the nearest opening brace before "l1_categories".
            key_idx = content.rfind('"l1_categories"')
            # Search backward from the key for the opening brace.
            start_idx = content.rfind('{', 0, key_idx)
            if start_idx != -1:
                return content[start_idx : end_idx+1]
    
    return None

def save_output_node(state, config):
    """
    Save the final Markdown document and the extracted JSON knowledge graph.
    """
    repo_path = Path(state["repo_path"])
    output_dir = repo_path / config.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save the Markdown document.
    doc_content = state.get("current_document", "")
    md_path = output_dir / "requirements.md"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(doc_content)
    logger.info(f"📄 Requirements Doc saved to: {md_path}")
    
    # 2. Save the JSON knowledge graph.
    json_path = output_dir / "knowledge_graph_detail.json"
    data = None

    kb_data = state.get("knowledge_graph") or state.get("metadata")
    if kb_data:
        data = kb_data
        logger.info("🧠 Knowledge Graph found in state memory.")

    if not data:
        json_str = extract_json_from_markdown(doc_content)
        if json_str:
            try:
                data = json.loads(json_str)
                logger.info("🧠 Knowledge Graph extracted from Markdown content.")
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON Parse Error: {e}")

    if data:
        if isinstance(data, dict) and "file_mappings" not in data:
            data["file_mappings"] = []

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Successfully saved KB to: {json_path}")
    else:
        logger.warning(f"⚠️ Failed to obtain KB data. {json_path} was NOT created.")
        
    state["status"] = "completed"
    return state

def _generate_report(state: AgentState, config: Config) -> str:
    """Generate the processing report"""
    lines = [
        "# Repo2Doc Agent Processing Report",
        "",
        f"**Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Repository Path**: {state['repo_path']}",
        "",
        "## Statistics",
        "",
        f"- Iteration count: {state.get('iteration_count', 0)}",
        f"- Final confidence: {state.get('confidence_score', 0):.2f}",
        f"- Document length: {len(state.get('current_document', ''))} characters",
        f"- Document versions: {len(state.get('document_versions', []))}",
        "",
        "## Exploration History",
        "",
    ]
    
    for record in state.get("exploration_history", []):
        lines.append(f"### Iteration {record.iteration}")
        lines.append(f"- Action: {record.action}")
        lines.append(f"- Findings: {record.findings}")
        
        if record.tool_calls:
            lines.append("- Tool calls:")
            for tc in record.tool_calls:
                status = "✓" if tc.success else "✗"
                lines.append(f"  - {status} {tc.tool_name}")
        lines.append("")
    
    if state.get("missing_parts"):
        lines.append("## Remaining Attention Items")
        lines.append("")
        for part in state["missing_parts"]:
            lines.append(f"- {part}")
        lines.append("")
    
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Model: {config.llm.model}")
    lines.append(f"- Max iterations: {config.agent.max_iterations}")
    lines.append(f"- Max tool calls per iteration: {config.agent.max_tool_calls_per_iteration}")
    
    return "\n".join(lines)


def _generate_stats_json(state: AgentState, config: Config, timestamp: str) -> dict:
    """
    Generate statistics in JSON format.
    
    Args:
        state: Current status
        config: Config object
        timestamp: Timestamp.
    
    Returns:
        Statistics dictionary.
    """
    exploration_history = state.get("exploration_history", [])
    llm_usage = state.get("llm_usage", {})
    
    # Count tool calls.
    total_tool_calls = 0
    successful_tool_calls = 0
    tool_usage = {}
    
    for record in exploration_history:
        for tc in record.tool_calls:
            total_tool_calls += 1
            if tc.success:
                successful_tool_calls += 1
            tool_usage[tc.tool_name] = tool_usage.get(tc.tool_name, 0) + 1
    
    return {
        "meta": {
            "timestamp": timestamp,
            "generated_at": datetime.now().isoformat(),
            "repo_path": state["repo_path"],
            "tool": "repo2docAgent",
            "version": "0.1.0"
        },
        "execution": {
            "total_iterations": state.get("iteration_count", 0),
            "llm_calls": len(llm_usage.get("calls", [])),
            "is_complete": state.get("is_complete", False),
        },
        "llm_tokens": {
            "total_prompt_tokens": llm_usage.get("total_prompt_tokens", 0),
            "total_completion_tokens": llm_usage.get("total_completion_tokens", 0),
            "total_tokens": llm_usage.get("total_tokens", 0),
            "calls": llm_usage.get("calls", []),
        },
        "files": {
            "tool_calls": total_tool_calls,
            "successful_calls": successful_tool_calls,
            "by_tool": tool_usage,
        },
        "document": {
            "final_length": len(state.get("current_document", "")),
            "versions_count": len(state.get("document_versions", [])),
            "confidence_score": state.get("confidence_score", 0),
        },
        "config": {
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "max_iterations": config.agent.max_iterations,
        }
    }
