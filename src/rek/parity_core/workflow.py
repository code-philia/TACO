from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from typing import Any

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
except ImportError:  # pragma: no cover
    AIMessage = BaseMessage = HumanMessage = SystemMessage = ToolMessage = object  # type: ignore
    ChatOpenAI = None  # type: ignore
    create_react_agent = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover
    ChatAnthropic = None

from rek.parity_core.git_time_travel import BackwardTimeMachine
from rek.parity_core.graph_db import GraphDatabase
from rek.parity_core.light_vector import LightVectorStore
from rek.parity_core.pr_distiller import GitTimeMachine, PRKnowledgeDistiller
from rek.parity_core.skill_loader import skill_registry
from rek.parity_core.tools_registry import MUTATION_TOOLS, build_inference_tools, set_inference_db, set_mutation_db, set_repo_path

logger = logging.getLogger("EvolutionEngine")


class EvolutionGraph:
    """
    Local parity-oriented copy of the original retrieval workflow.

    The implementation stays close to the source project while isolating all
    imports and runtime state inside the CLI workspace.
    """

    def __init__(self, repo_path: str, kb_path: str, elements_path: str, model_name: str = "gpt-4o"):
        self.repo_path = repo_path
        openai_api_base = os.getenv("OPENAI_API_BASE")
        self.clean_model_name = model_name.split("/", 1)[1] if "/" in model_name else model_name
        self.is_claude = False

        set_repo_path(self.repo_path)

        if ChatOpenAI is None or create_react_agent is None:
            raise ImportError(
                "Full parity workflow requires langchain_core, langchain_openai, and langgraph."
            )

        if "claude" in self.clean_model_name.lower():
            if ChatAnthropic is None:
                raise ImportError("Full parity workflow requires langchain-anthropic for Claude models.")
            self.is_claude = True
            self.llm = ChatOpenAI(
                model=self.clean_model_name,
                temperature=0.0,
                max_retries=3,
                timeout=180.0,
                base_url=openai_api_base,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        else:
            self.llm = ChatOpenAI(
                model=self.clean_model_name,
                temperature=0.0,
                max_retries=3,
                request_timeout=180,
                base_url=openai_api_base,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        self.graph_db = GraphDatabase(kb_path, elements_path)
        set_inference_db(self.graph_db)
        set_mutation_db(self.graph_db)
        self.vector_store = LightVectorStore(self.graph_db)

    def _serialize_trace(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        serialized = []
        for message in messages:
            role = "unknown"
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, ToolMessage):
                role = "tool"
            elif isinstance(message, SystemMessage):
                role = "system"
            item = {"role": role, "content": getattr(message, "content", ""), "type": getattr(message, "type", "")}
            if hasattr(message, "tool_calls") and message.tool_calls:
                item["tool_calls"] = message.tool_calls
            if hasattr(message, "tool_call_id"):
                item["tool_call_id"] = message.tool_call_id
            serialized.append(item)
        return serialized

    def get_formatted_trace(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        steps = []
        current_step = None

        for message in messages:
            if isinstance(message, (SystemMessage, HumanMessage)):
                continue

            if isinstance(message, AIMessage):
                if current_step:
                    steps.append(current_step)

                thought = getattr(message, "content", "") or ""
                tool_calls = getattr(message, "tool_calls", [])
                current_step = {
                    "thought": thought if thought else "AI is processing...",
                    "action": json.dumps(tool_calls, ensure_ascii=False) if tool_calls else "None",
                    "observation": "N/A",
                }
                if not tool_calls and thought:
                    steps.append(current_step)
                    current_step = None

            elif isinstance(message, ToolMessage) and current_step:
                if current_step["observation"] == "N/A":
                    current_step["observation"] = str(message.content)
                else:
                    current_step["observation"] += f"\n--- Next Tool ---\n{message.content}"

        if current_step:
            steps.append(current_step)

        return steps

    def run_inference_for_single_sample(
        self,
        issue: str,
        ground_truth_files: list[str],
        base_commit: str = "HEAD",
        patch_text: str = "",
    ) -> dict[str, Any]:
        import ast

        skill_content = skill_registry.load_skill_content("retrieval-navigation")
        current_tools = build_inference_tools(self.repo_path, base_commit)
        agent = create_react_agent(self.llm, current_tools)

        logger.info("⚡ [Track B] Executing Initial Vector Search...")
        track_b_initial = self.vector_store.search(issue, top_k=5)
        track_b_text = json.dumps(track_b_initial, indent=2, ensure_ascii=False)
        track_b_files = {item["file"] for item in track_b_initial}

        track_a_response = ""
        full_trace: list[dict[str, Any]] = []
        current_issue = issue

        for attempt in range(3):
            recursion_limit = 50 if attempt == 0 else (20 if attempt == 1 else 10)
            base_prompt = f"""
{skill_content}
# CONTEXT VARIABLES
- **Current Issue**:
{current_issue}
# INSTRUCTION
Execute the Search Protocol entirely within the MODERN Knowledge Graph.
You MUST use `get_functions_in_file` to pinpoint exact functions.
Output your thought process. DO NOT output final JSON.
"""
            input_messages: list[Any] = [SystemMessage(content=base_prompt)]
            if attempt > 0:
                warning_message = (
                    "🚨 SYSTEM WARNING: Your previous execution failed because it exceeded the model's maximum context length (128k tokens). "
                    "You explored too many paths or requested functions from excessively large files. "
                    "FOR THIS RETRY, YOU MUST CHANGE YOUR STRATEGY TO BE EXTREMELY TOKEN-EFFICIENT: "
                    "1. AVOID broad exploration. Do not list categories or features unless absolutely necessary. "
                    "2. PREFER `search_by_tags` for immediate and direct targeting. "
                    "3. DO NOT check more than 1 or 2 files using `get_functions_in_file`. "
                    "4. Conclude your search rapidly to save tokens."
                )
                input_messages.append(SystemMessage(content=warning_message))
            if self.is_claude:
                input_messages.append(HumanMessage(content="Please execute the search protocol now."))

            try:
                logger.info(
                    "🔄 [Track A] Executing Agent (Attempt %s/3, Max Steps: %s)...",
                    attempt + 1,
                    recursion_limit,
                )
                response = agent.invoke({"messages": input_messages}, config={"recursion_limit": recursion_limit})
                track_a_response = response["messages"][-1].content
                full_trace = self.get_formatted_trace(response["messages"])
                logger.info("   🧭 Track A Graph navigation completed.")
                break
            except Exception as exc:
                error_message = str(exc)
                logger.error("❌ [Track A] Error: %s", error_message)
                if attempt == 2:
                    return {
                        "is_solved": False,
                        "error": error_message,
                        "trace": [],
                        "ground_truth": ground_truth_files,
                    }
                if "context_length_exceeded" in error_message.lower() or "maximum context length" in error_message.lower():
                    logger.warning(
                        "   ⚠️ Context limit hit! Compressing issue length and altering agent strategy for next attempt..."
                    )
                    if len(current_issue) > 8000:
                        current_issue = current_issue[:8000] + "\n\n... [ISSUE TRUNCATED TO SAVE CONTEXT LIMIT] ..."
                else:
                    time.sleep(5)

        parsed_track_a = self._parse_candidates(track_a_response)
        track_a_files = set(parsed_track_a["files"])

        logger.info("🤝 [Synergy] Analyzing Track A and Track B consensus (Modern Space)...")
        overlap_modern = track_a_files.intersection(track_b_files)
        track_a_only = track_a_files - overlap_modern
        track_b_only = track_b_files - overlap_modern
        final_candidate_modern_files = track_a_files.union(track_b_files)

        synergy_context = ""
        track_b_synergy: list[dict[str, Any]] = []
        track_a_synergy = ""

        if overlap_modern:
            logger.info(
                "   ✅ High Confidence Overlap detected: %s, but retaining exclusive files for shallow check.",
                overlap_modern,
            )
            track_b_synergy = self.vector_store.search(issue, top_k=10, target_files=list(overlap_modern))
            non_overlap = track_a_only.union(track_b_only)
            if non_overlap:
                track_b_synergy.extend(self.vector_store.search(issue, top_k=3, target_files=list(non_overlap)))

            deep_prompt = f"""
Issue: {issue}
Track A and B agreed on these highly suspicious files: {overlap_modern}.
However, Track A exclusively suggested: {track_a_only}.
Track B exclusively suggested: {track_b_only}.

TASK:
1. Deeply analyze the overlapping files: What exact functions inside handle the logic described in the issue?
2. Play devil's advocate: briefly review the exclusive files. Could the overlap be false consensus while the bug hides in an exclusive file?
Provide step-by-step reasoning.
"""
            deep_messages: list[Any] = [SystemMessage(content=deep_prompt)]
            if self.is_claude:
                deep_messages.append(HumanMessage(content="Please provide your deep analysis and critical review now."))
            track_a_synergy = self.llm.invoke(deep_messages).content
            synergy_context = (
                "### SCENARIO: PARTIAL OVERLAP (Consensus + Devil's Advocate)\n"
                f"Tracks converged on {overlap_modern}.\n"
                f"Non-overlapping: Track A ({track_a_only}), Track B ({track_b_only}).\n"
                f"[Track B Deep & Shallow Search]:\n{json.dumps(track_b_synergy, indent=2)}\n"
                f"[Track A Reasoning & Critical Review]:\n{track_a_synergy}"
            )
        else:
            logger.info("   ⚔️ No Overlap. Initiating Cross-Validation...")
            track_b_synergy = self.vector_store.search(issue, top_k=5, target_files=list(track_a_files))
            cross_prompt = (
                f"Issue: {issue}\n"
                f"Track A proposed: {track_a_files}. Track B proposed: {track_b_files}. NO overlap.\n"
                "Critically review Track B's files based on the issue. Could Track B be right? Why or why not?"
            )
            cross_messages: list[Any] = [SystemMessage(content=cross_prompt)]
            if self.is_claude:
                cross_messages.append(HumanMessage(content="Please provide your critical review now."))
            track_a_synergy = self.llm.invoke(cross_messages).content
            synergy_context = (
                "### SCENARIO: CROSS-VALIDATION (NO OVERLAP)\n"
                f"Track A chose {track_a_files}, Track B chose {track_b_files}.\n"
                f"[Track B exploring Track A's files]:\n{json.dumps(track_b_synergy, indent=2)}\n"
                f"[Track A's critical review of Track B]:\n{track_a_synergy}"
            )

        logger.info("🛡️ [Projection] Projecting Modern Candidates back to %s...", base_commit[:7])
        backward_machine = BackwardTimeMachine(self.repo_path)
        historic_roster: dict[str, dict[str, Any]] = {}

        list_command = ["git", "ls-tree", "-r", "--name-only", base_commit]
        all_hist_files = set(
            subprocess.run(list_command, cwd=self.repo_path, capture_output=True, text=True).stdout.splitlines()
        )

        for modern_file in final_candidate_modern_files:
            historic_file = backward_machine.trace_prediction_backward(modern_file, base_commit)
            if historic_file in all_hist_files:
                show_command = ["git", "show", f"{base_commit}:{historic_file}"]
                show_result = subprocess.run(
                    show_command,
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    errors="replace",
                )
                hist_funcs: list[str] = []
                if show_result.returncode == 0:
                    try:
                        import ast

                        tree = ast.parse(show_result.stdout)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                hist_funcs.append(node.name)
                            elif isinstance(node, ast.ClassDef):
                                for sub_node in node.body:
                                    if isinstance(sub_node, ast.FunctionDef):
                                        hist_funcs.append(f"{node.name}.{sub_node.name}")
                    except Exception:
                        pass
                historic_roster[modern_file] = {
                    "Mapped_Historic_File": historic_file,
                    "Valid_Functions_At_Base_Commit": list(set(hist_funcs))[:50],
                    "Status": "Directly Mapped",
                }
            else:
                logger.info("   ⚠️ %s missing historically. Triggering global grep...", modern_file)
                modern_funcs = list(self.graph_db.file_map.get(modern_file, {}).get("functions", {}).keys())
                found_ancestors: dict[str, int] = {}
                for modern_func in modern_funcs:
                    clean_name = modern_func.split(".")[-1]
                    grep_result = subprocess.run(
                        ["git", "grep", "-l", f"def {clean_name}", base_commit],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True,
                    )
                    for line in grep_result.stdout.splitlines():
                        if line.endswith(".py"):
                            found_ancestors[line] = found_ancestors.get(line, 0) + 1

                if found_ancestors:
                    best_ancestor = max(found_ancestors, key=found_ancestors.get)
                    show_command = ["git", "show", f"{base_commit}:{best_ancestor}"]
                    show_result = subprocess.run(
                        show_command,
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True,
                        errors="replace",
                    )
                    hist_funcs = modern_funcs
                    if show_result.returncode == 0:
                        try:
                            import ast

                            tree = ast.parse(show_result.stdout)
                            hist_funcs = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                        except Exception:
                            pass
                    historic_roster[modern_file] = {
                        "Mapped_Historic_File": best_ancestor,
                        "Valid_Functions_At_Base_Commit": list(set(hist_funcs))[:50],
                        "Status": f"Recovered Ancestor: {best_ancestor}",
                    }
                else:
                    historic_roster[modern_file] = {
                        "Mapped_Historic_File": "NOT_FOUND",
                        "Valid_Functions_At_Base_Commit": [],
                        "Status": "Failed to trace.",
                    }

        arbiter_prompt = f"""
        You are the Chief Software Architect and Final Arbiter for Bug Localization.
        Your responsibility is to synthesize the findings from two independent retrieval tracks, reason deeply about the root cause based on the Issue, and make the definitive final decision.

        # 1. ISSUE OVERVIEW (THE PROBLEM STATEMENT)
        {issue}

        # 2. INITIAL RETRIEVAL RESULTS
        [Track A (Graph Agent) Initial Reasoning]:
        {track_a_response}

        [Track B (Vector Search) Top Hits]:
        {track_b_text}

        # 3. MODERN CONCEPT SYNERGY & CROSS-VALIDATION
        We forced Track A and Track B to cross-validate their findings in the MODERN codebase. Here are their debates and deep dives:
        {synergy_context}

        # 4. 🚨 HISTORIC PROJECTION (STRICT CONSTRAINT) 🚨
        The bug happened in the past at commit `{base_commit}`. The modern files discussed above might NOT have existed.
        I have automatically projected the modern files to their TRUE HISTORICAL equivalents.
        {json.dumps(historic_roster, indent=2)}

        # INSTRUCTION & JUDGMENT WORKFLOW
        As the Final Arbiter, you must not act as a strict filter that throws away good candidates. You must balance Precision and Recall.

        **Step 1: Independent Judgment & Confidence Check**: Read the Issue again and evaluate the Synergy Context.
          - If Track A and B agreed (Overlap), verify if the overlapping file truly contains the buggy logic.
          - If they disagreed (No Overlap), do NOT arbitrarily discard one. Analyze their INITIAL results: did Track A or Track B show high internal confidence? If a track has high initial confidence, strongly consider keeping its top candidate. If both tracks present compelling distinct cases, you are ENCOURAGED to output MULTIPLE files to maximize recall.
        **Step 2: Function Selection & Scope Sweeping**: Dive into the specific functions.
          - **Scope Sweeping**: Look at the `Valid_Functions_At_Base_Commit`. If you determine that a specific class is involved (e.g., `ClassName.method`), you MUST explicitly scan the roster for ALL other methods belonging to that same `ClassName`. Bugs often span sibling methods (e.g., if `__init__` changes, `fit` or `set_params` might need updates too). Include any highly related sibling methods to prevent omissions.
          - **Global Scope**: Similarly, if the bug is in a global function, scan other global functions in that file for linked logic.
          - **Balance Precision & Recall**: You are encouraged to output the **2 to 4 most suspicious functions** from the relevant class/scope. However, MUST maintain precision: DO NOT just dump all functions in the file or class. Every `element` you output must have a solid logical justification.
        **Step 3: Historic Mapping**: Once you have decided on the logical file(s) and function(s), look at the Historic Roster.
          - The `file` you output MUST be the `Mapped_Historic_File` from the Roster.
          - The `element` MUST exist exactly in the `Valid_Functions_At_Base_Commit` list for that historical file.

        Output valid JSON inside a code block:
        ```json
        {{
            "analysis": "Step 1: Track B showed extremely high confidence in function X (Top 1), and Track A emphasized function Y in the same file. I will protect both. Step 2: These functions form an execution chain... Step 3: Mapping to historic roster...",
            "candidates": [
                {{
                    "file": "historic_path_1.py",
                    "element": "BaseClass.function_X",
                    "rationale": "Retained due to high initial confidence in Track B and its role as the underlying base mechanism..."
                }},
                {{
                    "file": "historic_path_1.py",
                    "element": "ChildClass.function_Y",
                    "rationale": "Retained as the specific entry point discussed by Track A..."
                }}
            ]
        }}
        ```
        """

        try:
            arbiter_messages: list[Any] = [SystemMessage(content=arbiter_prompt)]
            if self.is_claude:
                arbiter_messages.append(HumanMessage(content="Please provide your critical synthesize now."))
            final_response = self.llm.invoke(arbiter_messages)
            parsed_result = self._parse_candidates(final_response.content)
            candidate_files = parsed_result["files"]
            candidate_details = parsed_result["details"]

            clean_gt = {
                file_path
                for file_path in ground_truth_files
                if "test" not in file_path and "conftest" not in file_path
            }
            hits = set(candidate_files).intersection(clean_gt)
            recall = len(hits) / len(clean_gt) if clean_gt else 0.0

            full_trace.append(
                {
                    "thought": f"=== THE ARBITER DECISION ===\n{parsed_result.get('analysis', '')}",
                    "action": "Final Output",
                    "observation": str(candidate_details),
                }
            )

            return {
                "issue": issue,
                "ground_truth": list(clean_gt),
                "candidates": candidate_files,
                "candidates_details": candidate_details,
                "track_b": track_b_initial,
                "track_b_synergy": track_b_synergy,
                "track_a_synergy": track_a_synergy,
                "recall": recall,
                "is_solved": recall >= 1.0,
                "trace": full_trace,
                "base_commit": base_commit,
                "ignored": False,
                "patch_text": patch_text,
            }
        except Exception as exc:
            logger.error("❌ [Arbiter] Error: %s", exc)
            return {
                "is_solved": False,
                "error": str(exc),
                "trace": full_trace,
                "ground_truth": ground_truth_files,
            }

    def _parse_candidates(self, text: str) -> dict[str, Any]:
        files = set()
        details = []
        analysis_text = "No CoT analysis provided."
        cleaned_text = text.strip()
        if "```" in cleaned_text:
            match = re.search(r"```(?:json)?(.*?)```", cleaned_text, re.DOTALL)
            if match:
                cleaned_text = match.group(1).strip()
        try:
            json_start = cleaned_text.find("{")
            json_end = cleaned_text.rfind("}")
            if json_start != -1 and json_end != -1:
                json_text = cleaned_text[json_start : json_end + 1]
                data = json.loads(json_text)
                analysis_text = data.get("analysis", "No CoT analysis provided.")
                for candidate in data.get("candidates", []):
                    file_path = candidate.get("file")
                    if file_path:
                        files.add(file_path)
                        details.append(
                            {
                                "file": file_path,
                                "element": candidate.get("element", "Entire File"),
                                "rationale": candidate.get("rationale", ""),
                            }
                        )
                if files:
                    return {"files": list(files), "details": details, "analysis": analysis_text}
        except Exception:
            pass

        raw_files = re.findall(r"[\w/._-]+\.(?:py|pyx|pxd|tp)", text)
        for file_path in raw_files:
            files.add(file_path)
            details.append({"file": file_path, "element": "Entire File", "rationale": "Regex Fallback"})
        return {
            "files": list(files),
            "details": details,
            "analysis": analysis_text if analysis_text != "No CoT analysis provided." else text[:500] + "...",
        }

    def run_batch_injection(self, samples: list[dict[str, Any]]) -> None:
        if not samples:
            return

        injection_skill_content = skill_registry.load_skill_content("graph-injection")
        time_machine = GitTimeMachine(self.repo_path)
        distiller = PRKnowledgeDistiller(model_name=self.clean_model_name)

        injection_report = ""
        for index, sample in enumerate(samples):
            commit_hash = sample.get("base_commit", "HEAD")
            issue_text = sample.get("issue", "")
            patch_text = sample.get("patch_text", "")
            ground_truth = list(set(sample.get("ground_truth", [])))
            if not patch_text:
                continue

            time_map = time_machine.build_time_mapping(patch_text, commit_hash, expected_files=ground_truth)
            if not time_map:
                continue

            try:
                distilled_knowledge = distiller.distill(issue_text, patch_text, time_map)
            except Exception as exc:
                logger.error("Distillation failed for sample %s: %s", index, exc)
                continue

            distilled_text = json.dumps(distilled_knowledge, indent=2) if distilled_knowledge else "[]"
            injection_report += f"""
--- PR Sample {index + 1} ---
[ISSUE]: {issue_text[:1000]}...
[DISTILLED KNOWLEDGE TO INJECT]:
{distilled_text}
-------------------------
"""

        if not injection_report.strip():
            return

        mutation_agent = create_react_agent(self.llm, MUTATION_TOOLS)
        prompt = f"""
{injection_skill_content}

# BATCH INJECTION REPORT
{injection_report}

Apply the injection skill to update the dual-index knowledge graph.
"""
        messages: list[Any] = [SystemMessage(content=prompt)]
        if self.is_claude:
            messages.append(HumanMessage(content="Please execute the graph injection now."))
        mutation_agent.invoke({"messages": messages}, config={"recursion_limit": 30})

    def distill_knowledge_for_sample(
        self,
        issue: str,
        ground_truth_files: list[str],
        base_commit: str,
        patch_text: str,
    ) -> list[dict[str, Any]]:
        if not patch_text:
            return []

        time_machine = GitTimeMachine(self.repo_path)
        distiller = PRKnowledgeDistiller(model_name=self.clean_model_name)
        time_map = time_machine.build_time_mapping(
            patch_text,
            base_commit,
            expected_files=ground_truth_files,
        )
        if not time_map:
            return []

        try:
            return distiller.distill(issue, patch_text, time_map)
        except Exception as exc:
            logger.error("Distillation failed: %s", exc)
            return []

    def run_fast_injection(self, batch_data: list[dict[str, Any]]) -> None:
        if not batch_data:
            return

        import os
        import time

        injection_report = ""
        for item in batch_data:
            distilled_text = json.dumps(item.get("distilled_data", []), indent=2, ensure_ascii=False)
            injection_report += f"--- PR Sample {item.get('instance_id')} ---\n"
            injection_report += f"[ISSUE]: {item.get('issue', '')[:1000]}...\n"
            injection_report += f"[🧠 DISTILLED KNOWLEDGE TO INJECT]:\n{distilled_text}\n"
            injection_report += "-------------------------\n"

        log_root = os.getenv("TRAINER_LOG_ROOT", "YOUR_LOG_DIR_HERE")
        debug_dir = os.path.join(log_root, "debug_injection_inputs")
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, f"injection_input_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(debug_file, "w", encoding="utf-8") as handle:
            handle.write(injection_report)
        logger.info("💉 [Debug] Injection Input saved to: %s", debug_file)

        replay_log = []
        replay_log.append(f"# 🔧 Fast Injection Replay ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
        replay_log.append(
            "> Note: This is a high-speed Python injection. The logs simulate Agent Tool Calls for readability.\n"
        )

        injected_count = 0
        for item in batch_data:
            distilled_items = item.get("distilled_data", [])
            if not distilled_items:
                continue

            instance_id = item.get("instance_id", "Unknown")
            replay_log.append(f"\n## 🚀 Processing PR: {instance_id}")

            for node in distilled_items:
                file_path = node.get("file")
                function_name = node.get("function")
                keywords = node.get("keywords", [])
                rationale = node.get("rationale", "")
                if not file_path or not rationale:
                    continue

                junk_phrases = [
                    "no changes",
                    "not modified",
                    "no functional changes",
                    "no direct changes",
                ]
                if any(phrase in rationale.lower() for phrase in junk_phrases) and not keywords:
                    replay_log.append(
                        f"\n### ⏭️ Skipped (Junk Detected):\n"
                        f"- **File**: `{file_path}`\n"
                        f"- **Function**: `{function_name}`\n"
                        f"- **Reason**: Rationale matched junk filters."
                    )
                    continue

                try:
                    success = self.graph_db.update_node_knowledge(
                        file_path,
                        function_name,
                        keywords,
                        rationale,
                    )
                    if success:
                        injected_count += 1
                        replay_log.append("\n### ⚡ Tool Call (Action): `inject_dual_index_knowledge`")
                        replay_log.append(f"- **File**: `{file_path}`")
                        replay_log.append(f"- **Function**: `{function_name}`")
                        replay_log.append(f"- **Keywords**: `{keywords}`")
                        replay_log.append(f"- **Rationale**: {rationale[:300]}...")
                        replay_log.append(
                            f"\n### 👁️ Tool Output (Observation):\n```text\n"
                            f"✅ Successfully injected {len(keywords)} tags and knowledge into "
                            f"{file_path}::{function_name}.\n```\n"
                        )
                except Exception as exc:
                    logger.error("❌ DB Write Error for %s: %s", file_path, exc)
                    replay_log.append(f"\n### ❌ Error:\n```text\n{exc}\n```\n")

        logger.info("💉 [Fast Injection] Successfully injected %s knowledge nodes in this batch.", injected_count)

        replay_dir = os.path.join(log_root, "injection_replays")
        os.makedirs(replay_dir, exist_ok=True)
        replay_file = os.path.join(replay_dir, f"replay_{time.strftime('%Y%m%d_%H%M%S')}.md")
        with open(replay_file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(replay_log))

        logger.info("✅ Full Injection Replay saved to: %s", replay_file)
        self.graph_db.save_state()
