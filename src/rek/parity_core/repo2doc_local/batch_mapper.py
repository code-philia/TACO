from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover
    ChatAnthropic = None

from rek.parity_core.repo2doc_local.config_loader import Config, setup_logging
from rek.parity_core.repo2doc_local.prompts.agent_prompts import (
    BATCH_MAPPING_SYSTEM_PROMPT,
    BATCH_MAPPING_USER_PROMPT,
)

logger = logging.getLogger(__name__)
console = Console()


class BatchMapper:
    def __init__(self, repo_path: str, config_path: str | None = None, kb_path: str | None = None):
        self.repo_path = Path(repo_path)
        self.config = Config.load(config_path)
        setup_logging(self.config.logging)
        self.llm = self._create_llm()
        folder_name = os.environ.get("REPO2DOC_OUTPUT_DIR", "repo2docAgent-output")
        self.output_dir = self.repo_path / folder_name
        self.kb_path = Path(kb_path) if kb_path else self.output_dir / "knowledge_graph_detail.json"
        self.batch_output_file = self.output_dir / "batch_mapping_results_detail.json"
        self.l2_taxonomy: list[dict] = []
        self.processed_files: set[str] = set()
        self.mappings: list[dict] = []

    def _create_llm(self):
        model_name = self.config.llm.model
        clean_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
        openai_base = os.getenv("OPENAI_API_BASE") or self.config.llm.base_url
        api_key = self.config.llm.api_key or os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        if "claude" in clean_model_name.lower():
            if ChatAnthropic is None:
                raise ImportError("Please install `langchain-anthropic`.")
            console.print(f"[dim]Init LLM: {clean_model_name} (Anthropic Native)[/dim]")
            return ChatOpenAI(
                model=clean_model_name,
                api_key=anthropic_api_key,
                base_url=openai_base,
                temperature=0.0,
                timeout=120.0,
                streaming=True,
            )
        console.print(f"[dim]Init LLM: {clean_model_name} (OpenAI Compatible)[/dim]")
        return ChatOpenAI(
            model=clean_model_name,
            api_key=api_key,
            base_url=openai_base,
            temperature=0.0,
            request_timeout=120,
            max_retries=3,
            max_tokens=4096,
        )

    def load_resources(self) -> None:
        if not self.kb_path.exists():
            console.print(f"[bold red]Error:[/bold red] Knowledge Base not found at {self.kb_path}.")
            console.print("Please run Phase A (Agent Mode) first to generate the L2 skeleton.")
            raise FileNotFoundError(f"Knowledge base not found: {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as handle:
            kb_data = json.load(handle)
            self.l2_taxonomy = [
                {key: value for key, value in item.items() if key in ["id", "name", "description"]}
                for item in kb_data.get("l2_features", [])
            ]
            for mapping in kb_data.get("file_mappings", []):
                self.processed_files.add(mapping["file_path"])

        console.print(f"[green]Loaded {len(self.l2_taxonomy)} L2 features from KB.[/green]")

        if self.batch_output_file.exists():
            with open(self.batch_output_file, "r", encoding="utf-8") as handle:
                try:
                    saved_data = json.load(handle)
                except json.JSONDecodeError:
                    saved_data = []
                restored_count = 0
                for mapping in saved_data:
                    if mapping["file_path"] not in self.processed_files:
                        self.processed_files.add(mapping["file_path"])
                        self.mappings.append(mapping)
                        restored_count += 1
                if restored_count > 0:
                    console.print(f"[green]Restored {restored_count} mappings from previous batch run.[/green]")

    def scan_unmapped_files(self) -> list[str]:
        all_files: list[str] = []
        exclude_dirs = {".git", "__pycache__", "venv", "build", "dist", "tests", "doc", "docs", ".idea", ".vscode"}
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [item for item in dirs if item not in exclude_dirs]
            for file_name in files:
                if file_name.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file_name), self.repo_path)
                    if rel_path not in self.processed_files:
                        all_files.append(rel_path)
        return sorted(all_files)

    def _get_file_snippet(self, rel_path: str, max_lines: int = 50) -> str:
        try:
            with open(self.repo_path / rel_path, "r", encoding="utf-8", errors="ignore") as handle:
                return "".join(handle.readlines()[:max_lines])
        except Exception:
            return "(Could not read file content)"

    def run_batch_loop(self, batch_size: int = 10) -> None:
        unmapped_files = self.scan_unmapped_files()
        total_files = len(unmapped_files)
        console.print(f"[bold cyan]Found {total_files} unmapped files. Starting batch processing...[/bold cyan]")
        if total_files == 0:
            console.print("[green]All files are already mapped![/green]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            console=console,
        ) as progress:
            task_id = progress.add_task("[green]Mapping files...", total=total_files)
            for index in range(0, total_files, batch_size):
                batch_files = unmapped_files[index : index + batch_size]
                self._process_single_batch(batch_files)
                progress.advance(task_id, len(batch_files))
                self._save_checkpoint()

    def _process_single_batch(self, file_paths: list[str]) -> None:
        batch_context = ""
        for file_path in file_paths:
            batch_context += f"### File: {file_path}\n```python\n{self._get_file_snippet(file_path)}\n```\n\n"

        taxonomy_str = json.dumps(self.l2_taxonomy, indent=2)
        prompt = BATCH_MAPPING_USER_PROMPT.format(
            l2_taxonomy=taxonomy_str,
            file_batch_list=batch_context,
        )

        try:
            messages = [
                SystemMessage(content=BATCH_MAPPING_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            import re

            json_match = re.search(r"(\{[\s\S]*\})", content)
            if json_match:
                json_str = json_match.group(1)
                try:
                    data = json.loads(json_str)
                    new_items = data.get("new_mappings", [])
                    added_in_this_batch = 0
                    for item in new_items:
                        file_path = item.get("file_path", "").strip()
                        if file_path in file_paths:
                            self.mappings.append(item)
                            self.processed_files.add(file_path)
                            added_in_this_batch += 1
                    if added_in_this_batch == 0:
                        logger.warning("Batch LLM output did not match requested file paths: %s", file_paths)
                except json.JSONDecodeError as exc:
                    logger.error("JSON Decode Error: %s. Raw: %s...", exc, json_str[:100])
            else:
                logger.warning("No JSON block found in LLM output for: %s", file_paths[0])
        except Exception as exc:
            logger.error("Error processing batch: %s", exc)
            if "model_dump" in str(exc):
                logger.error(
                    "Detected a Pydantic compatibility error. Check the langchain-openai version or API key state."
                )
            time.sleep(2)

    def _save_checkpoint(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.batch_output_file, "w", encoding="utf-8") as handle:
            json.dump(self.mappings, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo2Doc batch mapper")
    parser.add_argument("repo_path", help="Path to the repository analyzed")
    parser.add_argument("-c", "--config", default=None, help="Path to config file")
    parser.add_argument("--kb-path", default=None, help="Optional KB path override")
    parser.add_argument("--batch-size", type=int, default=10, help="Files per batch")
    args = parser.parse_args()

    mapper = BatchMapper(args.repo_path, config_path=args.config, kb_path=args.kb_path)
    mapper.load_resources()
    mapper.run_batch_loop(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
