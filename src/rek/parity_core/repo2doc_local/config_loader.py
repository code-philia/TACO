from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class AgentConfig(BaseModel):
    max_iterations: int = Field(default=5, description="Maximum exploration iterations.")
    max_tool_calls_per_iteration: int = Field(default=10, description="Maximum tool calls per iteration.")
    confidence_threshold: float = Field(default=0.75, description="Completion confidence threshold.")


class FileFilterConfig(BaseModel):
    include_extensions: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    max_file_size: int = Field(default=102400, description="Maximum file size in bytes.")


class LLMConfig(BaseModel):
    model: str = Field(default="qwen3-max")
    temperature: float = Field(default=0.3)
    api_key: str | None = Field(default=None)
    base_url: str | None = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        env_model = os.getenv("REPO2DOC_LLM_MODEL")
        if env_model:
            self.model = env_model
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")


class OutputConfig(BaseModel):
    output_dir: str = Field(default="./repo2doc-output")
    filename: str = Field(default="requirements.md")
    save_intermediate: bool = Field(default=True)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Config(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    file_filter: FileFilterConfig = Field(default_factory=FileFilterConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: str | None = None) -> "Config":
        config_data: dict = {}
        if config_path:
            path = Path(config_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as handle:
                    config_data = yaml.safe_load(handle) or {}
        config_obj = cls(**config_data)

        custom_output_dir = os.environ.get("REPO2DOC_OUTPUT_DIR")
        if custom_output_dir:
            if hasattr(config_obj, "output_dir"):
                config_obj.output_dir = custom_output_dir
            elif hasattr(config_obj, "output") and hasattr(config_obj.output, "output_dir"):
                config_obj.output.output_dir = custom_output_dir

        return config_obj


def setup_logging(config: LoggingConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        format=config.format,
    )
