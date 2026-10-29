from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
DEFAULT_ELEMENTS_FILENAME = "program_elements.jsonl"


@dataclass(slots=True)
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 180
    max_retries: int = 3


@dataclass(slots=True)
class EmbeddingConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str | None = None


@dataclass(slots=True)
class RetrievalConfig:
    mode: str = "hybrid"
    top_k: int = 5
    graph_top_k: int = 5
    vector_top_k: int = 10
    fusion: str = "rrf"


@dataclass(slots=True)
class PathsConfig:
    kb_path: str | None = None
    elements_path: str | None = None
    repo_path: str | None = None


@dataclass(slots=True)
class GithubConfig:
    token: str | None = None


@dataclass(slots=True)
class NetworkConfig:
    http_proxy: str | None = None
    https_proxy: str | None = None


@dataclass(slots=True)
class LoggingConfig:
    level: str = "info"
    log_file: str | None = None


@dataclass(slots=True)
class REKConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    github: GithubConfig = field(default_factory=GithubConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _expand_env_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        return os.getenv(match.group(1), "")

    return ENV_PATTERN.sub(replace, value)


def _expand_env_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_mapping(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_mapping(item) for item in value]
    return _expand_env_value(value)


def load_config(config_path: str | None = None) -> REKConfig:
    config = REKConfig()
    if not config_path:
        return config
    if tomllib is None:
        raise RuntimeError("tomllib is not available in this Python runtime.")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as handle:
        raw_data = tomllib.load(handle)
    data = _expand_env_mapping(raw_data)

    if "llm" in data:
        config.llm = LLMConfig(**data["llm"])
    if "embedding" in data:
        config.embedding = EmbeddingConfig(**data["embedding"])
    if "retrieval" in data:
        config.retrieval = RetrievalConfig(**data["retrieval"])
    if "paths" in data:
        config.paths = PathsConfig(**data["paths"])
    if "github" in data:
        config.github = GithubConfig(**data["github"])
    if "network" in data:
        config.network = NetworkConfig(**data["network"])
    if "logging" in data:
        config.logging = LoggingConfig(**data["logging"])
    return config


def apply_runtime_environment(config: REKConfig) -> None:
    if config.llm.api_key:
        os.environ["OPENAI_API_KEY"] = config.llm.api_key
    if config.llm.base_url:
        os.environ["OPENAI_API_BASE"] = config.llm.base_url
    if config.embedding.api_key:
        os.environ.setdefault("OPENAI_EMBEDDING_API_KEY", config.embedding.api_key)
    if config.embedding.base_url:
        os.environ.setdefault("OPENAI_EMBEDDING_API_BASE", config.embedding.base_url)
    if config.github.token:
        os.environ["GITHUB_TOKEN"] = config.github.token
    if config.network.http_proxy:
        os.environ["http_proxy"] = config.network.http_proxy
    if config.network.https_proxy:
        os.environ["https_proxy"] = config.network.https_proxy


def resolve_value(cli_value: str | None, config_value: str | None) -> str | None:
    if cli_value:
        return cli_value
    if config_value:
        return config_value
    return None


def resolve_elements_path(
    kb_path: str | None,
    cli_value: str | None = None,
    config_value: str | None = None,
) -> str | None:
    candidate = resolve_value(cli_value, config_value)
    if candidate:
        return _normalize_elements_candidate(candidate)
    if kb_path:
        return str(Path(kb_path).with_name(DEFAULT_ELEMENTS_FILENAME))
    return None


def _normalize_elements_candidate(value: str) -> str:
    path = Path(value).expanduser()
    raw_value = value.strip()
    if path.exists() and path.is_dir():
        return str(path / DEFAULT_ELEMENTS_FILENAME)
    if raw_value.endswith((os.sep, "/", "\\")):
        return str(path / DEFAULT_ELEMENTS_FILENAME)
    return str(path)
