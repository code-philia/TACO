from __future__ import annotations

import json
import re
from typing import Any

from rek.config import resolve_elements_path
from rek.parity_core.graph_db import GraphDatabase
from rek.parity_core.light_vector import LightVectorStore


def load_kb_json(kb_path: str) -> dict[str, Any]:
    with open(kb_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {"file_mappings": data, "l1_categories": [], "l2_features": []}
    return data


def create_graph_db(kb_path: str, elements_path: str | None = None) -> Any:
    resolved_elements_path = resolve_elements_path(kb_path, cli_value=elements_path)
    return GraphDatabase(kb_path, resolved_elements_path)


def create_light_vector_store(
    graph_db: Any,
    embedding_model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    return LightVectorStore(
        graph_db=graph_db,
        embedding_model=embedding_model,
        api_key=api_key,
        base_url=base_url,
    )


def score_text(text: str, query: str) -> float:
    query_tokens = re.findall(r"\b[a-zA-Z_]\w+\b", query.lower())
    if not query_tokens:
        return 0.0
    lowered = text.lower()
    return float(sum(lowered.count(token) for token in query_tokens))
