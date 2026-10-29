from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rek.config import REKConfig
from rek.models import CandidateMatch
from rek.services.parity_adapter import create_light_vector_store


@dataclass(slots=True)
class VectorRetrievalResult:
    candidates: list[CandidateMatch]


class VectorRetriever:
    def __init__(self, graph_db: Any, config: REKConfig, kb_path: str):
        self.graph_db = graph_db
        self.config = config
        self.kb_path = kb_path

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        target_files: list[str] | None = None,
    ) -> VectorRetrievalResult:
        vector_store = create_light_vector_store(
            graph_db=self.graph_db,
            embedding_model=self.config.embedding.model,
            api_key=self.config.embedding.api_key or self.config.llm.api_key,
            base_url=self.config.embedding.base_url or self.config.llm.base_url,
        )
        raw_results = vector_store.search(query, top_k=top_k, target_files=target_files)
        candidates = [
            CandidateMatch(
                file_path=item["file"],
                symbol=None if item["element"] == "Entire File" else item["element"],
                score=item.get("score"),
                rationale=item.get("preview"),
            )
            for item in raw_results
        ]
        return VectorRetrievalResult(candidates=candidates)
