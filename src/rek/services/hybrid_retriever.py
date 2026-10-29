from __future__ import annotations

from collections import defaultdict

from rek.config import REKConfig
from rek.models import CandidateMatch
from rek.services.graph_retriever import GraphRetriever
from rek.services.vector_retriever import VectorRetriever


class HybridRetriever:
    def __init__(self, graph_db: object, config: REKConfig, kb_path: str):
        self.graph_db = graph_db
        self.config = config
        self.kb_path = kb_path
        self.graph_retriever = GraphRetriever(graph_db)
        self.vector_retriever = VectorRetriever(graph_db, config, kb_path)

    def retrieve(self, query: str, top_k: int | None = None) -> list[CandidateMatch]:
        final_top_k = top_k or self.config.retrieval.top_k
        graph_results = self.graph_retriever.retrieve(
            query,
            top_k=max(final_top_k, self.config.retrieval.graph_top_k),
        ).candidates
        graph_target_files = list(dict.fromkeys(candidate.file_path for candidate in graph_results))
        vector_results = self.vector_retriever.retrieve(
            query,
            top_k=max(final_top_k, self.config.retrieval.vector_top_k),
            target_files=graph_target_files,
        ).candidates
        if not vector_results:
            return graph_results[:final_top_k]
        return reciprocal_rank_fusion(graph_results, vector_results, limit=final_top_k)


def reciprocal_rank_fusion(
    graph_results: list[CandidateMatch],
    vector_results: list[CandidateMatch],
    limit: int,
    k: int = 60,
) -> list[CandidateMatch]:
    fused_scores: dict[tuple[str, str | None], float] = defaultdict(float)
    candidate_index: dict[tuple[str, str | None], CandidateMatch] = {}

    for rank, candidate in enumerate(graph_results):
        key = (candidate.file_path, candidate.symbol)
        fused_scores[key] += 1.0 / (k + rank + 1)
        candidate_index[key] = candidate

    for rank, candidate in enumerate(vector_results):
        key = (candidate.file_path, candidate.symbol)
        fused_scores[key] += 1.0 / (k + rank + 1)
        if key not in candidate_index:
            candidate_index[key] = candidate

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    results: list[CandidateMatch] = []
    for (file_path, symbol), score in ranked[:limit]:
        base = candidate_index[(file_path, symbol)]
        results.append(
            CandidateMatch(
                file_path=base.file_path,
                symbol=base.symbol,
                score=score,
                rationale=base.rationale,
                line=base.line,
            )
        )
    return results
