from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rek.models import CandidateMatch
from rek.services.parity_adapter import score_text


@dataclass(slots=True)
class GraphRetrievalResult:
    candidates: list[CandidateMatch]


class GraphRetriever:
    def __init__(self, graph_db: Any):
        self.graph_db = graph_db

    def retrieve(self, query: str, top_k: int = 10) -> GraphRetrievalResult:
        ranked_matches: list[CandidateMatch] = []

        for file_path, file_node in self.graph_db.file_map.items():
            ancestry = self.graph_db.get_file_ancestry(file_path)
            l1_name = ancestry.get("l1_id", "")
            l2_name = ancestry.get("l2_id", "")
            file_text = " ".join(
                [
                    file_path,
                    file_node.get("role", "") or "",
                    file_node.get("rationale", "") or "",
                    " ".join(file_node.get("rationale_keywords", []) or []),
                    l1_name,
                    l2_name,
                ]
            )
            file_score = score_text(file_text, query)
            if file_score > 0:
                ranked_matches.append(
                    CandidateMatch(
                        file_path=file_path,
                        score=file_score,
                        rationale=file_node.get("rationale"),
                    )
                )

            for function_name, function_node in file_node.get("functions", {}).items():
                function_text = " ".join(
                    [
                        file_path,
                        function_name,
                        function_node.get("rationale", "") or "",
                        " ".join(function_node.get("rationale_keywords", []) or []),
                        l1_name,
                        l2_name,
                    ]
                )
                function_score = score_text(function_text, query)
                if function_score > 0:
                    ranked_matches.append(
                        CandidateMatch(
                            file_path=file_path,
                            symbol=function_name,
                            score=function_score,
                            rationale=function_node.get("rationale"),
                        )
                    )

        ranked_matches.sort(key=lambda item: item.score or 0.0, reverse=True)
        return GraphRetrievalResult(candidates=ranked_matches[:top_k])
