from pathlib import Path

from rek.config import REKConfig, resolve_elements_path
from rek.parity_core.workflow import EvolutionGraph
from rek.models import (
    CandidateMatch,
    CommandMeta,
    KBHintRequest,
    KBHintResponse,
    KBQueryRequest,
    KBQueryResponse,
    KBRetrieveRequest,
    KBRetrieveResponse,
    KBSearchRequest,
    KBSearchResponse,
    KBStatsResponse,
)
from rek.services.graph_retriever import GraphRetriever
from rek.services.hint_formatter import format_candidates_as_agent_hint
from rek.services.hybrid_retriever import HybridRetriever
from rek.services.parity_adapter import create_graph_db, load_kb_json


class KBService:
    def __init__(self, kb_path: str, config: REKConfig | None = None):
        self.kb_path = kb_path
        self.config = config or REKConfig()

    def query(self, request: KBQueryRequest) -> KBQueryResponse:
        if not Path(self.kb_path).exists():
            return KBQueryResponse(
                meta=CommandMeta(status="error", message="Knowledge base file not found."),
                kb_path=self.kb_path,
                question=request.question,
                candidates=[],
                summary=None,
            )

        try:
            candidates = self._query_candidates(request.question, request.top_k)
        except Exception as exc:
            return KBQueryResponse(
                meta=CommandMeta(status="error", message=str(exc)),
                kb_path=self.kb_path,
                question=request.question,
                candidates=[],
                summary=None,
            )
        return KBQueryResponse(
            meta=CommandMeta(status="ok", message="Query completed."),
            kb_path=self.kb_path,
            question=request.question,
            candidates=candidates,
            summary=f"Returned {len(candidates)} candidate matches.",
        )

    def search(self, request: KBSearchRequest) -> KBSearchResponse:
        if not Path(self.kb_path).exists():
            return KBSearchResponse(
                meta=CommandMeta(status="error", message="Knowledge base file not found."),
                kb_path=self.kb_path,
                symbol=request.symbol,
                file_path=request.file_path,
                matches=[],
            )

        graph_db = create_graph_db(self.kb_path)
        matches: list[CandidateMatch] = []

        if request.file_path:
            file_node = graph_db.file_map.get(request.file_path)
            if file_node:
                matches.append(
                    CandidateMatch(
                        file_path=request.file_path,
                        rationale=file_node.get("rationale"),
                    )
                )
                for function_name, function_node in file_node.get("functions", {}).items():
                    matches.append(
                        CandidateMatch(
                            file_path=request.file_path,
                            symbol=function_name,
                            rationale=function_node.get("rationale"),
                        )
                    )

        if request.symbol:
            symbol_query = request.symbol.lower()
            for file_path, file_node in graph_db.file_map.items():
                if symbol_query in file_path.lower():
                    matches.append(
                        CandidateMatch(
                            file_path=file_path,
                            rationale=file_node.get("rationale"),
                        )
                    )
                for function_name, function_node in file_node.get("functions", {}).items():
                    if symbol_query in function_name.lower():
                        matches.append(
                            CandidateMatch(
                                file_path=file_path,
                                symbol=function_name,
                                rationale=function_node.get("rationale"),
                            )
                        )

        trimmed_matches = matches[: request.top_k]
        return KBSearchResponse(
            meta=CommandMeta(status="ok", message="Search completed."),
            kb_path=self.kb_path,
            symbol=request.symbol,
            file_path=request.file_path,
            matches=trimmed_matches,
        )

    def retrieve(self, request: KBRetrieveRequest) -> KBRetrieveResponse:
        if not Path(self.kb_path).exists():
            return KBRetrieveResponse(
                meta=CommandMeta(status="error", message="Knowledge base file not found."),
                kb_path=self.kb_path,
                issue=request.issue,
                candidates=[],
            )

        try:
            candidates = self._retrieve_from_request(request)
        except Exception as exc:
            return KBRetrieveResponse(
                meta=CommandMeta(status="error", message=str(exc)),
                kb_path=self.kb_path,
                issue=request.issue,
                candidates=[],
            )
        return KBRetrieveResponse(
            meta=CommandMeta(status="ok", message="Retrieve completed."),
            kb_path=self.kb_path,
            issue=request.issue,
            candidates=candidates,
        )

    def hint(self, request: KBHintRequest) -> KBHintResponse:
        retrieve_result = self.retrieve(
            KBRetrieveRequest(
                kb_path=request.kb_path,
                issue=request.issue,
                repo_path=request.repo_path,
                elements_path=request.elements_path,
                base_commit=request.base_commit,
                patch_text=request.patch_text,
                ground_truth_files=request.ground_truth_files,
                top_k=request.top_k,
            )
        )
        if retrieve_result.meta.status != "ok":
            return KBHintResponse(
                meta=retrieve_result.meta,
                kb_path=retrieve_result.kb_path,
                issue=retrieve_result.issue,
                hint="",
                candidates=retrieve_result.candidates,
            )
        return KBHintResponse(
            meta=CommandMeta(status="ok", message="Hint generated."),
            kb_path=retrieve_result.kb_path,
            issue=retrieve_result.issue,
            hint=format_candidates_as_agent_hint(
                request.issue,
                retrieve_result.candidates,
                max_candidates=request.max_hint_candidates,
            ),
            candidates=retrieve_result.candidates,
        )

    def stats(self) -> KBStatsResponse:
        if not Path(self.kb_path).exists():
            return KBStatsResponse(
                meta=CommandMeta(status="error", message="Knowledge base file not found."),
                kb_path=self.kb_path,
                file_count=None,
                function_count=None,
                element_count=None,
            )

        data = load_kb_json(self.kb_path)
        file_mappings = data.get("file_mappings", [])
        function_count = sum(len(item.get("functions", {})) for item in file_mappings)
        graph_db = create_graph_db(self.kb_path, self.config.paths.elements_path)
        return KBStatsResponse(
            meta=CommandMeta(status="ok", message="Stats collected."),
            kb_path=self.kb_path,
            file_count=len(file_mappings),
            function_count=function_count,
            element_count=len(getattr(graph_db, "elements_data", {})),
        )

    def _retrieve_candidates(
        self,
        query: str,
        top_k: int,
        elements_path: str | None = None,
    ) -> list[CandidateMatch]:
        graph_db = create_graph_db(self.kb_path, elements_path or self.config.paths.elements_path)
        mode = self.config.retrieval.mode.lower()
        if mode == "graph":
            return GraphRetriever(graph_db).retrieve(query, top_k=top_k).candidates
        if mode == "hybrid":
            return HybridRetriever(graph_db, self.config, self.kb_path).retrieve(
                query, top_k=top_k
            )
        if mode != "parity":
            raise RuntimeError(
                f"Unsupported retrieval mode '{self.config.retrieval.mode}'. "
                "Expected one of: graph, hybrid, parity."
            )
        return HybridRetriever(graph_db, self.config, self.kb_path).retrieve(
            query, top_k=top_k
        )

    def _query_candidates(self, query: str, top_k: int) -> list[CandidateMatch]:
        if self.config.retrieval.mode.lower() in {"graph", "hybrid"}:
            return self._retrieve_candidates(query, top_k)
        return self._parity_candidates(query, top_k)

    def _retrieve_from_request(self, request: KBRetrieveRequest) -> list[CandidateMatch]:
        if self.config.retrieval.mode.lower() in {"graph", "hybrid"}:
            return self._retrieve_candidates(
                request.issue,
                request.top_k,
                elements_path=request.elements_path,
            )
        return self._parity_candidates(
            request.issue,
            request.top_k,
            repo_path=request.repo_path,
            elements_path=request.elements_path,
            base_commit=request.base_commit,
            patch_text=request.patch_text,
            ground_truth_files=request.ground_truth_files or [],
        )

    def _parity_candidates(
        self,
        query: str,
        top_k: int,
        repo_path: str | None = None,
        elements_path: str | None = None,
        base_commit: str = "HEAD",
        patch_text: str = "",
        ground_truth_files: list[str] | None = None,
    ) -> list[CandidateMatch]:
        repo_path = repo_path or self.config.paths.repo_path
        if not repo_path:
            raise RuntimeError("Parity mode requires paths.repo_path in the config.")
        elements_path = resolve_elements_path(
            self.kb_path,
            cli_value=elements_path,
            config_value=self.config.paths.elements_path,
        )
        workflow = EvolutionGraph(
            repo_path=repo_path,
            kb_path=self.kb_path,
            elements_path=elements_path,
            model_name=self.config.llm.model,
        )
        result = workflow.run_inference_for_single_sample(
            issue=query,
            ground_truth_files=ground_truth_files or [],
            base_commit=base_commit,
            patch_text=patch_text,
        )
        candidates = [
            CandidateMatch(
                file_path=item.get("file", ""),
                symbol=None if item.get("element") == "Entire File" else item.get("element"),
                rationale=item.get("rationale"),
            )
            for item in result.get("candidates_details", [])
            if item.get("file")
        ]
        return candidates[:top_k]
