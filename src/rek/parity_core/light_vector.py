"""
Local copy adapted from `Repo_Reqirement_Knowledge/src/retrieval/light_vector.py`.

This version preserves the original split between sparse and dense retrieval,
but replaces heavyweight local dependencies with a standard-library-friendly
implementation plus an OpenAI-compatible embedding endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


vector_logger = logging.getLogger("LightVectorStore")
vector_logger.setLevel(logging.INFO)
if not vector_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    vector_logger.addHandler(handler)


class OpenAICompatibleEmbeddingClient:
    def __init__(self, model: str, api_key: str | None, base_url: str | None):
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.model and self.api_key and self.base_url)

    def embed_text(self, text: str) -> list[float]:
        if not self.is_configured():
            raise RuntimeError("Embedding client is not configured.")

        safe_text = text[:8000]
        body = json.dumps({"model": self.model, "input": safe_text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding request failed: {details}") from exc

        items = payload.get("data", [])
        if not items:
            raise RuntimeError("Embedding response did not include any vectors.")
        return items[0]["embedding"]


class LightVectorStore:
    """
    Track B retrieval: sparse lexical search plus optional dense embeddings.
    """

    def __init__(
        self,
        graph_db: Any,
        embedding_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.graph_db = graph_db
        self.documents: list[dict[str, Any]] = []
        self.doc_lengths: list[int] = []
        self.avgdl = 0.0
        self.df = Counter()
        self.N = 0
        self.embedding_client = OpenAICompatibleEmbeddingClient(
            model=embedding_model or "",
            api_key=api_key,
            base_url=base_url,
        )
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        tokens = re.findall(r"\b[a-zA-Z_]\w+\b", text.lower())
        stop_words = {
            "the",
            "is",
            "in",
            "and",
            "to",
            "of",
            "a",
            "for",
            "this",
            "that",
            "with",
            "on",
            "as",
            "it",
            "be",
            "by",
        }
        return [token for token in tokens if token not in stop_words and len(token) > 2]

    def _build_index(self) -> None:
        total_len = 0
        for file_path, node in self.graph_db.file_map.items():
            file_rationale = node.get("rationale", "")
            file_keywords = node.get("rationale_keywords", [])
            if file_rationale or file_keywords:
                boosted_text = file_rationale + " " + " ".join(file_keywords * 5)
                tokens = self._tokenize(boosted_text)
                self.documents.append(
                    {
                        "id": f"{file_path}::FILE_SUMMARY",
                        "file": file_path,
                        "function": None,
                        "text": file_rationale if file_rationale else " ".join(file_keywords),
                        "tokens": tokens,
                    }
                )

            for function_name, function_node in node.get("functions", {}).items():
                function_rationale = function_node.get("rationale", "")
                function_keywords = function_node.get("rationale_keywords", [])
                if function_rationale or function_keywords:
                    boosted_text = (
                        f"{function_name} {function_name} {function_rationale} "
                        + " ".join(function_keywords * 5)
                    )
                    tokens = self._tokenize(boosted_text)
                    self.documents.append(
                        {
                            "id": f"{file_path}::{function_name}",
                            "file": file_path,
                            "function": function_name,
                            "text": function_rationale
                            if function_rationale
                            else " ".join(function_keywords),
                            "tokens": tokens,
                        }
                    )

        self.N = len(self.documents)
        for document in self.documents:
            tokens = document["tokens"]
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            for token in set(tokens):
                self.df[token] += 1
        self.avgdl = total_len / self.N if self.N else 1.0

    def _bm25_search(self, query_tokens: list[str], top_k: int) -> list[tuple[float, int]]:
        k1, b = 1.5, 0.75
        scored_documents: list[tuple[float, int]] = []
        for index, document in enumerate(self.documents):
            score = 0.0
            document_counter = Counter(document["tokens"])
            document_length = self.doc_lengths[index]
            for term in query_tokens:
                if term not in document_counter:
                    continue
                n_q = self.df.get(term, 0)
                idf = math.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1)
                tf = document_counter[term]
                score += idf * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * (document_length / self.avgdl))
                )
            if score > 0:
                scored_documents.append((score, index))
        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return scored_documents[:top_k]

    def _dense_search(
        self,
        query: str,
        top_k: int,
        target_files: list[str] | None = None,
    ) -> list[tuple[float, int]]:
        if not self.embedding_client.is_configured():
            return []

        query_vector = self.embedding_client.embed_text(query)
        scores: list[tuple[float, int]] = []
        for index, document in enumerate(self.documents):
            if target_files and document["file"] not in target_files:
                continue
            if target_files and document["function"] is not None:
                continue
            embedding = self._get_cached_embedding(document["id"], document["text"])
            score = cosine_similarity(query_vector, embedding)
            if score > 0:
                scores.append((score, index))
        scores.sort(key=lambda item: item[0], reverse=True)
        return scores[:top_k]

    def _get_cached_embedding(self, doc_id: str, text: str) -> list[float]:
        cache_dir = Path(self.graph_db.kb_path).parent / ".vector_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(
            f"{self.embedding_client.model}|{doc_id}|{text}".encode("utf-8")
        ).hexdigest()
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as handle:
                return json.load(handle)["embedding"]

        embedding = self.embedding_client.embed_text(text)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump({"embedding": embedding}, handle)
        return embedding

    def search(
        self,
        query: str,
        top_k: int = 5,
        target_files: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        bm25_results = self._bm25_search(query_tokens, top_k=50)
        dense_results: list[tuple[float, int]] = []
        try:
            dense_results = self._dense_search(query, top_k=50, target_files=target_files)
        except Exception as exc:
            vector_logger.warning("Dense retrieval failed, using sparse results only: %s", exc)

        rrf_scores: dict[int, float] = defaultdict(float)
        k_rrf = 60

        for rank, (_, index) in enumerate(bm25_results):
            rrf_scores[index] += 1.0 / (k_rrf + rank + 1)
        for rank, (_, index) in enumerate(dense_results):
            rrf_scores[index] += 1.0 / (k_rrf + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda index: rrf_scores[index], reverse=True)
        results: list[dict[str, Any]] = []
        for index in sorted_indices:
            document = self.documents[index]
            if target_files and document["file"] not in target_files:
                continue
            element = document["function"] if document["function"] else "Entire File"
            results.append(
                {
                    "file": document["file"],
                    "element": element,
                    "score": rrf_scores[index],
                    "preview": document["text"][:150] + "...",
                }
            )
            if len(results) >= top_k:
                break
        return results


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
