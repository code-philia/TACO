from rek.models import KBHintRequest, KBQueryRequest, KBSearchRequest
from rek.services.kb_service import KBService


def query_repository_knowledge(kb_path: str, question: str) -> dict:
    return KBService(kb_path=kb_path).query(
        KBQueryRequest(kb_path=kb_path, question=question)
    ).to_dict()


def search_symbol(kb_path: str, symbol: str) -> dict:
    return KBService(kb_path=kb_path).search(
        KBSearchRequest(kb_path=kb_path, symbol=symbol)
    ).to_dict()


def get_kb_stats(kb_path: str) -> dict:
    return KBService(kb_path=kb_path).stats().to_dict()


def retrieve_issue_hint(
    kb_path: str,
    issue: str,
    repo_path: str | None = None,
    elements_path: str | None = None,
    top_k: int = 5,
    max_hint_candidates: int = 5,
) -> dict:
    return KBService(kb_path=kb_path).hint(
        KBHintRequest(
            kb_path=kb_path,
            issue=issue,
            repo_path=repo_path,
            elements_path=elements_path,
            top_k=top_k,
            max_hint_candidates=max_hint_candidates,
        )
    ).to_dict()
