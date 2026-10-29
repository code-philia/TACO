from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ResponseModel:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommandMeta(ResponseModel):
    status: str
    message: str
    generated_at: str = field(default_factory=utc_timestamp)


@dataclass(slots=True)
class CandidateMatch(ResponseModel):
    file_path: str
    symbol: str | None = None
    score: float | None = None
    rationale: str | None = None
    line: int | None = None


@dataclass(slots=True)
class BuildResponse(ResponseModel):
    meta: CommandMeta
    repo_path: str
    stage: str
    output_path: str | None = None
    artifacts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrainResponse(ResponseModel):
    meta: CommandMeta
    repo_path: str
    kb_path: str
    train_data_path: str
    model: str
    planned_epochs: int


@dataclass(slots=True)
class EvalLocalizationResponse(ResponseModel):
    meta: CommandMeta
    repo_path: str
    kb_path: str
    test_data_path: str
    instance_id: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class KBQueryResponse(ResponseModel):
    meta: CommandMeta
    kb_path: str
    question: str
    candidates: list[CandidateMatch] = field(default_factory=list)
    summary: str | None = None


@dataclass(slots=True)
class KBSearchResponse(ResponseModel):
    meta: CommandMeta
    kb_path: str
    symbol: str | None = None
    file_path: str | None = None
    matches: list[CandidateMatch] = field(default_factory=list)


@dataclass(slots=True)
class KBRetrieveResponse(ResponseModel):
    meta: CommandMeta
    kb_path: str
    issue: str
    candidates: list[CandidateMatch] = field(default_factory=list)


@dataclass(slots=True)
class KBHintResponse(ResponseModel):
    meta: CommandMeta
    kb_path: str
    issue: str
    hint: str
    candidates: list[CandidateMatch] = field(default_factory=list)
    format: str = "agent_hint_v1"


@dataclass(slots=True)
class KBStatsResponse(ResponseModel):
    meta: CommandMeta
    kb_path: str
    file_count: int | None = None
    function_count: int | None = None
    element_count: int | None = None


@dataclass(slots=True)
class InspectPathsResponse(ResponseModel):
    meta: CommandMeta
    repo_path: str
    repo_exists: bool
    kb_path: str | None = None
    kb_exists: bool | None = None
    elements_path: str | None = None
    elements_exists: bool | None = None
