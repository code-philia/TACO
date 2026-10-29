from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RequestModel:
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class BuildRequest(RequestModel):
    repo_path: str
    stage: str = "all"
    output: str | None = None
    config: str | None = None
    repo_name: str | None = None
    base_repos: str | None = None
    train_data_dir: str | None = None
    model: str | None = None
    epochs: int = 1
    batch_size: int = 8
    workers: int = 8
    output_dir_name: str | None = None
    log_root: str | None = None
    build_kb_only: bool = False
    assume_pr_ready: bool = False
    train_only: bool = False
    verbose: bool = False


@dataclass(slots=True)
class TrainRequest(RequestModel):
    repo_path: str
    kb_path: str
    train_data_path: str
    model: str
    elements_path: str | None = None
    epochs: int = 1
    batch_size: int = 1
    workers: int = 1
    log_root: str | None = None


@dataclass(slots=True)
class EvalLocalizationRequest(RequestModel):
    repo_path: str
    kb_path: str
    test_data_path: str
    elements_path: str | None = None
    model: str | None = None
    instance_id: str | None = None
    swebench_gt_path: str | None = None


@dataclass(slots=True)
class KBQueryRequest(RequestModel):
    kb_path: str
    question: str
    repo_path: str | None = None
    top_k: int = 5


@dataclass(slots=True)
class KBSearchRequest(RequestModel):
    kb_path: str
    symbol: str | None = None
    file_path: str | None = None
    top_k: int = 10


@dataclass(slots=True)
class KBRetrieveRequest(RequestModel):
    kb_path: str
    issue: str
    repo_path: str | None = None
    elements_path: str | None = None
    base_commit: str = "HEAD"
    patch_text: str = ""
    ground_truth_files: list[str] | None = None
    top_k: int = 5


@dataclass(slots=True)
class KBHintRequest(RequestModel):
    kb_path: str
    issue: str
    repo_path: str | None = None
    elements_path: str | None = None
    base_commit: str = "HEAD"
    patch_text: str = ""
    ground_truth_files: list[str] | None = None
    top_k: int = 5
    max_hint_candidates: int = 5


@dataclass(slots=True)
class InspectPathsRequest(RequestModel):
    repo_path: str
    kb_path: str | None = None
    elements_path: str | None = None
