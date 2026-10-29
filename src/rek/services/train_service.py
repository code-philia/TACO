from rek.config import REKConfig, resolve_elements_path
from rek.parity_core.trainer import run_training
from rek.models import CommandMeta, TrainRequest, TrainResponse


class TrainService:
    def __init__(self, config: REKConfig | None = None):
        self.config = config or REKConfig()

    def run(self, request: TrainRequest) -> TrainResponse:
        elements_path = resolve_elements_path(
            request.kb_path,
            cli_value=request.elements_path,
            config_value=self.config.paths.elements_path,
        )
        run_training(
            repo_path=request.repo_path,
            kb_path=request.kb_path,
            elements_path=elements_path,
            train_data_path=request.train_data_path,
            model_name=request.model,
            epochs=request.epochs,
            batch_size=max(request.batch_size, 1),
            workers=max(request.workers, 1),
            log_root=request.log_root,
        )
        return TrainResponse(
            meta=CommandMeta(status="ok", message="Training completed."),
            repo_path=request.repo_path,
            kb_path=request.kb_path,
            train_data_path=request.train_data_path,
            model=request.model,
            planned_epochs=request.epochs,
        )
