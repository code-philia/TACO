from rek.config import REKConfig, resolve_elements_path
from rek.parity_core.eval_localization import run_localization_eval
from rek.models import CommandMeta, EvalLocalizationRequest, EvalLocalizationResponse


class EvalService:
    def __init__(self, config: REKConfig | None = None):
        self.config = config or REKConfig()

    def localization(self, request: EvalLocalizationRequest) -> EvalLocalizationResponse:
        elements_path = resolve_elements_path(
            request.kb_path,
            cli_value=request.elements_path,
            config_value=self.config.paths.elements_path,
        )
        result = run_localization_eval(
            repo_path=request.repo_path,
            kb_path=request.kb_path,
            elements_path=elements_path,
            test_data_path=request.test_data_path,
            model_name=request.model or self.config.llm.model,
            instance_id=request.instance_id,
            swebench_gt_path=request.swebench_gt_path,
        )
        return EvalLocalizationResponse(
            meta=CommandMeta(status="ok", message="Localization evaluation completed."),
            repo_path=request.repo_path,
            kb_path=request.kb_path,
            test_data_path=request.test_data_path,
            instance_id=request.instance_id,
            metrics=result.get("metrics", {}),
        )
