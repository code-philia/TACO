import typer

from rek.config import apply_runtime_environment, load_config, resolve_value
from rek.models import EvalLocalizationRequest
from rek.services.eval_service import EvalService


app = typer.Typer(help="Run evaluation workflows.")


@app.command("localization")
def eval_localization(
    repo_path: str | None = typer.Option(None, "--repo-path", help="Path to target repository."),
    kb: str | None = typer.Option(None, "--kb", help="Knowledge base path."),
    elements_path: str | None = typer.Option(None, "--elements-path", help="Program elements path."),
    config: str | None = typer.Option(None, "--config", help="Path to TOML config file."),
    test_data: str = typer.Option(..., "--test-data", help="Evaluation dataset path."),
    instance_id: str | None = typer.Option(None, "--instance-id", help="Evaluate only one instance."),
    swebench_gt_path: str | None = typer.Option(None, "--swebench-gt-path", help="Optional SWE-bench GT JSONL path."),
) -> None:
    runtime_config = load_config(config)
    apply_runtime_environment(runtime_config)
    resolved_repo_path = resolve_value(repo_path, runtime_config.paths.repo_path)
    resolved_kb_path = resolve_value(kb, runtime_config.paths.kb_path)
    if not resolved_repo_path:
        raise typer.BadParameter("A repo path is required via --repo-path or --config.")
    if not resolved_kb_path:
        raise typer.BadParameter("A KB path is required via --kb or --config.")
    service = EvalService(config=runtime_config)
    result = service.localization(
        EvalLocalizationRequest(
            repo_path=resolved_repo_path,
            kb_path=resolved_kb_path,
            test_data_path=test_data,
            elements_path=resolve_value(elements_path, runtime_config.paths.elements_path),
            instance_id=instance_id,
            swebench_gt_path=swebench_gt_path,
        )
    )
    typer.echo(result.to_dict())
