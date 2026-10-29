import typer

from rek.config import apply_runtime_environment, load_config, resolve_value
from rek.models import TrainRequest
from rek.services.train_service import TrainService


app = typer.Typer(help="Run training workflows.")


@app.command("run")
def run_train(
    repo_path: str | None = typer.Option(None, "--repo-path", help="Path to target repository."),
    kb: str | None = typer.Option(None, "--kb", help="Knowledge base path."),
    config: str | None = typer.Option(None, "--config", help="Path to TOML config file."),
    train_data: str = typer.Option(..., "--train-data", help="Training dataset path."),
    model: str = typer.Option("gpt-5-mini", "--model", help="Model name."),
    elements_path: str | None = typer.Option(None, "--elements-path", help="Optional program elements JSONL path."),
    epochs: int = typer.Option(1, "--epochs", help="Number of training epochs."),
    batch_size: int = typer.Option(5, "--batch-size", help="Batch size."),
    workers: int = typer.Option(1, "--workers", help="Reserved worker count for future use."),
    log_root: str | None = typer.Option(None, "--log-root", help="Reserved log root path."),
) -> None:
    runtime_config = load_config(config)
    apply_runtime_environment(runtime_config)
    resolved_repo_path = resolve_value(repo_path, runtime_config.paths.repo_path)
    resolved_kb_path = resolve_value(kb, runtime_config.paths.kb_path)
    if not resolved_repo_path:
        raise typer.BadParameter("A repo path is required via --repo-path or --config.")
    if not resolved_kb_path:
        raise typer.BadParameter("A KB path is required via --kb or --config.")
    service = TrainService(config=runtime_config)
    result = service.run(
        TrainRequest(
            repo_path=resolved_repo_path,
            kb_path=resolved_kb_path,
            train_data_path=train_data,
            model=model,
            elements_path=resolve_value(elements_path, runtime_config.paths.elements_path),
            epochs=epochs,
            batch_size=batch_size,
            workers=workers,
            log_root=log_root,
        )
    )
    typer.echo(result.to_dict())
