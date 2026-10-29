import typer

from rek.config import apply_runtime_environment, load_config, resolve_value
from rek.models import BuildRequest
from rek.services.build_service import BuildService


app = typer.Typer(help="Build or upgrade knowledge base artifacts.")


@app.callback()
def build_callback() -> None:
    """Build command group."""


@app.command("run")
def run_build(
    repo_path: str | None = typer.Option(None, "--repo-path", help="Path to target repository."),
    config: str | None = typer.Option(None, "--config", help="Path to TOML config file."),
    stage: str = typer.Option("all", "--stage", help="Build stage to run: scan, complete, two-index, all."),
    output: str | None = typer.Option(None, "--output", help="Output directory."),
    repo_name: str | None = typer.Option(None, "--repo-name", help="Repository name for parity pipeline."),
    base_repos: str | None = typer.Option(None, "--base-repos", help="Base directory that contains repositories."),
    train_data_dir: str | None = typer.Option(None, "--train-data-dir", help="Directory containing PR training data."),
    model: str | None = typer.Option(None, "--model", help="LLM model for parity build pipeline."),
    epochs: int = typer.Option(1, "--epochs", help="Epoch count when build includes training."),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size when build includes training."),
    workers: int = typer.Option(8, "--workers", help="Worker count when build includes training."),
    output_dir_name: str | None = typer.Option(None, "--output-dir-name", help="Output directory name under the repository root."),
    log_root: str | None = typer.Option(None, "--log-root", help="Training log root."),
    build_kb_only: bool = typer.Option(False, "--build-kb-only", help="Stop after KB build for parity stage all."),
    assume_pr_ready: bool = typer.Option(False, "--assume-pr-ready", help="Skip PR collection and threshold checks."),
    train_only: bool = typer.Option(False, "--train-only", help="Run only the training stage in the parity pipeline."),
) -> None:
    runtime_config = load_config(config)
    apply_runtime_environment(runtime_config)
    resolved_repo_path = resolve_value(repo_path, runtime_config.paths.repo_path)
    if not resolved_repo_path:
        raise typer.BadParameter("A repo path is required via --repo-path or --config.")
    service = BuildService(runtime_config)
    result = service.run(
        BuildRequest(
            repo_path=resolved_repo_path,
            stage=stage,
            output=output,
            config=config,
            repo_name=repo_name,
            base_repos=base_repos,
            train_data_dir=train_data_dir,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            workers=workers,
            output_dir_name=output_dir_name,
            log_root=log_root,
            build_kb_only=build_kb_only,
            assume_pr_ready=assume_pr_ready,
            train_only=train_only,
        )
    )
    typer.echo(result.to_dict())
