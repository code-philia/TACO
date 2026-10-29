import json

import typer

from rek.config import load_config, resolve_value
from rek.models import InspectPathsRequest
from rek.services.inspect_service import InspectService


app = typer.Typer(help="Inspect configuration, paths, and outputs.")


@app.command("paths")
def inspect_paths(
    repo_path: str = typer.Option(..., "--repo-path", help="Path to target repository."),
    kb: str | None = typer.Option(None, "--kb", help="Knowledge base path."),
    elements_path: str | None = typer.Option(None, "--elements-path", help="Program elements path or directory."),
    config: str | None = typer.Option(None, "--config", help="Path to TOML config file."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    runtime_config = load_config(config)
    service = InspectService()
    result = service.paths(
        InspectPathsRequest(
            repo_path=repo_path,
            kb_path=resolve_value(kb, runtime_config.paths.kb_path),
            elements_path=resolve_value(elements_path, runtime_config.paths.elements_path),
        )
    )
    if json_output:
        typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    typer.echo(f"repo_path: {result.repo_path}")
    typer.echo(f"repo_exists: {result.repo_exists}")
    typer.echo(f"kb_path: {result.kb_path}")
    typer.echo(f"kb_exists: {result.kb_exists}")
    typer.echo(f"elements_path: {result.elements_path}")
    typer.echo(f"elements_exists: {result.elements_exists}")
