import json

import typer

from rek.config import apply_runtime_environment, load_config, resolve_value
from rek.models import KBHintRequest, KBQueryRequest, KBRetrieveRequest, KBSearchRequest
from rek.services.kb_service import KBService


app = typer.Typer(help="Access knowledge base capabilities.")


@app.command("query")
def query(
    kb: str | None = typer.Option(None, "--kb", help="Knowledge base path."),
    config: str | None = typer.Option(None, "--config", help="Path to TOML config file."),
    question: str = typer.Option(..., "--question", help="Natural language question."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    runtime_config = load_config(config)
    apply_runtime_environment(runtime_config)
    kb_path = resolve_value(kb, runtime_config.paths.kb_path)
    if not kb_path:
        raise typer.BadParameter("A KB path is required via --kb or --config.")
    service = KBService(kb_path=kb_path, config=runtime_config)
    result = service.query(KBQueryRequest(kb_path=kb_path, question=question))
    if json_output:
        typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    typer.echo(f"question: {result.question}")
    typer.echo(f"status: {result.meta.status}")
    typer.echo(f"message: {result.meta.message}")


@app.command("search")
def search(
    kb: str | None = typer.Option(None, "--kb", help="Knowledge base path."),
    config: str | None = typer.Option(None, "--config", help="Path to TOML config file."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol name to search."),
    file_path: str | None = typer.Option(None, "--file", help="File path to inspect."),
    top_k: int = typer.Option(10, "--top-k", help="Maximum number of matches."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    runtime_config = load_config(config)
    apply_runtime_environment(runtime_config)
    kb_path = resolve_value(kb, runtime_config.paths.kb_path)
    if not kb_path:
        raise typer.BadParameter("A KB path is required via --kb or --config.")
    service = KBService(kb_path=kb_path, config=runtime_config)
    result = service.search(
        KBSearchRequest(kb_path=kb_path, symbol=symbol, file_path=file_path, top_k=top_k)
    )
    if json_output:
        typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    typer.echo(f"symbol: {result.symbol}")
    typer.echo(f"status: {result.meta.status}")
    typer.echo(f"message: {result.meta.message}")


@app.command("retrieve")
def retrieve(
    kb: str | None = typer.Option(None, "--kb", help="Knowledge base path."),
    config: str | None = typer.Option(None, "--config", help="Path to TOML config file."),
    issue: str = typer.Option(..., "--issue", help="Issue statement or bug description."),
    repo_path: str | None = typer.Option(None, "--repo-path", help="Repository path override."),
    elements_path: str | None = typer.Option(None, "--elements-path", help="Program elements path override."),
    base_commit: str = typer.Option("HEAD", "--base-commit", help="Historic base commit for parity mode."),
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of candidates."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    runtime_config = load_config(config)
    apply_runtime_environment(runtime_config)
    kb_path = resolve_value(kb, runtime_config.paths.kb_path)
    if not kb_path:
        raise typer.BadParameter("A KB path is required via --kb or --config.")
    service = KBService(kb_path=kb_path, config=runtime_config)
    result = service.retrieve(
        KBRetrieveRequest(
            kb_path=kb_path,
            issue=issue,
            repo_path=resolve_value(repo_path, runtime_config.paths.repo_path),
            elements_path=resolve_value(elements_path, runtime_config.paths.elements_path),
            base_commit=base_commit,
            top_k=top_k,
        )
    )
    if json_output:
        typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    typer.echo(f"issue: {result.issue}")
    typer.echo(f"status: {result.meta.status}")
    typer.echo(f"message: {result.meta.message}")


@app.command("hint")
def hint(
    kb: str | None = typer.Option(None, "--kb", help="Knowledge base path."),
    config: str | None = typer.Option(None, "--config", help="Path to TOML config file."),
    issue: str = typer.Option(..., "--issue", help="Issue statement or bug description."),
    repo_path: str | None = typer.Option(None, "--repo-path", help="Repository path override."),
    elements_path: str | None = typer.Option(None, "--elements-path", help="Program elements path override."),
    base_commit: str = typer.Option("HEAD", "--base-commit", help="Historic base commit for parity mode."),
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of retrieval candidates."),
    max_hint_candidates: int = typer.Option(5, "--max-hint-candidates", help="Maximum number of candidates rendered into the hint."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    runtime_config = load_config(config)
    apply_runtime_environment(runtime_config)
    kb_path = resolve_value(kb, runtime_config.paths.kb_path)
    if not kb_path:
        raise typer.BadParameter("A KB path is required via --kb or --config.")
    service = KBService(kb_path=kb_path, config=runtime_config)
    result = service.hint(
        KBHintRequest(
            kb_path=kb_path,
            issue=issue,
            repo_path=resolve_value(repo_path, runtime_config.paths.repo_path),
            elements_path=resolve_value(elements_path, runtime_config.paths.elements_path),
            base_commit=base_commit,
            top_k=top_k,
            max_hint_candidates=max_hint_candidates,
        )
    )
    if json_output:
        typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    typer.echo(result.hint)


@app.command("stats")
def stats(
    kb: str | None = typer.Option(None, "--kb", help="Knowledge base path."),
    config: str | None = typer.Option(None, "--config", help="Path to TOML config file."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    runtime_config = load_config(config)
    apply_runtime_environment(runtime_config)
    kb_path = resolve_value(kb, runtime_config.paths.kb_path)
    if not kb_path:
        raise typer.BadParameter("A KB path is required via --kb or --config.")
    service = KBService(kb_path=kb_path, config=runtime_config)
    result = service.stats()
    if json_output:
        typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    typer.echo(f"kb: {result.kb_path}")
    typer.echo(f"status: {result.meta.status}")
    typer.echo(f"message: {result.meta.message}")
