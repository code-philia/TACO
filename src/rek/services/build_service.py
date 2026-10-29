from pathlib import Path

from rek.config import REKConfig
from rek.parity_core.auto_runner import (
    AutoPipeline,
    DEFAULT_BASE_REPOS,
    DEFAULT_TRAIN_DATA_DIR,
    run_auto_pipeline,
)
from rek.parity_core.scanner import save_elements, save_statistics, scan_repo
from rek.parity_core.trainer import setup_logger
from rek.models import BuildRequest, BuildResponse, CommandMeta


class BuildService:
    def __init__(self, config: REKConfig | None = None):
        self.config = config or REKConfig()

    def run(self, request: BuildRequest) -> BuildResponse:
        repo_path = Path(request.repo_path)
        repo_name = request.repo_name or repo_path.name
        base_repos = request.base_repos or str(repo_path.parent) or DEFAULT_BASE_REPOS
        train_data_dir = request.train_data_dir or DEFAULT_TRAIN_DATA_DIR
        model_name = request.model or self.config.llm.model
        output_dir_name = request.output_dir_name or (Path(request.output).name if request.output else None)

        if request.stage == "all":
            result = run_auto_pipeline(
                repo_name=repo_name,
                repo_path=str(repo_path),
                base_repos=base_repos,
                train_data_dir=train_data_dir,
                model_name=model_name,
                epochs=max(request.epochs, 1),
                batch_size=max(request.batch_size, 1),
                workers=max(request.workers, 1),
                output_dir_name=output_dir_name,
                output_dir_path=request.output,
                log_root=request.log_root,
                build_kb_only=request.build_kb_only,
                assume_pr_ready=request.assume_pr_ready,
                train_only=request.train_only,
            )
            return BuildResponse(
                meta=CommandMeta(status="ok", message="Parity auto pipeline completed."),
                repo_path=str(repo_path),
                stage=request.stage,
                output_path=result["output_dir"],
                artifacts=result["artifacts"],
            )

        output_root = Path(request.output) if request.output else repo_path / (
            output_dir_name or "repo2docAgent-output"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        artifacts: list[str] = []

        if request.stage == "scan":
            elements = scan_repo(str(repo_path))
            elements_path = output_root / "program_elements.jsonl"
            stats_path = output_root / "scanned_files_stats.jsonl"
            save_elements(elements, elements_path)
            save_statistics(elements, stats_path)
            artifacts.extend([str(elements_path), str(stats_path)])

        if request.stage == "complete":
            logger, _ = setup_logger(request.log_root)
            pipeline = AutoPipeline(
                repo_name=repo_name,
                repo_path=str(repo_path),
                base_repos=base_repos,
                train_data_dir=train_data_dir,
                model_name=model_name,
                logger=logger,
                output_dir_name=output_dir_name or output_root.name,
                output_dir_path=str(output_root),
                log_root=request.log_root or None,
            )
            pipeline.prepare_kb()
            artifacts.extend(
                [
                    str(path)
                    for path in [
                        pipeline.kb_detail,
                        pipeline.output_dir / "knowledge_graph_complete_detail.json",
                        pipeline.elements_path,
                    ]
                    if Path(path).exists()
                ]
            )

        if request.stage == "two-index":
            logger, _ = setup_logger(request.log_root)
            pipeline = AutoPipeline(
                repo_name=repo_name,
                repo_path=str(repo_path),
                base_repos=base_repos,
                train_data_dir=train_data_dir,
                model_name=model_name,
                logger=logger,
                output_dir_name=output_dir_name or output_root.name,
                output_dir_path=str(output_root),
                log_root=request.log_root or None,
            )
            input_kb = pipeline.output_dir / "knowledge_graph_complete_detail.json"
            if input_kb.exists():
                pipeline.upgrade_to_two_index()
                artifacts.append(str(pipeline.output_dir / "knowledge_graph_two_index.json"))
            else:
                artifacts.append(f"missing:{input_kb}")

        return BuildResponse(
            meta=CommandMeta(status="ok", message="Build stage completed."),
            repo_path=str(repo_path),
            stage=request.stage,
            output_path=str(output_root),
            artifacts=artifacts,
        )
