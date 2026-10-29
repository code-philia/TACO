"""Repo2Doc Agent entrypoint."""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from rek.parity_core.repo2doc_local.agent_workflow import create_workflow
from rek.parity_core.repo2doc_local.config_loader import Config, setup_logging

console = Console()
logger = logging.getLogger(__name__)


def main():
    """Run the local Repo2Doc CLI."""
    parser = argparse.ArgumentParser(
        description="Repo2Doc Agent - agent-driven repository requirements generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a repository with the default config
  python main.py YOUR_REPO_PATH_HERE

  # Use a custom config
  python main.py YOUR_REPO_PATH_HERE -c config.yaml

  # Override the output directory
  python main.py YOUR_REPO_PATH_HERE -o ./output
        """
    )
    
    parser.add_argument(
        "repo_path",
        type=str,
        help="Repository path to analyze.",
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to the config file (defaults to the bundled config).",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory override.",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    
    parser.add_argument(
        "-m", "--max-iterations",
        type=int,
        default=None,
        help="Maximum iterations override.",
    )
    
    args = parser.parse_args()
    
    # Validate the repository path
    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists():
        console.print(f"[red]Error: repository path does not exist: {repo_path}[/red]")
        sys.exit(1)
    
    if not repo_path.is_dir():
        console.print(f"[red]Error: path is not a directory: {repo_path}[/red]")
        sys.exit(1)
    
    # Resolve the config file path
    config_path = args.config
    if config_path is None:
        # Try the bundled default config
        default_config = Path(__file__).parent / "config.yaml"
        if default_config.exists():
            config_path = str(default_config)
    
    # Show the welcome panel
    console.print(Panel.fit(
        "[bold blue]Repo2Doc Agent[/bold blue]\n"
        "[dim]Agent-driven repository requirements generation tool[/dim]",
        border_style="blue"
    ))
    
    console.print(f"\n[bold]Repository Path:[/bold] {repo_path}")
    if config_path:
        console.print(f"[bold]Config File:[/bold] {config_path}")
    console.print()
    
    # Load config
    config = Config.load(config_path)
    
    # Override log level if requested
    if args.verbose:
        config.logging.level = "DEBUG"
    
    setup_logging(config.logging)
    
    # Override config values from CLI flags
    if args.output:
        config.output.output_dir = args.output

    if args.max_iterations:
        config.agent.max_iterations = args.max_iterations
    
    # Run the workflow
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing...[/cyan]", total=None)
            
            # Create and run the workflow
            workflow = create_workflow(config_path)
            
            # Override config values from CLI flags
            if args.output:
                workflow.config.output.output_dir = args.output
            if args.max_iterations:
                workflow.config.agent.max_iterations = args.max_iterations
            
            final_state = workflow.run(str(repo_path), config_path)
            
            progress.update(task, completed=True)
        
        # Display result summary
        if final_state.get("status") == "completed":
            console.print("\n[bold green]✅ Processing completed.[/bold green]\n")
            
            console.print("[bold]Statistics:[/bold]")
            console.print(f"  • Iterations: {final_state.get('iteration_count', 0)}")
            console.print(f"  • Document length: {len(final_state.get('current_document', '')):,} characters")
            console.print(f"  • Confidence: {final_state.get('confidence_score', 0):.2f}")
            console.print(f"  • Document versions: {len(final_state.get('document_versions', []))}")
            
            # Display output paths
            output_dir = repo_path / config.output.output_dir
            console.print(f"\n[bold]Output Directory:[/bold] {output_dir}")
            console.print(f"[bold]Requirements Document:[/bold] {output_dir / config.output.filename}")
            
        else:
            console.print("\n[bold red]❌ Processing failed.[/bold red]")
            console.print(f"[red]Error: {final_state.get('error')}[/red]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print("\n[bold red]❌ An error occurred.[/bold red]")
        console.print(f"[red]{e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
