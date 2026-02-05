"""Command-line interface for CryoEMAgent."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from cryoemagent import CryoEMAgent, Config


console = Console()


def setup_logging(level: str = "INFO"):
    """Configure logging with rich handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx, verbose: bool):
    """CryoEMAgent - Autonomous Cryo-EM Structure Determination."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging("DEBUG" if verbose else "INFO")


@main.command()
@click.option("--project", "-p", required=True, help="CryoSPARC project UID (e.g., P3)")
@click.option("--workspace", "-w", required=True, help="CryoSPARC workspace UID (e.g., W1)")
@click.option("--movies", "-m", required=True, help="Path to movie files (glob pattern)")
@click.option("--pixel-size", type=float, help="Pixel size in Angstroms")
@click.option("--voltage", type=int, help="Acceleration voltage in kV")
@click.option("--dose", type=float, help="Total dose in e-/Å²")
@click.option("--target-resolution", type=float, help="Target resolution in Angstroms")
@click.pass_context
def run(
    ctx,
    project: str,
    workspace: str,
    movies: str,
    pixel_size: Optional[float],
    voltage: Optional[int],
    dose: Optional[float],
    target_resolution: Optional[float],
):
    """Run the full GPCR structure determination pipeline."""
    console.print(Panel.fit(
        "[bold blue]CryoEMAgent[/bold blue]\n"
        "Autonomous Cryo-EM Structure Determination",
        border_style="blue",
    ))
    
    config_table = Table(title="Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Project", project)
    config_table.add_row("Workspace", workspace)
    config_table.add_row("Movies", movies)
    config_table.add_row("Pixel Size", f"{pixel_size or 'default'} Å")
    config_table.add_row("Voltage", f"{voltage or 'default'} kV")
    config_table.add_row("Target Resolution", f"{target_resolution or 'default'} Å")
    console.print(config_table)
    
    try:
        config = Config.from_env()
        agent = CryoEMAgent(config)
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)
    
    console.print("\n[bold]Starting pipeline...[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=None)
        
        result = agent.run(
            project_uid=project,
            workspace_uid=workspace,
            movies_path=movies,
            pixel_size=pixel_size,
            voltage=voltage,
            total_dose=dose,
            target_resolution=target_resolution,
        )
    
    console.print()
    
    if result.success:
        result_table = Table(title="[green]Pipeline Completed Successfully[/green]")
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Value", style="green")
        result_table.add_row("Final Job", result.final_job_uid or "N/A")
        result_table.add_row("Resolution", f"{result.resolution:.2f} Å" if result.resolution else "N/A")
        result_table.add_row("Particle Count", f"{result.particle_count:,}")
        result_table.add_row("Total Jobs", str(result.total_jobs))
        result_table.add_row("Execution Time", f"{result.execution_time_seconds:.1f}s")
        console.print(result_table)
    else:
        console.print(f"[red]Pipeline failed: {result.error}[/red]")
        sys.exit(1)


@main.command()
@click.option("--project", "-p", required=True, help="CryoSPARC project UID")
@click.pass_context
def interactive(ctx, project: str):
    """Run in interactive mode with step-by-step control."""
    console.print(Panel.fit(
        "[bold blue]CryoEMAgent Interactive Mode[/bold blue]\n"
        f"Project: {project}",
        border_style="blue",
    ))
    
    console.print("[yellow]Interactive mode not yet implemented.[/yellow]")
    console.print("Use 'cryoem-agent run' for automated processing.")


@main.command()
def status():
    """Check CryoSPARC connection status."""
    console.print("[bold]Checking CryoSPARC connection...[/bold]")
    
    try:
        config = Config.from_env()
        
        from cryoemagent.tools.cryosparc import CryoSPARCTools
        tools = CryoSPARCTools(config.cryosparc)
        result = tools.test_connection()
        
        if result.is_success():
            console.print(f"[green]✓ Connected to CryoSPARC at {config.cryosparc.url}[/green]")
        else:
            console.print(f"[red]✗ Connection failed: {result.error}[/red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--project", "-p", required=True, help="CryoSPARC project UID")
@click.option("--workspace", "-w", help="CryoSPARC workspace UID")
@click.option("--status-filter", help="Filter by job status")
def jobs(project: str, workspace: Optional[str], status_filter: Optional[str]):
    """List jobs in a project."""
    try:
        config = Config.from_env()
        
        from cryoemagent.tools.cryosparc import CryoSPARCTools
        tools = CryoSPARCTools(config.cryosparc)
        
        result = tools.find_jobs(
            project_uid=project,
            workspace_uid=workspace,
            status=status_filter,
        )
        
        if not result.is_success():
            console.print(f"[red]Error: {result.error}[/red]")
            sys.exit(1)
        
        jobs_data = result.data.get("jobs", [])
        
        table = Table(title=f"Jobs in {project}")
        table.add_column("UID", style="cyan")
        table.add_column("Type", style="white")
        table.add_column("Status", style="green")
        table.add_column("Title", style="dim")
        
        for job in jobs_data:
            status_style = {
                "completed": "green",
                "running": "yellow",
                "failed": "red",
                "killed": "red",
                "queued": "blue",
            }.get(job.get("status", ""), "white")
            
            table.add_row(
                job.get("uid", ""),
                job.get("type", ""),
                f"[{status_style}]{job.get('status', '')}[/{status_style}]",
                job.get("title", "")[:40],
            )
        
        console.print(table)
        console.print(f"\nTotal: {len(jobs_data)} jobs")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
def version():
    """Show version information."""
    from cryoemagent import __version__
    console.print(f"CryoEMAgent version {__version__}")


if __name__ == "__main__":
    main()
