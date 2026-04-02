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

console = Console()


def setup_logging(level: str = "INFO"):
    """Configure logging with rich handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _find_config(explicit: Optional[str] = None) -> str:
    """
    Locate the MCP server config YAML.

    Priority:
    1. Explicit --config argument
    2. CRYOEM_AGENT_MCP_CONFIG environment variable
    3. config.yaml in the current directory
    4. profile.yaml in the current directory
    """
    import os  # noqa: PLC0415

    if explicit:
        return explicit
    env = os.getenv("CRYOEM_AGENT_MCP_CONFIG", "")
    if env:
        return env
    for name in ("config.yaml", "profile.yaml", "config.yml"):
        if Path(name).exists():
            return name
    return "config.yaml"  # will fail later with a clear error


def _load_orchestrator_config(config_path: str) -> dict:
    """Load the MCP server YAML config into a dict."""
    try:
        import yaml  # noqa: PLC0415
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except FileNotFoundError:
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Failed to load config '{config_path}': {exc}[/red]")
        sys.exit(1)


def _build_agent(config_path: str, agent_cfg=None):
    """Construct a CryoEMAgent from the given config path."""
    from cryoemagent.core.agent import CryoEMAgent  # noqa: PLC0415

    orch_config = _load_orchestrator_config(config_path)
    return CryoEMAgent(orch_config, agent_config=agent_cfg)


def _checkpoint_resume_loop(agent, result):
    """
    Handle checkpoint loop: print instructions, wait for human, resume.

    Returns the final AgentResult once the run completes or fails.
    """
    while result.checkpoint_required:
        console.print()
        console.print(Panel(
            result.checkpoint_instructions,
            title="[bold yellow]MANUAL ACTION REQUIRED[/bold yellow]",
            border_style="yellow",
        ))
        console.print()
        click.prompt(
            "Complete the steps above in CryoSPARC, then press ENTER to resume",
            default="done",
            show_default=True,
        )
        console.print("[bold]Resuming pipeline...[/bold]")
        result = agent.resume(result.run_id)

    return result


def _print_result(result):
    """Pretty-print the final AgentResult."""
    from cryoemagent.core.agent import AgentResult  # noqa: PLC0415

    if result.success:
        table = Table(title="[green]Pipeline Completed Successfully[/green]")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Run ID", result.run_id)
        table.add_row("Final Step", result.final_step)
        if result.report_paths:
            for k, v in result.report_paths.items():
                table.add_row(k.replace("_", " ").title(), v)
        console.print(table)
        if result.summary:
            console.print()
            console.print(Panel(result.summary, title="Run Summary", border_style="green"))
    else:
        console.print(f"[red]Pipeline failed: {result.error}[/red]")
        if result.run_id:
            console.print(f"[dim]Run ID: {result.run_id}[/dim]")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx, verbose: bool):
    """CryoEMAgent - Autonomous Cryo-EM Structure Determination."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging("DEBUG" if verbose else "INFO")


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to MCP server YAML config file (default: searches for config.yaml)",
)
@click.option("--movies", "-m", default=None, help="Override movie blob path")
@click.option("--pixel-size", type=float, default=None, help="Override pixel size in Angstroms")
@click.option("--voltage", type=int, default=None, help="Override acceleration voltage in kV")
@click.option("--dose", type=float, default=None, help="Override total dose in e-/Å²")
@click.option(
    "--resume-if-exists",
    is_flag=True,
    default=False,
    help="If a recent run exists, resume it instead of starting a new one",
)
@click.pass_context
def run(
    ctx,
    config: Optional[str],
    movies: Optional[str],
    pixel_size: Optional[float],
    voltage: Optional[int],
    dose: Optional[float],
    resume_if_exists: bool,
):
    """Run the full GPCR structure determination pipeline."""
    console.print(Panel.fit(
        "[bold blue]CryoEMAgent[/bold blue]\n"
        "Autonomous Cryo-EM Structure Determination",
        border_style="blue",
    ))

    config_path = _find_config(config)
    console.print(f"[dim]Using config: {config_path}[/dim]")

    # Build runtime overrides from CLI arguments
    data_overrides: dict = {}
    if movies is not None:
        data_overrides.setdefault("data", {})["movie_blob_path"] = movies
    if pixel_size is not None:
        data_overrides.setdefault("data", {})["psize_A"] = pixel_size
    if voltage is not None:
        data_overrides.setdefault("data", {})["accel_kv"] = voltage
    if dose is not None:
        data_overrides.setdefault("data", {})["total_dose_e_per_A2"] = dose

    try:
        agent = _build_agent(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to create agent: {exc}[/red]")
        sys.exit(1)

    # Handle --resume-if-exists
    if resume_if_exists:
        existing_runs = agent.list_runs()
        if existing_runs:
            run_id = existing_runs[-1]
            console.print(f"[yellow]Found existing run {run_id}, resuming...[/yellow]")
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                p.add_task("Resuming pipeline...", total=None)
                result = agent.resume(run_id)
            result = _checkpoint_resume_loop(agent, result)
            _print_result(result)
            if not result.success:
                sys.exit(1)
            return

    console.print("\n[bold]Starting pipeline...[/bold]\n")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        p.add_task("Running pipeline...", total=None)
        result = agent.run(runtime_overrides=data_overrides if data_overrides else None)

    result = _checkpoint_resume_loop(agent, result)
    _print_result(result)

    if not result.success and not result.checkpoint_required:
        sys.exit(1)


# ---------------------------------------------------------------------------
# resume command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--run-id", "-r", required=True, help="Run ID to resume")
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to MCP server YAML config file",
)
@click.pass_context
def resume(ctx, run_id: str, config: Optional[str]):
    """Resume a paused or checkpoint-stopped pipeline run."""
    config_path = _find_config(config)
    console.print(f"[dim]Using config: {config_path}[/dim]")

    try:
        agent = _build_agent(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to create agent: {exc}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Resuming run {run_id}...[/bold]\n")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        p.add_task("Resuming...", total=None)
        result = agent.resume(run_id)

    result = _checkpoint_resume_loop(agent, result)
    _print_result(result)

    if not result.success and not result.checkpoint_required:
        sys.exit(1)


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--run-id", "-r", required=True, help="Run ID to report on")
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to MCP server YAML config file",
)
@click.pass_context
def report(ctx, run_id: str, config: Optional[str]):
    """Write and display the report for a completed run."""
    config_path = _find_config(config)

    try:
        agent = _build_agent(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to create agent: {exc}[/red]")
        sys.exit(1)

    paths = agent.report(run_id)

    if "error" in paths:
        console.print(f"[red]{paths['error']}[/red]")
        sys.exit(1)

    table = Table(title=f"Report for run {run_id}")
    table.add_column("File", style="cyan")
    table.add_column("Path", style="green")
    for k, v in paths.items():
        table.add_row(k.replace("_", " ").title(), v)
    console.print(table)


# ---------------------------------------------------------------------------
# list-runs command
# ---------------------------------------------------------------------------

@main.command("list-runs")
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to MCP server YAML config file",
)
@click.pass_context
def list_runs(ctx, config: Optional[str]):
    """List all pipeline runs."""
    config_path = _find_config(config)

    try:
        agent = _build_agent(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to create agent: {exc}[/red]")
        sys.exit(1)

    runs = agent.list_runs()

    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(title="Pipeline Runs")
    table.add_column("#", style="dim", width=4)
    table.add_column("Run ID", style="cyan")

    for i, run_id in enumerate(runs, start=1):
        # Try to load status
        status_str = ""
        try:
            state = agent.orch_client.load_state(run_id)
            if state:
                colour = {
                    "completed": "green",
                    "failed": "red",
                    "running": "yellow",
                }.get(state.status, "white")
                status_str = f"[{colour}]{state.status}[/{colour}]  {state.current_step}"
        except Exception:
            status_str = "[dim]unknown[/dim]"

        table.add_row(str(i), run_id)
        if status_str:
            # Add status as a second row indented
            table.add_row("", f"  {status_str}")

    console.print(table)
    console.print(f"\nTotal: {len(runs)} run(s)")


# ---------------------------------------------------------------------------
# status command (rewritten)
# ---------------------------------------------------------------------------

@main.command()
@click.option("--run-id", "-r", required=True, help="Run ID to check")
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to MCP server YAML config file",
)
@click.pass_context
def status(ctx, run_id: str, config: Optional[str]):
    """Check the status of a specific pipeline run."""
    config_path = _find_config(config)

    try:
        agent = _build_agent(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to create agent: {exc}[/red]")
        sys.exit(1)

    state_dict = agent.status(run_id)

    if "error" in state_dict:
        console.print(f"[red]{state_dict['error']}[/red]")
        sys.exit(1)

    colour = {
        "completed": "green",
        "failed": "red",
        "running": "yellow",
    }.get(state_dict.get("status", ""), "white")

    table = Table(title=f"Run Status: {run_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Status", f"[{colour}]{state_dict.get('status', 'unknown')}[/{colour}]")
    table.add_row("Stage", state_dict.get("current_stage", ""))
    table.add_row("Step", state_dict.get("current_step", ""))
    table.add_row("Workflow", state_dict.get("workflow_id", ""))
    table.add_row("Created", (state_dict.get("created_at", "") or "")[:19])
    table.add_row("Updated", (state_dict.get("updated_at", "") or "")[:19])

    checkpoint = state_dict.get("checkpoint_required", False)
    if checkpoint:
        table.add_row(
            "Checkpoint",
            f"[yellow]REQUIRED: {state_dict.get('checkpoint_message', '')}[/yellow]",
        )

    console.print(table)

    jobs = state_dict.get("jobs", {})
    if jobs:
        jobs_table = Table(title="Completed Steps")
        jobs_table.add_column("Step", style="cyan")
        jobs_table.add_column("Job UID", style="green")
        for k, v in jobs.items():
            jobs_table.add_row(k, v)
        console.print(jobs_table)

    errors = state_dict.get("errors", {})
    if errors:
        console.print()
        console.print("[red]Errors:[/red]")
        for k, v in errors.items():
            console.print(f"  [red]{k}[/red]: {v}")


# ---------------------------------------------------------------------------
# interactive command (kept for compatibility)
# ---------------------------------------------------------------------------

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
    console.print("[yellow]Use 'cryoem-agent run' for automated processing.[/yellow]")
    console.print("[yellow]Use 'cryoem-agent status --run-id <ID>' to check run status.[/yellow]")
    console.print("[yellow]Use 'cryoem-agent resume --run-id <ID>' to resume a run.[/yellow]")


# ---------------------------------------------------------------------------
# jobs command (kept for compatibility)
# ---------------------------------------------------------------------------

@main.command()
@click.option("--project", "-p", required=True, help="CryoSPARC project UID")
@click.option("--workspace", "-w", help="CryoSPARC workspace UID")
@click.option("--status-filter", help="Filter by job status")
@click.option("--config", "-c", default=None, help="Path to MCP server YAML config file")
def jobs(project: str, workspace: Optional[str], status_filter: Optional[str], config: Optional[str]):
    """List jobs in a CryoSPARC project."""
    config_path = _find_config(config)

    try:
        agent = _build_agent(config_path)
        cs_client = agent.orch_client.get_cs_client()
    except Exception as exc:
        console.print(f"[red]Failed to connect: {exc}[/red]")
        sys.exit(1)

    try:
        proj = cs_client.find_project(project)
        all_jobs = []

        if workspace:
            ws = proj.find_workspace(workspace)
            for job in ws.find_jobs():
                all_jobs.append(job)
        else:
            for ws in proj.find_workspaces():
                for job in ws.find_jobs():
                    all_jobs.append(job)

        if status_filter:
            all_jobs = [j for j in all_jobs if getattr(j, "status", "") == status_filter]

        table = Table(title=f"Jobs in {project}")
        table.add_column("UID", style="cyan")
        table.add_column("Type", style="white")
        table.add_column("Status", style="green")
        table.add_column("Title", style="dim")

        for job in all_jobs:
            job_status = getattr(job, "status", "")
            status_style = {
                "completed": "green",
                "running": "yellow",
                "failed": "red",
                "killed": "red",
                "queued": "blue",
            }.get(job_status, "white")

            job_type = getattr(job, "type", "") or getattr(job, "job_type", "")
            job_title = getattr(job, "title", "") or ""

            table.add_row(
                getattr(job, "uid", ""),
                job_type,
                f"[{status_style}]{job_status}[/{status_style}]",
                job_title[:40],
            )

        console.print(table)
        console.print(f"\nTotal: {len(all_jobs)} jobs")

    except Exception as exc:
        console.print(f"[red]Error listing jobs: {exc}[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------

@main.command()
def version():
    """Show version information."""
    from cryoemagent import __version__  # noqa: PLC0415
    console.print(f"CryoEMAgent version {__version__}")


if __name__ == "__main__":
    main()
