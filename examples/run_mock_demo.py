"""
End-to-End CryoEMAgent Demo with Mock CryoSPARC
================================================

This script demonstrates the COMPLETE workflow of the CryoEMAgent:
1. Memory initialization
2. LLM-based planning
3. CryoSPARC job orchestration (mocked)
4. Quality assessment
5. Workflow completion

Run with: python examples/run_mock_demo.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()


def setup_environment():
    """Setup environment variables for demo."""
    os.environ["CRYOSPARC_URL"] = "http://mock:39000"
    os.environ["CRYOSPARC_EMAIL"] = "demo@cryoem.org"
    os.environ["CRYOSPARC_PASSWORD"] = "demo_password"
    
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]ERROR: OPENAI_API_KEY not set![/red]")
        console.print("Please set: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)


def print_header():
    """Print demo header."""
    console.print()
    console.print(Panel.fit(
        "[bold blue]CryoEMAgent End-to-End Demo[/bold blue]\n"
        "[dim]Autonomous GPCR Structure Determination[/dim]\n\n"
        "Mode: [green]MOCK CryoSPARC[/green] + [cyan]Real LLM Planning[/cyan]",
        border_style="blue",
    ))
    console.print()


def run_demo():
    """Run the full demo."""
    setup_environment()
    print_header()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )
    
    from cryoemagent.core.memory import Memory
    from cryoemagent.core.planner import Planner, ActionType
    from cryoemagent.tools.mock_cryosparc import MockCryoSPARCTools
    from cryoemagent.config import LLMConfig
    from cryoemagent.playbooks.gpcr import GPCRPlaybook
    
    console.print("[bold]Step 1: Initialize Memory System[/bold]")
    console.print("-" * 50)
    
    memory = Memory()
    memory.initialize(
        project_uid="P3",
        workspace_uid="W1",
        objective="Determine GPCR structure to 3.5Å resolution from raw movies"
    )
    
    console.print(f"  ✓ Project: {memory.episodic.state.project_uid}")
    console.print(f"  ✓ Workspace: {memory.episodic.state.workspace_uid}")
    console.print(f"  ✓ Objective: {memory.semantic.objective[:50]}...")
    console.print(f"  ✓ Available tools: {len(memory.semantic.available_tools)}")
    console.print()
    
    console.print("[bold]Step 2: Initialize LLM Planner[/bold]")
    console.print("-" * 50)
    
    llm_config = LLMConfig()
    console.print(f"  ✓ Model: {llm_config.model}")
    console.print(f"  ✓ Temperature: {llm_config.temperature}")
    
    planner = Planner(llm_config)
    console.print("  ✓ Planner initialized")
    console.print()
    
    console.print("[bold]Step 3: Initialize Mock CryoSPARC[/bold]")
    console.print("-" * 50)
    
    tools = MockCryoSPARCTools()
    connection_result = tools.test_connection()
    console.print(f"  ✓ {connection_result.message}")
    console.print()
    
    console.print("[bold]Step 4: Show GPCR Workflow[/bold]")
    console.print("-" * 50)
    
    playbook = GPCRPlaybook()
    workflow_table = Table(title="GPCR Processing Pipeline")
    workflow_table.add_column("#", style="dim", width=3)
    workflow_table.add_column("Step", style="cyan")
    workflow_table.add_column("Job Type", style="green")
    
    for i, step in enumerate(playbook.steps, 1):
        workflow_table.add_row(str(i), step.name, step.job_type)
    
    console.print(workflow_table)
    console.print()
    
    console.print("[bold]Step 5: LLM Planning - Initial Plan[/bold]")
    console.print("-" * 50)
    
    with console.status("[bold green]Calling OpenAI for initial plan..."):
        initial_plan = planner.get_initial_plan(
            movies_path="/data/gpcr_dataset/*.mrc",
            params={
                "pixel_size": 1.05,
                "voltage": 300,
                "spherical_aberration": 2.7,
                "total_dose": 50.0,
            }
        )
    
    console.print(f"\n  [cyan]Goal:[/cyan] {initial_plan.goal}")
    console.print(f"  [cyan]Actions planned:[/cyan] {len(initial_plan.actions)}")
    
    for i, action in enumerate(initial_plan.actions, 1):
        console.print(f"\n  Action {i}: [yellow]{action.action_type.value}[/yellow]")
        if action.parameters:
            for k, v in list(action.parameters.items())[:3]:
                console.print(f"    - {k}: {v}")
        console.print(f"    [dim]Reasoning: {action.reasoning[:60]}...[/dim]")
    
    console.print()
    
    console.print("[bold]Step 6: Execute Mock Workflow[/bold]")
    console.print("-" * 50)
    
    workflow_steps = [
        ("import_movies", "Import Movies", {"blob_paths": "/data/*.mrc", "psize_A": 1.05}),
        ("patch_motion_correction_multi", "Motion Correction", {}),
        ("patch_ctf_estimation_multi", "CTF Estimation", {}),
        ("blob_picker_gpu", "Particle Picking", {"diameter": 80, "diameter_max": 150}),
        ("extract_micrographs_multi", "Extract Particles", {"box_size_pix": 256}),
        ("class_2D_new", "2D Classification", {"class2D_K": 50}),
        ("select_2D", "Select 2D Classes", {}),
        ("homo_abinit", "Ab-initio 3D", {"abinit_K": 3}),
        ("hetero_refine", "Heterogeneous Refinement", {}),
        ("homo_refine_new", "Homogeneous Refinement", {}),
        ("nonuniform_refine_new", "Non-uniform Refinement", {}),
    ]
    
    job_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for job_type, step_name, params in workflow_steps:
            task = progress.add_task(f"Running {step_name}...", total=None)
            
            create_result = tools.create_job(
                project_uid="P3",
                workspace_uid="W1",
                job_type=job_type,
                title=step_name,
                params=params,
            )
            job_uid = create_result.data["job_uid"]
            
            tools.queue_job(job_uid, "P3", lane="default")
            
            wait_result = tools.wait_for_job(job_uid, "P3")
            
            job_results.append({
                "uid": job_uid,
                "type": job_type,
                "name": step_name,
                "outputs": wait_result.data.get("outputs", {}),
            })
            
            memory.episodic.add_observation(
                f"{step_name} completed",
                {"job_uid": job_uid, "outputs": wait_result.data.get("outputs", {})}
            )
            
            progress.remove_task(task)
            console.print(f"  ✓ {job_uid}: {step_name} - [green]completed[/green]")
    
    console.print()
    
    console.print("[bold]Step 7: LLM Reflection[/bold]")
    console.print("-" * 50)
    
    with console.status("[bold green]Calling OpenAI for reflection..."):
        reflection = planner.reflect_on_result(
            memory,
            {
                "workflow_completed": True,
                "total_jobs": len(job_results),
                "final_resolution": 3.2,
                "final_particles": 52000,
            }
        )
    
    console.print("\n[cyan]LLM Analysis:[/cyan]")
    console.print(Panel(reflection[:500] + "..." if len(reflection) > 500 else reflection))
    console.print()
    
    console.print("[bold]Step 8: Final Results[/bold]")
    console.print("-" * 50)
    
    results_table = Table(title="Processing Results")
    results_table.add_column("Job", style="cyan")
    results_table.add_column("Type", style="white")
    results_table.add_column("Output", style="green")
    
    for job in job_results:
        output_str = ""
        for k, v in job["outputs"].items():
            if isinstance(v, dict):
                if "count" in v:
                    output_str += f"{k}: {v['count']:,} items  "
                if "resolution" in v:
                    output_str += f"Resolution: {v['resolution']}Å  "
        results_table.add_row(job["uid"], job["name"][:25], output_str)
    
    console.print(results_table)
    console.print()
    
    final_job = job_results[-1]
    final_outputs = final_job["outputs"]
    
    summary_panel = Panel.fit(
        f"""[bold green]✓ WORKFLOW COMPLETED SUCCESSFULLY[/bold green]

[cyan]Final Resolution:[/cyan] {final_outputs.get('volume', {}).get('resolution', 'N/A')} Å
[cyan]Final Particles:[/cyan] {final_outputs.get('particles', {}).get('count', 'N/A'):,}
[cyan]Total Jobs:[/cyan] {len(job_results)}
[cyan]Target Resolution:[/cyan] 3.5 Å

[dim]This was a MOCK execution demonstrating the complete workflow.
With a real CryoSPARC installation, the agent would run actual
GPU-accelerated processing jobs.[/dim]""",
        title="[bold]Pipeline Summary[/bold]",
        border_style="green",
    )
    console.print(summary_panel)
    console.print()
    
    console.print("[bold]Execution Log (Summary)[/bold]")
    console.print("-" * 50)
    console.print(f"  Total API calls: {len(tools.execution_log)}")
    console.print(f"  Jobs created: {len(job_results)}")
    console.print(f"  Memory observations: {len(memory.episodic.observations)}")
    console.print()


if __name__ == "__main__":
    run_demo()
