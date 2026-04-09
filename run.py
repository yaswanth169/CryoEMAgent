"""
CryoEMAgent launcher — runs from your laptop via MCP over SSH.

Usage:
    python run.py                   # Interactive mode selector
    python run.py --autopilot       # Skip selector, go straight to autopilot
    python run.py --interactive     # Skip selector, go straight to interactive chat
    python run.py --local           # Local mode (must be on the GPU server)
    python run.py --config alt.yaml # Use a different profile
"""

import argparse
import os
import sys
from pathlib import Path

# ── ensure package importable without pip install ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── load .env ────────────────────────────────────────────────────────────────
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# ── imports ──────────────────────────────────────────────────────────────────
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from cryoemagent import CryoEMAgent, AgentConfig
from cryoemagent.config import LLMConfig

console = Console()


# ── mode selector ────────────────────────────────────────────────────────────

def select_mode() -> str:
    """Show mode selection screen and return 'autopilot' or 'interactive'."""
    console.print(Panel(
        "[bold cyan]CryoEMAgent v0.2[/bold cyan]\n"
        "[dim]Autonomous AI Agent for Cryo-EM Structure Determination[/dim]\n\n"
        "[bold white][1][/bold white]  Autopilot    — Fire-and-forget pipeline with human checkpoints\n"
        "                  Best for: overnight runs, standard workflows\n\n"
        "[bold white][2][/bold white]  Interactive  — Conversational control (like Claude Code)\n"
        "                  Best for: exploratory runs, learning, fine-grained control",
        title="Select Mode",
        border_style="cyan",
    ))

    while True:
        choice = Prompt.ask(
            "[bold]Choose mode[/bold]",
            choices=["1", "2"],
            default="2",
        )
        if choice == "1":
            return "autopilot"
        elif choice == "2":
            return "interactive"


# ── autopilot mode (original run logic) ─────────────────────────────────────

def run_autopilot(profile, agent_config, ssh_config):
    """Run the fully autonomous pipeline with checkpoint pauses."""
    mode_label = "REMOTE (MCP over SSH)" if ssh_config else "LOCAL (Python import)"

    console.print(Panel(
        f"[bold green]CryoEMAgent v0.2 — Autopilot — {mode_label}[/bold green]\n"
        f"Project: {profile.get('cryosparc', {}).get('project_uid', '?')}  "
        f"Movies: {profile.get('data', {}).get('movie_blob_path', '?')}",
        title="CryoEMAgent Autopilot",
    ))

    agent = CryoEMAgent(
        orchestrator_config=profile,
        agent_config=agent_config,
        ssh_config=ssh_config,
    )

    _setup_logging()

    console.print("[dim]Starting pipeline... (GPU jobs may take several minutes each)[/dim]")
    result = agent.run()

    # Checkpoint loop
    while result.checkpoint_required:
        console.print(Panel(
            result.checkpoint_instructions or "Review in CryoSPARC UI, then press ENTER.",
            title="[yellow]HUMAN CHECKPOINT REQUIRED[/yellow]",
            border_style="yellow",
        ))
        input("\nPress ENTER when done in CryoSPARC UI... ")
        console.print("[dim]Resuming pipeline...[/dim]")
        result = agent.resume(result.run_id)

    # Final result
    if result.success:
        console.print(Panel(
            f"[bold green]COMPLETED[/bold green]\n\n"
            f"{result.summary or 'Pipeline finished successfully.'}\n\n"
            f"Report: {result.report_paths}",
            title="Done",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold red]FAILED[/bold red]\n\n{result.error}",
            title="Error",
            border_style="red",
        ))
        sys.exit(1)


# ── interactive mode ─────────────────────────────────────────────────────────

def run_interactive(profile, agent_config, ssh_config):
    """Run the conversational interactive mode."""
    from cryoemagent.interactive import InteractiveSession

    _setup_logging()

    session = InteractiveSession(
        orchestrator_config=profile,
        agent_config=agent_config,
        ssh_config=ssh_config,
    )
    session.run()


# ── shared helpers ───────────────────────────────────────────────────────────

def _setup_logging():
    """Configure rich logging for live progress output."""
    import logging
    from rich.logging import RichHandler
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=True, show_path=False)],
    )


def load_profile(config_path_str: str) -> dict:
    """Load and return the profile YAML."""
    config_path = Path(config_path_str)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CryoEMAgent — Autonomous Cryo-EM Agent")
    parser.add_argument("--config", default="profile.yaml", help="Path to profile YAML")
    parser.add_argument("--local", action="store_true", help="Run in local mode (on GPU server)")
    parser.add_argument("--autopilot", action="store_true", help="Skip mode selector, run autopilot")
    parser.add_argument("--interactive", action="store_true", help="Skip mode selector, run interactive")
    args = parser.parse_args()

    # Load profile
    profile = load_profile(args.config)
    agent_cfg = profile.get("agent", {})
    llm_cfg = profile.get("llm", {})
    ssh_cfg = profile.get("ssh", None)

    agent_config = AgentConfig(
        llm=LLMConfig(
            provider=llm_cfg.get("provider", "openai"),
            model=llm_cfg.get("model", "gpt-4o"),
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
        ),
        mcp_server_src_path=agent_cfg.get("mcp_server_src_path", ""),
        root_dir=agent_cfg.get("root_dir", "runs"),
        max_agent_iterations=agent_cfg.get("max_agent_iterations", 200),
    )

    # Decide SSH mode
    if args.local or ssh_cfg is None:
        ssh_config = None
    else:
        ssh_config = ssh_cfg

    # Decide run mode
    if args.autopilot:
        mode = "autopilot"
    elif args.interactive:
        mode = "interactive"
    else:
        mode = select_mode()

    if mode == "autopilot":
        run_autopilot(profile, agent_config, ssh_config)
    else:
        run_interactive(profile, agent_config, ssh_config)


if __name__ == "__main__":
    main()
