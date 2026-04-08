"""
CryoEMAgent launcher — runs from your laptop via MCP over SSH.

Usage:
    python run.py                   # Remote mode (MCP over SSH, default)
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

from cryoemagent import CryoEMAgent, AgentConfig
from cryoemagent.config import LLMConfig

console = Console()


def main():
    parser = argparse.ArgumentParser(description="CryoEMAgent — Autonomous Cryo-EM Agent")
    parser.add_argument("--config", default="profile.yaml", help="Path to profile YAML")
    parser.add_argument("--local", action="store_true", help="Run in local mode (on GPU server)")
    args = parser.parse_args()

    # ── load profile ─────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    with open(config_path) as f:
        profile = yaml.safe_load(f)

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

    # ── decide mode ──────────────────────────────────────────────────────
    if args.local or ssh_cfg is None:
        # Local mode: direct Python import (must be on the GPU server)
        mode_label = "LOCAL (Python import)"
        ssh_config = None
    else:
        # Remote mode: MCP over SSH (runs from any laptop)
        mode_label = "REMOTE (MCP over SSH)"
        ssh_config = ssh_cfg

    console.print(Panel(
        f"[bold green]CryoEMAgent v0.2 — {mode_label}[/bold green]\n"
        f"Project: {profile.get('cryosparc', {}).get('project_uid', '?')}  "
        f"Movies: {profile.get('data', {}).get('movie_blob_path', '?')}",
        title="CryoEMAgent",
    ))

    # ── build agent ──────────────────────────────────────────────────────
    agent = CryoEMAgent(
        orchestrator_config=profile,
        agent_config=agent_config,
        ssh_config=ssh_config,
    )

    # ── run ───────────────────────────────────────────────────────────────
    console.print("[dim]Starting pipeline...[/dim]")
    result = agent.run()

    # ── checkpoint loop ──────────────────────────────────────────────────
    while result.checkpoint_required:
        console.print(Panel(
            result.checkpoint_instructions or "Review in CryoSPARC UI, then press ENTER.",
            title="[yellow]HUMAN CHECKPOINT REQUIRED[/yellow]",
            border_style="yellow",
        ))
        input("\nPress ENTER when done in CryoSPARC UI... ")
        console.print("[dim]Resuming pipeline...[/dim]")
        result = agent.resume(result.run_id)

    # ── final result ─────────────────────────────────────────────────────
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


if __name__ == "__main__":
    main()
