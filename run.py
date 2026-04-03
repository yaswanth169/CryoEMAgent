"""
Single-script launcher for CryoEMAgent.
Run with:  python run.py
"""

import os
import sys

# ── set API key (also read from .env if present) ──────────────────────────────
from pathlib import Path

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# ── ensure package is importable even without pip install ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── imports ───────────────────────────────────────────────────────────────────
import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()

from cryoemagent import CryoEMAgent, AgentConfig
from cryoemagent.config import LLMConfig

# ── load profile ──────────────────────────────────────────────────────────────
profile_path = Path(__file__).parent / "profile.yaml"
with open(profile_path) as f:
    profile = yaml.safe_load(f)

# ── build agent ───────────────────────────────────────────────────────────────
agent_cfg = profile.get("agent", {})
llm_cfg   = profile.get("llm", {})

agent_config = AgentConfig(
    llm=LLMConfig(
        provider=llm_cfg.get("provider", "openai"),
        model=llm_cfg.get("model", "gpt-4o"),
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
    ),
    mcp_server_src_path=agent_cfg.get("mcp_server_src_path"),
    root_dir=agent_cfg.get("root_dir", "runs"),
    max_agent_iterations=agent_cfg.get("max_agent_iterations", 200),
)

agent = CryoEMAgent(orchestrator_config=profile, agent_config=agent_config)

# ── run ───────────────────────────────────────────────────────────────────────
console.print(Panel(
    "[bold green]CryoEMAgent v0.2 — Starting W1→W2 Pipeline[/bold green]\n"
    f"Project: {profile['cryosparc']['project_uid']}  "
    f"Movies: {profile['data']['movie_blob_path']}",
    title="CryoEMAgent"
))

result = agent.run()

while result.checkpoint_required:
    console.print(Panel(
        result.checkpoint_instructions or "Review in CryoSPARC, then press ENTER.",
        title="[yellow]HUMAN CHECKPOINT REQUIRED[/yellow]",
        border_style="yellow",
    ))
    input("\nPress ENTER when done in CryoSPARC UI... ")
    result = agent.resume(result.run_id)

# ── final result ──────────────────────────────────────────────────────────────
if result.success:
    console.print(Panel(
        f"[bold green]COMPLETED[/bold green]\n\n"
        f"{result.summary or ''}\n\n"
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
