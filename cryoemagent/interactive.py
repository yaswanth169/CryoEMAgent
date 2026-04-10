"""
Interactive (conversational) mode for CryoEMAgent.

Design:
- User says "start" once — the agent auto-runs ALL autonomous GPU steps until a
  human checkpoint is reached (curate / inspect_blob / select2d / inspect_template
  / select2d_template). At those 5 points it pauses, shows instructions, waits.
- After "done", it resumes and auto-runs to the next checkpoint or completion.
- Every step gets an LLM narration: what just happened + what's coming next.
- Pipeline progress table shown at start, updated with checkmarks as steps complete.
- Final results table shown on completion.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from cryoemagent.config import LLMConfig
from cryoemagent.core.planner import Planner
from cryoemagent.core.memory import Memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Full pipeline definition (W1 + W2) in order
# Steps marked human=True are checkpoints requiring CryoSPARC UI interaction
# ---------------------------------------------------------------------------

PIPELINE_STEPS = [
    # ── W1 Blob pipeline ────────────────────────────────────────────────
    {"key": "import_movies",    "label": "Import Movies",            "human": False, "wave": "W1"},
    {"key": "patch_motion",     "label": "Patch Motion Correction",  "human": False, "wave": "W1"},
    {"key": "patch_ctf",        "label": "Patch CTF Estimation",     "human": False, "wave": "W1"},
    {"key": "curate",           "label": "Curate Exposures",         "human": True,  "wave": "W1"},
    {"key": "blob_pick",        "label": "Blob Picker",              "human": False, "wave": "W1"},
    {"key": "inspect_blob",     "label": "Inspect Blob Picks",       "human": True,  "wave": "W1"},
    {"key": "extract_blob",     "label": "Extract Particles (Blob)", "human": False, "wave": "W1"},
    {"key": "class2d_blob",     "label": "2D Classification (Blob)", "human": False, "wave": "W1"},
    {"key": "select2d_blob",    "label": "Select 2D Classes (Blob)", "human": True,  "wave": "W1"},
    {"key": "abinit_blob",      "label": "Ab-initio Reconstruction", "human": False, "wave": "W1"},
    {"key": "homo_blob",        "label": "Homogeneous Refinement",   "human": False, "wave": "W1"},
    # ── W2 Template pipeline ─────────────────────────────────────────────
    {"key": "template_pick",       "label": "Template Picker",               "human": False, "wave": "W2"},
    {"key": "inspect_template",    "label": "Inspect Template Picks",        "human": True,  "wave": "W2"},
    {"key": "extract_template",    "label": "Extract Particles (Template)",  "human": False, "wave": "W2"},
    {"key": "class2d_template",    "label": "2D Classification (Template)",  "human": False, "wave": "W2"},
    {"key": "select2d_template",   "label": "Select 2D Classes (Template)",  "human": True,  "wave": "W2"},
    {"key": "abinit_template",     "label": "Ab-initio (Template)",          "human": False, "wave": "W2"},
    {"key": "homo_template",       "label": "Homogeneous Refinement (W2)",   "human": False, "wave": "W2"},
    {"key": "nonuniform_template", "label": "Non-uniform Refinement",        "human": False, "wave": "W2"},
]

CHECKPOINT_STEPS = {s["key"] for s in PIPELINE_STEPS if s["human"]}

# ---------------------------------------------------------------------------
# System prompt for intent classification
# ---------------------------------------------------------------------------

INTERACTIVE_SYSTEM_PROMPT = """You are CryoEMAgent, an expert conversational assistant for cryo-EM structure determination using CryoSPARC.

You help users process cryo-EM data through natural language. The pipeline runs autonomously — it runs all GPU steps automatically and only pauses when human review is needed in CryoSPARC.

PIPELINE (W1 + W2, 19 steps total):
W1: import_movies → patch_motion → patch_ctf → [CHECKPOINT: curate] → blob_pick → [CHECKPOINT: inspect_blob] → extract_blob → class2d_blob → [CHECKPOINT: select2d_blob] → abinit_blob → homo_blob
W2: template_pick → [CHECKPOINT: inspect_template] → extract_template → class2d_template → [CHECKPOINT: select2d_template] → abinit_template → homo_template → nonuniform_template

GPCR DOMAIN KNOWLEDGE:
- GPCRs ~60 kDa membrane proteins | box size 256 px | diameter 80-150 A | symmetry C1
- Target resolution: <=3.5 A | CTF fit < 5 A is good | need >=50 particles/micrograph

OUTPUT FORMAT (strictly valid JSON):
{
    "intent": "<start|continue|confirm_checkpoint|status|quality|explain|adjust_param|pause|resume|report|quit|chat>",
    "response": "<2-4 sentence conversational response>",
    "action_params": {}
}

INTENT RULES:
- "start": user wants to begin a new pipeline
- "continue": user says go/continue/run/proceed (when NOT at checkpoint)
- "confirm_checkpoint": user says done/finished/complete/ready/ok (when AT checkpoint — they've finished in CryoSPARC)
- "status": user asks about progress/status
- "quality": user asks about data quality/metrics
- "explain": user wants to understand a step or concept
- "quit": user wants to exit
- "chat": anything else — answer conversationally

RULES:
- When at a checkpoint, "done"/"finished"/"ok" = confirm_checkpoint
- Never fabricate metrics — only report what you know from state
- Be encouraging and scientific, not robotic
"""


# ---------------------------------------------------------------------------
# Intent router
# ---------------------------------------------------------------------------

class IntentRouter:
    """LLM-backed intent classifier + narration generator."""

    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        self._openai_client = None
        self._anthropic_client = None

    # ── public ──────────────────────────────────────────────────────────

    def classify(self, user_msg: str, state_context: str) -> Dict[str, Any]:
        """Classify user message into an intent + generate conversational response."""
        prompt = (
            f"Pipeline state:\n{state_context}\n\n"
            f"User: {user_msg}\n\n"
            "Respond with strictly valid JSON."
        )
        try:
            text = self._llm(prompt, max_tokens=512, system=INTERACTIVE_SYSTEM_PROMPT)
            return self._parse_intent(text)
        except Exception as exc:
            logger.warning("classify failed: %s", exc)
            return {"intent": "chat", "response": str(exc), "action_params": {}}

    def narrate_step(self, completed_step: str, new_step: str, state_context: str) -> str:
        """
        3-5 sentence plain-text narration: what completed step did,
        what to expect, and what the next step does.
        """
        prompt = (
            f"A cryo-EM pipeline step just completed. In 3-5 plain sentences (no JSON, no bullets):\n"
            f"1. What '{completed_step}' did and why it matters for GPCR structure determination\n"
            f"2. What to expect from the results (use domain knowledge, not fabricated numbers)\n"
            f"3. What '{new_step}' will do and why it comes next\n\n"
            f"Pipeline state: {state_context}"
        )
        system = "You are an expert cryo-EM scientist explaining pipeline steps to a researcher. Be concise, scientific, and encouraging."
        try:
            return self._llm(prompt, max_tokens=400, system=system).strip()
        except Exception as exc:
            logger.warning("narrate_step failed: %s", exc)
            return f"Completed {completed_step}. Starting {new_step}."

    def generate_final_analysis(self, jobs: Dict[str, Any], state_context: str) -> str:
        """LLM analysis of the completed run — like the v1 'LLM Reflection' step."""
        prompt = (
            f"The cryo-EM GPCR processing pipeline just completed. In 4-6 plain sentences:\n"
            f"1. Assess whether the run was successful for GPCR structure determination\n"
            f"2. Comment on data quality based on the steps that ran\n"
            f"3. Recommend any next steps (further refinement, re-collection, publication)\n\n"
            f"Completed jobs: {json.dumps(jobs)}\n"
            f"State: {state_context}"
        )
        system = "You are an expert cryo-EM scientist writing a run assessment. Be concise and evidence-based."
        try:
            return self._llm(prompt, max_tokens=512, system=system).strip()
        except Exception as exc:
            logger.warning("generate_final_analysis failed: %s", exc)
            return "Pipeline completed. Review CryoSPARC results for resolution and quality assessment."

    # ── private ─────────────────────────────────────────────────────────

    def _llm(self, prompt: str, max_tokens: int, system: str) -> str:
        """Route to Anthropic or OpenAI and return raw text."""
        provider = (self.config.provider or "openai").lower()
        if provider == "anthropic":
            if self._anthropic_client is None:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=self.config.api_key)
            resp = self._anthropic_client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if hasattr(b, "text"))
        else:
            if self._openai_client is None:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.config.api_key)
            resp = self._openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

    def _parse_intent(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {"intent": "chat", "response": text.strip(), "action_params": {}}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {"intent": "chat", "response": text.strip(), "action_params": {}}
        valid = {"start","continue","confirm_checkpoint","status","quality",
                 "explain","adjust_param","pause","resume","report","quit","chat"}
        intent = data.get("intent", "chat")
        if intent not in valid:
            intent = "chat"
        return {
            "intent": intent,
            "response": data.get("response", ""),
            "action_params": data.get("action_params", {}),
        }


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

class InteractiveSession:
    """
    Conversational cryo-EM agent.

    The agent auto-runs all GPU steps when the user says 'start'.
    It only pauses for human checkpoints (5 in W1+W2).
    After the user says 'done', it resumes and runs to the next stop.
    """

    def __init__(
        self,
        orchestrator_config: Dict[str, Any],
        agent_config,
        ssh_config: Optional[Dict[str, Any]] = None,
    ):
        self.console = Console()
        self._profile = orchestrator_config

        # Build orchestrator client
        if ssh_config is not None:
            from cryoemagent.mcp_client import MCPOrchestratorClient
            self.orch_client = MCPOrchestratorClient(ssh_config, orchestrator_config)
            self._remote_mode = True
        else:
            from cryoemagent.orchestrator_client import OrchestratorClient
            self.orch_client = OrchestratorClient(
                orchestrator_config,
                mcp_src_path=agent_config.mcp_server_src_path if agent_config else None,
            )
            self._remote_mode = False

        llm_config = agent_config.llm if agent_config and hasattr(agent_config, "llm") else LLMConfig()
        self.router = IntentRouter(llm_config)

        # Planner for ReAct reasoning at each step (same LLM, different role)
        self.planner = Planner(llm_config)
        self.memory = Memory()

        # VLM critic for intelligent checkpoint evaluation
        from cryoemagent.vlm_critic import VLMCritic
        self.vlm_critic = VLMCritic(llm_config)

        self._state = None
        self._run_id: Optional[str] = None
        self._completed_steps: List[str] = []   # steps done so far (for progress table)
        self._chat_history: List[Dict] = []
        self._reasoning_entries: List[Dict] = []

        project = orchestrator_config.get("cryosparc", {}).get("project_uid", "?")
        mode = "Remote (MCP over SSH)" if self._remote_mode else "Local"
        self.console.print(Panel(
            f"[bold cyan]CryoEMAgent Interactive Mode[/bold cyan]\n"
            f"[dim]Project: {project}  |  Mode: {mode}[/dim]\n\n"
            "Chat naturally to control your cryo-EM pipeline.\n"
            "Type [bold]help[/bold] for commands  |  [bold]quit[/bold] to exit.",
            title="CryoEMAgent v0.2 — Interactive",
            border_style="cyan",
        ))

    # ── main loop ────────────────────────────────────────────────────────

    def run(self):
        while True:
            try:
                user_input = Prompt.ask("\n[bold green]you[/bold green]")
            except (EOFError, KeyboardInterrupt):
                self._handle_quit({})
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            lower = user_input.lower()
            if lower in ("quit", "exit", "q"):
                self._handle_quit({})
                break
            if lower in ("help", "?"):
                self._show_help()
                continue
            if lower in ("memory", "mem", "show memory"):
                self._show_memory()
                continue
            if lower in ("reasoning", "log", "show reasoning"):
                self._show_reasoning_log()
                continue

            state_ctx = self._build_state_context()
            classified = self.router.classify(user_input, state_ctx)
            intent = classified["intent"]
            response = classified["response"]
            params = classified.get("action_params", {})

            logger.debug("intent=%s params=%s", intent, params)
            self._chat_history.append({"role": "user", "text": user_input})
            self._chat_history.append({"role": "agent", "text": response, "intent": intent})

            self.console.print(f"\n[bold cyan]agent[/bold cyan]: {response}")

            dispatch = {
                "start":               self._handle_start,
                "continue":            self._handle_continue,
                "confirm_checkpoint":  self._handle_confirm_checkpoint,
                "status":              self._handle_status,
                "quality":             self._handle_quality,
                "explain":             lambda p: None,
                "adjust_param":        self._handle_adjust,
                "pause":               self._handle_pause,
                "resume":              self._handle_resume,
                "report":              self._handle_report,
                "quit":                self._handle_quit,
                "chat":                lambda p: None,
            }
            try:
                dispatch.get(intent, lambda p: None)(params)
            except Exception as exc:
                logger.exception("handler failed")
                self.console.print(f"[red]Error: {exc}[/red]")

    # ── core auto-run loop ───────────────────────────────────────────────

    def _sync_completed_steps(self):
        """
        Sync _completed_steps to match actual server-side pipeline progress.

        After resume_checkpoint(), the server may have run several steps internally
        (e.g. blob_pick, extract_blob, class2d_blob) before stopping at inspect_blob.
        We never called step() for those, so our local list is stale.
        Walk PIPELINE_STEPS up to (but not including) current_step and fill in gaps.
        """
        if self._state is None:
            return
        current = self._state.current_step
        if not current:
            return
        for s in PIPELINE_STEPS:
            if s["key"] == current:
                break  # stop before the current (not-yet-done) step
            if s["key"] not in self._completed_steps:
                self._completed_steps.append(s["key"])

    def _run_until_checkpoint(self):
        """
        Core engine: step() in a loop, narrating each completion,
        until checkpoint_required=True, status=completed, or status=failed.
        """
        total = len(PIPELINE_STEPS)

        while True:
            if self._state is None:
                break

            # Sync completed steps with server state (catches steps run during resume)
            self._sync_completed_steps()

            # Reached a checkpoint — stop and wait for human
            if self._state.checkpoint_required:
                self._show_progress_bar()
                self._show_checkpoint()
                break

            # Run ended
            if self._state.status in ("completed", "failed"):
                self._show_progress_bar()
                self._show_final_status()
                break

            prev_step = self._state.current_step or "unknown"
            step_num  = next((i+1 for i, s in enumerate(PIPELINE_STEPS) if s["key"] == prev_step), "?")

            # ── ReAct: LLM decides whether/how to proceed ──────────────
            state_summary = self._build_state_context()
            quality_ctx = self.memory.episodic.get_quality_context()
            dec_history = "\n".join(
                f"[{e.get('step','?')}] {e.get('decision','?')}: {e.get('reasoning','')[:80]}"
                for e in self.memory.episodic.decision_log[-3:]
            )
            with self.console.status("  [dim]LLM reasoning...[/dim]", spinner="dots"):
                decision = self.planner.react_decide(state_summary, quality_ctx, dec_history)

            # Show reasoning panel
            self._show_llm_reasoning(prev_step, decision)

            # Log the reasoning entry
            self._reasoning_entries.append({
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "step": prev_step,
                "decision": decision,
            })

            # Escalation
            if decision["decision"] == "ESCALATE":
                self.console.print(Panel(
                    f"[bold red]LLM ESCALATION at step: {prev_step}[/bold red]\n\n"
                    f"{decision.get('recommendation', decision['reasoning'])}",
                    title="Escalated — Human Intervention Required",
                    border_style="red",
                ))
                break

            # Record in memory
            self.memory.episodic.add_decision(
                step=prev_step,
                decision=decision["decision"],
                reasoning=decision["reasoning"],
            )

            # Run one step on the GPU server
            with self.console.status(
                f"  [bold green]▶ Step {step_num}/{total}:[/bold green] [cyan]{prev_step}[/cyan]"
                f"  [dim](running on GPU — may take several minutes)[/dim]",
                spinner="dots",
            ):
                self._state = self.orch_client.step(self._state)

            new_step = self._state.current_step or "done"
            self._completed_steps.append(prev_step)

            # Extract job UID for this step (if server returned it)
            job_uid = ""
            if hasattr(self._state, "jobs") and self._state.jobs:
                job_uid = self._state.jobs.get(prev_step, "")

            uid_label  = f"  [dim]({job_uid})[/dim]" if job_uid else ""
            next_label = f"  →  [cyan]{new_step}[/cyan]" if new_step != "done" else ""
            done_count = len(self._completed_steps)

            # Progress line
            self.console.print(
                f"  [bold green]✓[/bold green] [cyan]{prev_step}[/cyan]"
                f"{uid_label}{next_label}"
                f"  [dim][{done_count}/{total}][/dim]"
            )

            # Narration panel
            with self.console.status("  [dim]Analyzing step...[/dim]", spinner="dots"):
                narration = self.router.narrate_step(
                    completed_step=prev_step,
                    new_step=new_step,
                    state_context=self._build_state_context(),
                )
            uid_title = f" {job_uid}" if job_uid else ""
            self.console.print(Panel(
                narration,
                title=(
                    f"[green]✓ {prev_step}{uid_title}[/green]"
                    + (f"  →  [cyan]{new_step}[/cyan]" if new_step != "done" else "")
                ),
                border_style="green",
                padding=(0, 2),
            ))

            # Loop continues — will hit checkpoint check or completion at top

    # ── handlers ────────────────────────────────────────────────────────

    def _handle_start(self, params: Dict[str, Any]):
        if self._state is not None and self._state.status == "running":
            self.console.print("[yellow]A run is already active. Say 'status' to check.[/yellow]")
            return

        # Validate
        self.console.print("[dim]  Validating inputs...[/dim]")
        val = self.orch_client.validate_inputs()
        if not val.get("ok", False):
            issues = val.get("issues", ["Unknown error"])
            self.console.print(f"[red]Validation failed: {'; '.join(str(i) for i in issues)}[/red]")
            return

        # Show pipeline overview table
        self._show_pipeline_table()

        # Start the run (import_movies kicks off on GPU server)
        with self.console.status(
            "  [bold green]Starting pipeline on GPU server...[/bold green]",
            spinner="dots",
        ):
            self._state = self.orch_client.new_run(params.get("overrides"))

        self._run_id = self._state.run_id
        cs_url = self._profile.get("cryosparc", {}).get("base_url", "http://localhost:39000")
        project = self._profile.get("cryosparc", {}).get("project_uid", "")
        workspace = self._profile.get("cryosparc", {}).get("workspace_w1_title", "")

        self.console.print(Panel(
            f"[bold green]Pipeline started![/bold green]\n\n"
            f"  Run ID:    [dim]{self._run_id}[/dim]\n"
            f"  CryoSPARC: [cyan]{cs_url}/browse/{project}-*[/cyan]\n"
            f"  Workspace: [cyan]{workspace}[/cyan]\n\n"
            "[dim]GPU jobs are running on the remote server. Each step will narrate when it completes.[/dim]",
            title="Running — W1 + W2 Pipeline",
            border_style="green",
        ))

        # Auto-run all steps until first checkpoint
        self._run_until_checkpoint()

    def _handle_continue(self, params: Dict[str, Any]):
        if self._state is None:
            self.console.print("[yellow]No active run. Say 'start' to begin.[/yellow]")
            return
        if self._state.checkpoint_required:
            self._show_checkpoint()
            return
        if self._state.status in ("completed", "failed"):
            self.console.print(f"[yellow]Run already {self._state.status}.[/yellow]")
            return
        self._run_until_checkpoint()

    def _handle_confirm_checkpoint(self, params: Dict[str, Any]):
        if self._state is None:
            self.console.print("[yellow]No active run.[/yellow]")
            return
        if not self._state.checkpoint_required:
            self.console.print("[yellow]No checkpoint is pending right now.[/yellow]")
            return

        with self.console.status(
            "  [bold green]Resuming pipeline after checkpoint...[/bold green]",
            spinner="dots",
        ):
            self._state = self.orch_client.resume_checkpoint(self._state)

        # Sync completed steps — server may have run intermediate steps during resume
        self._sync_completed_steps()

        step = self._state.current_step or "next step"
        self.console.print(f"\n  [green]Checkpoint cleared![/green] Resuming from [cyan]{step}[/cyan]...\n")

        # Auto-run again until next checkpoint or completion
        self._run_until_checkpoint()

    def _handle_status(self, params: Dict[str, Any]):
        if self._state is None:
            self.console.print("[dim]No active run. Say 'start a new run'.[/dim]")
            return

        if self._run_id:
            refreshed = self.orch_client.load_state(self._run_id)
            if refreshed:
                self._state = refreshed

        table = Table(title=f"Pipeline Status  —  Run {self._run_id or '?'}", border_style="cyan")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("Status", self._state.status)
        table.add_row("Stage", self._state.current_stage or "—")
        table.add_row("Current Step", self._state.current_step or "—")
        table.add_row("Awaiting Human", "Yes" if self._state.checkpoint_required else "No")
        table.add_row("Steps Completed", str(len(self._completed_steps)))

        if hasattr(self._state, "jobs") and self._state.jobs:
            table.add_row("CryoSPARC Jobs", "\n".join(f"{k}: {v}" for k, v in self._state.jobs.items()))
        if hasattr(self._state, "errors") and self._state.errors:
            table.add_row("[red]Errors[/red]", "\n".join(f"{k}: {v}" for k, v in self._state.errors.items()))

        self.console.print(table)

        # Show progress table alongside
        self._show_pipeline_table(compact=True)

    def _handle_quality(self, params: Dict[str, Any]):
        self.console.print(Panel(
            "[bold]Quality Checklist for GPCR cryo-EM[/bold]\n\n"
            "[cyan]CTF Estimation[/cyan]\n"
            "  ✓ Mean CTF fit < 5 Å  (check CryoSPARC CTF plots)\n"
            "  ✓ >= 70% micrographs pass curation\n\n"
            "[cyan]Particle Picking[/cyan]\n"
            "  ✓ >= 50 particles / micrograph after inspection\n"
            "  ✓ Particles centred on protein, not ice / carbon\n\n"
            "[cyan]2D Classification[/cyan]\n"
            "  ✓ Clear secondary structure visible in best classes\n"
            "  ✓ Multiple views (top, side, tilted) represented\n"
            "  ✓ <= 50% empty / junk classes\n\n"
            "[cyan]3D Refinement[/cyan]\n"
            "  ✓ Resolution <= 3.5 Å for publishable structure\n"
            "  ✓ Gold-standard FSC curve reaches 0.143 cutoff cleanly\n\n"
            "[dim]All metrics visible in the CryoSPARC UI at localhost:39000[/dim]",
            title="Quality Guidelines — GPCR Structure Determination",
            border_style="cyan",
        ))

    def _handle_adjust(self, params: Dict[str, Any]):
        self.console.print(
            "[yellow]Parameters are configured in profile.yaml before a run. "
            "Mid-run parameter adjustments will be added in a future version.[/yellow]"
        )

    def _handle_pause(self, params: Dict[str, Any]):
        self.console.print("[yellow]Noted. The current GPU job will finish, then the pipeline will wait. "
                           "Say 'continue' or 'resume' when you're ready.[/yellow]")

    def _handle_resume(self, params: Dict[str, Any]):
        if self._state and self._state.checkpoint_required:
            self._handle_confirm_checkpoint(params)
            return

        run_id = params.get("run_id", "")
        if run_id:
            loaded = self.orch_client.load_state(run_id)
            if loaded:
                self._state = loaded
                self._run_id = run_id
                self.console.print(f"[green]Loaded run {run_id}[/green] — at step [cyan]{self._state.current_step}[/cyan]")
                self._run_until_checkpoint()
            else:
                self.console.print(f"[red]Run {run_id} not found.[/red]")
            return

        runs = self.orch_client.list_runs()
        if runs:
            self.console.print("[dim]Recent runs:[/dim]")
            for r in runs[-5:]:
                self.console.print(f"  [dim]-[/dim] {r}")
            self.console.print("[dim]Tell me the run ID to resume.[/dim]")
        else:
            self.console.print("[dim]No previous runs found. Say 'start' to begin.[/dim]")

    def _handle_report(self, params: Dict[str, Any]):
        if self._state is None:
            self.console.print("[yellow]No active run.[/yellow]")
            return
        try:
            with self.console.status("[dim]Generating report...[/dim]", spinner="dots"):
                report = self.orch_client.write_report(self._state)
            if "error" in report:
                self.console.print(f"[red]{report['error']}[/red]")
            else:
                self.console.print(Panel(
                    f"[bold green]Report saved[/bold green]\n"
                    f"  Markdown: {report.get('markdown_report', 'N/A')}\n"
                    f"  JSON:     {report.get('json_report', 'N/A')}",
                    title="Run Report",
                    border_style="green",
                ))
        except Exception as exc:
            self.console.print(f"[red]Report failed: {exc}[/red]")

    def _handle_quit(self, params: Dict[str, Any]):
        self.console.print("\n[dim]Closing MCP connection...[/dim]")
        try:
            self.orch_client.close()
        except Exception:
            pass
        self.console.print("[bold cyan]Goodbye![/bold cyan]")

    # ── display helpers ──────────────────────────────────────────────────

    def _show_llm_reasoning(self, step: str, decision: Dict[str, Any]):
        """Display the LLM's ReAct reasoning chain prominently."""
        d = decision["decision"]
        colour = {"CONTINUE": "green", "ADJUST": "yellow", "ESCALATE": "red"}.get(d, "white")

        obs    = decision.get("observation", "")
        thought = decision.get("thought", "")
        tool   = decision.get("tool_selected", "")
        reasoning = decision.get("reasoning", "")
        rec    = decision.get("recommendation", "")

        body = ""
        if obs:
            body += f"[bold]Observation:[/bold] {obs}\n"
        if thought:
            body += f"[bold]Thought:[/bold]     {thought}\n"
        if tool:
            body += f"[bold]Tool:[/bold]         [cyan]{tool}[/cyan]\n"
        body += f"[bold]Decision:[/bold]    [{colour}]{d}[/{colour}]\n"
        if reasoning and reasoning not in (obs + thought):
            body += f"[bold]Reasoning:[/bold]   {reasoning}\n"
        if rec:
            body += f"[bold]Recommendation:[/bold] [dim]{rec}[/dim]"

        self.console.print(Panel(
            body.strip(),
            title=f"[bold]LLM Reasoning — {step}[/bold]",
            border_style=colour,
            padding=(0, 2),
        ))

    def _show_progress_bar(self):
        """Print a compact one-line progress bar: ✓✓✓✓□□□□ (4/19)."""
        total = len(PIPELINE_STEPS)
        done  = len(self._completed_steps)
        bar   = ""
        for s in PIPELINE_STEPS:
            if s["key"] in self._completed_steps:
                bar += "[bold green]█[/bold green]"
            elif self._state and self._state.current_step == s["key"]:
                bar += "[bold yellow]▶[/bold yellow]"
            else:
                bar += "[dim]░[/dim]"
        pct = int(done / total * 100)
        self.console.print(
            f"\n  Progress: {bar}  [bold]{done}/{total}[/bold] steps  [dim]({pct}%)[/dim]\n"
        )

    def _show_pipeline_table(self, compact: bool = False):
        """Print the full W1+W2 pipeline table with live checkmarks."""
        table = Table(
            title="GPCR Processing Pipeline  —  W1 + W2  (19 steps)",
            border_style="dim",
            show_lines=False,
            padding=(0, 1),
        )
        table.add_column("#",     style="dim",   width=3,  no_wrap=True)
        table.add_column("Wave",  style="dim",   width=4,  no_wrap=True)
        table.add_column("Step",  style="white", width=34, no_wrap=True)
        table.add_column("Type",  style="cyan",  width=28, no_wrap=True)
        table.add_column("Human", style="dim",   width=6,  no_wrap=True)

        for i, s in enumerate(PIPELINE_STEPS, start=1):
            done = s["key"] in self._completed_steps
            active = (
                self._state is not None
                and self._state.current_step == s["key"]
                and not done
            )
            if done:
                marker = "[bold green]✓[/bold green]"
                label = f"[dim]{s['label']}[/dim]"
                key   = f"[dim]{s['key']}[/dim]"
            elif active:
                marker = "[bold yellow]▶[/bold yellow]"
                label = f"[bold yellow]{s['label']}[/bold yellow]"
                key   = f"[yellow]{s['key']}[/yellow]"
            else:
                marker = "[dim] [/dim]"
                label  = s["label"]
                key    = f"[dim]{s['key']}[/dim]"

            human_flag = "[yellow]✋[/yellow]" if s["human"] else ""
            table.add_row(f"{marker} {i}", s["wave"], label, key, human_flag)

            if not compact and i in (11,):   # separator between W1 and W2
                table.add_row("", "", "[dim]── W2 Template Pipeline ──[/dim]", "", "")

        self.console.print(table)
        if not compact:
            self.console.print(
                "[dim]  ✋ = Human checkpoint (review in CryoSPARC UI required)[/dim]\n"
            )

    def _show_checkpoint(self):
        """Show VLM assessment + checkpoint instructions and stop the auto-run loop."""
        step = self._state.current_step if self._state else "unknown"
        job_uid = getattr(self._state, "checkpoint_job_uid", "") or ""
        msg     = getattr(self._state, "checkpoint_message", "") or ""

        # ── VLM Assessment ───────────────────────────────────────────────
        jobs = getattr(self._state, "jobs", {}) or {}
        metrics = {"step": step, "job_uid": job_uid, "completed_jobs": list(jobs.keys())}

        self.console.print(
            f"\n  [dim]Running VLM quality assessment for [bold]{step}[/bold]...[/dim]"
        )
        with self.console.status("  [dim]VLM analyzing checkpoint...[/dim]", spinner="dots"):
            vlm = self.vlm_critic.assess_checkpoint(step, metrics)

        # VLM verdict colours
        verdict_colour = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(vlm.verdict, "white")
        tier_label = "Tier 2 (Vision)" if vlm.used_images else "Tier 1 (Metrics)"

        vlm_body = (
            f"[bold]Verdict:[/bold]    [{verdict_colour}]{vlm.verdict}[/{verdict_colour}]  "
            f"({tier_label})  confidence={vlm.confidence:.0%}\n\n"
            f"[bold]Observations:[/bold]\n"
            + "\n".join(f"  • {o}" for o in vlm.observations)
            + f"\n\n[bold]Reasoning:[/bold]\n  {vlm.reasoning}"
        )
        if vlm.recommended_actions:
            vlm_body += "\n\n[bold]Recommendations:[/bold]\n" + "\n".join(
                f"  → {a}" for a in vlm.recommended_actions
            )
        if vlm.suggested_params:
            vlm_body += f"\n\n[bold]Suggested params:[/bold] {vlm.suggested_params}"

        self.console.print(Panel(
            vlm_body,
            title=f"[bold]VLM Assessment — {step}[/bold]",
            border_style=verdict_colour,
        ))

        # Auto-approval path
        if vlm.auto_approved:
            self.console.print(
                f"  [bold green]✓ VLM AUTO-APPROVED[/bold green] this checkpoint "
                f"(confidence={vlm.confidence:.0%} >= 85%). "
                f"No human review needed. Continuing pipeline...\n"
            )
            return  # don't show manual instructions or wait for human input

        # Manual review instructions
        instructions = _checkpoint_instructions(step, job_uid)
        self.console.print(Panel(
            instructions + (f"\n\n[dim]Server note: {msg}[/dim]" if msg else ""),
            title=f"[bold yellow]✋  Human Checkpoint Required: {step}[/bold yellow]",
            border_style="yellow",
        ))
        self.console.print(
            "[dim]  Complete the step above in CryoSPARC, then type "
            "[bold white]done[/bold white] to continue.[/dim]"
        )

    def _show_final_status(self):
        """Show final results table + LLM analysis on completion."""
        if self._state.status == "completed":
            jobs = getattr(self._state, "jobs", {})

            # Jobs table
            if jobs:
                table = Table(title="Processing Results", border_style="green")
                table.add_column("Job", style="cyan", no_wrap=True)
                table.add_column("Step", style="white")
                table.add_column("Output", style="green")
                for step_key, job_uid in jobs.items():
                    label = next((s["label"] for s in PIPELINE_STEPS if s["key"] == step_key), step_key)
                    table.add_row(str(job_uid), label, "completed")
                self.console.print(table)

            # LLM analysis
            with self.console.status("[dim]Generating run analysis...[/dim]", spinner="dots"):
                analysis = self.router.generate_final_analysis(jobs, self._build_state_context())

            self.console.print(Panel(
                f"[bold]LLM Analysis[/bold]\n\n{analysis}",
                title="[green]Pipeline Analysis[/green]",
                border_style="green",
            ))

            # Summary box
            self.console.print(Panel(
                f"[bold green]✓  WORKFLOW COMPLETED SUCCESSFULLY[/bold green]\n\n"
                f"  Total steps:   {len(self._completed_steps)}\n"
                f"  CryoSPARC jobs: {len(jobs)}\n\n"
                f"  Say [bold white]report[/bold white] to save markdown + JSON report.",
                title="Pipeline Summary",
                border_style="green",
            ))

        elif self._state.status == "failed":
            err = ""
            if hasattr(self._state, "errors") and self._state.errors:
                err = list(self._state.errors.values())[-1]
            self.console.print(Panel(
                f"[bold red]Pipeline failed[/bold red]\n\n{err}\n\n"
                "[dim]Check CryoSPARC UI for details. Say 'status' for error log.[/dim]",
                border_style="red",
            ))

    def _show_memory(self):
        """Display the agent's episodic memory (jobs, decisions, quality)."""
        table = Table(title="Agent Episodic Memory", border_style="cyan")
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Content", style="white")

        # Completed steps
        table.add_row("Completed Steps", ", ".join(self._completed_steps) or "none")

        # Decision log
        if self.memory.episodic.decision_log:
            dec_lines = "\n".join(
                f"[{e.get('step','?')}] {e.get('decision','?')}: {e.get('reasoning','')[:80]}..."
                for e in self.memory.episodic.decision_log[-5:]
            )
            table.add_row("Recent Decisions (last 5)", dec_lines)
        else:
            table.add_row("Decisions", "None yet")

        # Quality timeline
        if self.memory.episodic.quality_timeline:
            q_lines = "\n".join(
                f"[{s.get('step','?')}] {s.get('verdict','?')}"
                for s in self.memory.episodic.quality_timeline[-3:]
            )
            table.add_row("Quality Timeline (last 3)", q_lines)

        self.console.print(table)

    def _show_reasoning_log(self):
        """Display the full reasoning log for this session."""
        if not self._reasoning_entries:
            self.console.print("[dim]No reasoning entries yet.[/dim]")
            return

        for i, entry in enumerate(self._reasoning_entries, 1):
            dec = entry.get("decision", {})
            d   = dec.get("decision", "?")
            colour = {"CONTINUE": "green", "ADJUST": "yellow", "ESCALATE": "red"}.get(d, "white")
            self.console.print(Panel(
                f"[bold]Step:[/bold]        {entry.get('step','?')}\n"
                f"[bold]Decision:[/bold]    [{colour}]{d}[/{colour}]\n"
                f"[bold]Tool:[/bold]        [cyan]{dec.get('tool_selected','')}[/cyan]\n"
                f"[bold]Observation:[/bold] {dec.get('observation','')}\n"
                f"[bold]Thought:[/bold]     {dec.get('thought','')}\n"
                f"[bold]Reasoning:[/bold]   {dec.get('reasoning','')[:300]}",
                title=f"Reasoning Entry {i}/{len(self._reasoning_entries)}",
                border_style=colour,
                padding=(0, 1),
            ))

    def _show_help(self):
        self.console.print(Panel(
            "[bold]Natural Language Commands[/bold]\n\n"
            '  [cyan]"start a new run"[/cyan]          — Launch the full W1+W2 pipeline\n'
            '  [cyan]"continue"[/cyan]                 — Resume auto-run (if paused)\n'
            '  [cyan]"done" / "finished"[/cyan]        — Confirm you completed a checkpoint in CryoSPARC\n'
            '  [cyan]"status"[/cyan]                   — Show pipeline progress table\n'
            '  [cyan]"quality"[/cyan]                  — Show quality guidelines\n'
            '  [cyan]"what is CTF?"[/cyan]             — Ask any cryo-EM question\n'
            '  [cyan]"report"[/cyan]                   — Save final report\n'
            '  [cyan]"quit"[/cyan]                     — Exit\n\n'
            "[bold]Agent Introspection[/bold]\n"
            "  [cyan]memory[/cyan]                     — Show agent episodic memory\n"
            "  [cyan]reasoning[/cyan]                  — Show full LLM reasoning log\n\n"
            "[bold]How it works — Agentic Framework[/bold]\n"
            "  Planning:  LLM reasons (Observe→Think→Act) at every step\n"
            "  Memory:    Episodic log of jobs, quality assessments, decisions\n"
            "  Action:    MCP tool calls to CryoSPARC GPU server\n"
            "  VLM:       Vision LLM evaluates checkpoints automatically\n\n"
            "  5 checkpoint steps: curate · inspect_blob · select2d_blob\n"
            "                      inspect_template · select2d_template\n"
            "  VLM auto-approves if confidence >= 85%, otherwise asks you.",
            title="CryoEMAgent Help",
            border_style="cyan",
        ))

    def _build_state_context(self) -> str:
        if self._state is None:
            return "No active run."
        s = self._state
        parts = [
            f"run_id={s.run_id}",
            f"status={s.status}",
            f"stage={s.current_stage}",
            f"step={s.current_step}",
            f"checkpoint_required={s.checkpoint_required}",
            f"completed_steps={self._completed_steps}",
        ]
        if getattr(s, "checkpoint_message", None):
            parts.append(f"checkpoint_message={s.checkpoint_message}")
        if getattr(s, "jobs", None):
            parts.append(f"jobs={list(s.jobs.keys())}")
        if getattr(s, "errors", None):
            parts.append(f"errors={s.errors}")
        if getattr(s, "operator_instruction", None):
            parts.append(f"operator_hint={s.operator_instruction}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Checkpoint instruction templates
# ---------------------------------------------------------------------------

def _checkpoint_instructions(step: str, job_uid: str) -> str:
    uid = f" (Job {job_uid})" if job_uid else ""
    templates = {
        "curate": (
            f"[bold]Curate Exposures{uid}[/bold]\n\n"
            "  1. Open CryoSPARC  →  navigate to the Curate Exposures job\n"
            "  2. Sort by CTF fit — exclude micrographs with fit > 5 Å\n"
            "  3. Remove images with crystalline ice, thick ice, or contamination\n"
            "  4. Aim to keep ≥ 80% of micrographs if data quality is good\n"
            "  5. Click [bold]Finish[/bold] in CryoSPARC"
        ),
        "inspect_blob": (
            f"[bold]Inspect Blob Picks{uid}[/bold]\n\n"
            "  1. Open the Inspect Picks job in CryoSPARC\n"
            "  2. Adjust NCC score thresholds — remove carbon edges, ice crystals\n"
            "  3. Target: ≥ 50 particles / micrograph after filtering\n"
            "  4. Verify particles are centred on protein, not on background\n"
            "  5. Click [bold]Finish[/bold] in CryoSPARC"
        ),
        "select2d_blob": (
            f"[bold]Select 2D Classes — Blob Picks{uid}[/bold]\n\n"
            "  1. Open the Select 2D job in CryoSPARC\n"
            "  2. Select classes with clear secondary structure (visible α-helices)\n"
            "  3. Deselect junk: rings, aggregates, noise, edge picks\n"
            "  4. Keep ≥ 30–40% of particles from the best classes\n"
            "  5. Click [bold]Finish[/bold] in CryoSPARC"
        ),
        "inspect_template": (
            f"[bold]Inspect Template Picks{uid}[/bold]\n\n"
            "  1. Open the Inspect Picks job in CryoSPARC (W2 template picker)\n"
            "  2. Template picks are more accurate — adjust NCC thresholds carefully\n"
            "  3. Remove false positives: ice, carbon, overlapping particles\n"
            "  4. Target: ≥ 50 particles / micrograph\n"
            "  5. Click [bold]Finish[/bold] in CryoSPARC"
        ),
        "select2d_template": (
            f"[bold]Select 2D Classes — Template Picks{uid}[/bold]\n\n"
            "  1. Open the Select 2D job in CryoSPARC (W2)\n"
            "  2. Be more stringent than the blob selection\n"
            "  3. Ensure orientation diversity: top, side, and tilted views present\n"
            "  4. Keep 40–60% of particles from the best classes\n"
            "  5. Click [bold]Finish[/bold] in CryoSPARC"
        ),
    }
    if step in templates:
        return templates[step]
    return (
        f"[bold]Interactive Step: {step}{uid}[/bold]\n\n"
        f"  1. Open job {job_uid} in CryoSPARC\n"
        "  2. Complete the required interactive step\n"
        "  3. Click [bold]Finish[/bold] in CryoSPARC"
    )
