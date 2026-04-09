"""Main CryoEMAgent - Autonomous cryo-EM structure determination."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryoemagent.config import LLMConfig
from cryoemagent.core.memory import Memory
from cryoemagent.core.planner import Planner
from cryoemagent.core.quality_critics import QualityCriticChain

logger = logging.getLogger(__name__)


def _get_reasoning_log_path(run_id: str, root_dir: str = "runs") -> Path:
    """Return path for the JSON reasoning log file for a run."""
    log_dir = Path(root_dir) / "reasoning_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"reasoning_{run_id[:8]}.json"


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------

def _format_decision_log(log_list: List[Dict[str, Any]]) -> str:
    """Format the last N decision log entries as a human-readable string."""
    if not log_list:
        return "No prior decisions recorded."
    lines = []
    for entry in log_list:
        ts = entry.get("timestamp", "")[:19]
        step = entry.get("step", "?")
        decision = entry.get("decision", "?")
        reasoning = entry.get("reasoning", "")[:120]
        lines.append(f"[{ts}] {step} -> {decision}: {reasoning}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Result from a single agent run or resume cycle."""

    success: bool
    run_id: str
    final_step: str = ""
    summary: str = ""
    error: str = ""
    checkpoint_required: bool = False
    checkpoint_instructions: str = ""
    report_paths: Optional[Dict[str, str]] = None
    quality_timeline: Optional[List[Dict[str, Any]]] = None
    decision_log: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "run_id": self.run_id,
            "final_step": self.final_step,
            "summary": self.summary,
            "error": self.error,
            "checkpoint_required": self.checkpoint_required,
            "checkpoint_instructions": self.checkpoint_instructions,
            "report_paths": self.report_paths,
            "quality_timeline": self.quality_timeline or [],
            "decision_log": self.decision_log or [],
        }


# ---------------------------------------------------------------------------
# CryoEMAgent
# ---------------------------------------------------------------------------

class CryoEMAgent:
    """
    Autonomous Cryo-EM structure determination agent.

    Bridges the LLM reasoning layer (Planner) with the MCP orchestrator
    (OrchestratorClient) and quality critics (QualityCriticChain).
    """

    def __init__(
        self,
        orchestrator_config: Dict[str, Any],
        agent_config=None,
        ssh_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        orchestrator_config : dict
            Full MCP server config dict (loaded from profile YAML).
        agent_config : AgentConfig, optional
            Agent-specific config (LLM provider, paths, etc.).
        ssh_config : dict, optional
            If provided, use MCP-over-SSH (remote mode).
            Must contain "command" (str) and "args" (list of str).
            If None, use local Python import mode (OrchestratorClient).
        """
        if ssh_config is not None:
            # Remote mode: MCP over SSH
            from cryoemagent.mcp_client import MCPOrchestratorClient
            self.orch_client = MCPOrchestratorClient(ssh_config, orchestrator_config)
            self._remote_mode = True
            logger.info("Agent running in REMOTE mode (MCP over SSH)")
        else:
            # Local mode: direct Python import
            from cryoemagent.orchestrator_client import OrchestratorClient
            self.orch_client = OrchestratorClient(
                orchestrator_config,
                mcp_src_path=agent_config.mcp_server_src_path if agent_config else None,
            )
            self._remote_mode = False
            logger.info("Agent running in LOCAL mode (Python import)")

        llm_config = agent_config.llm if agent_config is not None and hasattr(agent_config, "llm") else LLMConfig()
        self.planner = Planner(llm_config)
        self.memory = Memory()
        self.quality_chain = QualityCriticChain()

        self.project_uid: str = orchestrator_config.get("cryosparc", {}).get("project_uid", "")
        self._max_iterations: int = (
            agent_config.max_agent_iterations if agent_config else 200
        )
        self._root_dir: str = (
            agent_config.root_dir if agent_config and agent_config.root_dir else "runs"
        )

        # Storage for the last checkpoint instructions (useful for CLI display)
        self._last_checkpoint_instructions: str = ""

        # Reasoning log (written to JSON file each iteration)
        self._reasoning_entries: List[Dict[str, Any]] = []
        self._reasoning_log_path: Optional[Path] = None

        logger.info(
            "CryoEMAgent initialised. project_uid=%s max_iter=%d",
            self.project_uid,
            self._max_iterations,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, runtime_overrides: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Validate inputs, create a new run, and execute the control loop.

        Parameters
        ----------
        runtime_overrides : dict, optional
            Key/value pairs to merge into the orchestrator config for this run.

        Returns
        -------
        AgentResult
        """
        # Validate inputs before starting
        validation = self.orch_client.validate_inputs(runtime_overrides)
        if not validation.get("ok", False):
            issues = validation.get("issues", ["Unknown validation error"])
            logger.error("Input validation failed: %s", issues)
            return AgentResult(
                success=False,
                run_id="",
                error="Input validation failed: " + "; ".join(issues),
            )

        # Create a new run
        state = self.orch_client.new_run(runtime_overrides)
        logger.info("Starting new run: %s", state.run_id)

        return self._control_loop(state)

    def resume(self, run_id: str) -> AgentResult:
        """
        Load an existing run and continue from where it left off.

        Parameters
        ----------
        run_id : str

        Returns
        -------
        AgentResult
        """
        state = self.orch_client.load_state(run_id)
        if state is None:
            return AgentResult(
                success=False,
                run_id=run_id,
                error=f"Run {run_id} not found",
            )

        logger.info(
            "Resuming run %s at stage=%s step=%s checkpoint=%s",
            run_id,
            state.current_stage,
            state.current_step,
            state.checkpoint_required,
        )

        # If there's a pending checkpoint, resume it first.
        if state.checkpoint_required:
            logger.info("Resuming checkpoint for step: %s", state.current_step)
            state = self.orch_client.resume_checkpoint(state)

        return self._control_loop(state)

    def status(self, run_id: str) -> Dict[str, Any]:
        """
        Return the current state of a run as a dict.

        Parameters
        ----------
        run_id : str

        Returns
        -------
        dict
        """
        state = self.orch_client.load_state(run_id)
        if state is None:
            return {"error": f"Run {run_id} not found"}
        return state.__dict__

    def report(self, run_id: str) -> Dict[str, str]:
        """
        Write and return paths to the run report files.

        Parameters
        ----------
        run_id : str

        Returns
        -------
        dict with "markdown_report" and "json_report" keys.
        """
        state = self.orch_client.load_state(run_id)
        if state is None:
            return {"error": f"Run {run_id} not found"}
        return self.orch_client.write_report(state)

    def list_runs(self) -> List[str]:
        """Return a sorted list of all run IDs."""
        return self.orch_client.list_runs()

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self, initial_state) -> AgentResult:
        """
        Main agent control loop.

        Re-reads canonical state at the top of every iteration.  The checkpoint
        gate is absolute and unconditional.

        Parameters
        ----------
        initial_state : RunState

        Returns
        -------
        AgentResult
        """
        iteration = 0
        max_iter = self._max_iterations
        run_id = initial_state.run_id

        # Set up reasoning log file for this run
        self._reasoning_log_path = _get_reasoning_log_path(run_id, self._root_dir)
        self._reasoning_entries = []
        self._append_reasoning_log({"event": "run_started", "run_id": run_id,
                                    "timestamp": datetime.now().isoformat()})

        while iteration < max_iter:
            iteration += 1

            # Always re-read canonical state from disk at the top of each iteration.
            state = self.orch_client.load_state(run_id)
            if state is None:
                logger.error("Run %s disappeared from store at iteration %d", run_id, iteration)
                return AgentResult(
                    success=False,
                    run_id=run_id,
                    error=f"Run state disappeared at iteration {iteration}",
                    quality_timeline=list(self.memory.episodic.quality_timeline),
                    decision_log=list(self.memory.episodic.decision_log),
                )

            logger.debug(
                "Iteration %d/%d: run=%s stage=%s step=%s status=%s checkpoint=%s",
                iteration,
                max_iter,
                run_id,
                state.current_stage,
                state.current_step,
                state.status,
                state.checkpoint_required,
            )

            # ==================================================================
            # ABSOLUTE GATE - DO NOT MODIFY
            # Checkpoint gate must be the first condition checked every iteration.
            # ==================================================================
            if state.checkpoint_required:
                instructions = self.planner.generate_checkpoint_instructions(
                    step_name=state.current_step,
                    job_uid=state.checkpoint_job_uid,
                    quality_context=self.memory.episodic.get_quality_context(),
                )
                logger.info(
                    "CHECKPOINT required at step '%s' (job %s). Pausing loop.",
                    state.current_step,
                    state.checkpoint_job_uid,
                )
                self._last_checkpoint_instructions = instructions
                return AgentResult(
                    success=False,
                    run_id=run_id,
                    final_step=state.current_step,
                    checkpoint_required=True,
                    checkpoint_instructions=instructions,
                    quality_timeline=list(self.memory.episodic.quality_timeline),
                    decision_log=list(self.memory.episodic.decision_log),
                )
            # ==================================================================
            # END ABSOLUTE GATE
            # ==================================================================

            # Completion check
            if state.status == "completed":
                logger.info("Run %s completed at step %s", run_id, state.current_step)
                paths = self.orch_client.write_report(state)
                summary = self.planner.summarize_run(
                    state.__dict__,
                    self.memory.episodic.quality_timeline,
                )
                return AgentResult(
                    success=True,
                    run_id=run_id,
                    final_step=state.current_step,
                    summary=summary,
                    report_paths=paths,
                    quality_timeline=list(self.memory.episodic.quality_timeline),
                    decision_log=list(self.memory.episodic.decision_log),
                )

            # Failure check
            if state.status == "failed":
                last_err = "Unknown error"
                if state.errors:
                    last_err = list(state.errors.values())[-1]
                logger.error("Run %s failed: %s", run_id, last_err)
                return AgentResult(
                    success=False,
                    run_id=run_id,
                    final_step=state.current_step,
                    error=last_err,
                    quality_timeline=list(self.memory.episodic.quality_timeline),
                    decision_log=list(self.memory.episodic.decision_log),
                )

            # ------------------------------------------------------------------
            # Quality assessment
            # ------------------------------------------------------------------
            snap = None
            cs_client = self.orch_client.get_cs_client()
            if cs_client is not None:
                # Local mode: run quality critics via direct CryoSPARC API
                try:
                    snap = self.quality_chain.assess_step(
                        step_name=state.current_step,
                        cs_client=cs_client,
                        project_uid=self.project_uid,
                        jobs_dict=state.jobs,
                    )
                except Exception as exc:
                    logger.warning(
                        "Quality assessment raised unexpectedly: %s", exc, exc_info=True
                    )
            else:
                # Remote mode: quality critics unavailable, LLM decides from state alone
                logger.debug("Quality critics skipped (remote MCP mode)")

            if snap is not None:
                self.memory.episodic.add_quality_snapshot(snap)
                logger.info("Quality assessment: %s", snap.summary())

            # ------------------------------------------------------------------
            # LLM decision
            # ------------------------------------------------------------------
            state_summary = (
                f"run_id={state.run_id} "
                f"stage={state.current_stage} "
                f"step={state.current_step} "
                f"status={state.status} "
                f"jobs_done={list(state.jobs.keys())}"
            )
            # In remote mode, include MCP operator instruction for richer context
            if self._remote_mode and hasattr(state, "operator_instruction") and state.operator_instruction:
                state_summary += f"\noperator_context={state.operator_instruction}"
            quality_ctx = self.memory.episodic.get_quality_context()
            dec_history = _format_decision_log(self.memory.episodic.decision_log[-5:])

            # Use ReAct-style decision for richer reasoning
            decision = self.planner.react_decide(state_summary, quality_ctx, dec_history)

            quality_evidence = snap.summary() if snap is not None else None
            self.memory.episodic.add_decision(
                step=state.current_step,
                decision=decision["decision"],
                reasoning=decision["reasoning"],
                quality_evidence=quality_evidence,
            )

            # Always show LLM reasoning prominently (not just DEBUG)
            logger.info(
                "\n╔═══ LLM REASONING [Step: %s] ═══\n"
                "║ Observation: %s\n"
                "║ Thought:     %s\n"
                "║ Tool:        %s\n"
                "║ Decision:    %s\n"
                "║ Rationale:   %s\n"
                "╚══════════════════════════════",
                state.current_step,
                decision.get("observation", "")[:120],
                decision.get("thought", "")[:120],
                decision.get("tool_selected", ""),
                decision["decision"],
                decision["reasoning"][:200],
            )

            # Write to reasoning log file
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration,
                "step": state.current_step,
                "stage": state.current_stage,
                "state_summary": state_summary,
                "quality_context": quality_ctx[:500],
                "llm_decision": {
                    "observation": decision.get("observation", ""),
                    "thought": decision.get("thought", ""),
                    "tool_selected": decision.get("tool_selected", ""),
                    "decision": decision["decision"],
                    "reasoning": decision["reasoning"],
                    "recommendation": decision.get("recommendation", ""),
                    "parameter_adjustments": decision.get("parameter_adjustments", {}),
                },
            }
            self._append_reasoning_log(log_entry)

            # Escalation: stop and return
            if decision["decision"] == "ESCALATE":
                escalation_msg = (
                    f"Escalated by LLM at step '{state.current_step}': "
                    f"{decision.get('recommendation', decision.get('reasoning', 'No reason'))}"
                )
                logger.warning("ESCALATING: %s", escalation_msg)
                return AgentResult(
                    success=False,
                    run_id=run_id,
                    final_step=state.current_step,
                    error=escalation_msg,
                    quality_timeline=list(self.memory.episodic.quality_timeline),
                    decision_log=list(self.memory.episodic.decision_log),
                )

            # CONTINUE or ADJUST — execute the next step
            if decision["decision"] == "ADJUST":
                logger.info(
                    "ADJUST recommended: %s",
                    decision.get("recommendation", "")[:100],
                )
                # Log but do not block — proceed with step execution.

            # ------------------------------------------------------------------
            # Execute one pipeline step
            # ------------------------------------------------------------------
            logger.info(
                "Executing step: stage=%s step=%s",
                state.current_stage,
                state.current_step,
            )
            state = self.orch_client.step(state)
            logger.info(
                "After step: stage=%s step=%s checkpoint=%s status=%s",
                state.current_stage,
                state.current_step,
                state.checkpoint_required,
                state.status,
            )

        # Exhausted max iterations
        logger.warning(
            "Max iterations (%d) reached for run %s without completion.", max_iter, run_id
        )
        return AgentResult(
            success=False,
            run_id=run_id,
            final_step=initial_state.current_step,
            error=f"Max iterations ({max_iter}) reached without completion.",
            quality_timeline=list(self.memory.episodic.quality_timeline),
            decision_log=list(self.memory.episodic.decision_log),
        )

    # ------------------------------------------------------------------
    # Reasoning log helpers
    # ------------------------------------------------------------------

    def _append_reasoning_log(self, entry: Dict[str, Any]) -> None:
        """Append an entry to the in-memory reasoning log and flush to JSON file."""
        self._reasoning_entries.append(entry)
        if self._reasoning_log_path is None:
            return
        try:
            payload = {
                "run_id": entry.get("run_id", ""),
                "log_path": str(self._reasoning_log_path),
                "total_entries": len(self._reasoning_entries),
                "entries": self._reasoning_entries,
            }
            with open(self._reasoning_log_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.debug("Could not write reasoning log: %s", exc)

    @property
    def reasoning_log_path(self) -> Optional[str]:
        """Return path to the current reasoning log file, or None."""
        return str(self._reasoning_log_path) if self._reasoning_log_path else None
