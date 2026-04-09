"""Planner module for CryoEMAgent - LLM-based planning and reasoning."""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from cryoemagent.config import LLMConfig
from cryoemagent.core.memory import Memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formal CryoSPARC tool definitions (shown to LLM in system prompt)
# ---------------------------------------------------------------------------

CRYOSPARC_TOOLS = [
    {
        "name": "run_import_movies",
        "description": "Import raw cryo-EM movie files (.tif/.mrc) into CryoSPARC workspace. Reads pixel size, voltage, spherical aberration from profile.",
        "when_to_use": "Always first. Required before any other processing.",
        "outputs": "imported_movies dataset"
    },
    {
        "name": "run_patch_motion_correction",
        "description": "Apply patch-based motion correction to raw movies. Aligns all frames per movie to remove beam-induced motion blur.",
        "when_to_use": "After import_movies. Critical for high-resolution GPCR data.",
        "outputs": "micrographs dataset, motion_corrected_movies"
    },
    {
        "name": "run_patch_ctf_estimation",
        "description": "Estimate the Contrast Transfer Function (CTF) for each micrograph. Determines defocus, astigmatism, and fit resolution.",
        "when_to_use": "After motion correction. Required for accurate particle alignment.",
        "outputs": "exposures_ctf dataset with fit_res, df1, df2 per micrograph"
    },
    {
        "name": "curate_exposures",
        "description": "[CHECKPOINT] Human reviews CTF quality. Exclude micrographs with CTF fit > 5 Å, excessive ice, or contamination.",
        "when_to_use": "After patch_ctf. VLM can auto-evaluate if confidence >= 85%.",
        "outputs": "curated exposures subset"
    },
    {
        "name": "run_blob_picker",
        "description": "Automatically pick particle candidates using a Laplacian-of-Gaussian blob detector. No template required.",
        "when_to_use": "After curation. First picking pass for initial 2D references.",
        "outputs": "picks dataset with NCC scores, ~50-200 particles/micrograph"
    },
    {
        "name": "inspect_blob_picks",
        "description": "[CHECKPOINT] Human/VLM reviews particle picks. Adjust NCC thresholds to remove false positives and missed particles.",
        "when_to_use": "After blob_picker. VLM can auto-adjust NCC thresholds.",
        "outputs": "filtered picks with adjusted thresholds"
    },
    {
        "name": "extract_particles",
        "description": "Extract particle image stacks from micrographs using pick coordinates. Box size should be 256px for GPCR.",
        "when_to_use": "After inspect picks. Prepares particles for classification.",
        "outputs": "particle_stacks dataset"
    },
    {
        "name": "run_2d_classification",
        "description": "Classify particles into 2D class averages. Reveals particle heterogeneity and quality. Run 50 classes for GPCR.",
        "when_to_use": "After extraction. Generates templates and enables quality assessment.",
        "outputs": "class_averages, particles_classified"
    },
    {
        "name": "select_2d_classes",
        "description": "[CHECKPOINT] Human/VLM selects high-quality 2D classes (clear secondary structure, correct size) and discards junk.",
        "when_to_use": "After 2D classification. VLM uses vision to identify good classes.",
        "outputs": "selected_particles subset"
    },
    {
        "name": "run_abinit_reconstruction",
        "description": "Generate an initial 3D reconstruction ab initio (no reference required). Uses heterogeneous reconstruction to find multiple conformations.",
        "when_to_use": "After 2D selection with >= 5000 particles. Generates first 3D model.",
        "outputs": "volume maps, particle_assignments"
    },
    {
        "name": "run_homogeneous_refinement",
        "description": "Refine the 3D reconstruction assuming a single homogeneous conformation. Improves resolution iteratively.",
        "when_to_use": "After ab-initio. Targets < 4 Å resolution for GPCR.",
        "outputs": "refined_volume, FSC_curve, estimated_resolution_A"
    },
    {
        "name": "run_template_picker",
        "description": "Pick particles using 2D class averages as templates. More accurate than blob picking.",
        "when_to_use": "W2 pipeline. After W1 generates 2D class templates.",
        "outputs": "template_picks with higher specificity"
    },
    {
        "name": "run_nonuniform_refinement",
        "description": "Final high-resolution refinement accounting for local conformational heterogeneity. Best for flexible GPCRs.",
        "when_to_use": "Final step. Achieves highest resolution (target <= 3.5 Å).",
        "outputs": "final_volume, gold_standard_FSC, final_resolution_A"
    },
    {
        "name": "assess_quality",
        "description": "Evaluate quality metrics from a completed step and decide whether to continue, adjust parameters, or escalate.",
        "when_to_use": "After any computational step before proceeding.",
        "outputs": "quality_verdict: PASS/WARN/FAIL"
    },
    {
        "name": "escalate_to_human",
        "description": "Stop the pipeline and request human expert intervention. Use only when quality is critically below threshold.",
        "when_to_use": "Resolution > 8 Å AND particles < 1000, OR same step failed 3+ times.",
        "outputs": "pipeline_paused with explanation"
    },
]

TOOLS_SUMMARY = "\n".join(
    f"  [{i+1}] {t['name']}: {t['description'][:80]}"
    for i, t in enumerate(CRYOSPARC_TOOLS)
)


# ---------------------------------------------------------------------------
# Backwards-compatible action / plan types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Types of actions the agent can take."""

    CREATE_JOB = "create_job"
    SET_PARAM = "set_param"
    CONNECT_INPUT = "connect_input"
    QUEUE_JOB = "queue_job"
    WAIT_JOB = "wait_job"
    LOAD_OUTPUT = "load_output"
    ASSESS_QUALITY = "assess_quality"
    FINISH = "finish"


@dataclass
class PlannedAction:
    """A single planned action."""

    action_type: ActionType
    parameters: Dict[str, Any]
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
        }


@dataclass
class Plan:
    """Execution plan with multiple actions."""

    goal: str
    actions: List[PlannedAction]
    contingency: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "actions": [a.to_dict() for a in self.actions],
            "contingency": self.contingency,
        }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are an expert cryo-EM data processing agent specialized in GPCR structure determination.

AVAILABLE TOOLS (CryoSPARC operations you can invoke):
{TOOLS_SUMMARY}

AGENTIC FRAMEWORK:
You operate using a Planning → Memory → Action loop:
  Planning:  Reason about current state, quality metrics, and which tool to invoke next
  Memory:    You receive episodic context (jobs completed, quality timeline, prior decisions)
  Action:    Select a tool and return your decision with full reasoning chain



CONTEXT:
GPCRs (G-protein-coupled receptors) are membrane proteins of ~60 kDa that are challenging
to image by cryo-EM due to their small size and preferred orientations in detergent micelles
or lipid nanodiscs. Key processing considerations:

- Recommended box size: 256 pixels (at ~1.05 Å/px pixel size)
- Particle diameter: 80–150 Å
- Expected symmetry: C1 (asymmetric)
- Target resolution: ≤3.5 Å for publishable structures
- CTF quality: mean CTF fit ≤5 Å; ≥70% micrographs should pass
- Particle count: ≥50 particles/micrograph after picking; ≥5000 total
- 2D class quality: ≤50% empty classes; look for clear secondary structure

STANDARD WORKFLOW (CryoSPARC):
W1 pipeline:
  import_movies → patch_motion → patch_ctf → curate [CHECKPOINT] →
  blob_picker → inspect_blob [CHECKPOINT] → extract_blob → class2d_blob →
  select2d_blob [CHECKPOINT] → abinit_blob → homo_blob

W2 pipeline (template-based):
  template_picker → inspect_template [CHECKPOINT] → extract_template →
  class2d_template → select2d_template [CHECKPOINT] → abinit_template →
  homo_template → nonuniform_template [DONE]

YOUR ROLE:
At each step, you receive:
1. Current run state (stage, step, jobs completed)
2. Quality context (recent quality snapshots with metrics)
3. Decision history (recent decisions and their outcomes)

RESPONSE FORMAT — You must respond with strictly valid JSON:
{{
    "observation": "What you observe from the current state and quality metrics (1-2 sentences)",
    "thought": "Your chain-of-thought reasoning about what should happen next (2-3 sentences)",
    "tool_selected": "name of the tool from AVAILABLE TOOLS that should execute next",
    "decision": "CONTINUE" | "ADJUST" | "ESCALATE",
    "reasoning": "Full explanation combining observation + thought + decision rationale (3-5 sentences)",
    "recommendation": "Specific actionable recommendation for the operator",
    "parameter_adjustments": {{
        "key": "value"
    }}
}}

DECISION RULES:
- Default to CONTINUE + next logical tool unless you have specific evidence of a problem
- ESCALATE only if: resolution > 8 Å AND particles < 1000, OR same step failed 3+ times
- ADJUST when quality metrics are marginal but not critically bad
- Never invent data — base decisions ONLY on provided quality context
- tool_selected should be the next tool from the AVAILABLE TOOLS list
- Keep reasoning evidence-based and cite specific metrics when available
"""


# ---------------------------------------------------------------------------
# Planner class
# ---------------------------------------------------------------------------

class Planner:
    """
    LLM-based planning and decision engine for the CryoEM autonomous agent.

    Supports OpenAI and Anthropic providers.  Falls back to CONTINUE on any
    LLM failure so the control loop is never crashed.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        # Lazy-initialised clients to avoid import errors if packages not installed.
        self._openai_client = None
        self._anthropic_client = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def decide(
        self,
        state_summary: str,
        quality_context: str,
        decision_history: str,
    ) -> Dict[str, Any]:
        """
        Ask the LLM to make a pipeline control decision.

        Parameters
        ----------
        state_summary : str
            One-line summary of current run state.
        quality_context : str
            Formatted quality timeline text.
        decision_history : str
            Formatted recent decision log text.

        Returns
        -------
        dict with keys: decision, reasoning, recommendation, parameter_adjustments
        """
        user_msg = (
            f"=== Current Run State ===\n{state_summary}\n\n"
            f"=== Quality Context ===\n{quality_context}\n\n"
            f"=== Recent Decisions ===\n{decision_history}\n\n"
            "Based on the above, what is your decision? Respond with strictly valid JSON."
        )

        try:
            provider = (self.config.provider or "openai").lower()
            if provider == "anthropic":
                result = self._call_anthropic(user_msg)
            else:
                result = self._call_openai(user_msg)
        except Exception as exc:
            logger.warning(
                "LLM call failed, falling back to CONTINUE: %s", exc, exc_info=True
            )
            result = self._fallback_decision(str(exc))

        return result

    def generate_checkpoint_instructions(
        self,
        step_name: str,
        job_uid: str,
        quality_context: str,
    ) -> str:
        """
        Return step-specific human instructions for a checkpoint step.

        Parameters
        ----------
        step_name : str
            The current pipeline step (e.g., "curate", "inspect_blob").
        job_uid : str
            The CryoSPARC job UID for the checkpoint job.
        quality_context : str
            Recent quality assessment context.

        Returns
        -------
        str — multi-line instructions for the human operator.
        """
        instructions_map = {
            "curate": (
                f"CHECKPOINT: Curate Exposures (Job {job_uid})\n"
                "--------------------------------------------\n"
                "1. Open CryoSPARC and navigate to the curate exposures job.\n"
                "2. Review CTF fit values — exclude micrographs with fit > 5 Å.\n"
                "3. Check ice thickness — exclude very thick (rel > 1.2) or very thin samples.\n"
                "4. Look for obvious contamination, crystalline ice, or poor contrast.\n"
                "5. Aim to keep ≥ 80% of micrographs if CTF quality is good.\n"
                "6. Click 'Finish' to complete the interactive job.\n"
                "7. Return here and confirm you are done.\n"
                f"\nQuality context:\n{quality_context}"
            ),
            "inspect_blob": (
                f"CHECKPOINT: Inspect Blob Picks (Job {job_uid})\n"
                "------------------------------------------------\n"
                "1. Open the Inspect Picks job in CryoSPARC.\n"
                "2. Review the particle picks overlaid on micrographs.\n"
                "3. Adjust the minimum and maximum NCC score thresholds to:\n"
                "   - Remove obvious contaminants (carbon edges, ice crystals)\n"
                "   - Retain good-looking protein particles\n"
                "4. Target: ≥ 50 particles/micrograph after filtering.\n"
                "5. Verify particles are centred on protein, not on background.\n"
                "6. Click 'Finish' to complete the interactive job.\n"
                "7. Return here and confirm you are done.\n"
                f"\nQuality context:\n{quality_context}"
            ),
            "select2d_blob": (
                f"CHECKPOINT: Select 2D Classes - Blob Picks (Job {job_uid})\n"
                "----------------------------------------------------------\n"
                "1. Open the Select 2D job in CryoSPARC.\n"
                "2. Review 2D class averages sorted by resolution/quality.\n"
                "3. Select classes that show:\n"
                "   - Clear secondary structure (alpha-helices visible)\n"
                "   - Consistent particle size (~100-150 Å for GPCR)\n"
                "   - Low background noise\n"
                "4. Deselect 'junk' classes: rings, aggregates, edge particles.\n"
                "5. Aim to keep ≥ 30-40% of the total particles.\n"
                "6. Click 'Finish' to complete the interactive job.\n"
                "7. Return here and confirm you are done.\n"
                f"\nQuality context:\n{quality_context}"
            ),
            "inspect_template": (
                f"CHECKPOINT: Inspect Template Picks (Job {job_uid})\n"
                "---------------------------------------------------\n"
                "1. Open the Inspect Picks job for the template picker in CryoSPARC.\n"
                "2. This uses templates derived from 2D class averages - the picks\n"
                "   should be more accurate than blob picks.\n"
                "3. Adjust NCC thresholds to remove false positives:\n"
                "   - Eliminate ice contamination and carbon edges\n"
                "   - Keep well-centred protein particles\n"
                "4. Target: ≥ 50 particles/micrograph after filtering.\n"
                "5. Click 'Finish' to complete the interactive job.\n"
                "6. Return here and confirm you are done.\n"
                f"\nQuality context:\n{quality_context}"
            ),
            "select2d_template": (
                f"CHECKPOINT: Select 2D Classes - Template Picks (Job {job_uid})\n"
                "----------------------------------------------------------------\n"
                "1. Open the Select 2D job in CryoSPARC (W2 template picks).\n"
                "2. This is the final 2D selection before ab-initio reconstruction.\n"
                "3. Be more stringent than the W1 blob selection:\n"
                "   - Select only classes with clear secondary structure\n"
                "   - Verify particle orientation diversity (different views)\n"
                "   - Ensure you have top/side/tilted views represented\n"
                "4. Aim to keep 40-60% of particles from the best classes.\n"
                "5. Click 'Finish' to complete the interactive job.\n"
                "6. Return here and confirm you are done.\n"
                f"\nQuality context:\n{quality_context}"
            ),
        }

        if step_name in instructions_map:
            return instructions_map[step_name]

        # Generic fallback for any unrecognised checkpoint step
        return (
            f"CHECKPOINT: {step_name} (Job {job_uid})\n"
            "--------------------------------------\n"
            "A manual interaction is required in CryoSPARC.\n"
            f"1. Open job {job_uid} in the CryoSPARC UI.\n"
            "2. Complete the required interactive step.\n"
            "3. Click 'Finish' to complete the job.\n"
            "4. Return here and confirm you are done.\n"
            f"\nQuality context:\n{quality_context}"
        )

    def summarize_run(
        self,
        state_dict: Dict[str, Any],
        quality_timeline: List[Dict[str, Any]],
    ) -> str:
        """
        Return a natural language summary of the completed run.

        Parameters
        ----------
        state_dict : dict
            Serialised RunState dict.
        quality_timeline : list
            List of QualitySnapshot dicts from memory.

        Returns
        -------
        str
        """
        run_id = state_dict.get("run_id", "unknown")
        status = state_dict.get("status", "unknown")
        stage = state_dict.get("current_stage", "unknown")
        step = state_dict.get("current_step", "unknown")
        jobs = state_dict.get("jobs", {})
        errors = state_dict.get("errors", {})
        created = state_dict.get("created_at", "")[:19]
        updated = state_dict.get("updated_at", "")[:19]

        lines = [
            f"=== Run Summary: {run_id} ===",
            f"Status: {status}",
            f"Final stage: {stage}, final step: {step}",
            f"Started: {created}  |  Last updated: {updated}",
            f"Jobs completed: {len(jobs)}",
        ]

        if jobs:
            lines.append("\nCompleted pipeline steps:")
            for k, v in jobs.items():
                lines.append(f"  {k}: {v}")

        if errors:
            lines.append(f"\nErrors encountered ({len(errors)}):")
            for k, v in errors.items():
                lines.append(f"  [{k}]: {v}")

        if quality_timeline:
            lines.append(f"\nQuality assessments ({len(quality_timeline)} total):")
            passes = sum(1 for s in quality_timeline if s.get("verdict") == "PASS")
            warns = sum(1 for s in quality_timeline if s.get("verdict") == "WARN")
            fails = sum(1 for s in quality_timeline if s.get("verdict") == "FAIL")
            lines.append(f"  PASS={passes}  WARN={warns}  FAIL={fails}")

            # Report last refinement resolution if present
            for snap in reversed(quality_timeline):
                metrics = snap.get("metrics", {})
                if "resolution_A" in metrics:
                    lines.append(f"  Final resolution: {metrics['resolution_A']:.2f} Å")
                    break

        if status == "completed":
            lines.append("\nOutcome: WORKFLOW COMPLETED SUCCESSFULLY")
        elif status == "failed":
            lines.append("\nOutcome: WORKFLOW FAILED - see errors above")
        else:
            lines.append(f"\nOutcome: Workflow ended with status '{status}'")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Backwards-compatibility shims
    # ------------------------------------------------------------------

    def get_initial_plan(self, movies_path: str, params: Dict[str, Any]) -> Plan:
        """Generate initial plan (compatibility shim - returns a static import plan)."""
        return Plan(
            goal="Import movies and begin GPCR processing pipeline",
            actions=[
                PlannedAction(
                    action_type=ActionType.CREATE_JOB,
                    parameters={
                        "job_type": "import_movies",
                        "params": {
                            "blob_paths": movies_path,
                            "psize_A": params.get("pixel_size", 1.05),
                            "accel_kv": params.get("voltage", 300),
                            "cs_mm": params.get("spherical_aberration", 2.7),
                            "total_dose": params.get("total_dose", 50.0),
                        },
                    },
                    reasoning="Starting pipeline with movie import",
                ),
                PlannedAction(
                    action_type=ActionType.QUEUE_JOB,
                    parameters={},
                    reasoning="Queue the import job for execution",
                ),
                PlannedAction(
                    action_type=ActionType.WAIT_JOB,
                    parameters={"error_on_incomplete": True},
                    reasoning="Wait for import to complete before proceeding",
                ),
            ],
            contingency="If import fails, check movie paths and parameters",
        )

    def plan_next_step(self, memory: Memory) -> Plan:
        """
        Generate the next execution plan (compatibility shim).

        Calls decide() internally using the memory context.
        """
        context = memory.get_full_context()
        quality_context = memory.episodic.get_quality_context()
        decision_log = memory.episodic.decision_log[-5:] if memory.episodic.decision_log else []

        state_summary = "Current processing session"
        if memory.episodic.state:
            s = memory.episodic.state
            state_summary = (
                f"stage={s.current_stage} micrographs={s.total_micrographs} "
                f"particles={s.total_particles}"
            )

        dec_history = "\n".join(
            f"  [{e.get('step', '?')}] {e.get('decision', '?')}: {e.get('reasoning', '')[:80]}"
            for e in decision_log
        )

        decision = self.decide(state_summary, quality_context, dec_history)

        if decision.get("decision") == "ESCALATE":
            return Plan(
                goal="Escalate to human",
                actions=[PlannedAction(
                    action_type=ActionType.FINISH,
                    parameters={"reason": decision.get("recommendation", "Escalated")},
                    reasoning=decision.get("reasoning", ""),
                )],
            )

        # CONTINUE or ADJUST — return a generic "assess and continue" plan
        return Plan(
            goal="Continue pipeline processing",
            actions=[PlannedAction(
                action_type=ActionType.ASSESS_QUALITY,
                parameters={},
                reasoning=decision.get("reasoning", "Continue processing"),
            )],
            contingency=decision.get("recommendation"),
        )

    def reflect_on_result(self, memory: Memory, result: Dict[str, Any]) -> str:
        """Reflect on a result (compatibility shim - returns brief analysis string)."""
        quality_context = memory.episodic.get_quality_context()
        state_summary = "Unknown state"
        if memory.episodic.state:
            s = memory.episodic.state
            state_summary = f"stage={s.current_stage}"

        decision = self.decide(
            state_summary,
            quality_context,
            f"Last result: {json.dumps(result)[:200]}",
        )
        return decision.get("reasoning", "Unable to reflect on result.")

    # ------------------------------------------------------------------
    # LLM backend calls
    # ------------------------------------------------------------------

    def _call_openai(self, user_msg: str) -> Dict[str, Any]:
        """Call the OpenAI chat completions API."""
        if self._openai_client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415
                self._openai_client = OpenAI(api_key=self.config.api_key)
            except ImportError as exc:
                raise RuntimeError(
                    "openai package is not installed. "
                    "Install it with: pip install openai"
                ) from exc

        response = self._openai_client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        text = response.choices[0].message.content or ""
        return self._parse_response(text)

    def _call_anthropic(self, user_msg: str) -> Dict[str, Any]:
        """Call the Anthropic messages API."""
        if self._anthropic_client is None:
            try:
                import anthropic  # noqa: PLC0415
                self._anthropic_client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic package is not installed. "
                    "Install it with: pip install anthropic"
                ) from exc

        response = self._anthropic_client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        return self._parse_response(text)

    def react_decide(
        self,
        state_summary: str,
        quality_context: str,
        decision_history: str,
    ) -> Dict[str, Any]:
        """
        ReAct-pattern decision: Observe → Think → Act.

        Returns the same dict as decide() but with richer reasoning fields.
        """
        user_msg = (
            f"=== OBSERVE: Current Run State ===\n{state_summary}\n\n"
            f"=== OBSERVE: Quality Context ===\n{quality_context}\n\n"
            f"=== MEMORY: Recent Decision Log ===\n{decision_history}\n\n"
            "Apply the ReAct framework: Observe the current state → Think about "
            "what it means for GPCR structure quality → Select the appropriate tool "
            "and Act. Output strictly valid JSON."
        )

        try:
            provider = (self.config.provider or "openai").lower()
            if provider == "anthropic":
                result = self._call_anthropic(user_msg)
            else:
                result = self._call_openai(user_msg)
        except Exception as exc:
            logger.warning("ReAct decision failed: %s", exc)
            result = self._fallback_decision(str(exc))

        return result

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parse the LLM response text into a structured decision dict.

        Handles both the new ReAct format (with observation/thought/tool_selected)
        and the legacy format (decision/reasoning only).
        Falls back to CONTINUE on any parse failure.
        """
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            logger.warning("No JSON found in LLM response, falling back to CONTINUE")
            return self._fallback_decision("No JSON in response")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error in LLM response: %s", exc)
            return self._fallback_decision(f"JSON parse error: {exc}")

        decision = str(data.get("decision", "CONTINUE")).upper().strip()
        if decision not in ("CONTINUE", "ADJUST", "ESCALATE"):
            logger.warning("Unknown decision '%s', defaulting to CONTINUE", decision)
            decision = "CONTINUE"

        # Build full reasoning from ReAct fields if present
        observation = str(data.get("observation", ""))
        thought = str(data.get("thought", ""))
        reasoning = str(data.get("reasoning", "No reasoning provided"))
        if observation and thought and not reasoning.startswith("Observation"):
            reasoning = f"Observation: {observation}\nThought: {thought}\nDecision: {reasoning}"

        return {
            "decision": decision,
            "observation": observation[:300],
            "thought": thought[:300],
            "tool_selected": str(data.get("tool_selected", ""))[:80],
            "reasoning": reasoning[:800],
            "recommendation": str(data.get("recommendation", ""))[:300],
            "parameter_adjustments": data.get("parameter_adjustments", {}),
        }

    def _fallback_decision(self, reason: str) -> Dict[str, Any]:
        """Return a safe CONTINUE decision when anything fails."""
        return {
            "decision": "CONTINUE",
            "observation": "LLM unavailable.",
            "thought": "Defaulting to CONTINUE as safe fallback.",
            "tool_selected": "",
            "reasoning": f"LLM unavailable or response unparseable ({reason}). Proceeding with default CONTINUE.",
            "recommendation": "Review logs for LLM errors.",
            "parameter_adjustments": {},
        }
