"""
VLM (Vision Language Model) Critic for CryoEMAgent.

Addresses the research contribution: automated intelligent checkpoint evaluation
using a Vision LLM instead of (or to assist) human review.

Two tiers:
  Tier 1 — Metrics-based LLM reasoning (always works, even in remote/SSH mode).
            Uses CryoSPARC job metrics + domain knowledge to evaluate quality.
  Tier 2 — Image-based VLM (requires local cs_client and GPT-4V / Claude Vision).
            Downloads actual CTF plots / micrograph thumbnails / 2D class images
            from CryoSPARC and feeds them to the vision model.

The VLMCritic is called at every checkpoint step (curate, inspect_blob,
select2d_blob, inspect_template, select2d_template) and returns a structured
VLMAssessment that can:
  - Fully automate the checkpoint (confidence >= AUTO_APPROVE_THRESHOLD)
  - Recommend specific parameter adjustments
  - Guide the human reviewer with targeted observations
"""

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from cryoemagent.config import LLMConfig

logger = logging.getLogger(__name__)

AUTO_APPROVE_THRESHOLD = 0.85   # confidence above which VLM auto-approves


# ---------------------------------------------------------------------------
# Assessment dataclass
# ---------------------------------------------------------------------------

@dataclass
class VLMAssessment:
    """Structured result from VLM checkpoint evaluation."""

    step: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Core verdict
    verdict: str = "PASS"          # PASS / WARN / FAIL
    decision: str = "approve"      # approve / approve_with_adjustments / escalate_to_human
    confidence: float = 0.7        # 0.0 – 1.0
    auto_approved: bool = False    # True if VLM acted autonomously (no human needed)

    # Reasoning chain
    observations: List[str] = field(default_factory=list)
    reasoning: str = ""
    recommended_actions: List[str] = field(default_factory=list)

    # Suggested parameter changes (passed back to pipeline)
    suggested_params: Dict[str, Any] = field(default_factory=dict)

    # Mode
    used_images: bool = False       # True if Tier 2 (vision) was used

    def summary(self) -> str:
        auto = " [AUTO-APPROVED]" if self.auto_approved else ""
        return (
            f"VLM[{self.step}] {self.verdict} | {self.decision}{auto} "
            f"(conf={self.confidence:.0%}) — {self.reasoning[:120]}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "decision": self.decision,
            "confidence": self.confidence,
            "auto_approved": self.auto_approved,
            "observations": self.observations,
            "reasoning": self.reasoning,
            "recommended_actions": self.recommended_actions,
            "suggested_params": self.suggested_params,
            "used_images": self.used_images,
        }


# ---------------------------------------------------------------------------
# Checkpoint-specific prompts
# ---------------------------------------------------------------------------

CHECKPOINT_PROMPTS = {
    "curate": {
        "tier1": """You are an expert cryo-EM structural biologist evaluating micrograph quality for GPCR structure determination.

Evaluate the following CTF estimation results:
{metrics}

GPCR quality thresholds:
- CTF fit resolution: GOOD < 5 Å | WARN 5-7 Å | FAIL > 7 Å
- Ice thickness (relative): GOOD < 1.2 | WARN 1.2-1.5 | FAIL > 1.5
- Acceptance rate: GOOD > 70% | WARN 50-70% | FAIL < 50%
- Defocus range: optimal 0.5-3.0 μm underfocus

Respond with JSON:
{{
  "verdict": "PASS|WARN|FAIL",
  "decision": "approve|approve_with_adjustments|escalate_to_human",
  "confidence": 0.0-1.0,
  "observations": ["observation 1", "observation 2"],
  "reasoning": "chain-of-thought reasoning in 3-4 sentences",
  "recommended_actions": ["specific action 1", "specific action 2"],
  "suggested_params": {{}}
}}""",
        "tier2": """You are an expert cryo-EM structural biologist. Analyze this CTF power spectrum image from a GPCR dataset.

Look for:
1. Thon rings — should be clearly visible and evenly spaced
2. CTF fit quality — the overlaid curve should closely follow the rings
3. Ice/contamination — dark spots, asymmetric patterns, streaks indicate problems
4. Resolution of rings — rings should be visible to at least 5 Å

Additional metrics: {metrics}

Decide: should this micrograph be KEPT or REJECTED?
Respond with JSON: {{verdict, decision, confidence, observations, reasoning, recommended_actions, suggested_params}}"""
    },
    "inspect_blob": {
        "tier1": """You are an expert cryo-EM structural biologist evaluating blob particle picks for GPCR data.

Blob picking results:
{metrics}

GPCR particle picking thresholds:
- Particles per micrograph: GOOD > 50 | WARN 20-50 | FAIL < 20
- NCC score distribution: want right-skewed (most picks are real particles)
- Expected particle diameter: 80-150 Å for GPCR in detergent/nanodisc
- Pick density: too sparse = missed particles, too dense = false positives

Respond with JSON:
{{
  "verdict": "PASS|WARN|FAIL",
  "decision": "approve|approve_with_adjustments|escalate_to_human",
  "confidence": 0.0-1.0,
  "observations": [...],
  "reasoning": "...",
  "recommended_actions": [...],
  "suggested_params": {{"ncc_threshold_min": 0.1, "ncc_threshold_max": 0.9}}
}}""",
        "tier2": """Analyze this micrograph with overlaid particle picks for a GPCR cryo-EM dataset.

Evaluate:
1. Are picks centred on protein density (not on background noise, ice, or carbon)?
2. Are obvious GPCR-sized (~100-130 Å) densities being missed?
3. Are there many false positives (picks on contamination, edge artifacts)?
4. Is the pick density reasonable (50-150 picks per micrograph)?

Additional metrics: {metrics}

Respond with JSON: {{verdict, decision, confidence, observations, reasoning, recommended_actions, suggested_params}}"""
    },
    "select2d_blob": {
        "tier1": """You are an expert cryo-EM structural biologist evaluating 2D classification results for GPCR data.

2D classification results:
{metrics}

GPCR 2D class quality criteria:
- Secondary structure visibility: α-helices (TM helices) should be visible as lines
- Particle diameter: ~100-150 Å for GPCR in nanodisc/detergent
- View diversity: need top, side, and tilted views
- Junk classes: ice rings, noise aggregates, edge particles = reject
- Selection target: keep 30-50% of particles from best classes
- Minimum classes to keep: at least 5 good classes with clear features

Respond with JSON:
{{
  "verdict": "PASS|WARN|FAIL",
  "decision": "approve|approve_with_adjustments|escalate_to_human",
  "confidence": 0.0-1.0,
  "observations": [...],
  "reasoning": "...",
  "recommended_actions": [...],
  "suggested_params": {{}}
}}""",
        "tier2": """Analyze this grid of 2D class averages from a GPCR cryo-EM dataset.

For each class, determine:
1. Does it show clear protein structure (TM helices visible as lines)?
2. Is the particle size consistent with a GPCR (~100-150 Å)?
3. Is this a good class (KEEP) or junk (DISCARD: ring, noise, smeared)?
4. Are different orientations (top, side, tilted) represented in the good classes?

Additional metrics: {metrics}

Respond with JSON: {{verdict, decision, confidence, observations, reasoning, recommended_actions, suggested_params}}"""
    },
    "inspect_template": {
        "tier1": """You are an expert cryo-EM structural biologist evaluating template-based particle picks.

Template picking results:
{metrics}

These picks use 2D class averages as templates, so they should be more accurate than blob picks.
Expected: higher specificity, similar or better particle counts per micrograph.

Quality thresholds:
- Particles per micrograph: GOOD > 60 | WARN 30-60 | FAIL < 30
- Template correlation score distribution: should be narrow and right-skewed
- False positive rate: should be lower than blob picking

Respond with JSON:
{{
  "verdict": "PASS|WARN|FAIL",
  "decision": "approve|approve_with_adjustments|escalate_to_human",
  "confidence": 0.0-1.0,
  "observations": [...],
  "reasoning": "...",
  "recommended_actions": [...],
  "suggested_params": {{"ncc_threshold_min": 0.1}}
}}""",
        "tier2": """Analyze this micrograph with template-based particle picks for a GPCR cryo-EM dataset.

Template picks should be more accurate than blob picks. Evaluate:
1. Are picks well-centred on GPCR protein density?
2. Are obvious protein particles being missed?
3. Are there false positives on ice/contamination?
4. Does pick density look appropriate?

Additional metrics: {metrics}

Respond with JSON: {{verdict, decision, confidence, observations, reasoning, recommended_actions, suggested_params}}"""
    },
    "select2d_template": {
        "tier1": """You are an expert cryo-EM structural biologist doing final 2D class selection for GPCR template picking.

2D classification results (W2 template pipeline):
{metrics}

This is the FINAL 2D selection before ab-initio reconstruction.
Be more stringent than the W1 blob selection:
- Select only classes with clear TM helix density
- Require orientation diversity (top AND side AND tilted views)
- Target 40-60% particle retention (higher quality bar)
- Need at least 10,000 particles for reliable ab-initio

Respond with JSON:
{{
  "verdict": "PASS|WARN|FAIL",
  "decision": "approve|approve_with_adjustments|escalate_to_human",
  "confidence": 0.0-1.0,
  "observations": [...],
  "reasoning": "...",
  "recommended_actions": [...],
  "suggested_params": {{}}
}}"""
    }
}


# ---------------------------------------------------------------------------
# VLMCritic class
# ---------------------------------------------------------------------------

class VLMCritic:
    """
    Intelligent checkpoint evaluator using Vision Language Models.

    Usage:
        critic = VLMCritic(llm_config)

        # Tier 1 — metrics only (always works):
        assessment = critic.assess_checkpoint("curate", metrics_dict)

        # Tier 2 — with images (local mode):
        assessment = critic.assess_checkpoint("curate", metrics_dict, images=[img_bytes, ...])
    """

    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        self._openai_client = None
        self._anthropic_client = None

    # ── public ──────────────────────────────────────────────────────────

    def assess_checkpoint(
        self,
        step: str,
        metrics: Dict[str, Any],
        images: Optional[List[bytes]] = None,
    ) -> VLMAssessment:
        """
        Evaluate a checkpoint and return a structured assessment.

        Parameters
        ----------
        step : str
            Checkpoint step name (curate / inspect_blob / select2d_blob /
            inspect_template / select2d_template).
        metrics : dict
            Job metrics from CryoSPARC (particle counts, CTF fit values, etc.).
        images : list of bytes, optional
            Raw image bytes (PNG/JPG) for Tier 2 vision assessment.
            If None, falls back to Tier 1 (metrics-only).

        Returns
        -------
        VLMAssessment
        """
        tier2_available = (
            images is not None
            and len(images) > 0
            and self._supports_vision()
        )

        try:
            if tier2_available:
                assessment = self._assess_with_images(step, metrics, images)
            else:
                assessment = self._assess_with_metrics(step, metrics)
        except Exception as exc:
            logger.warning("VLMCritic failed for step %s: %s", step, exc)
            # Safe fallback: WARN verdict, escalate to human
            assessment = VLMAssessment(
                step=step,
                verdict="WARN",
                decision="escalate_to_human",
                confidence=0.0,
                reasoning=f"VLM assessment failed ({exc}). Defaulting to human review.",
                recommended_actions=["Review in CryoSPARC UI manually"],
            )

        # Auto-approve if confidence is high enough and verdict is PASS
        if (
            assessment.confidence >= AUTO_APPROVE_THRESHOLD
            and assessment.verdict == "PASS"
            and assessment.decision in ("approve", "approve_with_adjustments")
        ):
            assessment.auto_approved = True
            logger.info(
                "VLM auto-approved checkpoint '%s' (confidence=%.0f%%)",
                step, assessment.confidence * 100
            )

        logger.info("VLM assessment: %s", assessment.summary())
        return assessment

    def get_image_from_cryosparc(
        self,
        cs_client,
        project_uid: str,
        job_uid: str,
        image_type: str = "thumbnail",
    ) -> Optional[bytes]:
        """
        Download an image from CryoSPARC for Tier 2 vision assessment.

        Parameters
        ----------
        cs_client : CryoSPARC
            Active cryosparc-tools client (local mode only).
        project_uid : str
        job_uid : str
        image_type : str
            "thumbnail" | "ctf_plot" | "class_averages"

        Returns
        -------
        bytes or None
        """
        try:
            project = cs_client.find_project(project_uid)
            job = project.find_job(job_uid)
            # Try to get the first output image
            for output in job.doc.get("output_results", []):
                if output.get("type") in ("image", "png", "jpg"):
                    data = job.download(output["name"])
                    if data:
                        return data
        except Exception as exc:
            logger.debug("Could not download image from job %s: %s", job_uid, exc)
        return None

    # ── private ─────────────────────────────────────────────────────────

    def _assess_with_metrics(self, step: str, metrics: Dict[str, Any]) -> VLMAssessment:
        """Tier 1: LLM reasoning from metrics only."""
        template = CHECKPOINT_PROMPTS.get(step, {}).get("tier1", "")
        if not template:
            # Generic fallback
            template = """Evaluate this cryo-EM checkpoint step '{step}' with metrics: {{metrics}}
            Respond with JSON: {{verdict, decision, confidence, observations, reasoning, recommended_actions, suggested_params}}"""

        prompt = template.replace("{metrics}", json.dumps(metrics, indent=2))
        system = "You are an expert cryo-EM structural biologist. Be precise, evidence-based, and use the provided thresholds."

        text = self._llm_text(prompt, system=system, max_tokens=1024)
        return self._parse_assessment(step, text, used_images=False)

    def _assess_with_images(
        self,
        step: str,
        metrics: Dict[str, Any],
        images: List[bytes],
    ) -> VLMAssessment:
        """Tier 2: Vision LLM with actual CryoSPARC images."""
        template = CHECKPOINT_PROMPTS.get(step, {}).get("tier2", "")
        if not template:
            template = CHECKPOINT_PROMPTS.get(step, {}).get("tier1", "")

        prompt = template.replace("{metrics}", json.dumps(metrics, indent=2))
        system = "You are an expert cryo-EM structural biologist analyzing microscopy images. Be precise and scientific."

        # Encode images as base64
        encoded = [base64.b64encode(img).decode("utf-8") for img in images[:3]]  # max 3 images

        text = self._llm_vision(prompt, system=system, images_b64=encoded)
        return self._parse_assessment(step, text, used_images=True)

    def _parse_assessment(self, step: str, text: str, used_images: bool) -> VLMAssessment:
        """Parse LLM response into a VLMAssessment."""
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return VLMAssessment(
                step=step,
                verdict="WARN",
                decision="escalate_to_human",
                confidence=0.5,
                reasoning=text.strip()[:500],
                used_images=used_images,
            )
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return VLMAssessment(
                step=step,
                verdict="WARN",
                decision="escalate_to_human",
                confidence=0.5,
                reasoning=text.strip()[:500],
                used_images=used_images,
            )

        verdict = str(data.get("verdict", "WARN")).upper()
        if verdict not in ("PASS", "WARN", "FAIL"):
            verdict = "WARN"

        decision = str(data.get("decision", "escalate_to_human"))
        if decision not in ("approve", "approve_with_adjustments", "escalate_to_human"):
            decision = "escalate_to_human"

        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))

        return VLMAssessment(
            step=step,
            verdict=verdict,
            decision=decision,
            confidence=conf,
            observations=list(data.get("observations", [])),
            reasoning=str(data.get("reasoning", ""))[:800],
            recommended_actions=list(data.get("recommended_actions", [])),
            suggested_params=dict(data.get("suggested_params", {})),
            used_images=used_images,
        )

    def _supports_vision(self) -> bool:
        """Check whether the configured LLM supports vision input."""
        provider = (self.config.provider or "openai").lower()
        model = (self.config.model or "").lower()
        if provider == "anthropic":
            return "claude" in model  # All Claude models support vision
        # OpenAI: gpt-4o, gpt-4-turbo, gpt-4-vision
        return any(m in model for m in ("gpt-4o", "gpt-4-turbo", "gpt-4v", "vision"))

    def _llm_text(self, prompt: str, system: str, max_tokens: int = 1024) -> str:
        """Call LLM for text-only response."""
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
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

    def _llm_vision(self, prompt: str, system: str, images_b64: List[str]) -> str:
        """Call vision LLM with images."""
        provider = (self.config.provider or "openai").lower()
        if provider == "anthropic":
            if self._anthropic_client is None:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=self.config.api_key)
            content = []
            for img_b64 in images_b64:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
                })
            content.append({"type": "text", "text": prompt})
            resp = self._anthropic_client.messages.create(
                model=self.config.model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": content}],
            )
            return "".join(b.text for b in resp.content if hasattr(b, "text"))
        else:
            if self._openai_client is None:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.config.api_key)
            content = []
            for img_b64 in images_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            content.append({"type": "text", "text": prompt})
            resp = self._openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
