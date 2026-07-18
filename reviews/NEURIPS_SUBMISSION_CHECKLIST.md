# CryoEMAgent NeurIPS 2026 Super Checklist

Based on comprehensive review of:
- NEURIPS_2026_PLAN.md (v2, Master Plan)
- CryoWizard deep-dive review
- Cryo-IEF deep-dive review
- Core CryoEMAgent code (agent.py, planner.py, vlm_critic.py, memory.py, quality_critics.py)

**Total items: 380+** | **Minimum completion target: 75% for M0/M1/M3** | **Full completion: All milestones M0–M8**

---

## M0: Clean Run Baseline (Reproducible End-to-End)

Exit criterion: On EMPIAR-10288 20-movie subset, raw movies → refined volume with all 5 checkpoints human-approved. Resume-after-kill works. Zero crashes outside designated checkpoints. MetricsRecorder + reasoning_log both produced.

### M0.1 MetricsRecorder Implementation
- [ ] Create `cryoemagent/core/metrics.py` with `MetricsRecorder` class
  - [ ] Fields: timestamp, step, duration_sec, job_uid, vlm_verdict, vlm_confidence, human_paused, particles, resolution_A
  - [ ] Per-step JSONL output to `runs/{run_id}/metrics.jsonl`
  - [ ] Append-only semantics (resumable)
  - [ ] Type hints for all fields
  - [ ] Docstring with example output format
  - [ ] Timestamp format: ISO 8601 with subsecond precision
  - [ ] Error handling: graceful degradation if job_uid not available
- [ ] Hook MetricsRecorder into `agent.py` main loop
  - [ ] Record after each tool completion
  - [ ] Record at each checkpoint (pre-VLM and post-VLM)
  - [ ] Record after failure with error_reason field
  - [ ] Ensure no data loss on process crash
- [ ] Test: 10-step run produces complete metrics.jsonl
- [ ] Test: Resume a killed run; verify metrics file appends correctly

### M0.2 Reasoning-Log Output Enhancement
- [ ] Verify `_get_reasoning_log_path()` creates correct directory structure
- [ ] Verify reasoning-log JSON format in `agent.py:23` is schema-correct
  - [ ] Per-step records include: timestamp, step, decision, tool, params, verdict, confidence
  - [ ] Full decision history preserved (not circular-buffered)
  - [ ] Reasoning trace readable as plain text
- [ ] Promote reasoning logging from DEBUG to INFO level in interactive mode
  - [ ] Edit `interactive.py` logging setup
  - [ ] Verify no INFO spam (cap at 5 entries per step)
  - [ ] Test: run with `--verbose` and inspect log output

### M0.3 REPL Export Command
- [ ] Add `export-reasoning <run_id>` command to interactive REPL
  - [ ] Reads `runs/reasoning_logs/reasoning_{run_id[:8]}.json`
  - [ ] Outputs markdown file with readable narrative
  - [ ] Includes: step-by-step decisions, VLM verdicts, parameter adjustments
  - [ ] Format: Markdown with h2 headers per step, inline JSON for complex fields
  - [ ] Output path: `runs/{run_id}/reasoning_narrative.md`
- [ ] Test: export a completed run; verify markdown is readable and complete

### M0.4 Resume-After-Kill Test
- [ ] Create `tests/test_resume_after_kill.py`
  - [ ] Inject SIGKILL at step 8 (class2d step for 10288)
  - [ ] Verify ProcessingState persisted before kill
  - [ ] Restart agent with same `run_id`
  - [ ] Verify step 9 resumes without re-running step 8
  - [ ] Verify metrics.jsonl has exactly one entry for step 8
- [ ] Run on 10288 20-movie subset (not full run)
- [ ] Document: timeout/wall-clock time

### M0.5 No Hardcoded Config Values
- [ ] Audit `agent.py`, `planner.py`, `vlm_critic.py` for hardcoded thresholds
  - [ ] Expected: AUTO_APPROVE_THRESHOLD = 0.85 (in vlm_critic.py:35) ✓
  - [ ] Expected: GPCR-specific thresholds in planner.py:186 (documented)
  - [ ] Any others found → move to `config.yaml` + load via `LLMConfig`
- [ ] Update docstrings with "This is configurable via `config.yaml`"
- [ ] Test: change a threshold in config, verify it takes effect

### M0.6 Type Hints Audit
- [ ] All functions in agent.py have complete type hints (→ ReturnType)
- [ ] All functions in memory.py have complete type hints
- [ ] All functions in planner.py have complete type hints
- [ ] All dataclasses have type hints on fields
- [ ] Test: `mypy --strict agent.py` passes with no errors
- [ ] Test: `mypy --strict planner.py` passes with no errors

### M0.7 Docstring Audit
- [ ] Every public function has a docstring (PEP 257)
  - [ ] One-liner summary
  - [ ] Parameters section (Args:)
  - [ ] Returns section (Returns:)
  - [ ] Raises section (if applicable)
  - [ ] Example (if non-obvious)
- [ ] Test: `pydocstyle cryoemagent/` finds no errors

### M0.8 Clean Run on 10288 20-Movie Subset
- [ ] Prepare 10288 data: select 20 movies, link to `data/EMPIAR-10288-20mov/`
- [ ] Run: `cryoemagent run --dataset EMPIAR-10288 --dry-run false --resume-on-error false`
- [ ] All 5 checkpoints: curate exposures, inspect blob picks, select 2D, final refinement, postprocessing
- [ ] Human approval at each checkpoint (no auto-approve; test human-in-loop first)
- [ ] Verify outputs:
  - [ ] Final volume (refined_volume.mrc)
  - [ ] FSC curve
  - [ ] Final resolution estimate
  - [ ] metrics.jsonl (>= 19 entries, one per step)
  - [ ] reasoning_{run_id}.json (full decision trace)
- [ ] Document: run duration (GPU hours), # particles, final resolution
- [ ] Save run_id for M1 comparison

### M0.9 Baseline Metrics Capture
- [ ] Extract from M0.8 run:
  - [ ] Final resolution (Å) via GS-FSC 0.143
  - [ ] Total particles in refinement
  - [ ] # checkpoints human needed to pause (0 if all auto)
  - [ ] Runtime (wall-clock GPU hours)
- [ ] Store in `benchmarks/m0_baseline.json`
- [ ] Format: `{"run_id": "...", "resolution_A": 3.5, "particles": 50000, "human_pauses": 5, "gpu_hours": 12.5}`

---

## M1: Autopilot Mode (Zero Mandatory Checkpoints)

Exit criterion: Same EMPIAR-10288 run, `--autopilot=full`, 0–1 human pauses. VLM auto-approves ≥4/5 checkpoints with confidence ≥0.85. Final resolution within 0.3 Å of M0.

### M1.1 Tier-2 Image Retrieval Pipeline
- [ ] MCP tool: `cs_fetch_thumbnail(job_uid: str, kind: str) → bytes`
  - [ ] Kinds: "ctf_plot", "micrograph_thumbnail", "class_averages", "3d_volume_slice"
  - [ ] Implemented in `Cryosparc_mcp_Server/src/.../server.py`
  - [ ] Uses CryoSPARC REST API to fetch metadata
  - [ ] Renders PNG (if CTF plot) or crops MRC (if micrograph)
  - [ ] Returns base64-encoded bytes
  - [ ] Error handling: graceful fallback to Tier-1 if image unavailable
  - [ ] Caching: local disk cache under `runs/{run_id}/thumbnails/`
- [ ] Test: fetch 5 different kinds of images, verify dimensions and format
- [ ] Performance test: 10 fetches over SSH, measure latency (target: < 5s)

### M1.2 Wire Tier-2 into VLMCritic.evaluate()
- [ ] Update `vlm_critic.py:evaluate()` signature
  - [ ] Add optional `fetch_images_callback: Callable` parameter
  - [ ] If provided and confidence < 0.95, fetch relevant image
  - [ ] Pass image as base64 to Claude Vision in system prompt
- [ ] Prompts updated to reference image:
  - [ ] "curate": CTF plot + metrics → "Are these micrographs good quality?"
  - [ ] "inspect_blob": micrograph thumbnail + picks → "Are these picks correct?"
  - [ ] "select2d_blob": class_averages images → "Select high-quality classes"
  - [ ] "inspect_template": similar to blob
  - [ ] "select2d_template": similar to blob
- [ ] Type hints: all functions in vlm_critic.py have type hints
- [ ] Test: VLMCritic.evaluate() with and without images; verify format correct

### M1.3 VELM Feedback→Refine Inner Loop
- [ ] Implement in `vlm_critic.py`: `refine(checkpoint_step, initial_assessment) → VLMAssessment`
  - [ ] Takes initial assessment + image + metrics
  - [ ] Asks: "Given your previous verdict was {initial}, and looking more carefully, do you still agree?"
  - [ ] Produces refined confidence and reasoning
  - [ ] Returns new VLMAssessment
- [ ] Wire into `evaluate()` decision tree:
  - [ ] If `0.60 <= confidence < 0.85` → call `refine()` once
  - [ ] If `refined_confidence >= 0.85` → auto-approve
  - [ ] Else → escalate to human
- [ ] Decision thresholds documented in docstring
- [ ] Test: 5 checkpoints with borderline confidence; verify refine improves 3+

### M1.4 Implement `--autopilot=full` Flag
- [ ] Add to CLI argument parser (cli.py)
  - [ ] Options: "off" (default, all checkpoints pause), "partial" (auto only if conf >= 0.85), "full" (auto + refine)
  - [ ] Document: which thresholds apply in each mode
- [ ] Wire into agent.py main loop
  - [ ] At checkpoint: check autopilot mode
  - [ ] If "full": skip human pause; auto-approve if VLM says ok
  - [ ] Log decision to reasoning_log
- [ ] Test: run with `--autopilot=off` (human pauses), then `--autopilot=full` (no pauses)

### M1.5 Per-Checkpoint Confidence Logging
- [ ] Enhance metrics.jsonl to include:
  - [ ] `checkpoint_name` (e.g., "curate_exposures")
  - [ ] `vlm_confidence_raw` (pre-calibration)
  - [ ] `vlm_confidence_calibrated` (post-calibration; set to raw for now)
  - [ ] `auto_approved` (boolean)
  - [ ] `human_paused` (boolean; mutually exclusive with auto_approved)
- [ ] Ensure all 5 checkpoints emit these fields
- [ ] Test: run 10288, extract all checkpoint confidences, plot histogram

### M1.6 M1 Run on EMPIAR-10288 (Full 100+ Movies if Time)
- [ ] If time permits, run full EMPIAR-10288 dataset with `--autopilot=full`
  - [ ] Goal: 0 human pauses (all auto-approved)
  - [ ] Checkpoint confidence targets: ≥0.85 for each
- [ ] If time limited: run 50-movie subset and document limitations
- [ ] Capture metrics:
  - [ ] Final resolution (target: within 0.3 Å of M0.8)
  - [ ] Auto-approval rate per checkpoint (target: 4/5)
  - [ ] Total human intervention (target: 0)
  - [ ] Runtime comparison vs M0

### M1.7 Confidence Visualization
- [ ] Create `benchmarks/m1_confidence_report.md`
  - [ ] Table: checkpoint, confidence, auto_approved, resolution_impact
  - [ ] Histogram: confidence distribution across all checkpoints
  - [ ] Narrative: which checkpoints were hardest to auto-approve

### M1.8 Calibration Readiness Check
- [ ] Prepare placeholder `runs/calibration/thresholds.json`
  - [ ] Structure: `{"curate": {"T": 1.0}, "inspect_blob": {"T": 1.0}, ...}`
  - [ ] For now, all T=1.0 (no calibration); will populate in M3
- [ ] Verify VLMCritic can load and apply thresholds
- [ ] Test: change T value, verify confidence is multiplied

---

## M2: Reflexion + Skill Library (Failure Recovery)

Exit criterion: On deliberately-broken 10288 run (e.g., corrupt one batch of movies), agent reflects, retrieves/invents recovery recipe, records to skill library, finishes run. On subsequent failure-injected run, library is consulted first.

### M2.1 Planner.reflect_on_failure() Prompt & Implementation
- [ ] Add method to `planner.py`:
  ```python
  def reflect_on_failure(
      self, 
      state: ProcessingState, 
      decision: Dict[str, Any], 
      error: Exception
  ) -> 'Reflection':
  ```
- [ ] Prompt template asks for:
  1. **Cause analysis**: "What went wrong? Reference the error message and state."
  2. **Root lesson**: "What does this teach about {step_name}?"
  3. **Recovery recommendation**: "What should we try next?"
- [ ] Return dataclass `Reflection` with fields:
  - [ ] `cause_analysis: str` (1–2 sentences)
  - [ ] `lesson: str` (short actionable principle)
  - [ ] `recommendation: str` (next action: "retry with X", "skip Y", "switch to Z")
  - [ ] `timestamp: str`
  - [ ] `error_type: str` (e.g., "OOM", "empty_classification", "low_resolution")
- [ ] Constraint: all fields must cite something from `state` or `error`; no hallucination
- [ ] Test: feed 5 synthetic failures, inspect reflections for grounding

### M2.2 Skill Library Data Structure
- [ ] Create `cryoemagent/core/skills.py` with `SkillLibrary` class
- [ ] Data structure (JSON on disk, `runs/skill_library/skills.jsonl`):
  ```json
  {
    "skill_id": "reextract_larger_box_for_failed_2d",
    "trigger_conditions": {
      "current_step": "class2d_blob",
      "failure_mode": "empty_classes > 50%",
      "observed_metrics": {"avg_class_resolution_A": "> 15"}
    },
    "recipe_actions": [
      {"tool": "cs_re_extract_with_box", "params": {"box_px": 320}},
      {"tool": "run_2d_classification", "params": {"num_classes": 100}}
    ],
    "observed_outcome": {
      "success_rate": 0.83,
      "n_observations": 6,
      "mean_resolution_gain_A": -0.42
    },
    "provenance": "Learned from EMPIAR-10059 run 2026-04-28, confirmed 10288 run 2026-05-02."
  }
  ```
- [ ] SkillLibrary methods:
  - [ ] `retrieve(stage: str, step: str, failure_mode: str) → List[Skill]` (nearest match by semantic similarity)
  - [ ] `record(skill: Skill) → str` (writes to disk, returns skill_id)
  - [ ] `get_success_rate(skill_id: str) → float`
  - [ ] `update_statistics(skill_id: str, success: bool, metrics: Dict)` (incremental)
- [ ] Ensure thread-safe file I/O (JSON appending)
- [ ] Type hints throughout

### M2.3 Three Seed Recovery Recipes
- [ ] Recipe 1: `re_extract_larger_box`
  - [ ] Trigger: class2d step, >40% empty classes
  - [ ] Action: extract particles with box_px=320 (vs standard 256)
  - [ ] Stored in `runs/skill_library/seed_01_larger_box.json`
- [ ] Recipe 2: `re_2d_with_more_classes`
  - [ ] Trigger: class2d step, low average resolution
  - [ ] Action: re-run 2D with num_classes=100 (vs 50)
  - [ ] Stored in `runs/skill_library/seed_02_more_classes.json`
- [ ] Recipe 3: `skip_bad_batch_and_continue`
  - [ ] Trigger: extraction step, OOM error on specific micrographs
  - [ ] Action: exclude bad micrographs, re-extract remainder
  - [ ] Stored in `runs/skill_library/seed_03_skip_batch.json`
- [ ] Each seed recipe has full JSON structure with dummy statistics

### M2.4 Wire Reflexion into Agent Loop
- [ ] Update `agent.py` main loop:
  ```python
  try:
      result = mcp.call(decision["tool"], params)
  except ToolFailure as e:
      reflection = planner.reflect_on_failure(state, decision, e)
      recipe = skill_lib.retrieve(state.stage, state.step, reflection.error_type)
      if recipe is not None:
          decision = planner.react_with_hint(state, recipe)
          result = mcp.call(decision["tool"], params)
      else:
          escalate_with_reason({"reason": str(e), "reflection": reflection})
  ```
- [ ] Logging: every reflection recorded to `runs/{run_id}/reflections.jsonl`
- [ ] Error handling: if reflect_on_failure() fails, escalate (don't hallucinate)

### M2.5 Three Failure-Injection Tests
- [ ] Test 1: Corrupt one MRC file in the movie set
  - [ ] `tests/test_failure_corrupt_movie.py`
  - [ ] Inject failure at step: motion correction
  - [ ] Expected reflection: "Detected corrupted input data"
  - [ ] Expected recipe retrieval: skip-bad-batch or re-import
  - [ ] Verify: agent recovers, run completes
- [ ] Test 2: Empty 2D classification output
  - [ ] `tests/test_failure_empty_2d.py`
  - [ ] Inject: set num_classes=50, force to produce zero particles in top classes
  - [ ] Expected: reflection + re_2d_with_more_classes recipe
  - [ ] Verify: agent re-runs 2D with num_classes=100, recovers
- [ ] Test 3: Skill library consultation on second run
  - [ ] Run 1: hit failure, learn skill (recorded to disk)
  - [ ] Run 2 (new dataset): hit similar failure, verify skill is retrieved first
  - [ ] Verify: skill is applied before re-planning from scratch
- [ ] All three tests: `pytest tests/test_failure_*.py`

### M2.6 Skill Library Evaluation
- [ ] After M2.5, report:
  - [ ] Skill retrieval accuracy: of 10 tested failure modes, how many matched correctly?
  - [ ] Recovery success rate: of matched skills, what % led to completion?
  - [ ] Semantic similarity metric: did retrieval pick the most relevant skill?
- [ ] Document any retrieval failures → refine trigger_conditions structure

---

## M3: Calibration Harness (VLM Calibration)

Exit criterion: `runs/calibration/` directory with thresholds.json, pr_curves.png, calibration CSV. ECE per checkpoint ≤ 0.05. VLMCritic uses temperature-scaled outputs in production. All 16 headline metrics reported.

### M3.1 CryoSift Dataset Integration
- [ ] Download CryoSift 3,220-class labeled dataset (public, bioRxiv 2025)
  - [ ] Link/unzip to `benchmarks/cryosift_data/`
  - [ ] Verify directory structure: images/, labels.csv (or equivalent)
- [ ] Create adapter: `benchmarks/cryosift_adapter.py`
  - [ ] Function: `load_cryosift_images_and_labels() → List[Tuple[PIL.Image, int]]`
  - [ ] Maps image files to integer labels (0–4 for A/B/C/D/F grades)
  - [ ] Handles missing/corrupted images gracefully
  - [ ] Docstring with format specification
- [ ] Test: load adapter, verify 3220 images loaded, label distribution is reasonable

### M3.2 Mostofa Label Request
- [ ] Email Mostofa (one-page polite request):
  - [ ] "We need 30 labeled examples: 15 CTF estimations + 15 2D class selections"
  - [ ] Format: for CTF, provide {ctf_plot.png, is_good_quality: bool}
  - [ ] Format: for 2D, provide {class_average.png, keep_or_discard: bool}
  - [ ] Deadline: before M3.3 starts (suggest 1 week)
  - [ ] Mention: these will be held-out test set (not used in calibration training)
- [ ] Upon receipt: store at `benchmarks/mostofa_labels/`
  - [ ] ctf_holdout.jsonl (15 entries)
  - [ ] class2d_holdout.jsonl (15 entries)

### M3.3 Calibration.py Implementation
- [ ] Create `cryoemagent/core/calibration.py`
- [ ] Functions:
  - [ ] `compute_ece(predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) → float`
  - [ ] `compute_brier(predictions: np.ndarray, labels: np.ndarray) → float`
  - [ ] `compute_pr_curve(predictions: np.ndarray, labels: np.ndarray) → Tuple[np.ndarray, np.ndarray, float]`
  - [ ] `fit_temperature(predictions: np.ndarray, labels: np.ndarray, method: str = "isotonic") → float`
  - [ ] `apply_temperature(logits: np.ndarray, T: float) → np.ndarray`
- [ ] All functions have type hints and docstrings
- [ ] Unit tests: test on synthetic data (known calibration properties)

### M3.4 Decoupled Perception vs Reasoning Confidence
- [ ] Per VL-Calibration 2026 (arXiv 2604.09529), ask VLM two separate questions:
  - [ ] Q1: "Based ONLY on the image (if available), is this high quality?" → p_perception
  - [ ] Q2: "Based ONLY on the metrics/numbers, is this high quality?" → p_reasoning
  - [ ] Combine: `p_combined = w * p_perception + (1-w) * p_reasoning` (w learned from data)
- [ ] Modify `vlm_critic.py:evaluate()`:
  - [ ] Return VLMAssessment with fields: `p_perception`, `p_reasoning`, `p_combined`
  - [ ] Weight w: for now, w=0.5 (uniform); will optimize in M3.6

### M3.5 Calibration Pipeline
- [ ] Script: `benchmarks/calibrate.py`
  - [ ] For each checkpoint type: run VLMCritic.evaluate() on 3220 CryoSift samples
  - [ ] Compute ECE, Brier, PR curve on training set
  - [ ] Fit temperature T using isotonic regression
  - [ ] Evaluate on Mostofa holdout
  - [ ] Output: `runs/calibration/thresholds.json`
  - [ ] Output: `runs/calibration/pr_curves.png` (5 subplots, 300 DPI)
  - [ ] Output: `runs/calibration/calibration_data.csv`

### M3.6 Update VLMCritic Production Thresholds
- [ ] Load `runs/calibration/thresholds.json` in `vlm_critic.py`
- [ ] Update decision thresholds based on calibration:
  - [ ] If ECE > 0.05 for any checkpoint → raise AUTO_APPROVE_THRESHOLD to 0.90
  - [ ] Otherwise, keep 0.85
- [ ] Test: verify calibrated thresholds change auto-approval decisions

### M3.7 Calibration Report
- [ ] Create `benchmarks/m3_calibration_report.md`
  - [ ] Table: checkpoint, ECE, Brier, AUPRC, temperature T
  - [ ] Figure: PR curves
  - [ ] Holdout results: "On Mostofa's 30 samples, achieved 26/30 correct"

### M3.8 16 Headline Metrics Baseline
- [ ] Collect from M0, M1, M3 runs all 16 headline metrics (see §Evaluation Harness below)
- [ ] Store in `benchmarks/headline_metrics.json`

---

## M4: Cryo-IEF Integration (Foundation-Model Particle Scoring)

Exit criterion: On a bad-quality run, agent calls `cs_score_particles_with_cryoief`, ranks particles, keeps top-K, reaches comparable resolution with ~40% fewer particles than CryoWizard's grid-search default.

### M4.1 Cryo-IEF GPU Install
- [ ] On GPU server (10.0.1.2):
  - [ ] Clone Cryo-IEF to `/models/Cryo-IEF`
  - [ ] Create conda env: `conda create -n cryo_ief python=3.10`
  - [ ] Install requirements
  - [ ] Download model weights: `cryo_ranker_v1.5_vit_b_model.safetensors` → `/models/cryo_ranker_v1.5/`
  - [ ] Smoke test: `python /models/Cryo-IEF/CryoRanker_inference.py --help`

### M4.2 MCP Tool: cs_score_particles_with_cryoief
- [ ] In server.py, new tool handler (~200 lines):
  ```python
  def cs_score_particles_with_cryoief(
      job_path: str,
      output_dir: str,
      num_select: int = 40000,
      batch_size: int = 512,
      model_weights_path: str = "/models/cryo_ranker_v1.5",
  ) -> Dict[str, Any]:
  ```
- [ ] Runs subprocess: `accelerate launch CryoRanker_inference.py ...`
- [ ] Returns JSON with scores_csv, selected_cs, selected_star, statistics
- [ ] Error handling: return `{"status": "error", "error": "..."}` on failure

### M4.3 Planner Heuristic for Cryo-IEF Invocation
- [ ] At refinement step, if (post-2D resolution) > 4.0 Å for 2+ iterations AND n_particles > 100,000:
  - [ ] Suggest `cs_score_particles_with_cryoief` with reasoning

### M4.4 Benchmark: M4 vs CryoWizard Grid-Search
- [ ] Compare on test dataset:
  - [ ] Final resolution (target: within 0.2 Å)
  - [ ] Particles used (target: 40% fewer)
  - [ ] GPU hours (target: 60% of CryoWizard)
- [ ] Store in `benchmarks/m4_cryoief_benchmark.md`

---

## M5: CryoPipelineQA Benchmark (60 Questions)

Exit criterion: 60 QA pairs, 5 types × 12, labeled. Claude baseline + CryoWizard heuristic. Gap ≥ 0.2 accuracy.

### M5.1 Benchmark Design Template
- [ ] Five question types:
  1. **Next-action QA** (Decision type)
  2. **Checkpoint decision QA** (Classification type)
  3. **Failure diagnosis QA** (Explanation type)
  4. **Reconfig QA** (Parameter tuning type)
  5. **Escalation QA** (Categorical type)
- [ ] 12 questions per type → 60 total
- [ ] JSON schema: id, type, scenario, context (metrics, stage, step), options (A–D), correct_answer, reasoning

### M5.2 Benchmark Question Authoring (with Mostofa)
- [ ] Schedule 1-week sprint with Mostofa
  - [ ] Yaswanth writes scenario descriptions (all 60)
  - [ ] Mostofa labels correct answers + reasoning (all 60)
  - [ ] Mostofa spot-checks 15-item random sample
- [ ] Quality control: no ambiguous questions, plausible wrong options

### M5.3 Benchmark Storage & Format
- [ ] File: `benchmarks/cryopipelineqa.jsonl` (one JSON per line)
- [ ] Schema validation script: `benchmarks/validate_cryopipelineqa.py`

### M5.4 Claude Baseline Accuracy
- [ ] Script: `benchmarks/evaluate_cryopipelineqa.py`
  - [ ] Run all 60 questions through Claude
  - [ ] Output: `benchmarks/cryopipelineqa_claude_results.json`
  - [ ] Expected: 80–95% accuracy

### M5.5 CryoWizard Heuristic Baseline
- [ ] `cryowizard_baseline()`: always continue, fixed thresholds
- [ ] Expected accuracy: 40–60%
- [ ] Output: `benchmarks/cryopipelineqa_cryowizard_results.json`

### M5.6 CryoEMAgent Evaluation
- [ ] Run/evaluate CryoEMAgent on 60 scenarios
- [ ] Target: ≥80% overall
- [ ] Output: `benchmarks/cryopipelineqa_cryoemagent_results.json`

### M5.7 Gap Analysis
- [ ] `benchmarks/m5_benchmark_report.md`
  - [ ] Table: question_type × system accuracy
  - [ ] Narrative: highlight advantage on failure-diagnosis and reconfig questions

---

## M6: Perception Audit (VLM_SEE Causal 2×2 Probe)

Exit criterion: MCD/TADS computed for VLMCritic, figure ready. Placed on VLM_SEE scatter plot.

### M6.1 Causal 2×2 Probe Dataset Construction
- [ ] Build 4-cell dataset, ~40 samples each (160 total):
  - [ ] Cell 1: Natural image + natural metadata
  - [ ] Cell 2: Natural image + shuffled metadata
  - [ ] Cell 3: Perturbed image + natural metadata
  - [ ] Cell 4: Perturbed image + shuffled metadata
- [ ] Store at: `benchmarks/perception_audit_dataset/`

### M6.2 MCD (Colormap Dependence) Computation
- [ ] `MCD = |E[p_predict | cell1] - E[p_predict | cell2]|`
- [ ] Target: MCD < 0.30

### M6.3 TADS (Text-Authority Deference) Computation
- [ ] `TADS = |E[p_predict | cell1] - E[p_predict | cell3]|`
- [ ] Target: TADS > 0.20

### M6.4 VLM_SEE Coordination
- [ ] Call with VLM_SEE authors before M6: confirm formulas, get reference plot positions

### M6.5 Perception Audit Script
- [ ] `benchmarks/perception_audit.py`
  - [ ] Load 4-cell dataset, run VLMCritic 40× per cell
  - [ ] Output: JSON with MCD, TADS, interpretation

### M6.6 Perception Audit Figure
- [ ] 2D scatter plot: x=TADS, y=MCD
- [ ] Mark our point, overlay other VLMs for comparison
- [ ] File: `benchmarks/perception_audit_figure.png`

---

## M7: Multi-Protein Generalization (EMPIAR-10059 & 10644)

Exit criterion: Clean end-to-end runs on EMPIAR-10059 (TRPV1) and EMPIAR-10644 (β-gal). Generalization figure ready.

### M7.1 EMPIAR-10059 Setup
- [ ] Download to `data/EMPIAR-10059/`; verify movie count, pixel size, voltage

### M7.2 EMPIAR-10644 Setup
- [ ] Download to `data/EMPIAR-10644/`

### M7.3 Dataset-Family Tagging
- [ ] Add `dataset_family: str = "gpcr"` to ProcessingState
- [ ] Planner specializes thresholds per family:
  - [ ] gpcr: num_classes=50, box_px=256, min_particles=5000
  - [ ] membrane: num_classes=75, box_px=300, min_particles=8000
  - [ ] soluble: num_classes=60, box_px=280, min_particles=6000

### M7.4 Run on EMPIAR-10059
- [ ] `cryoemagent run --dataset EMPIAR-10059 --dataset-family membrane --autopilot=full`
- [ ] Target: ≥3.2 Å resolution, 0 human pauses

### M7.5 Run on EMPIAR-10644
- [ ] `cryoemagent run --dataset EMPIAR-10644 --dataset-family soluble --autopilot=full`
- [ ] Target: ≥2.8 Å resolution, 0 human pauses

### M7.6 Generalization Analysis
- [ ] `benchmarks/m7_generalization_analysis.md`
  - [ ] Table: dataset, protein_type, published_res_A, our_res_A, delta, particles, runtime_hours

### M7.7 Threshold Ablation (If Time)
- [ ] Rerun EMPIAR-10059 with GPCR thresholds; measure resolution degradation

### M7.8 Generalization Figure
- [ ] x=protein_MW_kDa, y=resolution_A; three points + reference lines
- [ ] File: `benchmarks/generalization_figure.png`

---

## M8: Paper + Reproducibility

Exit criterion: LaTeX paper (~8 pages), supplementary PDF, Docker kit, anonymized GitHub.

### M8.1 Paper Structure
- [ ] Abstract (150 words, 4 claims)
- [ ] Introduction (1.5 pages): position vs Structura, CryoWizard, ChemCrow
- [ ] Related work (1 page): all papers from plan §13
- [ ] Methods (2.5 pages): ReAct+Reflexion+skills, VLMCritic, calibration, pipeline stages
- [ ] Results (2 pages): M1 table, M3 PR curves, M5 benchmark, M6 perception, M7 generalization, cost
- [ ] Discussion (1 page): limitations, future work
- [ ] Reproducibility statement (0.5 page)

### M8.2–M8.6 Figures
- [ ] Figure 1: Agent architecture diagram (ReAct + Reflexion + VLMCritic + MCP-over-SSH)
- [ ] Figure 2: M1 auto-approval bar chart (5 checkpoints, auto/human/escalate)
- [ ] Figure 3: M3 PR curves (5 subplots, one per checkpoint, AUPRC labeled)
- [ ] Figure 4: M6 perception audit scatter (TADS vs MCD)
- [ ] Figure 5: M7 generalization (MW vs resolution, 3 proteins)

### M8.7–M8.9 Tables
- [ ] Table 1: Competitive comparison (System, Automation, Audit, Recovery, VLM, Remote, Resolution)
- [ ] Table 2: CryoPipelineQA benchmark (Type, Claude, CryoWizard, CryoEMAgent)
- [ ] Table 3: 16 headline metrics

### M8.10 LaTeX Compilation
- [ ] `paper/cryoemagent_main.tex`: all sections, figures, tables, references
- [ ] Compile clean: `pdflatex → bibtex → pdflatex → pdflatex`
- [ ] Output: 8-page PDF

### M8.11 Supplementary PDF
- [ ] Extended methods (calibration algorithm, skill library design)
- [ ] Additional results (per-dataset breakdowns)
- [ ] Failure cases (3–5 examples, lessons learned)
- [ ] Risk register (R1–R15 with mitigation status)
- [ ] Hyperparameter sensitivity analysis

### M8.12 Docker Reproducibility Kit
- [ ] `Dockerfile.cryoemagent`: CUDA 11.8 base, Python 3.10, all deps
- [ ] `docker-compose.yml` (optional)
- [ ] `README_DOCKER.md`: build + run instructions
- [ ] Test: `docker build -t cryoemagent:v0.3 .` + `docker run --gpus all cryoemagent:v0.3`

### M8.13 Anonymized GitHub Setup
- [ ] New private repo with no real credentials, real run IDs, or identifiable names
- [ ] Dummy configs, example outputs, `.gitignore` for data/weights
- [ ] README: installation, quick start, data download, architecture, citation, license

### M8.14 README & Documentation
- [ ] Main README.md (>500 words): summary, features, install, quick start, data, config, architecture, results, citation, license
- [ ] docs/METHODS.md (1000+ words): agent loop, Reflexion, skill library, VLMCritic, calibration, all hyperparameters
- [ ] docs/INSTALL.md: Linux/macOS/Windows, GPU setup, CryoSPARC integration, troubleshooting

### M8.15 License & Attribution
- [ ] LICENSE: MIT or Apache 2.0
- [ ] ATTRIBUTION.md: Cryo-IEF, CryoWizard, CryoSift, ReAct/Reflexion/Voyager/ERL/VL-Calibration

### M8.16 Citation Block
- [ ] BibTeX + OpenReview format prepared

---

## Engineering Quality Across All Milestones

### Code Quality
- [ ] Type hints: all functions have `→ ReturnType`
- [ ] Docstrings: PEP 257 for all public functions
- [ ] No hardcoded values (except thresholds in config.yaml)
- [ ] `pylint` and `flake8` warnings fixed
- [ ] Test coverage: ≥70% on core modules

### Testing
- [ ] Unit tests (tests/test_*.py):
  - [ ] test_memory.py, test_planner.py, test_vlm_critic.py
  - [ ] test_quality_critics.py, test_skills.py, test_calibration.py
- [ ] Integration tests:
  - [ ] test_m0_clean_run.py, test_m1_autopilot.py, test_m2_failure_recovery.py
- [ ] `pytest tests/` passes with no errors

### CI/CD
- [ ] GitHub Actions: install → pytest → mypy → pylint
- [ ] Artifact: test report HTML

### Dependencies
- [ ] requirements.txt: all pinned (anthropic>=0.45.0, fastmcp>=0.1.0, torch>=2.0.0, pydantic>=2.0, pyyaml>=6.0)
- [ ] Python 3.10+, CUDA 11.8+, CryoSPARC v4.7.1

### Windows Compatibility
- [ ] `pathlib.Path` everywhere (not os.path)
- [ ] `subprocess.run()` with `shell=False`
- [ ] CI tested on Windows 11 Home

---

## Evaluation Harness: 16 Headline Metrics

### Domain Metrics
- [ ] **#1** Final resolution (Å) — GS-FSC 0.143 on W2 volume; all 3 datasets
- [ ] **#2** Picking P/R/F1 — against Mostofa labels (if available); target F1 ≥ 0.80
- [ ] **#3** # particles in final refinement — target ≤100k
- [ ] **#4** Resolution vs particle-count curve — compare slope to CryoWizard

### Agentic Metrics
- [ ] **#5** Human-intervention rate — (# pauses) / 5; target 0–1 for 10288
- [ ] **#6** Auto-approval precision — (# correct auto) / (# total auto); target ≥0.95
- [ ] **#7** Auto-approval recall — (# correct auto) / (# should be approved); target ≥0.90

### Audit
- [ ] **#8** Reasoning audit score — Mostofa grades 1–5; target ≥4 average

### Calibration (M3)
- [ ] **#9** ECE per checkpoint — target ≤0.05 for all 5
- [ ] **#10** AUPRC at 0.85 threshold — target ≥0.85 per checkpoint

### Benchmark (M5)
- [ ] **#11** CryoPipelineQA accuracy — overall ≥80%; next-action ≥85%

### Perception (M6)
- [ ] **#12** MCD — target <0.30
- [ ] **#13** TADS — target >0.20

### Generalization (M7)
- [ ] **#14** EMPIAR-10059 resolution — target ≥3.2 Å
- [ ] **#15** EMPIAR-10644 resolution — target ≥2.8 Å

### Cost
- [ ] **#16** Cost per run — target <$50 + <20 GPU hours for GPCR full run

---

## Baselines (B1–B5)

- [ ] **B1** Mostofa manual run on 10288 subset — baseline time, resolution
- [ ] **B2** Pure-script (no LLM): hardcoded decisions end-to-end
- [ ] **B3** CryoSPARC Live — Rakshitha owns this
- [ ] **B4** CryoWizard — brute-force 20-GPU grid search
- [ ] **B5** Structura — cite paper if code not available

---

## Coordination & Team

### Mostofa (Weekly 1-pager asks)
- [ ] Week 1: Verify M0.8 clean run (volumes, thresholds)
- [ ] Week 2: Label 30 CTF + 2D holdout (M3.2)
- [ ] Week 3: Spot-check 15 CryoPipelineQA items (M5.2)
- [ ] Week 4: Audit reasoning logs on M1 runs (M8 audit score)
- [ ] Week 5: Advise on EMPIAR-10644 / stress-test datasets
- [ ] Week 6: Label 15-item holdout for CryoPipelineQA drift measurement

### Rakshitha
- [ ] B3 CryoSPARC Live baseline end-to-end
- [ ] Visualization figure for reasoning audit scores

### Mei Sign-Offs (Milestone Gates)
- [ ] M0 → M1 gate: reproducible clean run confirmed
- [ ] M1 → M3 gate: autopilot mode working
- [ ] M3 → M2 gate: calibration complete (M2 may run in parallel)
- [ ] Paper draft review: abstract + methods before full paper

### Lab Coordination
- [ ] **Before M6**: call with VLM_SEE authors (confirm MCD/TADS definitions)
- [ ] **Before M5**: call with SpatialRLM authors (benchmark construction alignment)
- [ ] **Before M1**: call with VELM authors (Feedback→Refine loop contract)

---

## Risk Mitigations (R1–R15)

| # | Risk | Mitigation | Status |
|---|------|-----------|--------|
| R1 | Cryo-IEF weights don't load | Spike install on sandbox; fallback to heuristics | NOT STARTED |
| R2 | CryoSift labels format wrong | Write adapter; fallback to Mostofa 30-item only | IN PROGRESS |
| R3 | Tier-2 image fetch slow over SSH | 10s timeout + disk cache + Tier-1 fallback | NOT STARTED |
| R4 | Feedback→Refine gives same answer | A/B test; drop if no gain | NOT STARTED |
| R5 | Reflexion hallucinates causes | Constrain prompts to cite state/error; filter | NOT STARTED |
| R6 | Skill library matches wrong skill | Log similarity scores; require confirm in non-autopilot | NOT STARTED |
| R7 | Structura "you're not first" objection | Table 1 + §1 claim set; honest positioning | IN PROGRESS |
| R8 | Auto-approve silently OKs bad checkpoint | Conservative calibration; Mostofa spot-check 20 | READY |
| R9 | CryoSPARC v5 breaks schema | Pin v4.7.1; version check in MCP server | PREVENTIVE |
| R10 | Windows-specific bugs | CI weekly on windows-latest; pathlib everywhere | PREVENTIVE |
| R11 | EMPIAR-10059 takes 20+ GPU hours | Schedule overnight; fallback to 250-movie subset | PREVENTIVE |
| R12 | API quota/price spike | Prompt-cache; GPT-4 fallback; budget tracking | PREVENTIVE |
| R13 | MCD/TADS definition mismatch | Call VLM_SEE before M6 | SCHEDULED |
| R14 | LLM hallucinates tool name | Strict validator; wide tests; re-prompt loop | DEFENSIVE |
| R15 | Team bandwidth issues | One ask/week via email; 30-min checkins | COMM PLAN |

---

## Final Pre-Submission Checklist

### Code Repository
- [ ] README.md: >500 words, all sections
- [ ] docs/INSTALL.md, docs/METHODS.md
- [ ] requirements.txt: all pinned
- [ ] LICENSE, ATTRIBUTION.md
- [ ] .gitignore: data, weights, logs excluded
- [ ] Anonymized (no lab names, no real credentials)

### Docker
- [ ] `docker build -t cryoemagent:v0.3 .` succeeds
- [ ] Sample data (10288 20-movie subset) included
- [ ] Expected outputs documented

### Paper
- [ ] cryoemagent_main.pdf: 8 pages, all figures, all tables, all references, spell-checked, NeurIPS format
- [ ] cryoemagent_supplementary.pdf: 10–15 pages
- [ ] All 5 claims backed by experiment results:
  - [ ] "4/5 checkpoints auto-approved" → M1 table
  - [ ] "ECE ≤ 0.05" → M3 table
  - [ ] "Generalizes 37–460 kDa" → M7 figure
  - [ ] "Recovers from failures" → M2 results

### Data & Reproducibility
- [ ] EMPIAR-10288, 10059, 10644: public, cited
- [ ] CryoSift labels: public, cited
- [ ] Pre-computed calibration (thresholds.json) in repo
- [ ] `benchmarks/generate_figures.py` regenerates all figures
- [ ] No data leakage (no real credentials, anonymized)

### No Hallucinated Citations
- [ ] All papers exist and are correctly cited (10 random spot-check)
- [ ] All DOI/arXiv/URLs verified

### OpenReview Submission
- [ ] Main PDF + supplementary uploaded
- [ ] Metadata: title, anonymous authors, abstract, keywords
- [ ] Code availability statement
- [ ] Data statement (public EMPIAR + CryoSift)
- [ ] Submission deadline verified and met

---

## Checklist Completion Estimates

| Milestone | Est. Person-Days | Est. GPU Hours | Critical Path |
|-----------|-----------------|----------------|---------------|
| M0 | 4 | 12 | Orchestrator tests |
| M1 | 6 | 24 | Image fetching performance |
| M2 | 5 | 8 | Skill schema + prompt engineering |
| M3 | 6 | 40 (VLM API) | CryoSift download, Mostofa labels |
| M4 | 4 | 24 | GPU server Cryo-IEF install |
| M5 | 7 | 4 | Mostofa labeling + LLM evaluations |
| M6 | 3 | 2 | VLM_SEE coordination |
| M7 | 4 | 48 | Schedule overnight runs |
| M8 | 8 | 0 | Writing + coordination |
| **Total** | **47** | **162** | **M0 first, then M1‖M3 in parallel** |

---

**TOTAL ITEMS: 380+**

**Minimum viable NeurIPS ED submission:** M0 + M1 + M3 + M5 ≈ 100 checklist items.

**Full comprehensive submission:** All 8 milestones + all supporting sections = all 380+ items.

---

**Document Owner:** Yaswanth (Remote Intern, Xu Lab, CMU)  
**Last Updated:** 2026-05-04  
**Next Review:** Upon completion of M0 (expect 2026-05-08)  
**Approval Authority:** Mei Yuan

Pin this document. Update with `[x]` as items complete. Use it to drive every engineering decision through NeurIPS submission.
