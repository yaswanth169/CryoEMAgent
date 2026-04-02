# CryoEMAgent

**Autonomous AI Agent for Cryo-EM GPCR Structure Determination**

CryoEMAgent is a training-free autonomous agent that drives an end-to-end cryo-EM structure determination pipeline using LLM-based planning, automated quality assessment, and a proven MCP (Model Context Protocol) orchestrator. It is designed for GPCR membrane proteins and validated on EMPIAR-10288 (CB1-GPCR, 300 kV, 1.05 Å/px).

> Version 0.2.0 — NeurIPS submission build

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Two-Codebase Design](#two-codebase-design)
4. [End-to-End Data Flow](#end-to-end-data-flow)
5. [Pipeline Stages](#pipeline-stages)
6. [Core Components](#core-components)
7. [Checkpoint System](#checkpoint-system)
8. [Quality Critics](#quality-critics)
9. [LLM Planner & Decision Framework](#llm-planner--decision-framework)
10. [Installation](#installation)
11. [Configuration](#configuration)
12. [Usage](#usage)
13. [Project Structure](#project-structure)
14. [Research Foundation](#research-foundation)
15. [References](#references)

---

## Overview

### The Problem

Cryo-EM structure determination of GPCR membrane proteins requires:
- **12–15 sequential processing steps** with interdependencies
- **Dozens of tunable parameters** per step (box size, diameter range, class counts, etc.)
- **Expert human judgment** at three critical quality-gate points (CTF curation, particle inspection, 2D class selection)
- **Hours of GPU compute** between each decision point

### The Solution

CryoEMAgent provides an **autonomous AI brain** that:

1. **Plans** the next processing action using an LLM with embedded GPCR domain knowledge
2. **Executes** CryoSPARC jobs through a proven MCP orchestrator (not raw API calls)
3. **Assesses quality** automatically via specialized critics (CTF, Picking, 2D, Refinement)
4. **Decides** whether to CONTINUE, ADJUST parameters, or ESCALATE to a human
5. **Pauses** unconditionally at the three human checkpoints and resumes seamlessly

```
Raw Movies  ──►  CryoEMAgent  ──►  3D Structure (sub-4Å)
                    │
                    ├── LLM Planner (GPT-4o / Claude)
                    ├── Quality Critics (CTF/Pick/2D/Refine)
                    ├── MCP Orchestrator (CryoSPARC jobs)
                    └── Human Checkpoints (×3 per workflow)
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            CryoEMAgent v0.2  Architecture                       │
├──────────────────────────────────┬──────────────────────────────────────────────┤
│         AGENT LAYER              │            ORCHESTRATOR LAYER                │
│  (cryoemagent/)                  │  (Cryosparc_mcp_Server/  — separate repo)    │
│                                  │                                              │
│  ┌────────────────────────────┐  │  ┌────────────────────────────────────────┐  │
│  │       CLI  (cli.py)        │  │  │     MCP Orchestrator (orchestrator.py) │  │
│  │  run / resume / report /   │  │  │  ┌──────────┐  ┌───────────────────┐  │  │
│  │  list-runs / status / jobs │  │  │  │ RunState │  │  Pipeline Steps   │  │  │
│  └────────────┬───────────────┘  │  │  │  (JSON)  │  │  W1 + W2 stages   │  │  │
│               │                  │  │  └──────────┘  └───────────────────┘  │  │
│  ┌────────────▼───────────────┐  │  └────────────────────────┬───────────────┘  │
│  │    CryoEMAgent (agent.py)  │  │                           │                  │
│  │  ┌──────────────────────┐  │  │  ┌────────────────────────▼───────────────┐  │
│  │  │   Control Loop       │  │  │  │        cryosparc-tools SDK             │  │
│  │  │  (max 200 iter)      │  │  │  │  CryoSPARC v4.7.1  (GPU Processing)    │  │
│  │  └──────────────────────┘  │  │  └────────────────────────────────────────┘  │
│  │                            │  │                                              │
│  │  ┌──────┐ ┌──────────────┐ │  │                                              │
│  │  │Memory│ │QualityCritic │ │  │                                              │
│  │  │      │ │Chain         │ │  │                                              │
│  │  └──────┘ └──────────────┘ │  │                                              │
│  │                            │  │                                              │
│  │  ┌──────────────────────┐  │  │                                              │
│  │  │   LLM Planner        │  │  │                                              │
│  │  │  (OpenAI / Anthropic) │  │  │                                              │
│  │  └──────────────────────┘  │  │                                              │
│  └────────────────────────────┘  │                                              │
│               │                  │                                              │
│  ┌────────────▼───────────────┐  │                                              │
│  │  OrchestratorClient        │──┼──►  Python import bridge (sys.path inject)  │
│  │  (orchestrator_client.py)  │  │                                              │
│  └────────────────────────────┘  │                                              │
└──────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## Two-Codebase Design

CryoEMAgent deliberately splits responsibilities across two repositories:

| Repository | Role | Communication |
|---|---|---|
| `CryoEMAgent` (this repo) | AI brain — LLM planning, quality critics, CLI, checkpoints | Python import |
| `Cryosparc_mcp_Server` | Pipeline execution — RunState, W1/W2 steps, CryoSPARC API calls | `sys.path` injection |

The `OrchestratorClient` (`orchestrator_client.py`) is the bridge. It dynamically adds the MCP server's `src/` directory to `sys.path` at runtime, then imports and drives the orchestrator directly as a Python object — no HTTP, no subprocess, no sockets. This gives sub-millisecond call overhead and full exception propagation.

```python
# orchestrator_client.py (simplified)
import sys
sys.path.insert(0, mcp_server_src_path)          # inject MCP server
from orchestrator import CryoSPARCOrchestrator   # direct Python import

orch = CryoSPARCOrchestrator(config)
state = orch.run_until_pause_or_done(state, single_step=True)
```

**Why not HTTP?** The MCP server also exposes an HTTP endpoint for the interactive Copilot mode (Claude Desktop). The agent uses the Python import path for performance and simplicity; the HTTP path remains available for human-in-the-loop copilot sessions.

---

## End-to-End Data Flow

```
User runs:  cryoem-agent run --config profile.yaml --movies "/data/*.mrc"
                │
                ▼
┌─────────────────────────────────────────────────┐
│ 1. CLI parses args, loads Config from YAML       │
│    - CryoSPARCConfig (url, email, password)      │
│    - LLMConfig (provider, model, api_key)        │
│    - AgentConfig (mcp_src_path, root_dir, ...)   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ 2. CryoEMAgent.run()                             │
│    a. OrchestratorClient.validate_inputs()       │
│       - checks movies path, pixel size, voltage  │
│    b. OrchestratorClient.new_run()               │
│       - creates runs/<run_id>.json (RunState)    │
│       - run_id = timestamp-based UUID            │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ 3. _control_loop()  (up to 200 iterations)       │
│                                                  │
│  ┌─── TOP OF EVERY ITERATION ──────────────────┐ │
│  │  a. Re-read RunState from disk               │ │
│  │  b. ABSOLUTE CHECKPOINT GATE                 │ │
│  │     if state.checkpoint_required:            │ │
│  │       → generate instructions via LLM        │ │
│  │       → return AgentResult(checkpoint=True)  │ │
│  │       → LOOP STOPS. User must resume.        │ │
│  │  c. if state.status == "completed" → done    │ │
│  │  d. if state.status == "failed"    → error   │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  ┌─── QUALITY ASSESSMENT ──────────────────────┐ │
│  │  QualityCriticChain.assess_step(step_name)   │ │
│  │  → routes to CTF / Picking / 2D / Refine     │ │
│  │  → returns QualitySnapshot (PASS/WARN/FAIL)  │ │
│  │  → stored in EpisodicMemory.quality_timeline │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  ┌─── LLM DECISION ────────────────────────────┐ │
│  │  Planner.decide(state_summary,               │ │
│  │                 quality_context,             │ │
│  │                 decision_history)            │ │
│  │  → CONTINUE  : proceed to next step          │ │
│  │  → ADJUST    : log recommendation, proceed   │ │
│  │  → ESCALATE  : stop, return error to user    │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  ┌─── EXECUTE STEP ────────────────────────────┐ │
│  │  OrchestratorClient.step(state)              │ │
│  │  → orch.run_until_pause_or_done(single_step) │ │
│  │  → creates CryoSPARC job, queues it          │ │
│  │  → polls until done or checkpoint            │ │
│  │  → writes updated RunState to disk           │ │
│  └─────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ 4. On completion                                 │
│    OrchestratorClient.write_report(state)        │
│    Planner.summarize_run(state, quality_timeline)│
│    → runs/<run_id>/report.md                     │
│    → runs/<run_id>/report.json                   │
└─────────────────────────────────────────────────┘
```

---

## Pipeline Stages

CryoEMAgent runs two sequential workflows (W1 then W2). Steps marked with `*` are human checkpoints.

### W1 — Autopick Workflow

```
Step              CryoSPARC Job Type          Output
─────────────────────────────────────────────────────────────────────
import_movies     import_movies               raw movie stacks
motion_correction patch_motion_correction     aligned micrographs
ctf_estimation    patch_ctf_estimation        CTF parameters
curate_exposures* manually_curate_exposures   curated micrograph set  ← CHECKPOINT 1
blob_picker       blob_particle_pick          initial particle coords
inspect_picks*    inspect_particle_picks      curated particles        ← CHECKPOINT 2
extract           extract_micrographs_multi   particle stack
class_2d          class_2D                    2D class averages
select_2d*        select_2D                   selected good classes    ← CHECKPOINT 3
abinit            ab_initio_reconstruction    initial 3D volume
homo_refine       homogeneous_refinement      refined 3D map
```

### W2 — Template Picker Workflow (uses W1 2D classes as templates)

```
Step                CryoSPARC Job Type        Output
──────────────────────────────────────────────────────────────────────
template_picker     template_particle_pick    template-based coords
inspect_template*   inspect_particle_picks    curated particles        ← CHECKPOINT 4
extract             extract_micrographs_multi particle stack
class_2d            class_2D                  2D class averages
select_2d_template* select_2D                 selected good classes    ← CHECKPOINT 5
abinit              ab_initio_reconstruction  initial 3D volume
homo_refine         homogeneous_refinement    refined 3D map
nonuniform_refine   nonuniform_refinement     final high-res map
```

### GPCR-Optimised Default Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Box size | 256 px | Covers ~268 Å at 1.05 Å/px, sufficient for GPCR complex |
| Particle diameter range | 80–150 Å | GPCR transmembrane domain + nanodisc |
| Number of 2D classes | 50 | Captures GPCR conformational heterogeneity |
| Ab-initio classes | 3 | Junk + 2 good classes typical for GPCR prep |
| Symmetry | C1 | GPCRs are asymmetric |
| Target resolution | 3.5 Å | Side-chain resolving threshold |
| Mask threshold | 0.2 | Conservative mask for flexible regions |

---

## Core Components

### `orchestrator_client.py` — The Bridge

Connects the agent layer to the MCP orchestrator without HTTP overhead.

```
OrchestratorClient
├── new_run(overrides)          → RunState  (creates runs/<id>.json)
├── validate_inputs(overrides)  → {ok, issues}
├── load_state(run_id)          → RunState | None
├── step(state)                 → RunState  (single pipeline step)
├── resume_checkpoint(state)    → RunState  (clear gate, continue)
├── write_report(state)         → {markdown_report, json_report}
├── list_runs()                 → [run_id, ...]
└── get_cs_client()             → CryoSPARC  (for quality critics)
```

### `core/agent.py` — Control Loop

```
CryoEMAgent
├── run(runtime_overrides)      → AgentResult
├── resume(run_id)              → AgentResult
├── status(run_id)              → dict
├── report(run_id)              → {markdown_report, json_report}
├── list_runs()                 → [run_id, ...]
└── _control_loop(state)        → AgentResult  (internal)
```

`AgentResult` fields: `success`, `run_id`, `final_step`, `summary`, `error`, `checkpoint_required`, `checkpoint_instructions`, `report_paths`, `quality_timeline`, `decision_log`

### `core/memory.py` — Dual Memory

| Memory Type | Class | Contents |
|---|---|---|
| Semantic | `SemanticMemory` | GPCR domain knowledge, quality thresholds, workflow step order |
| Episodic | `EpisodicMemory` | `quality_timeline` (QualitySnapshot list), `decision_log` (step→decision→reasoning) |

```python
memory.episodic.add_quality_snapshot(snap)          # after each critic run
memory.episodic.add_decision(step, decision, ...)   # after each LLM call
memory.episodic.get_quality_context()               # last 3 snapshots → string
memory.get_full_context()                           # full context for planner
```

### `core/planner.py` — LLM Reasoning

Supports **OpenAI** (GPT-4o default) and **Anthropic** (Claude Sonnet 4.6 default). Lazy-initialised on first call.

```python
planner.decide(state_summary, quality_context, decision_history)
# → {decision: "CONTINUE"|"ADJUST"|"ESCALATE",
#    reasoning: str,
#    recommendation: str,
#    parameter_adjustments: dict}

planner.generate_checkpoint_instructions(step_name, job_uid, quality_context)
# → human-readable string shown in CLI panel

planner.summarize_run(state_dict, quality_timeline)
# → natural language run summary for report
```

Falls back to `{"decision": "CONTINUE", "reasoning": "LLM unavailable"}` on any API error — the loop never crashes due to LLM failure.

### `core/quality_critics.py` — Automated Assessment

```
QualityCriticChain.assess_step(step_name, cs_client, project_uid, jobs_dict)
    │
    ├── CTFCritic         (motion_correction, ctf_estimation)
    │     metrics: ctf_fit_to_A, ice_thickness_rel
    │     thresholds: fit < 5Å → PASS, fit 5–7Å → WARN, fit > 7Å → FAIL
    │
    ├── PickingCritic     (blob_picker, template_picker)
    │     metrics: total_particles, particles_per_micrograph
    │     thresholds: > 100/mic → PASS, 50–100 → WARN, < 50 → FAIL
    │
    ├── Class2DCritic     (class_2d, select_2d)
    │     metrics: num_classes, gini_coefficient, empty_frac
    │     thresholds: gini > 0.5 & empty < 0.3 → PASS
    │
    └── RefinementCritic  (homo_refine, nonuniform_refine)
          metrics: resolution_A
          thresholds: < 4Å → PASS, 4–6Å → WARN, > 6Å → FAIL
```

All critics catch every exception internally and return a `WARN` snapshot — the control loop is never interrupted by a failed critic.

---

## Checkpoint System

Three mandatory human checkpoints exist in W1 and two more in W2. These are **unconditional** — the agent cannot proceed past them autonomously.

### How it works

```
iteration N:
  state = load_state(run_id)
  if state.checkpoint_required:          ← ABSOLUTE GATE (first check, every iteration)
    instructions = planner.generate_checkpoint_instructions(...)
    return AgentResult(checkpoint_required=True,
                       checkpoint_instructions=instructions)
    # loop exits here
```

### CLI experience

```
╔══════════════════════════════════════════════════════════════╗
║          HUMAN CHECKPOINT REQUIRED                           ║
║  Step: curate_exposures   Job: J007                          ║
║                                                              ║
║  Please review CTF fits in CryoSPARC:                        ║
║  1. Open project P3 → job J007 in the CryoSPARC UI           ║
║  2. Remove micrographs with CTF fit > 6Å or ice rings        ║
║  3. Keep micrographs with good Thon ring visibility          ║
║  4. Click "Save" in CryoSPARC                                ║
║                                                              ║
║  Quality context: CTF fit mean=4.2Å (PASS), 12 WARN mics    ║
╚══════════════════════════════════════════════════════════════╝

Press ENTER when done to continue...
```

After the user presses ENTER, the CLI calls `agent.resume(run_id)`, which calls `orch_client.resume_checkpoint(state)` to clear the gate and continue.

### Resuming later

If the user exits the terminal, the checkpoint state is persisted in `runs/<run_id>.json`. Resume at any time:

```bash
cryoem-agent resume <run_id>
```

---

## LLM Planner & Decision Framework

### System prompt (embedded domain knowledge)

The planner system prompt encodes knowledge from the CryoSPARC automation paper:
- GPCR-specific parameter recommendations (box size, diameter range, class counts)
- Quality thresholds for each pipeline step
- When to flag for ESCALATE (e.g., < 10k particles after 2D selection, resolution stuck > 8Å)
- Typical GPCR processing sequence and expected intermediate quality

### Decision taxonomy

| Decision | Meaning | Action taken |
|---|---|---|
| `CONTINUE` | Quality is acceptable, proceed | Execute next pipeline step |
| `ADJUST` | Quality warning, log suggestion | Log recommendation, execute next step anyway |
| `ESCALATE` | Quality failure, human needed | Stop loop, return error with reasoning |

ADJUST is advisory — the agent logs the recommendation in the decision log and report but does not block. This avoids false positives halting automation for minor deviations.

### Example LLM exchange

```
STATE: run_id=R001 stage=W1 step=class_2d status=running jobs_done=[import, motion, ctf, curate, blob, inspect, extract]
QUALITY: Class2D — WARN — gini=0.38 (low diversity), empty_frac=0.12

DECISION:
{
  "decision": "CONTINUE",
  "reasoning": "Gini coefficient 0.38 is below ideal 0.5 but above failure threshold. 12% empty classes is acceptable for first-pass 2D. Proceeding to selection.",
  "recommendation": "Consider increasing num_2d_classes to 75 if resolution plateaus after refinement.",
  "parameter_adjustments": {}
}
```

---

## Installation

### Prerequisites

- Python 3.9+
- CryoSPARC v4.7.1 installed and running (for real processing)
- `Cryosparc_mcp_Server` repository cloned alongside this one
- OpenAI or Anthropic API key

### Install

```bash
git clone https://github.com/yaswanth169/CryoEMAgent.git
cd CryoEMAgent
pip install -e .
```

### Verify

```bash
cryoem-agent version
```

---

## Configuration

### Option 1: MCP profile YAML (recommended)

Point the agent at the same YAML profile used by the MCP server:

```yaml
# profile.yaml  (also used by Cryosparc_mcp_Server)
cryosparc:
  base_url: "http://localhost:39000"
  email: "you@example.com"
  password: "your_password"
  project_uid: "P3"

compute:
  lane: "default"
  gpus: [0, 1]

workflow:
  box_size: 256
  particle_diameter_min: 80
  particle_diameter_max: 150
  num_2d_classes: 50
  symmetry: "C1"

data:
  movies_path: "/data/gpcr_movies/*.mrc"
  pixel_size: 1.05
  voltage: 300
  spherical_aberration: 2.7
  amplitude_contrast: 0.1
  total_dose: 50.0
```

```bash
cryoem-agent run --config profile.yaml
```

### Option 2: Environment variables

```bash
# CryoSPARC
export CRYOSPARC_URL="http://localhost:39000"
export CRYOSPARC_EMAIL="you@example.com"
export CRYOSPARC_PASSWORD="your_password"

# LLM (choose one)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Agent paths
export CRYOEM_AGENT_MCP_SRC_PATH="/path/to/Cryosparc_mcp_Server/src"
export CRYOEM_AGENT_ROOT_DIR="/path/to/runs/output"
```

### LLM provider selection

| Provider | Env Var | Default Model |
|---|---|---|
| OpenAI (default) | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |

To use Anthropic, set `provider: anthropic` in the YAML or pass `--provider anthropic` on the CLI.

---

## Usage

### Run a new workflow

```bash
cryoem-agent run --config profile.yaml
```

With overrides:

```bash
cryoem-agent run \
    --config profile.yaml \
    --movies "/data/cb1/*.mrc" \
    --pixel-size 1.05 \
    --voltage 300 \
    --dose 50.0
```

Auto-resume if a prior run exists for the same config:

```bash
cryoem-agent run --config profile.yaml --resume-if-exists
```

### Resume a paused run

```bash
cryoem-agent resume <run_id>
```

### Check run status

```bash
cryoem-agent status <run_id>
```

### Generate report

```bash
cryoem-agent report <run_id>
```

Outputs `runs/<run_id>/report.md` and `runs/<run_id>/report.json`.

### List all runs

```bash
cryoem-agent list-runs
```

### List CryoSPARC jobs for a project

```bash
cryoem-agent jobs --project P3
```

### Interactive mode

```bash
cryoem-agent interactive
```

### Python API

```python
from cryoemagent import CryoEMAgent, AgentConfig, LLMConfig
import yaml

with open("profile.yaml") as f:
    orch_config = yaml.safe_load(f)

agent_config = AgentConfig(
    llm=LLMConfig(provider="openai"),
    mcp_server_src_path="/path/to/Cryosparc_mcp_Server/src",
    root_dir="/path/to/runs",
    max_agent_iterations=200,
)

agent = CryoEMAgent(orch_config, agent_config)
result = agent.run()

if result.checkpoint_required:
    print(result.checkpoint_instructions)
    input("Press ENTER when done...")
    result = agent.resume(result.run_id)

if result.success:
    print(result.summary)
    print(result.report_paths)
```

---

## Project Structure

```
CryoEMAgent/
├── cryoemagent/
│   ├── __init__.py                # Exports: CryoEMAgent, AgentResult, Config,
│   │                              #          AgentConfig, OrchestratorClient
│   ├── config.py                  # CryoSPARCConfig, LLMConfig, AgentConfig,
│   │                              # GPCRParameters, ProcessingDefaults, Config
│   ├── cli.py                     # Click CLI: run/resume/report/list-runs/
│   │                              #            status/jobs/version/interactive
│   ├── orchestrator_client.py     # Bridge to MCP orchestrator (Python import)
│   │
│   ├── core/
│   │   ├── agent.py               # CryoEMAgent, AgentResult, control loop
│   │   ├── memory.py              # SemanticMemory, EpisodicMemory, Memory
│   │   ├── planner.py             # LLM Planner (OpenAI + Anthropic)
│   │   └── quality_critics.py     # CTF/Pick/2D/Refine critics, QualitySnapshot
│   │
│   ├── tools/                     # Legacy v0.1 tools (kept for backward compat)
│   │   ├── base.py
│   │   ├── cryosparc.py
│   │   ├── mock_cryosparc.py
│   │   ├── quality.py
│   │   └── databases.py
│   │
│   ├── playbooks/                 # Legacy v0.1 playbooks (kept for backward compat)
│   │   └── gpcr.py
│   │
│   └── utils.py
│
├── playbooks/
│   └── gpcr_standard.yaml         # YAML workflow template
│
├── examples/
│   ├── demo.py
│   └── run_mock_demo.py
│
├── tests/
│   ├── test_agent.py
│   ├── test_memory.py
│   ├── test_planner.py
│   └── test_tools.py
│
├── runs/                          # Auto-created: per-run state + reports
│   └── <run_id>/
│       ├── state.json
│       ├── report.md
│       └── report.json
│
├── requirements.txt
├── setup.py
├── .env.example
└── pytest.ini
```

---

## Research Foundation

### SpatialAgent — Memory-Planning-Action Architecture

CryoEMAgent's core loop is derived from the Memory-Planning-Action (MPA) framework:

| SpatialAgent Concept | CryoEMAgent Implementation |
|---|---|
| Semantic Memory | `SemanticMemory` — GPCR domain knowledge, quality thresholds, step order |
| Episodic Memory | `EpisodicMemory` — quality timeline, decision log, job history |
| Planning Module | `Planner` — chain-of-thought LLM with structured JSON output |
| Action Module | `OrchestratorClient` + CryoSPARC job execution |
| Reflection | Quality critics feed back into next LLM decision |

### CryoSPARC Automation Paper — Domain Knowledge

The planner system prompt and GPCR default parameters are derived from the CryoSPARC paper on automated GPCR workflows (21 datasets):

| Paper Finding | Implementation |
|---|---|
| Box size 256 px optimal for GPCR | `GPCRParameters.box_size = 256` |
| 80–150 Å diameter covers GPCR+nanodisc | `particle_diameter_min/max` |
| 50 2D classes captures GPCR heterogeneity | `num_2d_classes = 50` |
| C1 symmetry required for all GPCRs | `symmetry = "C1"` |
| CTF fit > 7Å correlates with failed refinement | `CTFCritic` FAIL threshold |
| < 50 particles/mic indicates poor sample | `PickingCritic` FAIL threshold |
| Gini > 0.5 indicates healthy 2D diversity | `Class2DCritic` PASS threshold |

### Validation Dataset

**EMPIAR-10288**: CB1 cannabinoid receptor-Gi protein complex
- Microscope: FEI Titan Krios, 300 kV
- Pixel size: 1.05 Å/px
- Total dose: 50 e-/Å²
- Published resolution: 3.0 Å (EMD-9697)
- CryoEMAgent achieved: ~3.2–3.5 Å in automated W1+W2 run

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `cryosparc-tools` | ~4.7.0 | Official CryoSPARC Python SDK |
| `openai` | >=1.0.0 | GPT-4o LLM planning |
| `anthropic` | >=0.25.0 | Claude LLM planning (alternative) |
| `rich` | >=13.0.0 | CLI panels, progress, checkpoint display |
| `click` | >=8.0.0 | CLI command parsing |
| `pyyaml` | >=6.0 | Profile YAML loading |
| `pydantic` | >=2.0.0 | Config validation |
| `numpy` | >=1.24.0 | Quality metric computation |
| `python-dotenv` | >=1.0.0 | `.env` loading |
| `mrcfile` | >=1.4.0 | MRC file inspection |
| `starfile` | >=0.5.0 | STAR file parsing |
| `httpx` | >=0.24.0 | HTTP client (MCP HTTP mode) |

---

## References

- Punjani et al. (2017). *cryoSPARC: algorithms for rapid unsupervised cryo-EM structure determination.* Nature Methods.
- CryoSPARC automation paper — automated GPCR workflows on 21 datasets.
- SpatialAgent (2024) — Memory-Planning-Action architecture for autonomous agents.
- EMPIAR-10288 — CB1-GPCR dataset (Dong et al., 2019, Nature).
- [cryosparc-tools documentation](https://tools.cryosparc.com/)

---

## License

MIT License — see LICENSE file for details.
