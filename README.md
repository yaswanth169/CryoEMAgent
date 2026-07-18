# CryoEMAgent v0.2

**Autonomous AI Agent for Cryo-EM GPCR Structure Determination**

CryoEMAgent is an autonomous agentic framework that drives an end-to-end cryo-EM structure determination pipeline using LLM-based planning (ReAct), episodic memory, automated quality assessment, VLM-based checkpoint evaluation, and a conversational interface — all connected to CryoSPARC on a remote GPU server via MCP over SSH.

Validated on **EMPIAR-10288 (CB1-GPCR, 300 kV, 1.05 Å/px)**.

> Version 0.2.0 — NeurIPS submission build

---

## Table of Contents

1. [Overview](#overview)
2. [Two Modes](#two-modes)
3. [System Architecture](#system-architecture)
4. [Agentic Framework — Planning · Memory · Action](#agentic-framework)
5. [VLM Critic — Intelligent Checkpoint Evaluation](#vlm-critic)
6. [Pipeline — W1 + W2 (19 Steps)](#pipeline)
7. [Checkpoint System](#checkpoint-system)
8. [MCP over SSH](#mcp-over-ssh)
9. [Installation](#installation)
10. [Configuration](#configuration)
11. [Usage](#usage)
12. [Project Structure](#project-structure)
13. [Reasoning Log](#reasoning-log)
14. [Research Foundation](#research-foundation)

---

## Overview

### The Problem

Cryo-EM structure determination of GPCR membrane proteins requires:
- **19 sequential processing steps** across two pipelines (W1 blob + W2 template)
- **Dozens of tunable parameters** per step (box size, NCC thresholds, class counts, etc.)
- **Expert human judgment** at 5 critical checkpoints (CTF curation, particle inspection ×2, 2D class selection ×2)
- **Hours of GPU compute** between each decision point
- Runs on a **remote GPU server** — not the scientist's laptop

### The Solution

CryoEMAgent provides an **autonomous AI agent** that:

1. **Plans** the next action using a ReAct-pattern LLM with 15 formal CryoSPARC tool definitions
2. **Executes** CryoSPARC jobs through MCP (Model Context Protocol) over SSH — runs from any laptop
3. **Assesses quality** via specialized critics (CTF, picking, 2D, refinement)
4. **Evaluates checkpoints** using a VLM critic — auto-approves when confidence ≥ 85%
5. **Explains** every decision with a full reasoning chain (Observation → Thought → Tool → Decision)
6. **Converses** naturally like Claude Code — users control it in plain English

---

## Two Modes

Launch with `python run.py` and choose at startup:

```
╭─────────────────── Select Mode ───────────────────╮
│  [1]  Autopilot    — Fire-and-forget pipeline      │
│                      Best for: overnight runs       │
│                                                     │
│  [2]  Interactive  — Conversational (Claude Code)  │
│                      Best for: demo, learning       │
╰─────────────────────────────────────────────────────╯
```

| Feature | Autopilot | Interactive |
|---------|-----------|-------------|
| GPU steps | Fully autonomous | Fully autonomous |
| Human checkpoints | Pause + print instructions | VLM assessment + optional human |
| LLM reasoning | Written to log file | Shown live on screen |
| User input | ENTER to resume | Natural language conversation |
| Best for | Production overnight runs | Demo, research, learning |

Both modes use the **same MCP engine** — only the UI layer differs.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAPTOP (any OS)                              │
│                                                                 │
│  python run.py                                                  │
│       │                                                         │
│       ├── [1] Autopilot ──► CryoEMAgent                        │
│       │                      ├── Planner (ReAct LLM)           │
│       │                      ├── Memory (Episodic + Semantic)   │
│       │                      ├── QualityCriticChain             │
│       │                      └── MCPOrchestratorClient ─────┐   │
│       │                                                      │  │
│       └── [2] Interactive ► InteractiveSession               │  │
│                              ├── IntentRouter (LLM chat)     │  │
│                              ├── Planner (ReAct reasoning)   │  │
│                              ├── Memory (Episodic)           │  │
│                              ├── VLMCritic (checkpoints)     │  │
│                              └── MCPOrchestratorClient ──────┤  │
│                                                              │  │
│  SSH tunnel (jump host)  ◄───────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┬──┘
                                                               │ stdio
┌──────────────────────────────────────────────────────────────▼──┐
│                  GPU SERVER (10.0.1.2)                          │
│                                                                 │
│  Cryosparc_mcp_Server/run_server.py                            │
│       │                                                         │
│       ├── FastMCP server (JSONL framing)                        │
│       ├── Orchestrator (W1 + W2 pipeline logic)                 │
│       ├── CryoSPARC Adapter (cryosparc-tools 4.7.1)            │
│       └── RunStore (JSON checkpoint persistence)                │
│                                                                 │
│  CryoSPARC v4.7.1  ◄────────────────── GPU jobs                │
│  (localhost:39000)       J180 · J181 · J182 · J183 · ...       │
└─────────────────────────────────────────────────────────────────┘
```

### Two-Codebase Design

| Codebase | Location | Runs on | Role |
|----------|----------|---------|------|
| `CryoEMAgent/` | This repo | Laptop | AI agent brain — planning, reasoning, UI |
| `Cryosparc_mcp_Server/` | GPU server | GPU server | MCP server — CryoSPARC orchestration |

The agent communicates with the MCP server via **SSH stdio** (same protocol as Cursor / Claude Desktop MCP).

---

## Agentic Framework

The agent implements a **Planning → Memory → Action** loop at every pipeline step.

### Planning — ReAct Pattern

At each step, the LLM reasons through:

```
Observation: patch_motion completed. 20 micrographs produced.
Thought:     Motion correction successful. CTF estimation is the logical
             next step to measure defocus and assess micrograph quality.
Tool:        run_patch_ctf_estimation
Decision:    CONTINUE
Reasoning:   Motion correction completed successfully on all 20 micrographs.
             CTF estimation is required before particle picking to characterise
             the contrast transfer function for each exposure...
```

The LLM has access to **15 formally defined CryoSPARC tools**:

| # | Tool | Description |
|---|------|-------------|
| 1 | `run_import_movies` | Import raw .tif/.mrc movies |
| 2 | `run_patch_motion_correction` | Align frames, remove beam-induced motion |
| 3 | `run_patch_ctf_estimation` | Estimate CTF per micrograph |
| 4 | `curate_exposures` | [CHECKPOINT] Remove bad micrographs |
| 5 | `run_blob_picker` | Laplacian-of-Gaussian particle detection |
| 6 | `inspect_blob_picks` | [CHECKPOINT] Filter picks by NCC score |
| 7 | `extract_particles` | Extract particle stacks (box=256px for GPCR) |
| 8 | `run_2d_classification` | 50-class 2D averaging |
| 9 | `select_2d_classes` | [CHECKPOINT] Keep protein, discard junk |
| 10 | `run_abinit_reconstruction` | Ab-initio 3D model (no reference) |
| 11 | `run_homogeneous_refinement` | Single-conformation refinement |
| 12 | `run_template_picker` | Template-based picking from 2D classes |
| 13 | `inspect_template_picks` | [CHECKPOINT] Filter template picks |
| 14 | `select_2d_template_classes` | [CHECKPOINT] Final 2D selection |
| 15 | `run_nonuniform_refinement` | High-resolution refinement (target ≤3.5 Å) |
| + | `assess_quality` | Evaluate step metrics |
| + | `escalate_to_human` | Stop for critical issues |

**Decision framework:**

| Decision | Meaning | Trigger |
|----------|---------|---------|
| `CONTINUE` | Proceed with next tool | Default — quality within threshold |
| `ADJUST` | Proceed but flag marginal quality | Metrics borderline but not critical |
| `ESCALATE` | Stop, alert human | Resolution >8 Å + particles <1000, OR 3+ consecutive failures |

### Memory

Two memory systems run in parallel:

**Episodic Memory** (session-specific):
- Job history (job UIDs, step names, completion times)
- Quality timeline (CTF fit, particle counts, resolution per step)
- Decision log (every LLM decision with reasoning)
- Observation log

**Semantic Memory** (domain knowledge, always loaded):
- GPCR characteristics (60 kDa, 256px box, 80-150 Å diameter, C1 symmetry)
- Quality thresholds (CTF <5 Å, ≥50 particles/micrograph, ≤3.5 Å target resolution)
- Available CryoSPARC tools

### Action

Actions are MCP tool calls over SSH:
```
cs_start_pipeline    → new_run()
cs_continue_pipeline → step()
cs_resume_pipeline   → resume_checkpoint()
cs_pipeline_status   → load_state()
cs_pipeline_report   → write_report()
cs_list_runs         → list_runs()
```

---

## VLM Critic

The VLM critic (`cryoemagent/vlm_critic.py`) replaces or assists human judgment at the 5 checkpoint steps.

**Two tiers:**

| Tier | Works in | What it sees | How |
|------|----------|-------------|-----|
| Tier 1 — Metrics | Always (remote SSH) | CryoSPARC job metrics (CTF fit, particle counts, etc.) | LLM reasoning from numbers |
| Tier 2 — Vision | Local mode only | Actual micrograph / CTF plot / 2D class images | GPT-4V / Claude Vision |

**Auto-approval logic:**
- If VLM confidence ≥ 85% AND verdict = PASS → checkpoint is **auto-approved**, no human needed
- Otherwise → shows assessment + manual instructions

**Example VLM output for `curate` checkpoint:**
```
╭─── VLM Assessment — curate ──────────────────────────────╮
│ Verdict:  PASS  (Tier 1 — Metrics)  confidence=95%       │
│                                                           │
│ Observations:                                             │
│   • Mean CTF fit 4.2 Å — below the 5 Å threshold (GOOD)  │
│   • 18/20 micrographs accepted — 90% acceptance rate     │
│   • Defocus range 0.8–2.5 μm — appropriate for GPCR      │
│                                                           │
│ Reasoning: CTF quality is excellent across this dataset.  │
│ The 90% acceptance rate exceeds the 70% minimum...        │
╰───────────────────────────────────────────────────────────╯
✓ VLM AUTO-APPROVED (confidence=95% ≥ 85%). Continuing...
```

**Checkpoint-specific evaluation criteria:**

| Checkpoint | VLM evaluates |
|------------|--------------|
| `curate` | CTF fit <5 Å, ice thickness, acceptance rate ≥70% |
| `inspect_blob` | Particles/micrograph ≥50, NCC distribution, false positive rate |
| `select2d_blob` | Secondary structure visibility, GPCR size ~100-150 Å, view diversity |
| `inspect_template` | Template pick accuracy, false positive rate |
| `select2d_template` | Stringent selection — orientation diversity, TM helix density |

---

## Pipeline

Full W1 + W2 pipeline — 19 steps:

```
W1 — Blob Pipeline
═══════════════════════════════════════════════════════════════
  1  import_movies          Import raw electron microscopy movies
  2  patch_motion           Patch-based motion correction (align 30 frames)
  3  patch_ctf              Patch CTF estimation (defocus, astigmatism)
  4  curate             ✋  [CHECKPOINT] Curate exposures — VLM auto-evaluates
  5  blob_pick              Laplacian-of-Gaussian blob particle picking
  6  inspect_blob       ✋  [CHECKPOINT] Inspect picks — VLM suggests thresholds
  7  extract_blob           Extract particle stacks (256px box)
  8  class2d_blob           2D classification (50 classes)
  9  select2d_blob      ✋  [CHECKPOINT] Select 2D classes — VLM vision analysis
 10  abinit_blob            Ab-initio 3D reconstruction
 11  homo_blob              Homogeneous refinement

W2 — Template Pipeline
═══════════════════════════════════════════════════════════════
 12  template_pick          Template picker (uses W1 2D class averages)
 13  inspect_template   ✋  [CHECKPOINT] Inspect template picks — VLM analysis
 14  extract_template       Extract particle stacks
 15  class2d_template       2D classification
 16  select2d_template  ✋  [CHECKPOINT] Final 2D selection — stringent VLM
 17  abinit_template        Ab-initio reconstruction
 18  homo_template          Homogeneous refinement
 19  nonuniform_template    Non-uniform refinement (final — target ≤3.5 Å)

✋ = Human checkpoint (VLM evaluates first, auto-approves if confidence ≥85%)
```

**Quality thresholds (GPCR-specific):**

| Metric | GOOD | WARN | FAIL |
|--------|------|------|------|
| CTF fit resolution | <5 Å | 5–7 Å | >7 Å |
| Acceptance rate | >70% | 50–70% | <50% |
| Particles/micrograph | >50 | 20–50 | <20 |
| Final resolution | ≤3.5 Å | 3.5–5 Å | >5 Å |

---

## Checkpoint System

At each of the 5 checkpoints, the agent:

1. **Runs VLM assessment** — analyzes metrics (+ images in local mode)
2. **Auto-approves** if VLM confidence ≥ 85% and verdict is PASS
3. Otherwise **shows manual instructions** and waits for user to complete in CryoSPARC UI
4. User types `done` → pipeline resumes automatically to next checkpoint or completion

The checkpoint state is **persisted to disk** (JSON) — if the session is interrupted, run `python run.py --autopilot` and it resumes exactly where it left off.

---

## MCP over SSH

The agent connects to the remote GPU server using the same protocol as Cursor / Claude Desktop:

```yaml
# profile.yaml — ssh section
ssh:
  command: "ssh"
  args:
    - "-J"
    - "username@xulab-login0.lan.cmu.edu:20022"
    - "-p"
    - "20022"
    - "username@10.0.1.2"
    - "/bin/sh"
    - "-c"
    - "cd /path/to/Cryosparc_mcp_Server && exec python3 -u run_server.py --config config/profile.yaml"
```

**JSONL framing** (newline-delimited JSON) — not Content-Length:
```
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}\n
{"jsonrpc":"2.0","id":1,"result":{...}}\n
```

Non-JSON lines (SSH banners, warnings) are silently skipped.

---

## Installation

### Prerequisites

- Python 3.10+ on your **laptop**
- CryoSPARC v4.7.1 on the **GPU server**
- `Cryosparc_mcp_Server` deployed on the GPU server
- SSH access to the GPU server (direct or via jump host)

### Laptop setup

```bash
git clone https://github.com/yaswanth169/CryoEMAgent.git
cd CryoEMAgent
pip install -e .
```

### Environment variables

Create `.env` in the `CryoEMAgent/` directory:

```env
# LLM (choose one)
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# CryoSPARC (only needed for local mode)
CRYOSPARC_EMAIL=you@university.edu
CRYOSPARC_PASSWORD=yourpassword
```

---

## Configuration

All settings live in `profile.yaml`:

```yaml
cryosparc:
  base_url: "http://localhost:39000"
  email: "user@university.edu"
  password: "password"
  project_uid: "P3"
  workspace_w1_title: "W1_blob_tools"
  workspace_w2_title: "W2_template_tools"
  lane: "default"

data:
  movie_blob_path: "/mnt/data/gpcr/*.tif"
  psize_A: 1.05
  accel_kv: 300
  cs_mm: 2.7
  amp_contrast: 0.1
  total_dose_e_A2: 50.0

llm:
  provider: "openai"        # openai | anthropic
  model: "gpt-4o"

ssh:
  command: "ssh"
  args: ["-J", "user@jumphost:22", "-p", "22", "user@gpu-server", "/bin/sh", "-c", "..."]
  timeout: 600
```

---

## Usage

### Launch (mode selector)
```bash
python run.py
```

### Direct mode flags
```bash
python run.py --autopilot      # Autopilot mode
python run.py --interactive    # Interactive/conversational mode
python run.py --local          # Local mode (run on GPU server directly)
python run.py --config alt.yaml
```

### Interactive mode commands

Natural language — the agent understands context:

```
you: start a new run
you: continue
you: done            ← after completing a checkpoint in CryoSPARC UI
you: status
you: quality
you: what is CTF estimation?
you: report
you: memory          ← show agent episodic memory
you: reasoning       ← show full LLM reasoning log
you: quit
```

### Expected interactive session

```
you: start a new run

agent: Starting the GPCR processing pipeline on the GPU server...

  GPCR Processing Pipeline — W1 + W2 (19 steps)
  ┌──┬──────┬───────────────────────────────┬──────────────┐
  │  │ Wave │ Step                          │ Human        │
  ├──┼──────┼───────────────────────────────┼──────────────┤
  │1 │  W1  │ Import Movies                 │              │
  │2 │  W1  │ Patch Motion Correction       │              │
  │3 │  W1  │ Patch CTF Estimation          │              │
  │4 │  W1  │ Curate Exposures              │   ✋          │
  │  │  ... │ ...                           │              │
  └──┴──────┴───────────────────────────────┴──────────────┘

  ╭── LLM Reasoning — import_movies ────────────────────────╮
  │ Observation: Pipeline started. No prior jobs completed.  │
  │ Thought:     Import movies is the mandatory first step.  │
  │ Tool:        run_import_movies                           │
  │ Decision:    CONTINUE                                    │
  ╰──────────────────────────────────────────────────────────╯

  ▶ Step 1/19: import_movies  (GPU job — may take several minutes)
  ✓ import_movies (J180)  →  patch_motion  [1/19]

  ╭── ✓ import_movies J180  →  patch_motion ────────────────╮
  │ Import Movies loaded 20 movies into CryoSPARC workspace  │
  │ (J180). Each movie contains 30 frames at 3710×3838...    │
  ╰──────────────────────────────────────────────────────────╯

  ▶ Step 2/19: patch_motion ...
  ✓ patch_motion (J181)  →  patch_ctf  [2/19]
  ▶ Step 3/19: patch_ctf ...
  ✓ patch_ctf (J182)  →  curate  [3/19]

  Progress:  ███░░░░░░░░░░░░░░░░░  3/19 steps (16%)

  [VLM analyzing curate checkpoint...]

  ╭── VLM Assessment — curate ──────────────────────────────╮
  │ Verdict: PASS  confidence=95%                            │
  │ Observations:                                            │
  │   • Mean CTF fit 4.2 Å — within GOOD threshold (<5 Å)   │
  │   • 18/20 micrographs accepted (90%)                    │
  ╰──────────────────────────────────────────────────────────╯
  ✓ VLM AUTO-APPROVED. Continuing pipeline...

  ▶ Step 5/19: blob_pick ...
```

---

## Project Structure

```
CryoEMAgent/
├── run.py                          # Launcher — mode selector + autopilot/interactive
├── profile.yaml                    # Configuration (data paths, LLM, SSH, CryoSPARC)
├── .env                            # API keys (not committed)
├── setup.py
│
├── cryoemagent/
│   ├── __init__.py
│   ├── config.py                   # LLMConfig, AgentConfig, CryoSPARCConfig
│   ├── interactive.py              # Interactive/conversational mode (NEW v0.2)
│   ├── vlm_critic.py               # VLM checkpoint evaluator (NEW v0.2)
│   ├── mcp_client.py               # MCP-over-SSH client (JSONL framing)
│   ├── orchestrator_client.py      # Local mode orchestrator client
│   │
│   └── core/
│       ├── agent.py                # CryoEMAgent — main control loop + reasoning log
│       ├── planner.py              # LLM planner — ReAct, 15 tool definitions
│       ├── memory.py               # EpisodicMemory + SemanticMemory
│       └── quality_critics.py      # CTF / Picking / 2D / Refinement critics
│
└── runs/
    ├── <run_id>/                   # CryoSPARC job records per run
    └── reasoning_logs/
        └── reasoning_<run_id>.json # Full LLM reasoning chain log (Mei Yuan's ask)
```

---

## Reasoning Log

Every LLM decision is written to `runs/reasoning_logs/reasoning_<run_id>.json`:

```json
{
  "run_id": "a5662be4-...",
  "entries": [
    {
      "timestamp": "2026-04-09T01:40:00",
      "step": "patch_ctf",
      "stage": "w1",
      "llm_decision": {
        "observation": "patch_motion completed. 20 micrographs produced.",
        "thought": "CTF estimation is required to characterise defocus per micrograph.",
        "tool_selected": "run_patch_ctf_estimation",
        "decision": "CONTINUE",
        "reasoning": "Motion correction completed on all 20 micrographs. CTF estimation is the mandatory next step before particle picking...",
        "recommendation": "Proceed to CTF estimation.",
        "parameter_adjustments": {}
      }
    }
  ]
}
```

View in interactive mode: type `reasoning` at any prompt.

---

## Research Foundation

### Novel Contributions

1. **ReAct-pattern LLM agent for cryo-EM** — formal tool definitions, Observe→Think→Act reasoning loop, full decision transparency
2. **VLM-based checkpoint evaluation** — automated visual quality assessment at the 5 human-required steps using GPCR-specific prompts; auto-approves with ≥85% confidence
3. **MCP-over-SSH architecture** — laptop-to-GPU-server agent communication using the same protocol as Cursor/Claude Desktop, enabling remote autonomous operation
4. **Dual-mode UX** — same MCP engine, two interfaces: autopilot (fire-and-forget) and interactive (conversational like Claude Code)

### Limitations and Future Work

- VLM Tier 2 (image-based) requires local CryoSPARC access; remote image download via MCP is planned
- Mid-run parameter adjustment requires restart; dynamic parameter updates are in development
- Truly zero-intervention mode requires automated 2D class selection AI (TOPAZ-style) — identified as next research step

### Dataset

EMPIAR-10288: CB1 cannabinoid receptor (GPCR class A), 300 kV, 1.05 Å/px, 20 movies (demo subset).

---

## References

- [CryoSPARC v4.7.1](https://cryosparc.com)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [cryosparc-tools SDK](https://tools.cryosparc.com)
- [EMPIAR-10288](https://www.ebi.ac.uk/empiar/EMPIAR-10288/)
- ReAct: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023
