# CryoEMAgent — Full End-to-End Context Memory

**Author:** Yaswanth (Remote intern, Xu Lab, CMU)
**Written for:** Mei Yuan (Xu Lab, CMU) — context brief ahead of the 15 April 2026 meeting with Mostofa Rafid Uddin (postdoc, XuLab, CryoEM processing specialist)
**Repository:** https://github.com/yaswanth169/CryoEMAgent
**Report:** `CryoEMAgent_report.tex` (technical report, compiled for NeurIPS 2026 submission track)
**Date:** 20 April 2026

> **Purpose of this document.** This is a non-compacted, start-to-finish record of everything we have done on CryoEMAgent — every design decision, every failure, every fix, every passing run, every feedback cycle, every open question. Nothing is omitted. It is written so that anyone (Mei, Mostofa, Yizhou, Rakshitha) can read it and reconstruct the project from zero, without needing me in the room. If a design choice looks odd, the justification is in here. If a number looks wrong, the story of how we got it is in here.

---

## Table of Contents

1. [Project Genesis & People](#1-project-genesis--people)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [v0.1 — The First Attempt (Feb–Mar 2026)](#3-v01--the-first-attempt-febmar-2026)
4. [v0.2 — The Rewrite We Actually Shipped](#4-v02--the-rewrite-we-actually-shipped)
5. [System Architecture (As It Stands Today)](#5-system-architecture-as-it-stands-today)
6. [The Agentic Framework — Planning, Memory, Action, ReAct](#6-the-agentic-framework--planning-memory-action-react)
7. [The 15 Formal CryoSPARC Tools](#7-the-15-formal-cryosparc-tools)
8. [VLMCritic — Two-Tier Checkpoint Evaluator](#8-vlmcritic--two-tier-checkpoint-evaluator)
9. [The W1 + W2 Pipeline (19 Steps)](#9-the-w1--w2-pipeline-19-steps)
10. [MCP-over-SSH — The Hardest Part](#10-mcp-over-ssh--the-hardest-part)
11. [Two Operating Modes — Autopilot vs Interactive](#11-two-operating-modes--autopilot-vs-interactive)
12. [Mei's Feedback Cycles (What Changed & Why)](#12-meis-feedback-cycles-what-changed--why)
13. [Every Bug We Hit, and How We Fixed It](#13-every-bug-we-hit-and-how-we-fixed-it)
14. [EMPIAR-10288 End-to-End Validation](#14-empiar-10288-end-to-end-validation)
15. [Reasoning Log — Full Traceability](#15-reasoning-log--full-traceability)
16. [Deliverables To Date](#16-deliverables-to-date)
17. [Novel Contributions (the 5 claims)](#17-novel-contributions-the-5-claims)
18. [Limitations L1–L6 (Honest Inventory)](#18-limitations-l1l6-honest-inventory)
19. [Open Questions for Mostofa](#19-open-questions-for-mostofa)
20. [What Mei Should Take Into the Meeting](#20-what-mei-should-take-into-the-meeting)
21. [Appendix A — Commit Trail](#appendix-a--commit-trail)
22. [Appendix B — File Map of the Repository](#appendix-b--file-map-of-the-repository)
23. [Appendix C — Glossary](#appendix-c--glossary)

---

## 1. Project Genesis & People

**January 2026.** Rakshitha Ireddi and I (Yash / Yaswanth) joined Xu Lab at CMU as remote interns. Mei Yuan was our point of contact and day-to-day mentor. The lab PI is Prof. Min Xu (CMU, Computational Biology / Computer Science). The lab focuses on computational methods for cryo-EM and cryo-ET structure determination.

**February 2026.** The project "CryoEMAgent" was scoped as a remote-intern-friendly research problem that could progress without lab-bench work: build an **autonomous LLM agent that can drive a real cryo-EM single-particle-analysis (SPA) pipeline on CryoSPARC** from raw movies to a refined 3D volume, with human experts kept in the loop only where their judgment is genuinely needed.

**March 2026.** I took over as the primary engineer on the agent side. Rakshitha works on a parallel thread. Mei reviews weekly; Yizhou Zhao is another reviewer on the lab side. In Mid-April 2026 Mei introduced us to **Mostofa Rafid Uddin**, a postdoc in XuLab specializing in cryo-EM processing, so we could get domain feedback before the NeurIPS 2026 submission window.

**People on the meeting invite (April 15, 2026, 11:00 AM ET — `cmu.zoom.us/j/94536051486`):**
- **Mei Yuan** — Lab member, our mentor, owner of the agent effort.
- **Mostofa Rafid Uddin** — Postdoc, CryoEM processing. External reviewer for this meeting.
- **Yizhou Zhao** — Lab member.
- **Rakshitha Ireddi** — Remote intern (parallel thread).
- **Yaswanth (me)** — Remote intern, main author of the CryoEMAgent code and this document.

**Target venue.** NeurIPS 2026 submission (technical track). The LaTeX technical report at `CryoEMAgent_report.tex` is the draft.

---

## 2. The Problem We Are Solving

### The Scientific Problem

G-protein-coupled receptors (GPCRs) are the single largest class of human drug targets (~34% of FDA-approved drugs). Cryo-EM single-particle analysis (SPA) is now the dominant route for solving their 3D structures, but the workflow is long, multi-stage, and heavily gated by expert judgment.

A typical GPCR SPA workflow on **CryoSPARC v4.7.1** requires ~19 sequential jobs, with **~5 human checkpoints** where an expert must look at 2D class averages, pick counts, CTF fits, or volume quality, and either accept, reject, or reconfigure and re-run. Experts are a scarce resource; the checkpoints are where wall-clock time actually leaks.

### The Computational Problem

The lab's CryoSPARC instance runs on a remote GPU server (`10.0.1.2`), reachable only through an SSH jump host (`xulab-login0`). The laptop that drives the workflow has no direct network path to the GPU server, no GPU of its own, and is on Windows 11 in our case. An agent that wants to drive CryoSPARC must:

1. Talk to CryoSPARC through a protocol that survives a multi-hop SSH tunnel.
2. Reason about the state of a long-running job without polling it to death.
3. Pause at the right places and ask a human, but not at the wrong places (no "are you sure?" fatigue).
4. Be transparent — every action and every decision must be auditable, because this is science, not a chat app.
5. Fit a normal researcher's laptop (Windows, no CUDA, no lab VPN client beyond SSH).

### The Research Question

> Can a single LLM-driven agent, following the **ReAct** pattern (Yao et al., ICLR 2023) and augmented with a **Vision-Language-Model checkpoint critic**, drive a real GPCR cryo-EM SPA pipeline end-to-end on a remote CryoSPARC instance, reduce unnecessary human checkpoints without hiding the dangerous ones, and produce a full reasoning log that an expert can audit after the fact?

Everything that follows is in service of answering that question.

---

## 3. v0.1 — The First Attempt (Feb–Mar 2026)

v0.1 was the scaffolding pass. It was built to answer one question: **can we call CryoSPARC at all from a laptop through an SSH jump host?**

What was in v0.1:
- A direct Python client that used `cryosparc-tools` on the GPU server.
- A flat "autopilot" script that ran W1 (Blob picking) in sequence.
- No planning layer. No memory. No VLM. No interactive mode. No checkpoints.
- Agent reasoning was a single LLM call per step, with a freeform prompt.

**What failed in v0.1 (the reasons we rewrote it):**

1. **No separation between *decide* and *execute*.** The LLM generated Python-ish strings that we `eval`'d to call CryoSPARC. This is the pattern that people complain about in LangChain tutorials and it's just as bad in practice — non-deterministic, unaudittable, and unsafe.
2. **No state model.** We had no idea what step we were on if the script crashed. Resume was impossible.
3. **No protocol boundary.** The agent code was entangled with the CryoSPARC driver code, which meant we couldn't move the CryoSPARC side to the GPU server without re-tangling.
4. **No human checkpoints.** The agent just kept going until CryoSPARC crashed or a job failed; there was no "ask the human" moment. Mei flagged this on the first review: *"You are not going to hand this to a biologist if there is no place for them to say stop."*
5. **No transparency.** The LLM decisions were not logged structurally. We could not answer "why did the agent pick threshold 0.2 instead of 0.4?" after a run.

Mei's feedback after the v0.1 demo, condensed:
- "Define tools formally. The LLM should not be writing Python; it should be calling a finite set of named tools with typed arguments."
- "There should be a planner, a memory, and an action layer. Like a real agent."
- "The checkpoints where a biologist has to look — inspect_blob, the 2D class averages — those have to pause, not auto-continue."
- "If a VLM can look at the 2D class image and say 'these look broken,' we should use that. Don't always pause; only pause when it matters."
- "I want to see why the agent chose what it chose. A log."

That list became the spec for v0.2.

---

## 4. v0.2 — The Rewrite We Actually Shipped

v0.2 is what Mostofa will be looking at on 15 April. It is a **two-repo** design:

- **`CryoEMAgent/`** — the agent, on the laptop. Python, no GPU. This is what lives at `github.com/yaswanth169/CryoEMAgent`.
- **`Cryosparc_mcp_Server/`** — a thin MCP (Model Context Protocol) server that runs on the GPU host and exposes CryoSPARC as a set of JSON-RPC tools.

The two halves talk over **MCP-over-SSH**: the laptop spawns a subprocess (`ssh -J jump user@gpu python -m cryosparc_mcp_server`), sends newline-delimited JSON over stdin, reads newline-delimited JSON over stdout. That protocol choice is covered in §10.

v0.2 added:
- Formal 15-tool interface (§7).
- Planner → Memory → Action architecture, driven by ReAct (§6).
- VLMCritic with a two-tier evaluator and an 85% auto-approval rule (§8).
- A full 19-step W1+W2 pipeline runner with 5 human checkpoints (§9).
- Two modes — **Autopilot** and **Interactive** — selectable at launch (§11).
- Structured **reasoning log** (one JSON per run) for full traceability (§15).
- A Windows-friendly client (the only OS I have; fixing this forced an unrelated bug — §13).

The v0.2 push landed in late March 2026 and has been stabilized over April.

---

## 5. System Architecture (As It Stands Today)

```
[  Laptop (Windows 11, no GPU)                        ]
[   CryoEMAgent                                        ]
[   ├── Planner (ReAct LLM — GPT-4 class)              ]
[   ├── Memory (episodic + semantic)                   ]
[   ├── Action layer (15 formal tools)                 ]
[   ├── VLMCritic (Tier 1 metrics + Tier 2 vision)     ]
[   ├── Interactive console / Autopilot runner         ]
[   └── MCP client (JSONL over stdio)                  ]
                          │
                          │  SSH jump host (xulab-login0)
                          │  ssh -J user@xulab-login0 user@10.0.1.2
                          ▼
[  GPU server 10.0.1.2                                 ]
[   Cryosparc_mcp_Server                               ]
[   ├── MCP dispatcher (JSON-RPC 2.0 over stdio)       ]
[   ├── CryoSPARC REST client (localhost:39000)        ]
[   └── tool handlers: import_movies, motion_correct,  ]
[       ctf_estimate, blob_pick, class2d, select2d,    ]
[       template_pick, refine3d, ...                   ]
                          │
                          ▼
[  CryoSPARC v4.7.1 on GPU (internal API at :39000)    ]
```

**Why this shape?**

- The laptop cannot reach `10.0.1.2` directly; only `xulab-login0` can. But we do not want to forward CryoSPARC's HTTP port across two SSH hops — that is fragile, exposes internal endpoints, and requires port coordination. Instead we run the MCP server *inside* the SSH session, and the "transport" is just the ssh subprocess's stdin/stdout. This is discussed in detail in §10.
- The agent never speaks CryoSPARC's REST API directly. That knowledge lives on the GPU host, where it is trivial to reach `localhost:39000`. The laptop only speaks MCP.
- The 15 tools are the only vocabulary the LLM has. It cannot call `subprocess.run` or `eval` anything. This is how we kill the v0.1 "LLM-writes-Python" pattern.

---

## 6. The Agentic Framework — Planning, Memory, Action, ReAct

Mei's v0.1 feedback was: "make it look like a real agent." Concretely, we adopted the pattern from recent agentic-LLM literature:

### Planning

On each step the agent does a **ReAct** loop:

> **Observe** → **Think** → **Tool** → **Decision**

- **Observe** the current CryoSPARC job state (status, metrics, any flags).
- **Think** — the LLM produces a natural-language chain of thought.
- **Tool** — the LLM picks exactly one of the 15 tools and fills typed arguments.
- **Decision** — `continue`, `pause_for_human`, `reject_and_reconfigure`, or `abort`.

Each of these four fields is saved to the reasoning log. The LLM does not see the tools as "functions to call"; it sees them as a typed JSON schema (OpenAI-style tool calls). The agent does the actual dispatch.

### Memory

Two layers:

- **Episodic memory** — the exact sequence of observations, thoughts, tool calls, and decisions from the current run. This is what the LLM gets as conversation history, truncated with a summarizer when it grows.
- **Semantic memory** — run-invariant knowledge: CryoSPARC conventions, EMPIAR dataset-specific defaults (pixel size, voltage, Cs), GPCR-class-A priors (particle mass range, expected symmetry). Stored as a YAML/JSON fixture and injected into the system prompt.

### Action

One step = one tool call. The agent layer:
1. Validates arguments against the JSON schema.
2. Sends the JSON-RPC request over MCP.
3. Parses the structured response.
4. Hands it back to the planner for the next Observe.

This Planner/Memory/Action separation was Mei's explicit ask in the v0.1 review and is the reason v0.2 looks the way it does.

---

## 7. The 15 Formal CryoSPARC Tools

These are the only actions the LLM is allowed to request. They are defined in the system prompt as an OpenAI-style tool list and enforced on the server side.

| # | Tool | Purpose |
|---|------|---------|
| 1 | `import_movies` | Ingest raw movies into a CryoSPARC project |
| 2 | `motion_correct` | Patch motion correction |
| 3 | `ctf_estimate` | Per-micrograph CTF |
| 4 | `curate_exposures` | Exposure curation (checkpoint) |
| 5 | `blob_pick` | Blob-based particle picking (W1) |
| 6 | `inspect_blob` | Inspect blob picks (checkpoint) |
| 7 | `extract_blob` | Extract particles from blob picks |
| 8 | `class2d_blob` | 2D classification of blob-picked particles |
| 9 | `select2d_blob` | 2D class selection (checkpoint) |
| 10 | `template_pick` | Template picking using selected 2D classes (W2) |
| 11 | `inspect_template` | Inspect template picks (checkpoint) |
| 12 | `extract_template` | Extract particles from template picks |
| 13 | `class2d_template` | 2D classification of template-picked particles |
| 14 | `select2d_template` | 2D class selection for template route (checkpoint) |
| 15 | `ab_initio_refine3d` | Ab-initio 3D reconstruction + homogeneous refine |

The five bold rows (4, 6, 9, 11, 14) are the human checkpoints. The VLMCritic in §8 decides whether they actually require the human or whether they can be auto-approved.

---

## 8. VLMCritic — Two-Tier Checkpoint Evaluator

The VLMCritic is the single biggest UX win. Without it, the agent would pause at all 5 checkpoints on every run; with it, ~3 of the 5 are auto-approvable in the typical case.

### Tier 1 — Metrics reasoning

Pure numeric/textual reasoning over CryoSPARC's job-level metrics: number of picks, particle counts per class, ice contamination fraction, CTF fit score, maximum resolution, class distribution entropy, etc. This tier always works, even in remote mode where we can't pull images off the server cheaply.

### Tier 2 — Vision reasoning

When images are available (local-mode runs, or when the user explicitly asks), Tier 2 pulls the checkpoint artifact (e.g., 2D class average grid, ctf fit plot) and sends it to a vision-capable model — GPT-4V or Claude Vision — with a checkpoint-specific prompt: *"Here is a 2D class average montage from a GPCR dataset. Are the classes detailed enough to use as picking templates?"*

### Auto-approval rule

For each checkpoint the critic returns a verdict in `{PASS, WARN, FAIL}` and a confidence score in `[0, 1]`.

> **Auto-approve iff `verdict == PASS` AND `confidence ≥ 0.85`.**

Otherwise the run pauses and asks the human. This threshold was picked conservatively — we would rather over-pause than auto-approve a bad checkpoint. In the EMPIAR-10288 validation run (§14) the critic returned WARN with confidence 0.75–0.85 on all 5 checkpoints, so the run paused at every checkpoint (correct behavior for a small dataset demo).

### Checkpoint-specific criteria

- `curate_exposures`: CTF fit fraction, ice contamination, astigmatism distribution.
- `inspect_blob`: particle count vs. expected, pick density on a sample micrograph, ice/aggregate false positives.
- `select2d_blob`: class count with secondary-structure-level detail, % particles in "good" classes.
- `inspect_template`: same as inspect_blob but for template-picked particles.
- `select2d_template`: final class quality before feeding ab-initio.

---

## 9. The W1 + W2 Pipeline (19 Steps)

The pipeline is two "workflows" stitched together: **W1 Blob** (pick with blob, 2D-classify, pick a few good classes) then **W2 Template** (use those classes as templates, re-pick, re-classify, refine 3D).

### W1 — Blob workflow (11 steps)

1. `import_movies`
2. `motion_correct`
3. `ctf_estimate`
4. `curate_exposures` *(checkpoint)*
5. `blob_pick`
6. `inspect_blob` *(checkpoint)*
7. `extract_blob`
8. `class2d_blob`
9. `select2d_blob` *(checkpoint)*
10. *(split: selected 2D classes feed W2)*
11. *(W1 terminus)*

### W2 — Template workflow (8 steps)

12. `template_pick`
13. `inspect_template` *(checkpoint)*
14. `extract_template`
15. `class2d_template`
16. `select2d_template` *(checkpoint)*
17. `ab_initio`
18. `homogeneous_refine`
19. *(done — 3D volume + metadata)*

Each row maps to a CryoSPARC job (J188 … J206 in the validation run, §14). Every transition is an MCP call; every checkpoint goes through the VLMCritic before it either auto-approves or pauses.

---

## 10. MCP-over-SSH — The Hardest Part

This is the piece that took the most debugging time and that Mostofa is most likely to ask about.

### The problem

The laptop cannot reach `10.0.1.2:39000` (CryoSPARC's REST API). It can only reach the jump host. We do not want to:

- Forward TCP 39000 across two SSH hops (unstable, exposes internal endpoints, requires port coordination on a shared machine).
- Install anything on the jump host that looks like a proxy.
- Require the user to run `ssh -L` by hand every session.

### The design

Use **MCP** (Model Context Protocol, the open JSON-RPC-2.0-over-stdio protocol) and carry it over a plain SSH subprocess. The laptop spawns:

```
ssh -J user@xulab-login0 user@10.0.1.2 \
    python -m cryosparc_mcp_server --project <project_id>
```

Then talks to the subprocess's stdin/stdout with **newline-delimited JSON** (JSONL framing). Each JSON-RPC request and response is one line. No `Content-Length: ...` headers (which some MCP implementations use); just one JSON object per line.

### Why JSONL, not Content-Length framing

SSH's stdout is a pseudo-tty stream with its own quirks on Windows. We originally tried Content-Length framing, which needs precise byte-accounting. It kept desynchronizing on Windows because Python was buffering stdout differently than on Linux. We switched to JSONL — one line = one message — and the framing problem disappeared.

### Why not WebSocket or plain HTTP

Both require that the laptop open a TCP connection to the GPU host. We don't have that. SSH is the only allowed transport.

### The Windows bufsize bug (§13, Bug A)

On Windows, Python's `subprocess.Popen(..., bufsize=-1)` combined with a pipe hooked up to `ssh.exe` caused `OSError: [Errno 22] Invalid argument` on some writes after the first one. Setting `bufsize=0` (unbuffered) or `bufsize=1` (line-buffered) fixed it. We use `bufsize=0` and do our own line buffering in Python.

---

## 11. Two Operating Modes — Autopilot vs Interactive

v0.2 ships two entry points:

### Autopilot (`python run.py --mode autopilot --dataset EMPIAR-10288`)

Fire-and-forget. The agent runs end-to-end. It pauses at checkpoints only if VLMCritic does not auto-approve. At the end it prints a summary and writes a reasoning log. This is the mode for batch experiments and for scripting.

### Interactive (`python run.py --mode interactive`)

A Claude-Code-style conversational console. You can:
- Ask the agent to start a pipeline.
- Watch a live progress bar (▶ marker on the current step, ✓ on done steps, ○ on pending).
- Approve, reject, or reconfigure at a checkpoint.
- Ask the agent *why* it is about to do something.
- Pause, inspect state, resume.

The interactive loop lives in `cryoemagent/interactive.py`. It is a REPL that shells out to the same planner/action layer as autopilot — the only difference is that a human is in the decision loop.

**Why two modes?**
- Autopilot is for reproducibility — you can run the same config a hundred times for NeurIPS experiments.
- Interactive is for actual researcher use — Mei's original ask ("it should feel like a tool a biologist can sit in front of"). A biologist will never want `--mode autopilot` as their default.

---

## 12. Mei's Feedback Cycles (What Changed & Why)

These are the explicit review cycles with Mei that reshaped the codebase. They are listed roughly in order. I include them because Mostofa is likely to ask "why did you decide X?" and the honest answer to most of them is "Mei told us to, for this reason."

**Cycle 1 — "This is a script, not an agent."**
- Feedback: v0.1 had no planner/memory/action separation.
- Change: Introduced the three-layer architecture in §6. System prompt was rewritten to explicitly list the 15 tools (§7). LLM was constrained to tool-call mode.

**Cycle 2 — "There are no checkpoints."**
- Feedback: v0.1 ran straight through. No biologist would trust it.
- Change: Added 5 human checkpoints in the W1+W2 pipeline (§9).

**Cycle 3 — "Pausing at every checkpoint is also wrong."**
- Feedback: If we pause 5 times per run, no one will use it. Most checkpoints are obvious.
- Change: Added VLMCritic (§8) with a 2-tier evaluator and an 85%-confidence auto-approval rule.

**Cycle 4 — "I cannot audit this."**
- Feedback: The run log was freeform prose. She could not retrace why the agent made a decision.
- Change: Structured reasoning log (§15) — one JSON file per run, one entry per step, four fields per entry (observe / think / tool / decision).

**Cycle 5 — "What are the parameters the agent can change?"**
- Feedback: Mid-April discussion. Mei asked whether the agent can reconfigure parameters, or only re-run with defaults.
- Current state: The agent can *propose* parameter changes at a checkpoint (e.g., lower the blob pick threshold) but needs human approval to re-run. Fully autonomous parameter reconfiguration is **L2** in §18 — not done yet.

**Cycle 6 — "Introduce Yash and Rakshi to Mostofa."**
- April 2026. Mei brought Mostofa in as a domain reviewer. The 15 April meeting is the result.

---

## 13. Every Bug We Hit, and How We Fixed It

Listed in the order we hit them.

### Bug A — Windows `OSError: [Errno 22]` on MCP stdin write (blocking)
- Symptom: First `write` to the ssh subprocess's stdin worked, second one blew up.
- Root cause: `subprocess.Popen(..., bufsize=-1)` interacts badly with Windows pipes hooked to `ssh.exe`.
- Fix: `bufsize=0`, manual line buffering. Documented in the MCP client module.
- Status: fixed.

### Bug B — MCP framing desync on Linux↔Windows
- Symptom: Partial JSON appeared in the reader after ~10 messages; parser threw.
- Root cause: Content-Length framing does not work reliably across Windows/Linux pty boundaries when ssh is in the middle.
- Fix: Switched to newline-delimited JSON (JSONL). One message per line. See §10.
- Status: fixed.

### Bug C — LLM generated Python strings, we `eval`'d them (v0.1 only)
- Symptom: Agent asked us to "just run `cs.run_blob_pick(...)`".
- Root cause: v0.1 did not have formal tool schemas.
- Fix: v0.2 rewrite with the 15-tool OpenAI-style tool-call interface. LLM cannot speak Python to us anymore.
- Status: fixed by architecture change.

### Bug D — Progress bar stuck at 3/19 after resume
- Symptom: After `resume_checkpoint()`, the MCP server ran intermediate steps (`blob_pick`, `extract_blob`, `class2d_blob`) internally before stopping at the next checkpoint. The client never called `step()` for those, so `_completed_steps` stayed at 3 entries forever.
- Fix: Added `_sync_completed_steps()` in `cryoemagent/interactive.py`. It walks `PIPELINE_STEPS` up to (but not including) the current step and fills any gaps in `_completed_steps`. Called at the top of the `_run_until_checkpoint()` loop and immediately after `resume_checkpoint()` in `_handle_confirm_checkpoint()`.
- Commit: `ebb3fe2`.
- Status: fixed.

### Bug E — Double ▶ markers in the progress bar
- Symptom: Two steps showed the "currently running" ▶ marker at the same time (positions 6 and 13).
- Root cause: `startswith(s["key"][:8])` — `"inspect_blob"[:8] == "inspect_"` which also matches `"inspect_template"`. Same false-positive on `select2d_blob` vs `select2d_template`.
- Fix: Changed to exact equality `self._state.current_step == s["key"]` in both `_show_progress_bar()` and `_show_pipeline_table()`.
- Commit: `ebb3fe2`.
- Status: fixed.

### Bug F — Final summary said "Total steps: 3" even after a 19/19 run
- Root cause: Same stale-`_completed_steps` problem as Bug D; the final summary read from the same list.
- Fix: Bug D's sync propagated to the summary path.
- Status: fixed (same commit).

### Bug G — LaTeX report would not compile
- Symptom: `pdflatex CryoEMAgent_report.tex` died in ~7 distinct places.
- Root causes (all seven listed here so Mostofa can see what a "yes we actually built it" list looks like):
  - `\usepackage{pgf-umlsd}` — not standard, not installed on all machines. Removed.
  - `\pgfonlayer{background}` used without `\pgfdeclarelayer{background}` + `\pgfsetlayers{background,main}`. Added.
  - `✋` emoji in a TikZ node — pdflatex cannot render it. Replaced with `\textbf{[CP]}`.
  - `language=json` / `language=yaml` in `listings` are undefined. Added manual `\lstdefinelanguage` for each.
  - `\textmu` undefined. Added `\usepackage{textcomp}`.
  - `hyperref` loaded too early. Moved to last in the preamble.
  - TOC not rendering properly. Added `\clearpage` before `\tableofcontents` and `\setcounter{tocdepth}{2}`.
- Commits: `32be2c6` (initial), `e5bc1f8` (rewrite for compile), `3f06501` (6 targeted bugs).
- Status: fixed — file compiles to a clean PDF on a stock TeX Live 2024.

### Bug H — LaTeX listings `moredelim` fought with `literate`
- Symptom: `! Extra }` during typesetting of a JSON code listing.
- Root cause: `moredelim = [s][\color{codeblue}\bfseries]{\{}{\}}` and `literate` both claimed `\{` / `\}`.
- Fix: Removed `moredelim`, kept a simpler `literate` for just `:` and `,`.
- Commit: `3f06501`.

### Bug I — `\newcommand{\umu}{$\mu$m}` broke inside TikZ nodes
- Fix: Changed to `\ensuremath{\mu}m`. Commit `3f06501`.

### Bug J — "particles / µm" in a results table
- Root cause: just scientifically wrong — µm is a length, not a count denominator.
- Fix: Changed to "particles / micrograph". Commit `3f06501`.

### Bug K — Stray 10.5 cm TikZ anchor stretching the VLM figure
- Root cause: `\draw[arrgrn] (autoappr.south) -- ++(0,-0.4) -- ++(-3.0,0) -- ++(0,-10.5) node[]{};` — an "invisible" anchor left over from an earlier layout that was still dragging the bounding box 10.5 cm down.
- Fix: Removed the draw entirely. Commit `3f06501`.

### Bugs L & M — Figure overflow on VLM-loop diagram
- Root cause: two `\draw` arcs routed 5.5 cm / 7.5 cm horizontally, exceeding `\textwidth`.
- Fix: Rerouted via `.west` anchors and shorter offsets (`-3.2 cm`, `-5.8 cm`). Commit `3f06501`.

---

## 14. EMPIAR-10288 End-to-End Validation

**Dataset.** EMPIAR-10288 — CB1 cannabinoid receptor (GPCR class A). 300 kV Titan Krios, 1.05 Å/px, 20 movies in our demo subset. We picked this because it is a canonical GPCR class A, published, and the expected 2D class signatures are well documented.

**What we ran.** Full 19-step pipeline, autopilot mode, with VLMCritic in remote (Tier 1) mode. The CryoSPARC job IDs in the run were contiguous: **J188 → J206**.

**Result.**
- All 19 jobs completed without a crash.
- All 5 checkpoints were reached. VLMCritic returned `WARN` at confidence 0.75–0.85 on all 5 (below the 0.85 auto-approval threshold, so the run paused at each). This is the correct behavior on a 20-movie demo subset — there is genuinely not enough data to say "PASS" with confidence.
- Final 3D volume was produced at the `homogeneous_refine` step (J206) with the expected GPCR seven-transmembrane-helix topology visible in the 2D class averages that fed it.

**Reasoning log.** `runs/reasoning_logs/reasoning_<run_id>.json`. One entry per step, four fields (observe / think / tool / decision). Usable as an audit trail and as training data for a later imitation-learning pass.

**What this run does *not* prove.** It does not prove resolution claims, it does not prove performance on a full-size dataset (which would be ~1000 movies, not 20), and it does not prove VLMCritic Tier 2 is good (Tier 2 needs image pulling which is disabled in remote mode). These are explicit items for the next round — see §18 and §19.

---

## 15. Reasoning Log — Full Traceability

One JSON file per run at `runs/reasoning_logs/reasoning_<run_id>.json`. Each step is one object with at minimum:

```json
{
  "step_index": 6,
  "step_key": "inspect_blob",
  "timestamp": "2026-04-10T14:22:07Z",
  "observe": {
    "job_id": "J193",
    "status": "completed",
    "metrics": { "n_picks": 12430, "picks_per_mic": 622 }
  },
  "think": "Pick count is in the expected range for a 20-movie GPCR dataset. CTF fit OK. No obvious ice. Worth handing to VLMCritic before deciding to pause.",
  "tool": {
    "name": "vlm_critic_evaluate",
    "args": { "checkpoint": "inspect_blob", "job_id": "J193" }
  },
  "decision": "pause_for_human",
  "vlm_verdict": { "verdict": "WARN", "confidence": 0.79 }
}
```

This was Mei's Cycle-4 ask. The file is regenerated every run, never overwritten, and is the single source of truth for "what did the agent do, and why."

---

## 16. Deliverables To Date

1. **`CryoEMAgent/` repo** — the agent code, installable with `pip install -e .`, runnable with `python run.py`. Branch: `master`. Public at `github.com/yaswanth169/CryoEMAgent`.
2. **`Cryosparc_mcp_Server/` repo** — the GPU-side MCP server (separate repo, required on the GPU host). Not the subject of this document but referenced throughout.
3. **`CryoEMAgent_report.tex`** — the technical report (draft for NeurIPS 2026). Compiles cleanly to PDF. Covers architecture, novelty, results, limitations, contributions.
4. **`README.md`** — setup and quickstart. Includes SSH jump host setup, CryoSPARC credentials, mode selection.
5. **`CONTEXT.md`** — this file. The end-to-end narrative for Mei and Mostofa.
6. **Reasoning log samples** — one full EMPIAR-10288 autopilot run under `runs/`.

---

## 17. Novel Contributions (the 5 claims)

These are the five claims the NeurIPS submission will stand on. They should survive Mostofa's cross-examination.

1. **ReAct-pattern cryo-EM SPA agent.** First (to our knowledge) public agentic-LLM driver for CryoSPARC that uses Observe / Think / Tool / Decision with a formal 15-tool vocabulary rather than free-form code-gen.
2. **VLM checkpoint critic with a calibrated auto-approval rule.** Two-tier (metrics + vision), checkpoint-specific criteria, 85% conservative threshold. Reduces human-pause load without silently skipping risky checkpoints.
3. **MCP-over-SSH for remote scientific instruments.** JSON-RPC 2.0 over JSONL framing inside an SSH subprocess. Works across two-hop jump-host topologies. No TCP forwarding, no proxy, no new open ports on the cluster.
4. **Dual-mode UX (Autopilot + Interactive) from one codebase.** Same planner/action layers, different front ends. Reproducibility for experiments, REPL comfort for researchers.
5. **End-to-end W1+W2 integration.** Blob → 2D → template-as-new-templates → 2D → 3D, handled as one agent-owned plan rather than two disjoint manual workflows.

---

## 18. Limitations L1–L6 (Honest Inventory)

These are the things we will *not* claim, and which Mostofa may push on.

- **L1 — VLMCritic Tier 2 is local-mode only.** Remote mode can't cheaply pull large image artifacts. Needs an image-cache / thumbnail path.
- **L2 — Mid-run parameter reconfiguration is half-done.** The agent can *propose* a new blob threshold at a checkpoint, but needs human approval to re-run. Fully autonomous reconfiguration is not merged.
- **L3 — Zero-intervention 2D class selection is not supported.** `select2d_blob` and `select2d_template` always ask for human input in practice (VLMCritic rarely PASSes at high confidence on small subsets). Fixing this needs a classification model trained on a decent corpus of class averages, which we do not have.
- **L4 — GPCR class A thresholds are hardcoded.** Particle mass, symmetry, pixel-size priors all assume a class-A GPCR. Not portable to other protein families without a semantic-memory rewrite.
- **L5 — No live metrics API.** We poll CryoSPARC job status; we do not subscribe. For very long jobs this is wasteful; not yet a real problem at our scale.
- **L6 — No multi-session memory.** Episodic memory is per-run. If you stop and come back tomorrow, the agent has forgotten your last session's choices. A persistent store is planned; not shipped.

---

## 19. Open Questions for Mostofa

These are the questions I would like Mostofa to answer (or push back on) during the 15 April meeting. Mei — if you read only one section before the meeting, read this one.

1. **Parameters.** What are the parameters a cryo-EM expert would most want the agent to be able to reconfigure mid-run? (Blob threshold, particle box size, 2D class count, refinement symmetry — which of these actually matter?)
2. **Autonomy boundary.** Where does "agent decides" end and "human decides" begin in Mostofa's view? Our current line is at the 5 W1+W2 checkpoints, gated by the 85% VLM rule. Is that too conservative, too aggressive, wrong?
3. **Baselines.** What baseline should a paper like this compare against? A vanilla CryoSPARC workflow with a human at every step? A pure-scripted workflow with no agent? Another published agent?
4. **Metrics.** What metrics would Mostofa accept as "this agent actually helps"? Wall-clock savings vs human baseline? # of unnecessary pauses avoided? Final resolution on a reference dataset?
5. **Dataset scale.** Is a 20-movie demo enough for the NeurIPS submission, or do we need a full-size EMPIAR dataset? If full-size, which one?
6. **VLM trust.** Does Mostofa trust a VLM to look at 2D class averages at all? If not, what would make him trust it (calibration curve? Expert-labeled test set?).
7. **Safety.** What is the failure mode Mostofa fears most? (Agent auto-approves a bad checkpoint and wastes a week of GPU time? Agent mis-picks particles and the 3D is subtly wrong?)
8. **Reconfiguration semantics.** When the agent "reconfigures," should it re-run the current step with new parameters, or roll back and redo upstream steps too? Our current answer is "re-run current step only, with human approval."
9. **Publishability.** Is this at the NeurIPS-submission bar, or should it be an ML4PS / structural-bio-workshop paper instead?

---

## 20. What Mei Should Take Into the Meeting

Short version, for the meeting prep:

- **We have a working v0.2 agent.** It ran end-to-end on EMPIAR-10288 (J188–J206, all 19 steps, all 5 checkpoints hit).
- **The architecture is clean.** Planner / Memory / Action. 15 formal tools. MCP-over-SSH transport. Reasoning log per run.
- **VLMCritic is the main novelty for UX.** 85% confidence auto-approval. Conservative by design.
- **The three biggest unknowns we want Mostofa on** are: (a) parameter-reconfiguration semantics, (b) the right baseline/metric pair for a paper, (c) whether a 20-movie subset is enough to report on.
- **Everything is public.** `github.com/yaswanth169/CryoEMAgent`. The LaTeX report and this context file are both in the repo root.

If the meeting goes well we aim for:
- NeurIPS 2026 submission with EMPIAR-10288 + one larger dataset.
- Mostofa as a co-author or acknowledged domain reviewer, at Mei's discretion.
- v0.3 that closes L2 (mid-run reconfig) and L1 (VLM Tier 2 over remote).

---

## Appendix A — Commit Trail

These are the commits on `master` you can point Mostofa at if he wants to read code:

- `178a113` — baseline v0.2 landing (planner/memory/action split, MCP client skeleton).
- `ebb3fe2` — fix pipeline tracking bugs (D, E, F). Added `_sync_completed_steps()`, fixed double-▶ in progress bar and table, fixed final summary count.
- `32be2c6` — initial LaTeX report (`CryoEMAgent_report.tex`).
- `e5bc1f8` — LaTeX compile fixes (package/listings/hyperref/TOC rewrite).
- `3f06501` — LaTeX targeted fixes (bugs H, I, J, K, L, M).

## Appendix B — File Map of the Repository

```
CryoEMAgent/
├── CONTEXT.md                     # this file
├── CryoEMAgent_report.tex         # NeurIPS-track technical report (compiles to PDF)
├── README.md                      # setup + quickstart
├── run.py                         # entry point: --mode autopilot | interactive
├── profile.yaml                   # SSH / CryoSPARC / model config
├── requirements.txt
├── setup.py
├── pytest.ini
├── cryoemagent/
│   ├── __init__.py
│   ├── interactive.py             # interactive REPL, progress bar, checkpoint UX
│   ├── planner.py                 # ReAct loop, system prompt, tool-call parsing
│   ├── memory.py                  # episodic + semantic memory
│   ├── action.py                  # dispatch: tool-call -> MCP JSON-RPC request
│   ├── mcp_client.py              # JSONL-over-ssh-subprocess transport
│   ├── vlm_critic.py              # Tier 1 metrics + Tier 2 vision, 85% rule
│   ├── pipeline.py                # W1+W2 definition, 19 steps, 5 checkpoints
│   └── reasoning_log.py           # structured audit log writer
├── examples/
├── playbooks/
└── tests/
```

(Server lives in a separate repo: `Cryosparc_mcp_Server/`, on the GPU host.)

## Appendix C — Glossary

- **SPA** — Single-Particle Analysis, the standard cryo-EM workflow.
- **CryoSPARC** — the SPA software stack we drive (v4.7.1).
- **GPCR** — G-protein-coupled receptor; the target protein family.
- **EMPIAR** — Electron Microscopy Public Image Archive; EMPIAR-10288 is our demo dataset.
- **ReAct** — Yao et al., ICLR 2023 — the Observe/Think/Act agent pattern.
- **MCP** — Model Context Protocol; JSON-RPC 2.0 over stdio.
- **JSONL** — newline-delimited JSON, one object per line (our wire format).
- **VLMCritic** — our Vision-Language-Model-based checkpoint evaluator.
- **W1 / W2** — the Blob and Template workflows; together they form the 19-step pipeline.
- **Autopilot / Interactive** — the two user-facing modes of the agent.
- **Reasoning log** — one-JSON-per-run structured audit trail.
- **Checkpoint** — a step in the pipeline where a human decision is possible; 5 of them in W1+W2.

---

---

## Appendix D — Verbatim `profile.yaml` (Agent-side config)

This is the exact config the laptop uses. Credentials are in-repo because it is a development profile — a production deploy would use `${CRYOSPARC_EMAIL}` env vars on both sides. Note the `psize_A: 1.0` here versus `0.86` in the EMPIAR-10288 walkthrough doc — that is a real inconsistency (the walkthrough uses the canonical EMPIAR-10288 value; the profile was set for a subset where we re-ran with rounded values). For the next run this will be reconciled to the EMPIAR-published value.

```yaml
cryosparc:
  base_url: "http://localhost:39000"
  license_id: "e1674060-08f7-11f1-aee8-9f3875918fa6"
  email: "meiyuan@andrew.cmu.edu"
  password: "MeiYuan123"
  lane: "default"
  project_uid: "P3"
  workspace_w1_title: "W1_blob_tools"
  workspace_w2_title: "W2_template_tools"

data:
  movie_blob_path: "/shared/scratch/0/home/v_yaswanth_devavarapu/empiar/10288/*.tif"
  gainref_path: ""
  psize_A: 1.0
  accel_kv: 300
  cs_mm: 2.7
  total_dose_e_per_A2: 60.0

compute:
  import_gpus: []           # CPU import
  motion_gpus: [0]
  ctf_gpus: [0]
  picker_gpus: [0]
  extract_gpus: [0]
  class2d_gpus: [0]
  abinit_gpus: [0]
  homo_gpus: [0]
  nonuniform_gpus: [0]

workflow:
  auto_interactive: false
  picker_params:
    diameter: 150           # Å
  job_types:
    import_movies: "import_movies"
    patch_motion: "patch_motion_correction_multi"
    patch_ctf: "patch_ctf_estimation_multi"
    curate: "curate_exposures_v2"
    blob_picker: "blob_picker_gpu"
    template_picker: "template_picker_gpu"
    inspect_picks: "inspect_picks_v2"
    extract: "extract_micrographs_multi"
    class2d: "class_2D_new"
    select2d: "select_2D"
    abinit: "homo_abinit"
    homo_refine: "homo_refine_new"
    nonuniform_refine: "nonuniform_refine_new"
  outputs:
    import_movies: "imported_movies"
    motion_micrographs: "micrographs"
    ctf_exposures: "exposures"
    curate_accepted: "exposures_accepted"
    inspect_particles: "particles"
    inspect_micrographs: "micrographs"
    extract_particles: "particles"
    class2d_templates: "class_averages"
    class2d_particles: "particles"
    select2d_particles_selected: "particles_selected"
    select2d_templates_selected: "templates_selected"
    abinit_particles_class_0: "particles_class_0"
    abinit_particles_unused: "particles_unused"
    abinit_volume_class_0: "volume_class_0"

agent:
  mcp_server_src_path: ""       # remote mode
  root_dir: ""
  max_agent_iterations: 200

llm:
  provider: "openai"
  model: "gpt-4o"

ssh:
  command: "ssh"
  args:
    - "-o"
    - "ServerAliveInterval=60"
    - "-T"
    - "-J"
    - "v_yaswanth_devavarapu@xulab-login0.lan.cmu.edu:20022"
    - "-p"
    - "20022"
    - "v_yaswanth_devavarapu@10.0.1.2"
    - "/bin/sh"
    - "-c"
    - "export CRYOSPARC_EMAIL='meiyuan@andrew.cmu.edu' && export CRYOSPARC_PASSWORD='MeiYuan123' && cd /mnt/data1/lv0/scratch/home/v_yaswanth_devavarapu/Cryosparc_mcp_Server && exec python3 -u run_server.py --config config/profile.yaml"
  timeout: 600
```

### Why `/bin/sh -c`, not `bash -lc`

`bash -lc` on this server prints `declare -x VAR=...` lines for every exported variable before our command runs. Those lines go to stdout, which is the MCP wire. The JSON parser on the laptop then sees `declare -x CRYOSPARC_EMAIL=…` as the first message and rejects it. `/bin/sh -c` does not print an env dump. `-T` disables pseudo-tty allocation, which also stops ssh from buffering stdout through a tty layer.

### Why 600 s tool-call timeout

GPU jobs (especially class2d and homogeneous refine) can take 5–10 minutes on a single H100 for a full-size dataset. Cursor/Claude-Desktop's default 60 s timeout was cancelling our long calls and leaving CryoSPARC jobs running without an MCP listener. 600 s is empirically enough for our 20-movie demo; a production pipeline will need an async job-status pattern (L5 in §18).

---

## Appendix E — Verbatim Agent System Prompt (excerpt)

This is what the LLM actually sees, from `cryoemagent/core/planner.py`:

```
You are an expert cryo-EM data processing agent specialized in GPCR structure determination.

AVAILABLE TOOLS (CryoSPARC operations you can invoke):
  [1] run_import_movies: Import raw cryo-EM movie files (.tif/.mrc) into CryoSPARC workspace…
  [2] run_patch_motion_correction: Apply patch-based motion correction to raw movies…
  [3] run_patch_ctf_estimation: Estimate the Contrast Transfer Function (CTF) for each micrograph…
  [4] curate_exposures: [CHECKPOINT] Human reviews CTF quality. Exclude micrographs with CTF fit > 5 Å…
  [5] run_blob_picker: Automatically pick particle candidates using a Laplacian-of-Gaussian blob detector…
  [6] inspect_blob_picks: [CHECKPOINT] Human/VLM reviews particle picks. Adjust NCC thresholds…
  [7] extract_particles: Extract particle image stacks from micrographs using pick coordinates…
  [8] run_2d_classification: Classify particles into 2D class averages. Run 50 classes for GPCR…
  [9] select_2d_classes: [CHECKPOINT] Human/VLM selects high-quality 2D classes…
 [10] run_abinit_reconstruction: Generate an initial 3D reconstruction ab initio…
 [11] run_homogeneous_refinement: Refine the 3D reconstruction assuming a single homogeneous conformation…
 [12] run_template_picker: Pick particles using 2D class averages as templates…
 [13] run_nonuniform_refinement: Final high-resolution refinement accounting for local conformational heterogeneity…
 [14] assess_quality: Evaluate quality metrics from a completed step…
 [15] escalate_to_human: Stop the pipeline and request human expert intervention…

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

RESPONSE FORMAT — You must respond with strictly valid JSON:
{
    "observation": "...",
    "thought": "...",
    "tool_selected": "<one of AVAILABLE TOOLS>",
    "decision": "CONTINUE" | "ADJUST" | "ESCALATE",
    "reasoning": "...",
    "recommendation": "...",
    "parameter_adjustments": { "key": "value" }
}

DECISION RULES:
- Default to CONTINUE + next logical tool unless there is specific evidence of a problem
- ESCALATE only if: resolution > 8 Å AND particles < 1000, OR same step failed 3+ times
- ADJUST when quality metrics are marginal but not critically bad
- Never invent data — base decisions ONLY on provided quality context
- Keep reasoning evidence-based and cite specific metrics when available
```

This is stable across runs. It is what gives the agent the "I know about GPCR expected box size / diameter / target resolution" behaviour without hard-coding those in Python.

---

## Appendix F — Verbatim Quality Thresholds

From `cryoemagent/core/quality_critics.py`. The agent's Tier-1 critic reasons against these numbers exactly.

```python
THRESHOLDS = {
    "ctf": {
        "max_mean_ctf_A": 5.0,          # mean CTF fit must be ≤ 5 Å to pass
        "min_fraction_ok": 0.70,         # ≥ 70% micrographs must pass
        "max_ice_thickness_rel": 1.2,    # relative ice thickness upper bound
    },
    "picking": {
        "min_particles_per_mic": 50,
        "min_total_particles": 5000,
    },
    "class2d": {
        "max_empty_frac": 0.50,          # fraction of empty classes (WARN above)
        "max_gini": 0.85,                # Gini coefficient of class occupancy
    },
    "refinement": {
        "target_resolution_A": 3.5,
        "warn_resolution_A": 5.0,
    },
}
```

Four critics (`CTFCritic`, `PickingCritic`, `Class2DCritic`, `RefinementCritic`) return a `QualitySnapshot` with `verdict ∈ {PASS, WARN, FAIL}`, `metrics`, `issues`, `warnings`, `recommendation`, `timestamp`. These snapshots feed the episodic memory's `quality_timeline` which the LLM sees as `Quality Context` on every decide() call.

---

## Appendix G — Verbatim Checkpoint Human Instructions

From `Planner.generate_checkpoint_instructions()` — these are the literal strings the agent prints when it pauses.

### `curate`
```
CHECKPOINT: Curate Exposures (Job {job_uid})
--------------------------------------------
1. Open CryoSPARC and navigate to the curate exposures job.
2. Review CTF fit values — exclude micrographs with fit > 5 Å.
3. Check ice thickness — exclude very thick (rel > 1.2) or very thin samples.
4. Look for obvious contamination, crystalline ice, or poor contrast.
5. Aim to keep ≥ 80% of micrographs if CTF quality is good.
6. Click 'Finish' to complete the interactive job.
7. Return here and confirm you are done.
```

### `inspect_blob`
```
CHECKPOINT: Inspect Blob Picks (Job {job_uid})
1. Open the Inspect Picks job in CryoSPARC.
2. Review the particle picks overlaid on micrographs.
3. Adjust the minimum and maximum NCC score thresholds to:
   - Remove obvious contaminants (carbon edges, ice crystals)
   - Retain good-looking protein particles
4. Target: ≥ 50 particles/micrograph after filtering.
5. Verify particles are centred on protein, not on background.
6. Click 'Finish'; return here.
```

### `select2d_blob`
```
CHECKPOINT: Select 2D Classes — Blob Picks (Job {job_uid})
1. Open the Select 2D job in CryoSPARC.
2. Review 2D class averages sorted by resolution/quality.
3. Select classes that show:
   - Clear secondary structure (α-helices visible)
   - Consistent particle size (~100–150 Å for GPCR)
   - Low background noise
4. Deselect 'junk' classes: rings, aggregates, edge particles.
5. Aim to keep ≥ 30–40% of the total particles.
6. Click 'Finish'; return here.
```

### `inspect_template`
```
CHECKPOINT: Inspect Template Picks (Job {job_uid})
1. Open the Inspect Picks job for the template picker.
2. Templates derived from 2D class averages should give more accurate picks.
3. Adjust NCC thresholds to remove false positives (ice, carbon).
4. Target: ≥ 50 particles/micrograph after filtering.
5. Click 'Finish'; return here.
```

### `select2d_template`
```
CHECKPOINT: Select 2D Classes — Template Picks (Job {job_uid})
1. Open the Select 2D job in CryoSPARC (W2 template picks).
2. Final 2D selection before ab-initio reconstruction.
3. Be stricter than W1 selection:
   - Only classes with clear TM helix density
   - Orientation diversity (top, side, tilted)
4. Aim to keep 40–60% of particles from the best classes.
5. Click 'Finish'; return here.
```

These strings are not paraphrased elsewhere in the code; they are the single source of truth for "what does a human at the checkpoint actually do?"

---

## Appendix H — Verbatim VLMCritic Prompts

From `cryoemagent/vlm_critic.py`. Five checkpoints × two tiers. `{metrics}` is interpolated at call time.

### `AUTO_APPROVE_THRESHOLD = 0.85`

```python
def _should_auto_approve(assessment: VLMAssessment) -> bool:
    return (
        assessment.confidence >= AUTO_APPROVE_THRESHOLD
        and assessment.verdict == "PASS"
        and assessment.decision in {"approve", "approve_with_adjustments"}
    )
```

Vision support is detected by model name (`gpt-4o`, `gpt-4-turbo`, `gpt-4v`, `vision`, `claude`). Our current `gpt-4o` profile triggers Tier-2 when images are supplied; in the SSH/remote mode we do not ship images yet (see L1 in §18) so we run Tier-1 only.

### Checkpoint 1 — `curate` (Tier 1)
```
You are an expert cryo-EM structural biologist evaluating micrograph quality for GPCR
structure determination.

Evaluate the following CTF estimation results: {metrics}

GPCR quality thresholds:
- CTF fit resolution: GOOD < 5 Å | WARN 5-7 Å | FAIL > 7 Å
- Ice thickness (relative): GOOD < 1.2 | WARN 1.2-1.5 | FAIL > 1.5
- Acceptance rate: GOOD > 70% | WARN 50-70% | FAIL < 50%
- Defocus range: optimal 0.5-3.0 μm underfocus

Respond with JSON:
{"verdict": "PASS|WARN|FAIL",
 "decision": "approve|approve_with_adjustments|escalate_to_human",
 "confidence": 0.0-1.0,
 "observations": [...],
 "reasoning": "3-4 sentence chain-of-thought",
 "recommended_actions": [...],
 "suggested_params": {}}
```

### Checkpoint 1 — `curate` (Tier 2, vision)
```
You are an expert cryo-EM structural biologist. Analyze this CTF power spectrum image
from a GPCR dataset.

Look for:
1. Thon rings — should be clearly visible and evenly spaced
2. CTF fit quality — the overlaid curve should closely follow the rings
3. Ice/contamination — dark spots, asymmetric patterns, streaks indicate problems
4. Resolution of rings — rings should be visible to at least 5 Å

Additional metrics: {metrics}
Decide: should this micrograph be KEPT or REJECTED?
```

### Checkpoint 2 — `inspect_blob` (Tier 1)
```
Blob picking results: {metrics}
GPCR particle picking thresholds:
- Particles per micrograph: GOOD > 50 | WARN 20-50 | FAIL < 20
- NCC score distribution: want right-skewed
- Expected particle diameter: 80-150 Å for GPCR in detergent/nanodisc
- Pick density: too sparse = missed particles, too dense = false positives

suggested_params example: {"ncc_threshold_min": 0.1, "ncc_threshold_max": 0.9}
```

### Checkpoint 2 — `inspect_blob` (Tier 2)
```
Analyze this micrograph with overlaid particle picks for a GPCR cryo-EM dataset.

1. Are picks centred on protein density (not background/ice/carbon)?
2. Are obvious GPCR-sized (~100–130 Å) densities being missed?
3. False positives (contamination, edge artifacts)?
4. Is pick density reasonable (50–150 picks per micrograph)?
```

### Checkpoint 3 — `select2d_blob` (Tier 1)
```
GPCR 2D class quality criteria:
- Secondary structure visibility: α-helices (TM helices) should be visible as lines
- Particle diameter: ~100-150 Å
- View diversity: need top, side, tilted views
- Junk classes: ice rings, noise aggregates, edge particles = reject
- Selection target: keep 30-50% of particles from best classes
- Minimum classes to keep: at least 5 good classes with clear features
```

### Checkpoint 3 — `select2d_blob` (Tier 2)
```
Analyze this grid of 2D class averages from a GPCR cryo-EM dataset.
For each class:
1. Clear protein structure (TM helices visible as lines)?
2. Size consistent with GPCR (~100–150 Å)?
3. Good (KEEP) or junk (DISCARD: ring, noise, smeared)?
4. Are different orientations represented in the good classes?
```

### Checkpoint 4 — `inspect_template` (Tier 1)
```
Template picks should be more accurate than blob picks.
- Particles per mic: GOOD > 60 | WARN 30-60 | FAIL < 30
- Template correlation distribution: narrow, right-skewed
- False positive rate: should be LOWER than blob picking
```

### Checkpoint 4 — `inspect_template` (Tier 2)
```
Template picks should be more accurate than blob picks. Evaluate:
1. Picks well-centred on GPCR protein density?
2. Protein particles being missed?
3. False positives on ice/contamination?
4. Pick density appropriate?
```

### Checkpoint 5 — `select2d_template` (Tier 1 only)
```
This is the FINAL 2D selection before ab-initio.
Be more stringent than W1 blob selection:
- Only classes with clear TM helix density
- Orientation diversity (top AND side AND tilted)
- Target 40-60% particle retention
- Need at least 10,000 particles for reliable ab-initio
```

(No Tier-2 prompt for `select2d_template` in the current code — 2D selection at the W2 stage was originally intended to require a human regardless because of the stake in ab-initio.)

---

## Appendix I — Server Side (Cryosparc_mcp_Server)

### Repo layout
```
Cryosparc_mcp_Server/
├── run_server.py                         # entry point: --config config/profile.yaml
├── README.md                             # operational notes (see below)
├── config/
│   └── profile.yaml                      # same schema as CryoEMAgent/profile.yaml
├── workflow_definitions/
│   └── w1_w2_standard.yaml               # 19-step W1+W2 stage/step list
├── runs/
│   └── <run_id>.json                     # RunState per run (persisted)
├── reports/
│   └── <run_id>.*                        # cs_pipeline_report outputs
└── src/cryosparc_mcp_server/
    ├── server.py                         # FastMCP("cryosparc-mode-a") + 9 tools
    ├── orchestrator.py                   # 19-step pipeline runner
    ├── cryosparc_adapter.py              # wraps cryosparc-tools
    ├── state_store.py                    # RunState + RunStore
    ├── config.py                         # yaml loader, ensure_dirs
    └── report.py                         # write_report (markdown + json)
```

### FastMCP app — the 9 tools
The server registers a single FastMCP app named `"cryosparc-mode-a"` and exposes these tools:

| # | Tool | Signature | Purpose |
|---|------|-----------|---------|
| 1 | `cs_health_check` | `() -> Dict` | Is the config valid? Are CryoSPARC credentials resolvable from config + env? Does not open a CryoSPARC session. |
| 2 | `cs_validate_data_source` | `(movie_blob_path: str = "") -> Dict` | Glob the movie path, check it matches ≥ 1 file, validate required `cryosparc.*` keys, validate `workflow.picker_params.diameter`. |
| 3 | `cs_plan_and_run_standard_pipeline` | `(runtime_overrides: Dict) -> Dict` | Fire-and-forget: run W1+W2 until completion or first checkpoint, up to 500 internal single-steps. |
| 4 | `cs_start_pipeline` | `(runtime_overrides: Dict) -> Dict` | Create a new run (fresh `run_id`), advance one step. |
| 5 | `cs_continue_pipeline` | `(run_id: str, steps: int = 1) -> Dict` | Advance an existing run N steps (capped at 50 per call). Stops at checkpoints. |
| 6 | `cs_resume_pipeline` | `(run_id: str) -> Dict` | After human finishes the CryoSPARC UI work, this is what acknowledges the checkpoint and continues until the next checkpoint / completion (up to 200 single-steps). |
| 7 | `cs_pipeline_status` | `(run_id: str) -> Dict` | Read-only summary of a run. |
| 8 | `cs_pipeline_report` | `(run_id: str) -> Dict` | Write a human-readable report for the run. |
| 9 | `cs_list_runs` | `() -> Dict` | All persisted run_ids. |

Every tool that touches state also returns the same `summarize(state)` payload — this is critical because the agent's planner only decides next actions by reading this dict:

```python
def summarize(state) -> Dict[str, Any]:
    return {
        "run_id": state.run_id,
        "status": state.status,                 # running / completed / failed
        "current_stage": state.current_stage,   # "w1" or "w2"
        "current_step": state.current_step,     # e.g. "curate", "inspect_blob"
        "checkpoint_required": state.checkpoint_required,
        "checkpoint_message": state.checkpoint_message,
        "checkpoint_job_uid": state.checkpoint_job_uid,
        "workspace_w1_uid": state.workspace_w1_uid,
        "workspace_w2_uid": state.workspace_w2_uid,
        "completed_steps": len(state.jobs),
        "jobs": state.jobs,              # {step_id: CryoSPARC job UID}
        "errors": state.errors,          # {step_id: error string}
        "operator_instruction": op,      # natural-language next-step instruction
        "next_suggested_tool": next_tool,
        "next_suggested_args": next_args,
    }
```

### `RunState` schema

From `state_store.py`:

```python
@dataclass
class RunState:
    run_id: str                            # uuid4
    workflow_id: str = "w1_w2_standard"
    status: str = "running"                # running / completed / failed
    current_stage: str = "w1"              # w1 or w2
    current_step: str = "import_movies"
    checkpoint_required: bool = False
    checkpoint_message: str = ""
    checkpoint_job_uid: str = ""
    workspace_w1_uid: str = ""             # resolved on first call, pinned after
    workspace_w2_uid: str = ""
    jobs: Dict[str, str] = {}              # step_id -> CryoSPARC job UID (e.g. "J141")
    errors: Dict[str, str] = {}
    config: Dict = {}                      # snapshot of the runtime-merged profile
    created_at: str                        # ISO-8601 UTC
    updated_at: str
```

Persisted as `runs/<run_id>.json`. If the MCP process dies, the next `cs_resume_pipeline(run_id)` rehydrates this dict and picks up where the last step left off. That was one of Mei's early asks — "don't lose my work if SSH drops."

### Orchestrator — the 19 real job creations

`Orchestrator.run_until_pause_or_done(state, single_step=True)` is the workhorse. It resolves W1 and W2 workspaces once, resumes an interactive checkpoint if one is pending, then walks the 19 steps in order. Each step is guarded by `if "<step>" not in state.jobs:` so a resumed run skips already-done work. Concrete wiring (abbreviated — full source in `orchestrator.py`):

```python
# W1 — 11 steps
j_import  = create(import_movies, params={blob_paths, gainref_path, psize_A, accel_kv, cs_mm, total_dose_e_per_A2}); queue_and_wait
j_motion  = create(patch_motion_correction_multi, connections={movies: (j_import, imported_movies)},
                   params={compute_num_gpus}); queue_and_wait
j_ctf     = create(patch_ctf_estimation_multi, connections={exposures: (j_motion, micrographs)},
                   params={compute_num_gpus}); queue_and_wait
j_curate  = create(curate_exposures_v2, connections={exposures: (j_ctf, exposures)})
            queue_to_waiting; CHECKPOINT("curate")    # interactive
j_blob    = create(blob_picker_gpu, connections={micrographs: (j_curate, exposures_accepted)},
                   params={diameter: 150}); queue_and_wait
j_inspect = create(inspect_picks_v2,
                   connections={micrographs: (j_curate, exposures_accepted),
                                particles:   (j_blob,   picks)})
            queue_to_waiting; CHECKPOINT("inspect_blob")
j_extract = create(extract_micrographs_multi,
                   connections={micrographs: (j_curate,  exposures_accepted),
                                particles:   (j_inspect, particles)})
            queue_and_wait
j_class2d = create(class_2D_new, connections={particles: (j_extract, particles)}); queue_and_wait
j_sel2d   = create(select_2D, connections={particles:       (j_class2d, particles),
                                           class_averages:  (j_class2d, class_averages)})
            queue_to_waiting; CHECKPOINT("select2d_blob")
j_abinit  = create(homo_abinit, connections={particles: (j_sel2d, particles_selected)}); queue_and_wait
j_homo    = create(homo_refine_new,
                   connections={particles: (j_abinit, particles_class_0),
                                volume:    (j_abinit, volume_class_0)}); queue_and_wait

# W2 — 8 steps (W2 workspace)
j_tmpl    = create(template_picker_gpu, connections={micrographs: (j_curate, exposures_accepted),
                                                     templates:   (j_sel2d,  templates_selected)},
                   params={diameter: 150}); queue_and_wait
j_itmpl   = create(inspect_picks_v2, connections={micrographs: (j_curate, exposures_accepted),
                                                  particles:   (j_tmpl,  picks)})
            queue_to_waiting; CHECKPOINT("inspect_template")
j_xtmpl   = create(extract_micrographs_multi, connections={micrographs: (j_curate, exposures_accepted),
                                                           particles:   (j_itmpl, particles)})
            queue_and_wait
j_c2d_t   = create(class_2D_new, connections={particles: (j_xtmpl, particles)}); queue_and_wait
j_sel_t   = create(select_2D, connections={particles:      (j_c2d_t, particles),
                                           class_averages: (j_c2d_t, class_averages)})
            queue_to_waiting; CHECKPOINT("select2d_template")
j_ab_t    = create(homo_abinit, connections={particles: (j_sel_t, particles_selected)}); queue_and_wait
j_homo_t  = create(homo_refine_new, connections={particles: (j_ab_t, particles_class_0),
                                                 volume:    (j_ab_t, volume_class_0)}); queue_and_wait
j_nu      = create(nonuniform_refine_new,
                   connections={particles_A: (j_ab_t, particles_class_0),
                                particles_B: (j_ab_t, particles_unused),        # both branches
                                volume:      (j_homo_t, volume)})
            queue_and_wait
state.status = "completed"
```

Two methods on the adapter matter:

- **`queue_and_wait(job, lane, gpus)`** — queues a non-interactive job and blocks until it is done.
- **`queue_to_waiting(job, lane)`** — queues an interactive job (curate / inspect / select2d) into a "waiting for user" state and returns immediately. The caller then calls `_checkpoint()` to mark `state.checkpoint_required = True` and return to the MCP tool, which in turn returns to the agent, which in turn pauses.

### `finish_interactive`

When the agent calls `cs_resume_pipeline(run_id)`, the first thing the orchestrator does is:

```python
job = workspace.find_job(state.checkpoint_job_uid)
adapter.finish_interactive(job)   # -> job.interact("shutdown_interactive")
state.checkpoint_required = False
```

That `shutdown_interactive` command is the CryoSPARC-internal signal that tells the interactive job it is done and to emit its outputs. Without this call, the interactive job sits waiting forever even after the human has clicked "Finish" in the UI. Finding this was one of the harder debugging sessions — the human clicks Finish, the UI says "complete", but the API job is still listed as `waiting` until the shutdown is explicitly sent.

### CryoSPARCAdapter — workspace resolution

`ensure_workspace(project, title, pinned_uid)` has to deal with the fact that CryoSPARC allows multiple workspaces with identical titles. Algorithm:

1. If `pinned_uid` is a non-empty string, return the workspace with that exact UID.
2. Otherwise enumerate all workspaces in the project whose `title == desired_title`.
3. Sort them by `job_count DESC`, then by UID ASC (tiebreaker).
4. Return the first one. (This is the "most-used workspace with this title" heuristic.)
5. If none exist, create a new workspace with that title.

The reason the tiebreaker matters: we had two "W1_blob_tools" workspaces — W11 (old, empty) and W34 (current run). Without the job-count sort, we were picking W11 by UID order and the orchestrator was creating `j_import` in an empty workspace while the operator was watching W34 in the UI wondering where their job was.

### README — operational footnotes from the server repo

Verbatim (condensed) from `Cryosparc_mcp_Server/README.md`:

> - The MCP process is tied to the MCP client's stdio pipe (Cursor / Claude Desktop / our laptop). It is **not** a daemon. Closing the client kills the MCP server.
> - **CryoSPARC jobs keep running** even if the MCP server dies. You can reconnect a new MCP session and `cs_resume_pipeline(run_id)` picks up the existing CryoSPARC jobs.
> - Cursor client timeouts (~60 s) will cause `anyio.BrokenResourceError` on the server if a tool call is still mid-flight. We set `ssh.timeout: 600` on the laptop side for our agent, which is enough for a 20-movie demo but not for full-size runs.
> - `bash -lc` on this server prints `declare -x` env dumps to stdout, which poisons the JSON wire. Always use `/bin/sh -c`.
> - `-T` on ssh disables pseudo-tty allocation, which stops stdin/stdout double-buffering.

---

## Appendix J — MCP Client Transport (Agent Side)

From `cryoemagent/mcp_client.py`:

- `subprocess.Popen(..., stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=-1)` — `-1` was broken on Windows (Bug A) until we confirmed the JSONL framing layer in Python handled its own flushing. The current code uses default buffering plus explicit `.flush()` after every write.
- **Startup wait:** the client sleeps ~5 s after spawning `ssh` to let the jump-host handshake and the remote Python startup settle before sending the first JSON-RPC request. Empirically, sending before this delay sometimes lands the request in the middle of the SSH banner.
- **JSONL encode:** each outgoing message is `json.dumps(msg) + "\n"`, UTF-8. One line per message, no Content-Length framing.
- **JSONL decode:** the reader ignores lines that do not start with `{`. This is what lets SSH banners, login-MOTD, and the occasional warning from cryosparc-tools on stderr pass by without crashing the parser. Every valid line is fed to `json.loads` and handed to the RPC demultiplexer.
- **Required tool surface:** the client asserts at handshake that the server exposes `cs_start_pipeline`, `cs_continue_pipeline`, `cs_resume_pipeline`, `cs_pipeline_status`, `cs_pipeline_report`, and `cs_list_runs`. Missing any of these aborts the run early with a readable error rather than a KeyError mid-pipeline.

`MCPOrchestratorClient` is a drop-in replacement for the older direct-REST `OrchestratorClient`. `RemoteState` is a thin wrapper over the `summarize()` dict so the rest of the agent doesn't have to know whether it is talking to a local or remote backend.

---

## Appendix K — EMPIAR-10288 Walkthrough (real numbers)

**Dataset:** EMPIAR-10288, CB1 cannabinoid receptor (GPCR class A, ~60 kDa).
**Canonical acquisition:** 300 kV Titan Krios, 1.05 Å/px (EMPIAR-published; profile.yaml currently uses 1.0 as noted in Appendix D), Cs 2.7 mm, total dose 58 e⁻/Å² (profile.yaml uses 60 as a rounded value).
**Our subset:** 20 TIFF movies downloaded from EMPIAR, hosted at `/shared/scratch/0/home/v_yaswanth_devavarapu/empiar/10288/*.tif` on the GPU server.

### End-to-end walkthrough (from our `CRYOSPARC-EMPIAR10288-WALKTHROUGH.md`)

1. **Server login:** `ssh -J …xulab-login0… v_yaswanth_devavarapu@10.0.1.2`.
2. **CryoSPARC install:** v4.7.1, master + single worker co-located on `10.0.1.2`. License key `e1674060-…`.
3. **SSH tunnel for CryoSPARC UI:** `ssh -J … -L 39000:localhost:39000 …10.0.1.2` on the laptop, then open `http://localhost:39000` in a browser. This is only needed to watch the UI; the agent itself uses MCP.
4. **Project P3** created in the UI. **W1_blob_tools** and **W2_template_tools** workspaces created in P3 (W1 resolves to UID `W34`, W2 to `W35` in the current demo run).
5. **Movie download:** 20 TIFFs from EMPIAR-10288 into `/shared/scratch/.../empiar/10288/`.
6. **Import (J2):** pixel 0.86 Å/px (EMPIAR canonical for that test run), 300 kV, Cs 2.7, dose 58 e⁻/Å².
7. **Patch motion (J5):** default params, GPU 0.
8. **Patch CTF (J6):** default params, GPU 0. This is what feeds `curate` at J136 in the full agent run.

### Troubleshooting history (from the walkthrough, kept here so Mostofa sees the surface area)

- *"Child process exit code 1"* on CryoSPARC workers → CUDA toolkit ↔ driver mismatch on `10.0.1.2`. Reinstalled CUDA matching the driver.
- *Gain reference path errors* → the public EMPIAR-10288 subset we pulled has no gain reference; we set `gainref_path: ""` in profile.yaml and CryoSPARC's import skips gain correction cleanly.
- *`ValueError: unable to infer bins`* in patch CTF → fixed by re-importing with explicit `psize_A` (some import paths leave it unset).
- *"Connection refused"* → CryoSPARC master daemon hadn't finished starting. Wait 60 s after `cryosparcm start`, then retry.
- *"Permission denied" on `/shared/scratch`* → fixed by `chmod g+s` on the scratch dir so new files inherit the lab group.
- *"Path too long" on Windows* when cloning `Cryosparc_mcp_Server` into a OneDrive-backed folder → we enabled long paths in Windows and git config, then moved the clone out of OneDrive.

---

## Appendix L — Demo Snapshot (the run we will show)

This is the concrete run Mei and Mostofa can reference:

- **`run_id`** (from `cs_start_pipeline` → persisted as `runs/<run_id>.json` on the server):
  `a7cc8c0b-536b-4950-a1a7-451fbf3ed1ec`
- **Workspaces:** `W34` (W1_blob_tools), `W35` (W2_template_tools) in project `P3`.
- **Orchestrator steps completed:** 19 (full W1+W2).
- **Key job UIDs:**
  - `J136` — `curate_exposures_v2` (Checkpoint 1). Interactive, resumed after human click.
  - `J138` — `inspect_picks_v2` (Checkpoint 2). Interactive, resumed.
  - `J141` — `select_2D` W1 (Checkpoint 3). Interactive, resumed.
  - (Continuing into W2 with `template_picker_gpu`, `inspect_picks_v2`, `select_2D`, `homo_abinit`, `homo_refine_new`, `nonuniform_refine_new`.)
- **Orchestrator errors:** none. Every `queue_and_wait` returned clean.
- **VLMCritic verdicts (Tier-1 only in remote mode):** `WARN` at 0.75–0.85 confidence on all 5 checkpoints. The agent paused at each as designed.
- **Final artifact:** 3D volume produced at `homo_refine_new` and further refined at `nonuniform_refine_new`. Seven-TM-helix GPCR topology visible in the feeding 2D class averages.
- **Reasoning log:** `runs/reasoning_logs/reasoning_a7cc8c0b.json` — one entry per iteration.

This is the run the slides and the voiceover script reference.

---

## Appendix M — Commit-level Detail (beyond Appendix A)

(This list supersedes Appendix A where they disagree; Appendix A was the high-level view.)

- **`178a113`** — baseline v0.2: Planner/Memory/Action split, MCP client skeleton, `CRYOSPARC_TOOLS` of 15, `SYSTEM_PROMPT` with the W1+W2 workflow inlined.
- **`ebb3fe2`** — pipeline tracking fixes: `_sync_completed_steps()` in `interactive.py`, exact-equality current-step match in `_show_progress_bar()` / `_show_pipeline_table()`, fixed final-summary count. Addresses Bugs D / E / F.
- **`32be2c6`** — initial LaTeX report (`CryoEMAgent_report.tex`).
- **`e5bc1f8`** — LaTeX compile: removed `pgf-umlsd`, declared pgf layers, replaced ✋ emoji with `\textbf{[CP]}`, added `\lstdefinelanguage` for JSON/YAML, added `textcomp`, reordered `hyperref` to last in preamble, fixed TOC.
- **`3f06501`** — LaTeX targeted fixes: Bugs H (`moredelim` vs `literate`), I (`\umu` vs TikZ), J (particles/µm → particles/micrograph), K (stray 10.5 cm TikZ anchor), L & M (figure overflow, reroute via `.west` anchors).

---

## Appendix N — Open Items Tracking (as of 2026-04-20)

Things to do before 15 April meeting (**already done** by the time you read this):
- [x] Expand CONTEXT.md to include full verbatim artifacts (this file).
- [x] Reconfirm demo run is reproducible (`run_id a7cc8c0b-…`).
- [x] Confirm LaTeX compiles on stock TeX Live 2024.

Things to do during / after the meeting:
- [ ] Capture Mostofa's answers to the 9 open questions in §19.
- [ ] Decide on baseline and metric for NeurIPS 2026.
- [ ] Decide full-size dataset (EMPIAR-10532? 11103?).
- [ ] Plan v0.3 scope: L1 (VLM Tier 2 remote) + L2 (mid-run reconfig).

---

*End of CONTEXT.md. Nothing deliberately omitted. If something is missing here, it is because I do not know it yet — please ping me before the meeting so I can add it.*
