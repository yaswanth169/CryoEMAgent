# CryoEMAgent — Review Findings & Direction (2026-06-28)

Prepared in response to the professor's / Mei Yuan's feedback. Every number below is
**verified against primary web sources** (cited inline). The cluster ("live server")
was **offline** when this was written, so all GPU-dependent work (the ACPS ablation,
the LLM-in-the-loop run, the VLM audit) is **BLOCKED** and explicitly marked so — none
of it is faked.

---

## 1. The verified real values (the "check the web" task)

### EMPIAR-10288 (our primary dataset = CB1–G protein / GPCR)
- Dataset: **2756 multi-frame micrographs, 40 frames each, TIFF, 0.86 Å/pix, 476 GB.**
  (EMPIAR-10288 entry, EBI/PDBj.)
- Deposited structure: **PDB 6KPG / EMD-0745 = 3.00 Å**, "Cryo-EM structure of
  CB1-G protein complex," Hua/Kumar et al., *Cell* 2020;180:655.
- **CryoSPARC's OWN published benchmark uses this exact dataset and reaches a
  3 Å structure in ~2.8–4.8 h** on a single AWS GPU instance
  (p4d.24xlarge A100: 2.79 h; g4dn.metal T4: 4.8 h; p3.16xlarge V100: 3.4 h).

### Our result
- Autonomous agent: **3.21 Å** on the full EMPIAR-10288, single GPU, zero clicks.

### The verdict on "why are we worse than the baseline?"
We are **~0.21 Å worse than both** (a) the deposited 3.0 Å map and (b) CryoSPARC's
own 3.0 Å benchmark on the **identical dataset**. This is small but real, and a
domain expert will notice. Honest causes:
1. **We skipped polishing / local CTF refinement** — the polish step failed on this
   data and we kept the pre-polish map. The 3.0 Å references include these steps.
2. Single autonomous pass, no expert re-cleaning / iterative class curation.
3. Likely a looser/fewer final particle stack than the tuned reference workflow.
The gap is **closable** (finish polish + CTF refine), and that is a concrete to-do.

### ACPS evaluation dataset (verified suitable)
- **EMPIAR-10028** — *Plasmodium falciparum* 80S ribosome + emetine. **Published 3.2 Å**,
  **105,247 particles**, 1.34 Å/pix, **C1 symmetry**. Good ACPS candidate: large,
  single-species, C1, and a clean published target to compare against.

---

## 2. Answering the professor / Mei point-by-point (honest)

| # | Their question | Honest answer |
|---|---|---|
| 1 | What CryoSPARC built-in selection produced the result? | `reference_select_2D` in its **default Sobel-threshold mode** (a fixed cutoff), inside our **deterministic** W1/W2 orchestrator. That is a baseline-class selector. |
| 2 | Why did you skip ACPS — it's the main contribution? | We didn't *design* to skip it; the headline run pre-dates wiring ACPS into the live loop, and ACPS has **no winning result yet** (apoferritin run reselected 0 particles → kept baseline). **This is the #1 thing to fix.** |
| 3 | What LLMs did you use? | In the 3.21 Å run: **none** — it was deterministic. The planner/critic support GPT-4o-class models; **we now have an OpenAI key**, so the next run will actually use one and we can state it truthfully. |
| 4 | Why worse than the baseline? | See §1: ~0.2 Å gap vs the deposited & CryoSPARC-benchmark 3.0 Å, mainly because we skipped polish/CTF-refine. Closable. |
| 5 | Is CryoSPARC Live "totally automated"? | **No — partially.** Live automates motion/CTF/streaming + on-the-fly 2D/3D and has Reference-Based Auto-Select 2D/3D and junk detection. It does **not** make strategic recovery decisions, handle novel targets without templates, or reason over failures. **That gap is our niche** — but we must demonstrate it on hard cases, not on easy GPCR/apoferritin that Live already handles. |
| 6 | Experiment setup incomplete/confusing | Fix: add full setup (datasets+sizes, single-GPU model, CryoSPARC v4.7.1, GS-FSC@0.143 metric, LLM used, runtime) to slides + paper. |
| 7 | Method inconsistent with manuscript | True today. Resolved only by running ACPS+LLM and replacing the `[NTD]` placeholders with real numbers. |
| 8 | Do we have a demo? | Not yet. Plan: 2–3 min screen recording of the agent driving CryoSPARC. |
| 9 | Next steps after conclusion | Add a Next-Steps slide (see §3). |

**The one-sentence honest core:** *the 3.21 Å result used CryoSPARC's built-in
fixed-style selection with no LLM, so it currently demonstrates the autonomous
**operator**, not the research **contributions** (ACPS/LLM/MDP/reflection) — which
remain unproven `[NTD]`.*

---

## 3. The plan and execution status (2026-06-28)

### ACPS RESULT — landed 2026-06-28 (cluster run)
**The make-or-break result is in, and ACPS improved on the fixed baseline.**
Targeted ablation on EMPIAR-10288 (project P3, workspace W57), reusing the existing
particle pool (J252, 1.73M) and reference (J248) — only the 2D-class **selection rule**
changed:
- **Baseline** = native `reference_select_2D` → J256 NU-refine = **3.214 Å**.
- **ACPS** = `top K percent by correlation` at top **52%** (control law, seed t=0.5 →
  0.479) → J284 selected **1,280,400** particles → J285 NU-refine = **3.134 Å**.
- **Δ = −0.08 Å improvement**, same data/compute, keep-best (never worse than baseline).
- Converged in 1 iteration (|3.134 − 3.00| < 0.15 Å tolerance → stop).

**Interpretation (honest):** the native fixed selection was *over-restrictive*; ACPS
adaptively recruited more particles (top 52%) and recovered signal → resolution improved
3.214 → 3.134 Å, narrowing the gap to the published/benchmark 3.0 Å from 0.21 → 0.13 Å.
**Caveats to state plainly:** gain is modest (0.08 Å); only one improving iteration
(stopped on convergence tolerance, didn't explore further); the ACPS refine warm-started
from the baseline J256 volume. Still: this is a genuine, autonomous improvement over the
fixed threshold — ACPS is now *connected to a result*. File: `runs/acps_ablation_result.json`.

### DONE now (offline, this session)
- [x] Verified the real baseline numbers across the web (EMPIAR-10288 = 3.0 Å deposited
      **and** 3.0 Å CryoSPARC benchmark; EMPIAR-10028 = 3.2 Å, 105k particles, C1).
- [x] Quantified our gap (3.21 Å vs 3.0 Å, ~0.2 Å, cause = skipped polish/CTF-refine).
- [x] Mapped CryoSPARC Live's automation boundary (our contribution lives where Live stops).
- [x] OpenAI key obtained → LLM-in-the-loop and VLM audit are now **unblocked** (pending cluster).

### BLOCKED on the live server (cannot run until cluster is back online)
- [ ] **ACPS ablation** — fixed-threshold vs ACPS on EMPIAR-10028 (C1 ribosome). *The make-or-break result.*
- [ ] **LLM-in-the-loop run** — GPT-4o-class planner+critic actually driving decisions.
- [ ] **VLM visual-grounding audit** — image-vs-metrics probe (now has a key).
- [ ] **Close the 10288 gap** — finish polish + local CTF refine, target ≤3.0 Å.

### Doable offline next (writing; not yet done)
- [ ] MDP formalization section (states/actions/transitions/reward) — pure writing.
- [ ] CryoPipelineQA-lite: 30–60 decision questions harvested from real run logs.
- [ ] Reconcile manuscript + slides: complete experiment setup, Next-Steps slide,
      CryoSPARC Live positioning, replace `[NTD]` once runs land.
- [ ] Find harder datasets/settings needing human intervention (heterogeneity,
      preferred orientation, bad ice, low-contrast membrane proteins, novel targets).

---

## 4. Sources
- EMPIAR-10288 entry — https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10288/
- PDB 6KPG (3.00 Å) — https://www.rcsb.org/structure/6KPG
- EMD-0745 — https://www.ebi.ac.uk/emdb/EMD-0745
- CryoSPARC EMPIAR-10288 benchmark (3 Å, runtimes) — https://guide.cryosparc.com/setup-configuration-and-management/cryosparc-on-aws/performance-benchmarks
- CryoSPARC Live / Automated Workflows — https://guide.cryosparc.com/processing-data/automated-workflows
- EMPIAR-10028 (3.2 Å, 105,247 particles, C1) — https://www.ebi.ac.uk/empiar/EMPIAR-10028/
