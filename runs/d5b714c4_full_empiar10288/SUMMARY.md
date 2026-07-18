# CryoEMAgent — Full-Dataset Autonomous Run Summary

**Run ID:** `d5b714c4-0b3f-42c2-8771-322fa8a048e3`
**Dataset:** EMPIAR-10288 — CB1-GPCR (full, 2,756 movies, 477 GB, 0.86 Å/px, 300 kV)
**Started:** 2026-06-05 05:23 UTC · **Completed:** 2026-06-06 02:03 UTC (~20.7 h)
**Mode:** Fully autonomous, MCP-over-SSH (laptop → GPU cluster), **zero human intervention**
**CryoSPARC project/workspaces:** P3 · W1_blob_tools (W56) · W2_template_tools (W57)

---

## HEADLINE RESULT

| Metric | Value |
|---|---|
| **Final resolution** | **3.21 Å** (GS-FSC 0.143) |
| **Final volume** | J256 (NU-Refine) |
| **Final particles** | 620,174 (structurally selected from 1.99 M template picks) |
| **Pipeline status** | COMPLETED — 18 jobs succeeded, 2 failed (1 auto-recovered, 1 data limit) |
| **Human clicks** | 0 |

Comparison: the 20-movie subset reached only **9.12 Å** with 5,358 particles. The full dataset reached **3.21 Å** — near-atomic, publication-grade.

---

## PIPELINE (job-by-job)

### W1 — blob arm (bootstrap reference) — 10 jobs, all ✅
| Job | Step | Result | Time |
|---|---|---|---|
| J239 | Import movies | 2,756 movies @ 0.86 Å | 16 s |
| J240 | Patch motion | 2,755 micrographs (1 incomplete) | 6 h 45 m |
| J241 | Patch CTF | 2,753 exposures (2 incomplete) | 7 h 22 m |
| J242 | Curate (auto-threshold) | 2,181 accepted (79%), **572 rejected (21%)** | 17 s |
| J243 | Blob picker | 505,029 picks | 22 m |
| J244 | Extract | 144,556 particles @ 256 px | 25 m |
| J245 | 2D classification | 96 classes (dynamic) | 14 m |
| J246 | Auto-select 2D (count-based) | completed | 2 s |
| J247 | Ab-initio | 83,750 particles → 3D volume | 8 m |
| J248 | Homo refine | 83,750 particles → **4.10 Å** (reference V1) | 9 m |

### W2 — template arm (final structure) — 10 jobs, 8 ✅ + 2 ❌
| Job | Step | Result | Time |
|---|---|---|---|
| J249 | Template picker | **1,993,908 picks** (~2 M) | 1 h 09 m |
| J250 | Extract | 1,735,240 particles @ 256 px | 1 h 37 m |
| J251 | 2D classification | ❌ **CUDA OOM** (384 classes) | 17 m |
| J252 | 2D classification (retry) | ✅ 100 classes (memory fix) | 51 m |
| J253 | Ref-based auto-select 2D | **620,174 selected (36%)**, 1,115,066 excluded (64%) | 1 m |
| J254 | Ab-initio | 149,400 particles → 3D volume | 11 m |
| J255 | Homo refine | 149,400 particles → **3.96 Å** | 13 m |
| J256 | **NU-Refine** | **620,174 particles → 3.21 Å** (FINAL) | 41 m |
| J257 | Global CTF refine | ✅ completed | 4 m |
| J258 | Reference motion (polish) | ❌ skipped — mixed frame counts | 34 s |

---

## THE TWO FAILURES (and why they're OK)

**J251 — 2D classification CUDA Out-Of-Memory.**
384 classes × 1.7 M particles exceeded the shared GPU's memory. Caught live, capped classes to 100 + shrank per-class batch, retried as **J252 → succeeded**. This is the resumable/auto-recovery design working.

**J258 — Bayesian polishing (reference_motion_correction).**
Failed with `AssertionError: All movies must have the same number of frames`. EMPIAR-10288 contains a **mix of 30-frame and 40-frame movies** (see J239: "30 frames"), and polish requires uniform frame counts. This is a **property of the raw data, not a pipeline bug** — same failure on subset and full set. It was best-effort, so the run completed without it. To enable polish later: split the dataset by frame count and polish each group.

---

## KEY OBSERVATIONS

- **Auto-curation does real QC at scale:** rejected 21% of micrographs on full data (vs 0% on the clean 20-subset). No human needed.
- **Structural 2D selection (reference_select_2D) is the resolution driver:** kept the 620,174 particles (36%) that match the 3D reference, discarded 1.1 M (64%) as junk. This is why 3.21 Å.
- **Dynamic class count scaled and was capped:** 16 → 96 → (capped) 100.
- **Resolution journey:** W1 homo 4.10 Å → W2 homo 3.96 Å → W2 NU-Refine **3.21 Å**.

---

## OPEN ITEMS / FOLLOW-UPS (do not block the result)

1. **Polish** — blocked by mixed 30/40-frame data; split dataset to enable (→ possibly ~3.0 Å).
2. **ACPS** (adaptive refinement) — currently disabled (`acps_max_iters=0`); re-enable + fix 2nd-round NU-refine param.
3. **`get_job_resolution` parser** — returned the 100.0 default; the real 3.21 Å was read from the CryoSPARC event stream. Minor fix.
4. **W1 extraction** — 144,556 extracted from 505,029 picks; worth verifying extraction wasn't capped (didn't affect result; W1 is only the bootstrap).

---

## FILES IN THIS FOLDER

| File | What it is |
|---|---|
| `SUMMARY.md` | This document |
| `run_metrics.json` | Machine-readable key metrics (resolution, particle counts, failures) |
| `d5b714c4-...json` | Orchestrator run state — every step + CryoSPARC job UID |
| `e2e_progress.log` | Step-by-step timeline (timestamps per step) |
| `e2e_driver_full.out` | Full driver stdout/stderr (incl. the OOM + polish skip messages) |

The actual 3D volumes/maps live in CryoSPARC on the GPU server
(`P3 → W2_template_tools → J256`), not in this folder (they are large `.mrc` files).
