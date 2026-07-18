# CryoEMAgent — Project Summary

## What this project is

This project builds an autonomous system that takes raw cryo-electron microscopy
data — thousands of noisy movie files straight off a microscope — and turns it
into a finished 3D structure of a protein, without a person clicking through the
usual chain of software steps. The underlying processing engine is CryoSPARC, the
standard tool labs use for this kind of reconstruction. Normally a person runs
CryoSPARC by hand: import the movies, correct for beam-induced motion, estimate
the microscope's optical distortion, pick out particle images, sort them into
2D classes, throw away the junk, build a rough 3D model, then refine it. Every
one of those steps involves a judgment call — which particles look clean, which
classes are junk, when to stop refining — and normally a trained person makes
those calls.

The system described here tries to make those calls automatically, and it also
tries to make them better than the fixed rules other automation tools use. Two
separate codebases came out of this: one is a working operator that actually
drives CryoSPARC end to end on a real dataset, and the other is a more ambitious
agent framework built around a large language model making the decisions. Both
are described below, along with what has actually been proven to work and what
hasn't.

---

## The dataset used to test everything

All the real experiments in this project run on EMPIAR-10288, a public dataset
of cryo-EM movies of the CB1 cannabinoid receptor bound to a G-protein complex.
It's 2,756 raw movies, about 476 gigabytes. This is real, already-published data
— the structure from this exact dataset was solved by the original researchers
and deposited as PDB 6N4B / EMD-0339, at a resolution of 3.0 Ångströms (measured
by the standard gold-standard Fourier Shell Correlation method at the 0.143
threshold — lower numbers mean a sharper, more detailed map). That published
number is the benchmark everything here gets compared against.

---

## Part one: the operator that actually runs on the cluster

### The core idea

The working codebase (kept in a repository called the MCP server) is a
deterministic pipeline driver. It knows the fixed sequence of CryoSPARC jobs
needed to go from raw movies to a finished structure, and it walks through that
sequence automatically — creating each job, submitting it to the GPU cluster,
waiting for it to finish, checking whether it succeeded, and moving to the next
one. There are two parallel tracks inside it, following an approach borrowed from
existing automation tools: a first pass that picks particles with a generic
blob detector to bootstrap a rough 3D reference, and a second, more accurate pass
that uses that reference as a template to re-pick particles more precisely and
carry them through to a final high-resolution structure.

None of the usual manual stopping points exist in this version. Where a person
would normally sit down and look at CTF quality plots, particle picks, or 2D
class averages and decide by eye what to keep, this pipeline either applies a
fixed numeric threshold automatically or (in the one case where CryoSPARC's own
software architecture allows it) uses a small custom job that filters
programmatically instead of waiting for someone to click through a review
screen. The result is that from the moment the raw movies are pointed at, the
pipeline can run unattended from start to finish.

### How resolution is actually read out

One early, easy-to-miss problem: when you ask CryoSPARC for the resolution of a
finished job through its normal API, the field that looks like it should hold
the answer often just contains a placeholder value (100.0), because CryoSPARC
doesn't reliably populate that field for every job type. The real, final
resolution number only shows up in the job's own text log, in a specific line
that says the map was filtered to the gold-standard resolution. The fix here was
to parse that log directly — pull the event stream for a job, search for that
specific phrase, and take the number out of it. This was checked against jobs
where the true answer was already known from the CryoSPARC web interface, and it
matched exactly. Without this fix, every automated decision downstream would have
been working from a fake number.

### The adaptive selection algorithm — the main scientific contribution

The one piece of genuinely new algorithmic work in this project is called ACPS
(adaptive closed-loop particle selection). The problem it solves: after 2D
classification, you have a large pool of particle images, some clean and some
junk, and you need to decide how many to keep for the final 3D reconstruction.
Every existing automated tool uses a fixed rule for this — keep the top X% by
some quality score, always the same X regardless of the dataset. That's brittle:
too strict and you throw away usable signal, too loose and junk drags the
resolution down.

ACPS instead treats this as a small feedback-control problem. It tracks a
threshold value, refines the particles it keeps, looks at the resulting
resolution, and adjusts the threshold up or down depending on whether the result
was better or worse than a target: if the resolution came out worse than target,
loosen the threshold and let more particles in; if it came out better, tighten
it and keep only the cleanest particles. It repeats this for a small, capped
number of iterations, and at every step it remembers the best result seen so
far — if a new attempt turns out worse, the algorithm quietly discards it and
keeps the earlier, better one. That last property matters a lot: it means
turning this feature on can never make the final result worse than not using it
at all, which is what makes it safe to run unattended.

The control law itself is a few lines of arithmetic (the new threshold is the
old one, nudged by a learning-rate constant times the gap between current and
target resolution, clipped to a safe range) and it has its own small test suite
that checks the basic behavior in isolation — that a worse-than-target result
lowers the threshold, a better one raises it, that it respects its upper and
lower bounds, and so on. That part was validated early and easily.

What took much longer, and is the part worth being honest about, is that the
knob the algorithm turns needed real investigation. CryoSPARC's reference-based
selection job has a parameter that looks like the natural thing to control (a
minimum correlation score), but under the default selection mode that parameter
turned out to have no effect at all on the outcome — tested directly on the live
system, two very different values of it produced the exact same particle count.
The knob that actually works is a different selection mode entirely, where you
specify what percentage of particles to keep by their reference-correlation
ranking. That one was checked and confirmed to behave predictably (a smaller
percentage kept meant a smaller, higher-quality set every time).

### What the real run on this dataset showed

Once that piece was sorted out, the algorithm was tested for real, not in
simulation. Using the same particle pool and the same 3D reference throughout, so
that the only thing changing between runs was the selection rule, several
selection percentages were tried and their resulting resolutions recorded:

- The fixed, non-adaptive selection rule (what the pipeline uses by default)
  produced 620,174 particles and a resolution of 3.214 Å.
- Keeping the top 52% by correlation gave 1,280,400 particles and 3.134 Å.
- Top 55% gave 1,307,208 particles and 3.120 Å.
- Top 58% gave 1,318,388 particles and 3.110 Å — the best point found.
- Top 62% gave 1,349,784 particles and 3.179 Å — worse again.

That last point matters as much as the improvement itself: it shows there's a
real optimum in the middle, not a straight line where more particles always
helps. The adaptive algorithm found that optimum and stopped there rather than
continuing to load in more (increasingly junky) particles, which is exactly the
behavior it was designed to produce. Compared to the fixed rule, this is a
genuine, measured improvement of about a tenth of an Ångström, achieved purely by
changing how particles are selected — same data, same compute, same everything
else.

That result on its own still left a gap to the published reference (3.110 Å
versus the deposited 3.0 Å). The published structure used two additional
refinement steps this pipeline hadn't completed: a global correction pass on the
per-particle optical model, and a per-particle motion re-tracking pass (often
called polishing) that re-derives how each individual particle moved during
imaging rather than assuming the whole micrograph moved together. Running those
two steps on top of the adaptive-selection result, then doing one more
reconstruction pass to read out the true resolution afterward, brought the final
number to 3.008 Å — a difference from the published 3.0 Å that's smaller than
the normal run-to-run noise in this kind of measurement. In plain terms: the
fully automated pipeline, with no person involved, reached essentially the same
resolution as the original manually-supervised structure determination.

It's worth being precise about what this does and doesn't claim. It does not
claim the automated pipeline beat the published result — it matched it, within
noise. And the fixed-selection baseline used for comparison was deliberately run
without those two extra refinement steps, specifically so the adaptive
algorithm's contribution could be isolated and measured on its own; comparing
that baseline directly to the published number was never a fair comparison, and
saying so plainly avoids the kind of overclaiming that undermines this kind of
work.

### The non-blocking quality critic

A second piece sits alongside the pipeline: at a handful of checkpoints
(exposure curation, particle selection, final refinement) the pipeline records a
quality verdict — pass, warning, or fail — based on the actual numbers it just
produced. This never stops or changes the pipeline's behavior; it's purely an
audit trail for a person to read afterward. In its simplest form this is a
handful of deterministic threshold rules (acceptance rate below fifty percent is
a warning, CTF fit worse than seven Ångströms is a fail, and so on) that need no
external service and always work. When an API key for a language model is
present, the same checkpoint is optionally re-evaluated by actually sending the
real numbers to the model and asking it to reason about them, and its answer —
verdict, confidence, and a short explanation — replaces the rule-based one. If
that call fails for any reason, the rule-based verdict is used instead; nothing
about the pipeline's execution depends on the language model being reachable.

This was tested with a real call to a hosted model, not just written and left
untested. It produced a real trace of the model reasoning over genuine pipeline
metrics, and it's worth flagging honestly that the model made at least one clear
factual mistake during that test — asked to compare two resolution numbers, it
stated the worse one was "better," inverting the basic convention that a smaller
number means a sharper map. That's a real, reproducible failure mode worth
documenting rather than hiding, since it's exactly the kind of thing that matters
if this kind of critic were ever trusted to make unsupervised decisions.

### A benchmark of processing decisions

A separate piece of work is a small multiple-choice benchmark: thirty short
scenarios describing a real situation that comes up during this kind of
processing (a particle count that's too low, a resolution number with a
suspicious mask artifact, a symmetry mismatch, and so on), each with four
possible responses and one correct one, with a short written justification for
why. Several of the scenarios use the actual numbers produced by the real runs
described above, rather than invented figures. Two language models were scored
against it and both got the same one question wrong out of thirty — a question
that specifically requires understanding which direction a control-law threshold
should move, rather than recalling a rule of thumb. It's worth noting that the
first version of this benchmark had a real flaw: every correct answer happened to
sit in the same position across all thirty questions, which meant a model could
score perfectly just by guessing that position every time. That was caught,
and the answer positions were shuffled randomly before the scores above were
taken as final.

### Infrastructure problems along the way, briefly

A meaningful chunk of the time on this project went into fighting infrastructure
rather than the actual research questions, and it's worth naming honestly since
it explains gaps in the timeline. The shared cluster's CryoSPARC installation had
its database quietly die at some point, apparently from an unclean shutdown, and
because of how the underlying database engine stores its files, that failure
turned into something that no amount of restarting or repairing from a normal
user account could fix — every recovery attempt got stuck at exactly the same
point. The eventual fix was to give the installation a brand new, empty database
and re-attach the existing project folder to it (the actual scientific results —
every particle stack, every 3D volume — live as ordinary files on disk, separate
from that database, and were never at risk). After that rebuild, two more mundane
things had to be re-registered before real jobs could run again: the compute
node needed to be told which GPUs it owned, and one particular job type's local
disk-caching option had to be turned off because the dataset was larger than the
available cache space allowed.

A second, smaller but genuinely interesting problem came up during the final
polishing step: it failed with a complaint that the input movies didn't all have
the same number of frames. Investigating properly (rather than guessing)
revealed that exactly two of the 2,753 movies genuinely did have a different
frame count — the very first two files in the original public dataset download,
for reasons that predate this project entirely. The fix was to build a small
filtered copy of the exposure list that excluded just those two, and feed that
into the polishing step instead of the original list.

---

## Part two: the language-model-driven agent framework

### What it's meant to do differently

Separately from the deterministic operator above, a second, more ambitious
codebase exists that puts a language model in charge of the actual
decision-making at every step, not just as an optional after-the-fact auditor.
The idea is a loop, in the same spirit as recent "reasoning and acting" agent
designs: the system observes the current state of the pipeline and the quality
metrics so far, asks a language model to reason about what that means and which
action to take next, and either proceeds automatically or pauses for a person if
the model isn't confident enough. This part supports both major LLM providers,
falls back safely to a default "keep going" decision if the model call fails or
returns something unparseable, and keeps a running log of every decision the
model made and why, so the whole run can be reviewed afterward as a reasoning
trace rather than just a list of completed jobs.

Alongside the text-only decision loop, there's a vision-capable version of the
same idea aimed specifically at the checkpoints where a person would normally
look at an actual image (a CTF power spectrum, a sheet of 2D class averages) and
judge it by eye. That component can take the real image from CryoSPARC and hand
it to a vision-capable model along with the relevant metrics, and only
auto-approves a checkpoint when the model's stated confidence clears a
deliberately high bar; anything less confident is left for a person, with
specific instructions on what to look for.

### What has and hasn't been proven for this half of the project

This framework is built out fairly completely — the planning logic, the memory
of past decisions, the checkpoint-specific prompts tuned to this specific kind of
protein and this specific pipeline, the vision-based assessment path, all exist
as real, callable code, and there's a unit test suite that exercises the
control-law and decision-parsing logic in isolation. What it does not have,
honestly, is a completed run against the live cluster where the language model
was actually in the driver's seat making every decision on real data end to end.
The real, hours-long run described in part one used the deterministic pipeline,
not this framework; this side has been validated with mocked and simulated
inputs, plus the standalone critic test against a real hosted model described
above, but not yet as the thing actually steering a full live reconstruction.
That's a fair and important distinction to keep straight, and stating it plainly
here is better than letting the two get blurred together.

---

## What's proven, what's built-but-unproven, and what doesn't exist yet

To keep the state of things unambiguous:

**Proven with real data and real numbers:** the deterministic pipeline runs
end-to-end on the full dataset with no human clicks; the resolution-reading fix
is verified against known-correct values; the adaptive selection algorithm has a
real, multi-point, honestly-reported result showing it beats the fixed
threshold and finds a genuine optimum; the full pipeline including the extra
refinement steps reaches a resolution matching the published structure within
normal measurement noise; the quality critic makes a real call to a hosted
language model and produces a genuine (imperfect) reasoning trace; the decision
benchmark is built, de-biased, and scored against two real models.

**Built and unit-tested, but not yet run live end to end:** the full
language-model-driven decision loop and its vision-based checkpoint assessment.
The logic exists and passes its own tests against simulated data; it has not yet
been the thing actually making every call on a real, full-scale cluster run.

**Not started:** a dedicated failure-recovery mechanism that reasons about why a
step failed using a structured picture of how failures in this domain typically
cascade (as opposed to simply catching an exception and moving on, which the
deterministic pipeline already does in a few places); and the full,
two-condition experiment specifically designed to test whether a vision model is
actually looking at an image versus just reading the accompanying numbers back
to itself (some suggestive evidence of the latter turned up by accident during
the critic testing described above, but the dedicated experiment to properly
characterize it hasn't been run).

---

## What to look at in the code

The deterministic pipeline lives in a repository referred to internally as the
MCP server. Its core logic is a small number of files: one that talks to
CryoSPARC directly (creating jobs, submitting them, and — importantly — reading
the true resolution out of each job's log rather than trusting the API field
that's often just a placeholder); one that encodes the fixed sequence of
pipeline steps and contains the adaptive-selection feedback loop; a short file
holding just the adaptive algorithm's control-law arithmetic on its own, cleanly
separated so it can be tested in isolation; the quality-critic file with both
the rule-based checks and the optional real-language-model escalation; a small
state-persistence layer so a long-running pipeline survives being interrupted
and resumed; and a thin wrapper exposing all of this as callable tools over the
Model Context Protocol, so a remote client can drive the whole thing. On top of
that core sit a handful of standalone scripts, written to run the specific
real-data experiments described above directly against the live cluster —
the adaptive-selection ablation, the multi-point resolution curve, the
gap-closing refinement run, and the language-model critic trace — each one
self-contained and runnable independently of the main pipeline.

The language-model-driven agent framework lives in a separate repository. Its
core is the control loop that alternates between asking the model for a decision
and executing a pipeline step; the prompting and decision-parsing logic that
talks to either of the two major LLM providers; the vision-based checkpoint
critic described above; a memory module that keeps track of past decisions and
quality snapshots so the model has context; and a domain-specific "playbook"
encoding the particular workflow and quality thresholds relevant to this protein
family. A mock version of the CryoSPARC interface exists specifically so this
side of the project can be exercised and tested without needing a live cluster
connection, which is how most of its current test coverage was obtained.

Alongside both of these sits the small decision-benchmark file described above,
and a written paper draft that lays out the algorithm, the theoretical
formulation of this whole problem as a sequential decision process, and the
experimental results, with the real numbers from these runs already filled in
where they exist and honest placeholders left wherever a result is still
pending.
