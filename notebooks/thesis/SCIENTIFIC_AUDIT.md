# Thesis experiment audit — 15 September 2026

## Verdict

**The notebooks can produce a defensible evaluation of the local TMCL-inspired
classifier procedure. They do not yet contain the evidence needed for a positive
efficacy claim, and they do not reproduce TMCL.** No `frozen_design.json`,
`*.completed.json` or `completed_runs.json` was found under `results/` during this
audit. Existing semantic smoke/pilot artifacts use four classes, two tasks and
validation outcomes; they cannot replace the full confirmation campaign.

The minimum question supported by these notebooks is:

> Does transferring temporary class-informed modulations into the ordinary
> classifier head improve retention and final accuracy relative to spending the
> same additional optimizer-update allowance on ordinary joint training?

A negative or inconclusive answer is a scientific result. Neither code review nor
passing tests can promise useful CIFAR learning, convergence, a positive effect,
short runtime or acceptance by a thesis committee.

## What was reviewed

All eleven `notebooks/thesis/*.ipynb` notebooks, both resolved recipes, and their
workflow, completion, presentation and collection helpers were read. The audit
traced these calls through `common.config`, `runtime`, `dataloader`, `model`,
`train`, `learner`, replay, recovery, continual reporting and experiment statistics;
the DiTClassifier and V1 diffusion wrapper; and semantic acquisition,
consolidation, controls, diagnostics, saved inference and study validation.

Repository-wide inventory also identified alternative allocation/gist studies,
V2 wrappers, U-Nets, encoder-decoder models, VAEs, CNN/DNN/HPO paths, historical
notebooks, pretrained feature files and saved weights. Those alternatives are
outside the selected thesis recipe. Broad test coverage and inventory do not
constitute line-by-line or numerical validation of every historical experiment.
The active route does reuse allocation-study artifact validation and gist-memory
tensor accounting; their source identities and helper contracts were inspected.

Search scope: the TMCL author manuscript v3, its published NeurIPS paper and
linked official implementation; local source/configuration, prior recipe source
ledger and saved-result artifacts. This was a methods/implementation audit,
not a systematic literature review or a new state-of-the-art comparison.

## Comparison with TMCL

TMCL studies sparse-label representation learning with internal affine
modulations, view/modulation invariance and post-hoc readouts. Its main CIFAR-100
protocol uses five sessions; Appendix F includes ten. The local experiment uses
full supervision, generated replay, bounded head modulations, squared-cosine
acquisition and CE plus instance InfoNCE. Consolidation changes the classifier
head while the shared backbone stays fixed. Its acquired target is an independent
frozen copy. These are substantial adaptations. [TMCL, Sections 3–4 and
Appendices A, C, F](https://arxiv.org/html/2505.14125v3)

| Proposed conclusion | Supported by this minimum design? |
|---|---|
| Effect on local final/incremental clean accuracy and forgetting | Yes, after all paired full streams complete. |
| Learned procedure versus extra ordinary training | Yes, for equal optimizer-update allowances. Images, FLOPs and time differ. |
| Learned gates versus random gates; alignment versus replacement CE | Supporting CIFAR-100 evidence; neither comparison alone isolates every mechanism. |
| Reproduces or exceeds TMCL's published numbers | No: objectives, architecture, supervision and evaluation differ. |
| Sparse-label efficiency, label-noise robustness or downstream transfer | No matching experiment in these notebooks. |
| Benefit from diffusion noise or reliability weighting | No: semantic `noise_levels=[0]` and uniform reliability. |
| Improvement of the shared denoising representation | No: default consolidation trains only the classifier projection/head. |
| Biological memory or cortical mechanism established | No: the evidence is computational and benchmark-specific. |

Local BWT is final-minus-acquisition accuracy. TMCL's Appendix B BT uses separately
trained single-task references and averages later checkpoints; its FT also needs
pre-learning measurements. Do not present local BWT as those paper metrics.
[TMCL, Appendix B](https://arxiv.org/html/2505.14125v3#A2)

## Scientific safeguards verified in source

- The training partition supplies the stratified 20% validation split. Fixed
  pixel scaling does not fit statistics to future tasks or test data.
- Development disables test evaluation. Confirmation uses the held-out test
  split; tuning and stopping must use development/validation only.
- Each task uses current real training examples; after the first task it also
  uses 2,048 generated old examples.
  Historical validation probes are observation access, not gradient rehearsal.
- The previous completed teacher is independent. Classifier distillation uses
  replay masks, temperature scaling and full expanded student support.
  Noise distillation also includes classifier-free null-conditioned rows,
  including new-class rows whose labels were dropped; its scope is different
  from replay-only classifier distillation.
- Acquisition changes new gates only. Consolidation freezes the gate bank and
  acquired target. The ordinary deployed classifier uses all seen classes,
  without task identity or true-label gate selection.
- Primary result collection authenticates complete task matrices and fixed
  stream identities, refuses incomplete final campaigns, and computes metrics
  within streams before pairing or averaging.

## Values to use in the defense

| Required evidence | Where it comes from | Interpretation |
|---|---|---|
| Final accuracy, incremental accuracy, signed forgetting, local BWT | Notebook 10 compact summary and per-stream numeric source rows | Report all seeds, mean, sample SD and actual n. Accuracy is percent; changes are percentage points. |
| Primary learned-minus-extra-joint effect | `tables/T90_primary_native_interval.csv` | Paired mean difference and 95% t interval, separately for each dataset. |
| Learning versus retention | Saved task accuracy matrices and optional trajectories | Check both old and new performance; low forgetting alone can mean failure to learn. |
| Modulation acquisition and transfer | Optional saved validation phase changes and gate visits | Descriptive checks on fixed examples, not independent efficacy replicates. |
| Update/time/memory costs | Saved resources and task timers | Equal updates are not equal compute; distinguish active time, checkpoint I/O and unavailable lost work. |
| Replay examples | Saved replay grids and diagnostics | Qualitative/internal consistency evidence, not independently verified semantic generation quality. |

Only three independent stream pairs are planned. Tasks, images, gates and epochs
do not increase n. The t interval assumes a suitable distribution of stream
differences; three pairs cannot establish that assumption. The dataset-specific
intervals are unadjusted. Avoid a combined claim selected because either dataset
looks favorable. A zero sample SD from identical paired differences does not
prove zero uncertainty. The test sets are fixed, so repeat variability does not
measure generalization across new datasets.

The eight validation examples per class and at most four probed gates support
small descriptive mechanism checks. Fixed-pixel KID and classifier agreement
are not independent semantic image-quality evaluations. Source hashes detect
changes relative to retained evidence; retain an independent frozen design and
source copy. Optional diagnostic files are bound at export and are not an
independently timestamped measurement archive.

## Minimum execution and writing route

1. Run notebook 00 with seed 17 for platform and learned on both datasets.
   Inspect complete streams, including the final CIFAR-100 task. Check useful
   new-class learning, old-class retention, replay, phase behavior and runtime.
   Record validation-driven recipe changes and keep the same selection process
   for compared methods. Do not choose settings from confirmation outcomes.
2. Freeze the settled recipe with notebook 01. Preserve the record and source.
3. Follow the 24-row checklist using notebooks 02–09, one fresh kernel per row.
   Retain failed, negative and unfavorable runs; resume interrupted streams.
4. Run notebook 10. Use its compact table and primary interval in the results
   chapter; request detailed saved diagnostics only when they answer a defense
   question. These conditions address the narrow comparison above; interpretation
   still depends on adequate learning, complete runs and the resulting uncertainty.
5. Describe the work as a **TMCL-inspired supervised continual diffusion
   classifier adaptation**. Align the thesis methods/claims with this final
   recipe; older chapter records still discuss broader candidate directions.

If the thesis requires noise-dependent consolidation, sparse-label learning,
general transfer or original-paper reproduction, this minimum campaign cannot
answer that claim. It requires a separately planned experiment or narrower
wording; adding a favorable post-hoc analysis would not repair the mismatch.

## Software validation and audit changes

The following issues were corrected without changing the experiment conditions
or training budgets:

- **Pilot checkpoint collision:** changing inherited validation settings kept
  the same checkpoint path. A reproduced 20%-to-25% split change demonstrated
  the collision. Development now binds resolved settings and executable source;
  earlier pilot directories remain untouched.
- **Incomplete final display:** the general metric API can average available
  cells, so a partial notebook matrix could appear as final scalar results.
  The notebook view now requires the complete scheduled matrix before writing
  or displaying results, and labels its split and task count.
- **Missing compact BWT:** the compact summary now includes the already computed
  local backward-transfer metric, with its difference from TMCL's BT explained.
- **Shared classifier API:** a positional `training=True` was interpreted as a
  logits request in `DiTClassifier.compute_class`. The original training-argument
  position is restored. Thesis callers already use keywords. Six explanatory
  branch comments also restore the repository's static source contract.
- **Notebook overhead:** training notebooks now use six small code cells, with
  four main steps and one grouped diagnostic section. Large model summaries and
  repeated optional steps were removed; native data/training/recovery calls remain
  visible. No new experiment conditions or sweeps were added.

See [VALIDATION.md](VALIDATION.md) for actual commands, results and runtime limits.
Full-budget real CIFAR confirmation was not run as part of this audit. The
notebooks and their test fixtures must not be presented as measured thesis
outcomes.

## Sources and assistance

- Tran, Neftci and Wybo. *Contrastive Consolidation of Top-Down Modulations
  Achieves Sparsely Supervised Continual Learning.* [Author manuscript v3,
  26 January 2026](https://arxiv.org/html/2505.14125v3) and
  [NeurIPS 2025 paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/dd1fef536655685898a6602bfbf16857-Paper-Conference.pdf).
- [TMCL official implementation](https://github.com/Dendritic-Learning-Group/tmcl).
  Local source-setting provenance remains in `recipe_sources.json`.
- This working audit used Codex with notebook, scientific-critical-thinking and
  statistical-analysis guidance. Procedural guidance citation: Kassis, Agarwal,
  He, Patel and Brueckner (2026), *Scientific Agent Skills: A Library of Procedural
  Knowledge for Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
  This acknowledges assistance; it is not evidence for the method's efficacy.
