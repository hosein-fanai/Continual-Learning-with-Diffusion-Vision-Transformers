# Held-out experimental diagnostics

`route.experimental` observes training without adding an optimization objective.
Its online inputs come from the validation portion of the training dataset.
The common learner owns final continual score matrices and separately controlled
test evaluation. Observation settings are saved with the complete route settings.

```yaml
route:
  experimental:
    enabled: true
    probe_per_class: 8
    generation_per_class: 8
    batch_size: 32
    ece_bins: 15
    feature_extractor: fixed_pixels
    learning_curves: true
```

Use `configs/section11_smoke.yaml` for a small executable configuration. Sample
counts are caps. A requested count does not imply that many observations were
available, and a smoke run does not establish benchmark performance.

## Observation lifecycle

1. `ExperimentalController.before_task` maps the supplied validation labels into
   the current classifier vocabulary and optionally appends an epoch callback.
2. Each requested learning-curve observation evaluates the clean, unconditional
   primary classifier and records the joint optimizer iteration count.
3. After all joint and semantic updates, `after_task` measures clean outcomes,
   fixed hidden-feature changes and actual generated replay candidates.
4. After the common learner installs the next-task teacher, `teacher_boundary`
   records the replacement teacher separately from the previous training teacher.
5. The runner closes the resource monitor and exports the completed records.

All observations use inference-mode network calls. Their time and retained data
are reported separately, while ordinary task elapsed time already includes the
online hooks. Adding nested times to common task time would double-count work.

## Classification and hidden features

Classification reports include accuracy, old/new accuracy, per-class recall,
confusion counts, negative log-likelihood, Brier score and expected calibration
error (ECE). Confusion rows are true dense classes and columns are predictions.
ECE uses the configured number of equal-width maximum-confidence bins. Missing
old/new populations have unavailable subgroup accuracy rather than an invented
zero.

`FixedHiddenProbe` chooses a content-canonical, seeded validation cohort when
each class first appears. It compares the same observations against both that
class's acquisition checkpoint and its previous measured checkpoint. The default
retains those validation pixels exclusively for diagnostics and reports their
bytes. The lower-level `retain_images=False` option retains fingerprints and
features; unavailable later observations are reported, never substituted.

The feature extractor is the real clean hidden projection immediately before
the primary classifier output. The unmodulated representation is measured without
the temporary consolidation predictor. Linear centered-kernel alignment (CKA)
can compare different feature widths. Coordinate and centroid drift require
matching widths, and relative norm drift requires a nonzero reference norm.
CKA is unavailable for fewer than three aligned rows or a constant centered
representation. Small-cohort CKA remains descriptive even above that threshold.
Class acquisition references occur at different checkpoints, so the observer
does not pool them into one cross-class acquisition CKA.

## Generated samples

The adapter captures images actually produced by the existing replay sampler.
A seeded uniform reservoir retains at most `generation_per_class` occurrences
per old class; it does not select favorable examples by classifier score.
The report separates the full generated occurrence counts from this sampled
diagnostic population. Captured class conditions identify the generator's
requested labels, not independently verified image semantics.

Per-class outputs include coverage, duplicates, bounded diversity, classifier
label agreement and a cubic-kernel unbiased two-sample MMD-squared estimate.
At least two generated and two validation observations are needed for that
distribution estimate. Negative finite-sample estimates are retained. The
default fixed-pixel feature space makes this a pixel-distribution diagnostic;
it is not standard Inception KID. A custom frozen extractor is available only
through the explicit diagnostic API and must declare identity, pretraining and
preprocessing. The evaluator never downloads or fits an extractor.

The macro distribution score averages measured classes equally. Missing classes
remain explicit. Neither classifier agreement nor a favorable distribution
score establishes that replay improves retention.

## Resources and artifacts

`section11.json` records settings, task observations, learning curves, tensor
inventory, retained diagnostic arrays and timing scopes. Optional
`learning_curves.csv` contains scalar epoch observations.
`generated_examples_task_*.npz` stores sampled image/label arrays for inspection.
These diagnostic arrays are retained until export and are included in the
reported retained-audit bytes.

Tensor inventory counts actual numerical payloads and deduplicates shared
variable identities. It is distinct from process memory. When available, a
background monitor samples process resident memory and records TensorFlow
allocator peaks. Sampling can miss short process peaks, and allocator peaks
are not total device occupancy. Unsupported measurements remain unavailable.

See [ASSESSMENT.md](ASSESSMENT.md) for verification scope and
[README.md](README.md) for the core method, inference and paired study workflow.
