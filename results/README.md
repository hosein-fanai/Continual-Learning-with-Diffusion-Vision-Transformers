# Experiment results

Training reserves one new directory atomically here, named with a timestamp
and optional `training.project_tag`. A unique suffix resolves concurrent or
same-second collisions. The reserved path is shared by callbacks, configuration,
reporting and default checkpoint writers. Depending on reporting settings, a run can
contain:

- the resolved `config.yaml` and `model.weights.h5`;
- training-history plots and CSV data;
- train/validation evaluation CSV data;
- generated image grids; and
- progressive denoising GIFs under `gifs/` or still images under `images/`.

HPO studies live under `results/hpo/<task>/<model>/<dataset>/` with a resumable
`study.db`, `trials.csv`, and the exact input YAML for every trial. Individual
runs are stored in that study's `runs/` directory. TensorBoard logs use compact
dataset-specific paths under `results/hpo/_tb/`; the parameter-value sequence
in each custom event filename follows alphabetical parameter-name order, while
the event text summary and resolved `config.yaml` retain the complete mapping.
Every successful HPO run also contains `objectives.csv` and an animated
training-history GIF, so classification-only trials have the same artifact
coverage as generative trials.

`common.train.train_model` creates and updates training artifacts;
`common.train.report` writes final reports and samples in that execution's reserved
directory. The allocator never reuses an occupied directory. Preserve existing
experiment records; direct report calls must use their own reserved path. Explicit
task-checkpoint recovery keeps its separately authenticated checkpoint root.
The `old/` and `tests/` groupings contain archived and trial runs respectively.

## Frozen thesis benchmark

The maintained campaign is `thesis_route_one/minimum_v6_tf220_21streams/`.
Its `frozen_design.json` binds two native manifests and 21 run YAMLs;
`execution_checklist.csv` gives their paired seed and randomized execution order.
A prepared design is not a completed result. Native completed-run records and
their validated artifacts establish progress. The user-selected scope is
notebooks 03–09: CIFAR-10 extra-joint/learned and all five CIFAR-100 conditions,
each with seeds 1103, 2207 and 3301. Optional notebook-10 collection requires all
21 streams. Notebook 02 is preserved but its CIFAR-10 platform runs are excluded;
no CIFAR-10 platform comparison can be reported from this plan.

This is a **test-informed benchmark** because earlier official-test HPO informed
the recipe. The current joint LR is 0.001 with cosine decay; primary timestep
ensemble weights are classifier 1.0 and distillation head 0.0. See the
[thesis recipe and interpretation](../notebooks/thesis/HYPERPARAMETER_RATIONALE.md)
for the complete settings and separate reference cosine durations.

The previous `minimum_v5_tf220` 24-run design remains a historical record and
requires its matching archived source for reproduction. The revised 21-run
design has a separate source/design ZIP. The original ZIP and documentation amendments are
retained under `thesis_route_one/frozen_records/`. A documentation amendment
records its prior design digest and changed document hashes; it does not
retroactively change scientific settings, run identities, or training evidence.
The original `freeze_validation.json` is a dated software-check receipt, not a
claim of completed experiments or validation of later document hashes.

Notebooks 11–12 save supplemental offline/naive runs under
`thesis_route_one/reference_benchmarks/`; these are outside the 21-run campaign
and notebook-10 collection. Most result artifacts are Git-ignored. A hosted
launch therefore needs the frozen campaign and matching source snapshot copied
to the active checkout, in addition to the notebook files.
