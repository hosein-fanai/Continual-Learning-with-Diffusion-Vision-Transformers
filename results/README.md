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
