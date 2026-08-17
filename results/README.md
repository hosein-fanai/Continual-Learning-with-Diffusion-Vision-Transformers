# Experiment results

Training callbacks create one project directory here, normally named with a
timestamp or the configured `training.project_tag`. Depending on reporting
settings, a run can contain:

- the resolved `config.yaml` and `model.weights.h5`;
- training-history plots and CSV data;
- train/validation evaluation CSV data;
- generated image grids; and
- progressive denoising GIFs under `gifs/` or still images under `images/`.

`common.train.train_model` creates and updates training artifacts;
`common.train.report` writes final reports and samples. Existing timestamped
directories are immutable experiment records and do not expose Python APIs.
The `old/` and `tests/` groupings contain archived and trial runs respectively.
