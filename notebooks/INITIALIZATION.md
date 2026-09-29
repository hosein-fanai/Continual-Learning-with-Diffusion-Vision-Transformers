# Shared notebook initialization

[`setup_cell.py`](setup_cell.py) is the canonical first-cell source copied into
maintained notebooks and generated HPO notebooks. [`init.py`](init.py) contains
checkout discovery, download, import paths and
runtime preparation. Shared-bootstrap notebooks have a small standalone loader: it loads the
local initializer or downloads that file first when the checkout is absent.
The experiment's own code then runs with its original settings and outputs.

GitHub reports the canonical repository as
[`hosein-fanai/Continual-Learning-with-Diffusion-Vision-Transformers`](https://github.com/hosein-fanai/Continual-Learning-with-Diffusion-Vision-Transformers).
`CHECKOUT_NAME` uses that exact spelling and `REPOSITORY` is an f-string built
from it. The old underscore URL redirects to this repository.

Their first code cell exposes `REVISION`, `RUNTIME` and `CUDA`. See the
[hosted runtime guide](thesis/README.md#hosted-runtimes) for provider setup.
An existing checkout is reused without pulling updates or changing its revision.
The loader needs Internet access only if its local initializer is absent;
dependency installation and missing dataset downloads may also need Internet.

Tracked notebooks begin with three Markdown launch buttons for Colab,
Kaggle and Binder, plus a Studio Lab text link for existing accounts. Each link
targets that notebook's own path on GitHub
`main`, including URL-encoded spaces. Publish the notebook, shared setup files
and `.binder` configuration together before using the remote launch links.
Unpublished notebooks can instead be uploaded to Colab or Kaggle; their setup
still requires the shared initializer to be available locally or on GitHub.
Binder is for small CPU checks, and Studio Lab requires an existing account.

The archived `thesis/runs/00_Development2.ipynb` retains local-only initialization. Its
first code cell requires an existing checkout and a prepared TensorFlow 2.20 /
Keras 3.11.2 kernel; it does not download files, install dependencies or expose
the shared runtime settings. Use `thesis/00_Development.ipynb` for automatic
hosted setup.

Historical `import init` calls remain supported from the repository root,
`notebooks` and `notebooks/thesis`. They set the working directory to the root,
and add the root and thesis helpers to `sys.path`, without importing TensorFlow
or changing device configuration. Explicit `prepare_notebook()` owns checkout
discovery and runtime preparation before the notebook's scientific imports.

Older notebooks retain their original scientific code and optional dependencies,
including Hyperas or Avalanche where used. Shared startup does not establish
that those historical experiments fully run with TensorFlow 2.20.

The startup regression inventory follows Git's tracked and nonignored notebooks.
Ignored run copies, checkpoint notebooks and scratch artifacts are historical
execution evidence and are not rewritten by maintenance. The test checks every
selected loader against `setup_cell.py`; a copied notebook can still be run
independently because the bootstrap source is embedded in its first cell.

Run `python -m unittest discover -s notebooks/thesis/tests` in a separate prepared
interpreter for bootstrap, synthetic workflow/recovery, result provenance and
bounded reference-fit checks. Full multi-seed research runs are a separate
experiment, and passing software tests does not establish their outcomes.
