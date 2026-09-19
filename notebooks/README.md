# Notebook examples

The root-level DiT, U-DiT, classifier, and variational examples explore the
project's existing model APIs. Their saved code and outputs may describe
different experiments. Select a TensorFlow 2.20 kernel and use a fresh process
when checking reproducibility; historical outputs are not new validation results.

## Execution

When starting from this directory, `import init` resolves the repository root,
changes the working directory, and makes project imports available. Its
setup uses only the standard library. Device settings belong to the runtime setup described in the
[development-container guide](../.devcontainer/README.md).

`DiffusionModel` and `DiffusionClassifier` use ordinary `fit` keyword arguments.
`DiffusionClassifierV2.fit` takes separate `gen_kwargs` and `clf_kwargs` mappings
and returns a merged history dictionary. Its `evaluate(eval_both=True, x=...)`
checks both phases. Supported depth curricula use the existing `fit_progressively`
API; see the [growth contract](../diffusion/README.md).

## Data and interpretation

The included MNIST examples retain local preprocessing helpers and use the
official test arrays as validation. That workflow is exploratory: displayed
validation results must not be presented as an untouched final test evaluation.
For controlled studies, the [common data pipeline](../common/README.md) and
[semantic study API](../semantic_consolidation/README.md) provide explicit
training/validation separation and confirmation protocols.

The [thesis workflow](thesis/README.md) provides the approved fixed,
test-informed benchmark through **notebooks 03–09 only**: three paired repeats
per notebook, one next unfinished repeat per fresh-kernel launch, for **21 streams**.
They use `results/thesis_route_one/minimum_v6_tf220_21streams/`. Notebook 01 can
reproduce preparation but is unnecessary when the prepared campaign is supplied;
notebook 10 provides optional saved-results collection. Notebook 02 remains the
unchanged historical 24-stream entry point. Supplemental notebooks 11/12 are
outside the current scope. The current plan includes no CIFAR-10 platform
comparison. The
[recipe rationale](thesis/HYPERPARAMETER_RATIONALE.md) distinguishes current settings,
historical settings, and software checks from measured scientific results.
Its primary endpoint is the raw primary-head timestep ensemble with uniform
averaging; supplemental references have their own cosine training budgets.
Root notebooks were preserved during the latest repairs;
`hpo`, `old`, `legacy`, and filenames containing `copy` remain outside that review.

The root `test.ipynb` is a historical Avalanche/PyTorch and legacy API scratchpad;
it is not a TensorFlow 2.20 entry point. Some preserved archive banners still link
to the removed `thesis_development.ipynb`. Use the current
[semantic route guide](../semantic_consolidation/README.md) for the supported
workflow. Notebook schema and syntax checks do not make those historical cells,
kernel metadata, or links current.
