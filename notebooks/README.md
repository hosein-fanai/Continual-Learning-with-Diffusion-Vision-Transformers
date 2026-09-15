# Notebook examples

The root-level DiT, U-DiT, classifier, and variational examples explore the
project's existing model APIs. Their saved code and outputs may describe
different experiments. Select a TensorFlow 2.20 kernel and use a fresh process
when checking reproducibility; historical outputs are not new validation results.

## Execution

When starting from this directory, `import init` resolves the repository root,
changes the working directory, and makes project imports available. Its
`common.utils.init()` compatibility call does not configure GPU memory. Device
settings belong to the runtime setup described in the
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

The [thesis workflow](thesis/README.md) provides the maintained, reduced
confirmation experiment and compact result collection. Its
[validation record](thesis/VALIDATION.md) separates software checks from measured
scientific results. Root notebooks were preserved during the latest repairs;
`hpo`, `old`, `legacy`, and filenames containing `copy` remain outside that review.

The root `test.ipynb` is a historical Avalanche/PyTorch and legacy API scratchpad;
it is not a TensorFlow 2.20 entry point. Some preserved archive banners still link
to the removed `thesis_development.ipynb`. Use the current
[semantic route guide](../semantic_consolidation/README.md) for the supported
workflow. Notebook schema and syntax checks do not make those historical cells,
kernel metadata, or links current.
