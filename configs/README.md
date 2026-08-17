# Training configurations

YAML files in this directory provide keyword overrides for the dataclasses in
`common.config`. `load_config(path)` converts nested mappings into a `Config`
tree; omitted sections and fields keep their dataclass defaults. Unknown keys
are rejected by the corresponding dataclass constructor.

The top-level sections are:

- `dataset`: `batch_size` and `shuffle_buffer` for the MNIST `tf.data` input.
- `model`: architecture selection, optional weight path, raw-network settings,
  and wrapper settings.
- `optimizer`: initial learning rate and optional cosine-decay step count.
- `training`: epoch count, validation use, result directory, image/GIF display,
  and weight persistence.
- `reporting`: history plots/CSV, final sample controls, and train/validation
  evaluation switches.

```python
from common.config import load_config, save_config

config = load_config("configs/default.yaml")
print(config.model.with_classifier)
print(config.model.dit_classifier.kwargs())
save_config(config, "results/resolved-config.yaml")
```

Nested model dataclasses expose `kwargs() -> dict[str, Any]`; those dictionaries
are forwarded to the corresponding transformer or wrapper constructor. Use the
constructor docstrings as the authoritative valid-value reference. Several
`mnist_config copy*.yaml` files and saved-model configs record experiments from
earlier API revisions and may contain legacy key names. Treat `default.yaml` as
a historical template and compare every forwarded field with the current
dataclasses and model constructors before training.

The YAML examples are experiment inputs, not an additional default layer:
`load_config(None)` uses only dataclass defaults. `default.yaml` also retains
legacy transformer/classifier names, so translate those fields using
`common.config` and the current constructor documentation before running it.

`project_tag: null` lets the image callback generate a timestamp. A non-null
tag reuses that named result directory. `weights_path: null` starts with newly
initialized weights; otherwise it must identify a Keras-compatible weights
file for the selected architecture.
