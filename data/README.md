# Pre-extracted datasets

This directory can store NumPy feature artifacts used by configured and direct
dataset loaders. The two large files currently present in this workspace
contain Xception global-average-pooled CIFAR-10 and CIFAR-100 features in
train/validation/test order; data payloads are ignored by Git and are not
included by a source checkout.

The two currently present files predate the safe archive format and are legacy
object-NPY payloads. The default loader intentionally refuses them because
unpickling can execute code. If and only if a file is trusted, migrate it once
through the explicit compatibility path:

```python
from common.utils import load_samples, save_samples

base = "data/cifar10_xception_gavgpooled_features_train_val_test"
legacy_bundle = load_samples(base, ".npy", allow_pickle=True)
save_samples(
    legacy_bundle,
    "data/cifar10_xception_gavgpooled_features_train_val_test_safe",
    ".npy",
)

train_features, val_features, test_features = load_samples(
    "data/cifar10_xception_gavgpooled_features_train_val_test_safe", ".npy"
)
```

`allow_pickle=True` emits a warning and must never be used for an untrusted
file. `save_samples` writes the heterogeneous numeric splits as a non-pickled
NPZ container while retaining the public `.npy` filename convention. The
dataset loaders do not expose the legacy opt-in, so feature-mode and HPO paths
can consume these legacy artifacts only after migration. They infer the
class count by looking for `mnist_`, `fmnist_`, `cifar10_`, or `cifar100_` in
the feature path, so preserve the matching marker when adding compatible
files. These artifacts are large inputs, not generated test fixtures.
