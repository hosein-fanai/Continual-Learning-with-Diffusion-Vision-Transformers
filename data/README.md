# Pre-extracted datasets

This directory stores NumPy feature artifacts used by the legacy continual-
learning data loaders. The tracked files contain Xception global-average-
pooled CIFAR-10 and CIFAR-100 features in train/validation/test order.

Load them through the public common helpers rather than assuming an on-disk
array layout:

```python
from common.utils import load_samples

train_features, val_features, test_features = load_samples(
    "data/cifar10_xception_gavgpooled_features_train_val_test",
    ".npy",
)
```

`load_samples` appends the extension and returns the stored NumPy object. The
continual-learning loaders infer the class count by looking for `cifar10_` or
`cifar100_` in the feature path, so preserve that marker when adding compatible
files. These artifacts are large and are inputs, not generated test fixtures.
