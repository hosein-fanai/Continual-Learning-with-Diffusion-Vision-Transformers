# Notebook initialization helper

Start each experiment notebook with:

```python
import init
```

All working-directory and Python search-path setup lives in `__init__.py` here.
It resolves the repository from its own file location, adds the repository and
thesis helpers to the import path, changes to the repository root, and calls
`common.utils.init()`. Small forwarding modules support starting from the
repository root or `notebooks/thesis` as well as `notebooks`.

The resolved path is available as `init.REPOSITORY_ROOT`. Initialization runs
on import; no separate `init()` call is needed in a notebook.

The runtime helper caps the first TensorFlow GPU at 6,144 MiB. If more GPU
memory is needed, change `memory_limit=6144` in `common/utils.py` before starting
a fresh kernel.
