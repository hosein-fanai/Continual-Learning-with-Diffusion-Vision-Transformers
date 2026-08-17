# Notebook initialization helper

Importing this package from a notebook whose working directory is
`notebooks/` moves the process to the repository root, imports the local
`autoencoder` and `diffusion` packages, and calls `common.utils.init()`. That
helper attempts to cap TensorFlow's first GPU logical device at 6,144 MiB.

```python
import init
```

The import has process-wide side effects and takes no arguments. Do not use it
when the working directory is already the repository root: its relative
`os.chdir("../")` would move one directory too far upward. Scripts should
instead use explicit imports and call `common.utils.init()` directly.
