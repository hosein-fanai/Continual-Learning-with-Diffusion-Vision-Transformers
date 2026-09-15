# TensorFlow 2.20 development container

The configuration builds on the official TensorFlow 2.20 GPU Jupyter image and
installs `requirements.txt`. It uses native Keras 3 with the TensorFlow backend.
GPU access requires a compatible NVIDIA driver and container runtime.

## Open and test

Open the repository in VS Code and choose **Dev Containers: Reopen in Container**.
For notebooks, select **TensorFlow 2.20 (Docker GPU)**. The project is mounted at
`/workspace`; the configured interpreter is `/usr/bin/python`. These container
conventions are independent of the host checkout location.

With the Dev Containers CLI, run from the repository root:

```sh
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . /usr/bin/python -m pip check
devcontainer exec --workspace-folder . /usr/bin/python -m unittest discover -s common/tests -t .
devcontainer exec --workspace-folder . /usr/bin/python -m unittest discover -s semantic_consolidation/tests
devcontainer exec --workspace-folder . /usr/bin/python test.py
```

Run tests in separate processes: they reset Keras and random state. The registry
runs the complete model/layer self-tests; start with a focused module when
debugging. Verify installed versions rather than relying on the image tag.

If `devcontainer` is unavailable, the VS Code Dev Containers extension includes
`dist/spec-node/devContainersSpecCLI.js`. Run that installed file with Node.js
and the same CLI arguments. Locate the extension on the current installation
instead of copying another user's absolute path.

## Runtime and persistence

GPU memory growth is enabled through `TF_FORCE_GPU_ALLOW_GROWTH=true`. The
notebook initializer does not impose a fixed cap. Configure logical-device
limits explicitly before the first GPU operation if needed; memory growth and
virtual-device limits cannot be configured together on one physical device.

Changes under `/workspace` persist in the checkout. Manually installed packages
survive restarts, but a rebuild replaces the container's writable layer. Preserve
active notebook sessions and packages when testing. Reuse a matching container;
rebuild only when intentionally applying an environment change. The configured
container stops when its last editor window closes.

Verify an existing container's checkout/config labels and workspace bind mount
before testing. For a manually created container without those labels, verify
the exact mount, installed packages, and image identity independently, and record
that difference in the execution evidence.

See the [compatibility guide](../compatibility_migration.md),
[TensorFlow Docker documentation](https://www.tensorflow.org/install/docker),
and [Dev Containers CLI documentation](https://code.visualstudio.com/docs/devcontainers/devcontainer-cli).
