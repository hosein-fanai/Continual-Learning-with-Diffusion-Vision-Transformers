# TensorFlow 2.17 GPU in VS Code

This configuration gives each project a VS Code managed container using the
already downloaded `tensorflow/tensorflow:2.17.0-gpu-jupyter` image.
The current project folder is shared with the container at `/workspace`.

## Open the environment

1. Start Docker Desktop and open the project folder in VS Code.
2. Press **Ctrl+Shift+P** and choose **Dev Containers: Reopen in Container**.
   VS Code creates or starts the project's container and opens `/workspace`.
   On the first connection, it installs the VS Code server and the Python,
   Pylance, and Jupyter extensions in the container; internet access is needed.
3. Open a notebook. If a kernel is not selected, use **Select Kernel** >
   **Select Another Kernel...** > **Jupyter Kernel...** and select
   **TensorFlow 2.17 (Docker GPU)**. Alternatively, choose **Python Environments**
   and the interpreter at `/usr/bin/python`. VS Code remembers kernel choices.

The container indicator in the lower-left corner should say
**Dev Container: TensorFlow 2.17 GPU**. New terminals and notebook kernels start
at `/workspace`, including notebooks stored in subfolders. No `%cd` cell or
Jupyter server URL is needed. Remove old hard-coded `%cd /D/...` cells when
using this environment: this container mounts the project at `/workspace`.

For later sessions, reopen the container entry from **File > Open Recent**, or
use **Dev Containers: Reopen in Container** from the local project window.
Docker Desktop must be running. The configuration stops the managed container
when its last VS Code window closes; reopen it to start it again. To keep
background jobs running after closing VS Code, set `shutdownAction` to `none`.
Stopping a container ends its running notebook kernels and their in-memory state.

## Check the notebook environment

```python
from pathlib import Path
import sys
import tensorflow as tf

print("Project root:", Path.cwd())  # /workspace
print("Python:", sys.executable)   # /usr/bin/python
print("TensorFlow:", tf.__version__)  # 2.17.0
print("GPUs:", tf.config.list_physical_devices("GPU"))
```

GPU memory grows as needed instead of being reserved up front. The container's
Python extension uses `/usr/bin/python`; its environment-manager integration is
disabled in this container to avoid the project's host-side Conda preference.

## Files, packages, and other projects

Edits and outputs under `/workspace` are written directly into the local project
folder. Files outside the project are not mounted by this configuration. Add
specific bind mounts to `devcontainer.json` if an external dataset is needed.

The existing `tf_env_dckr` container is separate. This configuration creates its
own container from the same image, so packages added manually to `tf_env_dckr`
are not copied. The original container does not need to run for this workflow.
Jupyter kernels run through VS Code; this configuration does not publish a
Jupyter web-server port.

Install extra packages into the selected notebook kernel with `%pip install`,
or use `python -m pip install` in a container terminal. Such installations survive
normal stops and starts, but are lost when the container is rebuilt or deleted.
For reproducible additions, declare them in a Dockerfile or a requirements file
installed by `postCreateCommand`, retaining the kernel-registration command.

The root `requirements.txt` now targets **TensorFlow 2.17.0 / Keras 3.4.1**.
It is not installed automatically; install it into this container when package
index access is available. Read `compatibility_migration.md` for tested paths,
remaining continual-growth/metadata blockers, and supporting dependency gaps.
Newer hosted-runtime combinations require a separate validation environment.

Copy the `.devcontainer` folder into another project to use the same workflow.
`${localWorkspaceFolder}` automatically mounts that project's root at
`/workspace`. Each project gets its own container; the downloaded image layers
are shared. After changing the configuration, run
**Dev Containers: Rebuild Container** to apply the changes.

References: [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/create-dev-container),
[notebook kernels](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management).

## Codex and command-line tests

The root `AGENTS.md` tells future Codex sessions to use this environment for
task-relevant tests. A local Codex session can use Docker from Windows without
opening the VS Code container window. A cloud session needs its own accessible
Docker environment; these instructions do not connect it to this computer.

If the standalone `devcontainer` command is available, run
`devcontainer up --workspace-folder .` from the project root, followed by
`devcontainer exec --workspace-folder . /usr/bin/python <arguments>`.

This Windows installation also includes the CLI with VS Code's Dev Containers
extension. When `devcontainer` is not on PATH, use the following from the project
root, selecting the installed extension's CLI file (update the paths if the
Windows user or VS Code installation changes):

```powershell
$taskCode = 'C:\Users\Hosein\AppData\Local\Programs\Microsoft VS Code\Code.exe'
$taskCli = Get-ChildItem -Path 'C:\Users\Hosein\.vscode\extensions\ms-vscode-remote.remote-containers-*\dist\spec-node\devContainersSpecCLI.js' |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $taskCli) { throw 'VS Code Dev Containers CLI was not found.' }
$taskPreviousElectronMode = $env:ELECTRON_RUN_AS_NODE
try {
    $env:ELECTRON_RUN_AS_NODE = '1'
    & $taskCode $taskCli up --workspace-folder . | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Dev Container startup failed.' }
    & $taskCode $taskCli exec --workspace-folder . /usr/bin/python -c 'import os; print(os.getcwd())' | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Container Python command failed.' }
} finally {
    $env:ELECTRON_RUN_AS_NODE = $taskPreviousElectronMode
}
```

The command prints `/workspace`. Replace the arguments after `/usr/bin/python`
with the desired test command, for example
`-m unittest discover -s common/tests -t .`. Preserve and check its exit status.
Piping the Code executable's output ensures PowerShell waits for this GUI
executable when it is used as the CLI's Node runtime. VS Code is not opened by
this invocation. This command-line workflow uses the container directly; it
does not require VS Code's server or editor extensions to be installed there.
