# TensorFlow Docker command guide

Copy and run the section you need in **Windows PowerShell**. This is a reference
file, not a script to execute from top to bottom.

## 1. Create the container once

Open PowerShell in the project folder before running this block. `$PWD.Path`
becomes the container's `/workspace` mount, and `$env:USERPROFILE` resolves your
Windows user directory. Each continuation backtick must be the last character
on its line, with no trailing spaces.

Docker Desktop must be running with Linux containers and GPU support. The image
must already exist locally because this command uses `--pull=never`. The name
`tf_env_220` and host ports 8888, 6006, and 8080 must be available.

~~~powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.keras" | Out-Null

docker run -d `
  --name tf_env_220 `
  --pull=never `
  --gpus all `
  --init `
  --restart unless-stopped `
  --stop-timeout=60 `
  --shm-size=5g `
  --log-driver=json-file `
  --log-opt max-size=10m `
  --log-opt max-file=3 `
  -p 127.0.0.1:8888:8888 `
  -p 127.0.0.1:6006:6006 `
  -p 127.0.0.1:8080:8080 `
  --mount "type=bind,source=$($PWD.Path),target=/workspace" `
  --mount "type=bind,source=$env:USERPROFILE\.keras,target=/root/.keras" `
  --mount "type=bind,source=C:\,target=/mnt/c" `
  --mount "type=volume,source=tf_pip_cache,target=/root/.cache/pip" `
  --workdir /workspace `
  -e TF_FORCE_GPU_ALLOW_GROWTH=true `
  -e PYTHONUNBUFFERED=1 `
  tensorflow/full-tensorflow:2.20.0-gpu-jupyter `
  jupyter lab `
  --ip=0.0.0.0 `
  --port=8888 `
  --no-browser `
  --allow-root `
  --ServerApp.root_dir=/workspace `
  --ServerApp.port_retries=0
~~~

The pip cache volume is created automatically if needed. The command applies
no container-specific RAM cap; Docker Desktop/WSL and available host resources
still limit memory. `--shm-size=5g` sets the capacity of `/dev/shm` without
allocating all 5 GiB immediately.

| Windows resource | Container location |
| --- | --- |
| Project folder used at creation | `/workspace` |
| Current user's `.keras` folder | `/root/.keras` |
| C drive, with read/write access | `/mnt/c` |
| Docker volume `tf_pip_cache` | `/root/.cache/pip` |

Jupyter's file browser starts at `/workspace`. Access `/mnt/c` through Python or
a container terminal. See [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/),
[volumes](https://docs.docker.com/engine/storage/volumes/), and
[memory settings](https://docs.docker.com/engine/containers/resource_constraints/).

## 2. Start an existing container

Use this on later visits if the container is stopped:

~~~powershell
docker start tf_env_220
~~~

Check its status and published ports:

~~~powershell
docker ps -a --filter name=^/tf_env_220$
docker port tf_env_220
~~~

`docker run` creates a new container; `docker start` reuses the saved one and its
configuration. [Docker start documentation](https://docs.docker.com/reference/cli/docker/container/start/)

## 3. Get the Jupyter link and check access

Show the current server URL, including its login token when token authentication
is enabled:

~~~powershell
docker exec tf_env_220 jupyter server list
~~~

Alternatively, read the recent startup logs:

~~~powershell
docker logs --tail 50 tf_env_220
~~~

Open [JupyterLab](http://127.0.0.1:8888/lab). Use the token from the current server
listing if the login page asks for it. If a printed URL contains the container
hostname or `0.0.0.0`, replace that host with `127.0.0.1` and preserve the token.

Check that the Jupyter login page responds; a healthy response is normally 200:

~~~powershell
(Invoke-WebRequest -Uri "http://127.0.0.1:8888/login" -UseBasicParsing).StatusCode
~~~

To follow server logs continuously:

~~~powershell
docker logs --follow --tail 50 tf_env_220
~~~

Press **Ctrl+C** to stop following the logs; the container keeps running.
The commands above retrieve the current token instead of saving one in this file.
[Jupyter server URLs and tokens](https://jupyter-server.readthedocs.io/en/stable/operators/security.html)

## 4. Start TensorBoard

Run this in a separate PowerShell terminal:

~~~powershell
docker exec -it -w /workspace tf_env_220 tensorboard --logdir=/workspace/results --host=0.0.0.0 --port=6006
~~~

Open [TensorBoard](http://127.0.0.1:6006). Leave that terminal open while using
TensorBoard; **Ctrl+C** stops this TensorBoard process.

The project's default result trees sit under `results`. Ordinary training must
enable `training.tensorboard` to write TensorBoard logs. If you configured a
different `training.tensorboard_path` or result directory, change `--logdir` to
the matching path inside the container.

For HPO-only logs, use this alternative instead of the command above:

~~~powershell
docker exec -it -w /workspace tf_env_220 tensorboard --logdir=/workspace/results/hpo/_tb --host=0.0.0.0 --port=6006
~~~

Start only one TensorBoard process on port 6006 at a time. Port publishing makes
the service reachable after you start it. TensorBoard launched with `docker exec`
must be started again after the container restarts.

Repository references: [results layout](results/README.md),
[default configuration](configs/default.yaml), and
[training log configuration](common/train.py).
[TensorBoard guide](https://www.tensorflow.org/tensorboard/get_started)

## 5. Run commands inside the container

Open an interactive Bash terminal in the project:

~~~powershell
docker exec -it -w /workspace tf_env_220 bash
~~~

Type `exit` to leave that shell. The container and Jupyter keep running.

Run a single command from PowerShell:

~~~powershell
docker exec -w /workspace tf_env_220 ls -lah
~~~

Run your own Python script; replace the example path with your script's path:

~~~powershell
docker exec -w /workspace tf_env_220 /usr/bin/python ./path/to/your_script.py
~~~

For several lines of Python, this PowerShell here-string avoids nested quoting
problems. It runs in a separate Python process:

~~~powershell
@'
import sys
import tensorflow as tf
import keras

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
'@ | docker exec -i -w /workspace tf_env_220 /usr/bin/python -
~~~

Run these commands while the container is running.
[Docker exec documentation](https://docs.docker.com/reference/cli/docker/container/exec/)

## 6. Check GPU, resources, packages, and mounts

GPU and driver status:

~~~powershell
docker exec tf_env_220 nvidia-smi
~~~

One snapshot of container CPU and memory usage:

~~~powershell
docker stats --no-stream tf_env_220
~~~

Shared-memory capacity and current usage:

~~~powershell
docker exec tf_env_220 df -h /dev/shm
~~~

Image and bind/volume mounts:

~~~powershell
docker inspect tf_env_220 --format '{{.Config.Image}}'
docker inspect tf_env_220 --format '{{json .Mounts}}'
~~~

Dependency consistency and pip cache information:

~~~powershell
docker exec tf_env_220 /usr/bin/python -m pip check
docker exec tf_env_220 /usr/bin/python -m pip cache info
~~~

GPU discovery and package checks establish only those specific checks.
Repository compatibility requires the relevant tests to pass.
[Docker resource statistics](https://docs.docker.com/reference/cli/docker/container/stats/)

## 7. Run repository tests when needed

First inspect the image/mounts using section 6 and verify that `/workspace` points
to this exact checkout. Select tests appropriate to the change. These commands
use separate Python processes so tests do not reset the live notebook kernel's
Keras or random-number state.

Common tests:

~~~powershell
docker exec -w /workspace tf_env_220 /usr/bin/python -m unittest discover -s common/tests -t .
~~~

Semantic consolidation tests:

~~~powershell
docker exec -w /workspace tf_env_220 /usr/bin/python -m unittest discover -s semantic_consolidation/tests
~~~

Complete source-contract and embedded model self-test registry; this can be
substantial work:

~~~powershell
docker exec -w /workspace tf_env_220 /usr/bin/python test.py
~~~

See [compatibility notes](compatibility_migration.md) for remaining limitations.
Keep Python assertions enabled when running these tests.

## 8. Use the spare port 8080

For example, serve result files through a temporary HTTP server:

~~~powershell
docker exec -it -w /workspace tf_env_220 /usr/bin/python -m http.server 8080 --bind 0.0.0.0 --directory /workspace/results
~~~

Open [the result file server](http://127.0.0.1:8080). **Ctrl+C** stops it.
For your own application, configure it to listen on `0.0.0.0:8080` inside the
container. This mapping publishes TCP traffic to the Windows loopback interface.
[Docker port publishing](https://docs.docker.com/engine/network/port-publishing/)

## 9. Stop or restart intentionally

Save notebooks and training checkpoints first. These commands end running
notebook kernels and other container processes.

Stop:

~~~powershell
docker stop --timeout 60 tf_env_220
~~~

Restart:

~~~powershell
docker restart --timeout 60 tf_env_220
~~~

After starting again, retrieve the current Jupyter URL with `jupyter server list`
as shown in section 3, and relaunch TensorBoard if needed.
[Docker shutdown behavior](https://docs.docker.com/reference/cli/docker/container/stop/)

## 10. What persists

- Files in `/workspace` and the C-drive mount are stored on Windows.
- The Keras cache is stored in the current user's Windows `.keras` folder.
- Pip downloads are stored in the named Docker volume `tf_pip_cache`.
- Packages installed inside the container survive stop/start and restart.
  Removing/replacing the container discards its installed-package changes unless
  they have been incorporated into an image. A pip cache stores downloads.
- Changing this guide does not change a container that already exists.

For a later configuration change, preserve needed container changes before
replacing it. The existing container should be reused for routine work.
[Container storage](https://docs.docker.com/engine/containers/run/#filesystem-mounts)
