# Minimum Route One thesis experiments

These notebooks use TensorFlow 2.20 and native Keras 3. Locally, select the project's TensorFlow kernel, restart it, and run one notebook from top to bottom. For hosted sessions, use the setup below. Each training notebook runs one complete seed/class-order stream.

**Defense scope:** this is a TMCL-inspired supervised classifier experiment, designated a **test-informed fixed benchmark** because earlier official-test HPO results informed recipe discussion. It can produce accuracy, forgetting, local backward-transfer and paired-effect values after complete runs, but not independent confirmation on those previously inspected test rows. It does not reproduce TMCL's published protocol or test noise-dependent consolidation. The [selection record](benchmark_selection.json) explains the known test exposure and interpretation limits. Registering the settings below does not itself freeze or execute a campaign.

Training notebooks keep four main steps: select a stream, load data/create the model, train, and save/read results. Development then reviews diagnostics; benchmark diagnostics are optional.

## Hosted runtimes

The shared-bootstrap notebooks use a small loader for [`notebooks/init.py`](../init.py),
which finds or downloads the checkout and prepares its runtime. The maintained
notebooks **00 through 12** use the same dependency manifest. Setup recognizes Colab, Kaggle and Binder. Other hosted
Jupyter services can use an explicit setting in the first code cell. All require
Python **3.11-3.13**, a writable project directory and Internet access when
dependencies, the repository or datasets need downloading.

Every notebook outside `notebooks/old` starts with a Markdown panel linking to
its own GitHub notebook through three buttons: Colab, Kaggle and Binder. Studio
Lab remains available as a text link for existing accounts. These links use
the published `main` branch: push the notebooks, shared setup files and Binder
configuration together before using them. For an unpublished notebook, upload
the `.ipynb` to Colab or Kaggle; the shared initializer must still be available
locally or on GitHub. Opening a runtime does not start training automatically.

Earlier development and HPO executions are retained under `runs/` as historical
artifacts. Use the numbered notebooks in this directory for the current recipe.

### Google Colab

[![Open development notebook in Colab](https://img.shields.io/badge/Open_in-Colab-F9AB00?logo=googlecolab&logoColor=F9AB00)](https://colab.research.google.com/github/hosein-fanai/Continual-Learning-with-Diffusion-Vision-Transformers/blob/main/notebooks/thesis/00_Development.ipynb)

1. Open a notebook with its **Open in Colab** button, which links to its own
   GitHub copy.
2. For training, connect to a **GPU** runtime. Use **Runtime > Change runtime
   type** if a GPU is not selected. GPU allocation depends on Colab availability.
3. Choose **Runtime > Run all**. The first code cell finds an existing checkout or
   clones the repository's `main` branch into
   `/content/Continual-Learning-with-Diffusion-Vision-Transformers`.
4. Setup checks installed package versions and installs missing or incompatible
   requirements before TensorFlow or project imports. Dataset loading downloads
   CIFAR through Keras when its cache is absent.

Colab supports loading notebooks directly from GitHub, but opening an `.ipynb`
does not provide the surrounding repository or its dependencies; the startup
cell supplies both. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

### Kaggle

[![Open development notebook in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https%3A%2F%2Fgithub.com%2Fhosein-fanai%2FContinual-Learning-with-Diffusion-Vision-Transformers%2Fblob%2Fmain%2Fnotebooks%2Fthesis%2F00_Development.ipynb)

1. Use the notebook's **Open in Kaggle** button, sign in, and complete the import.
   Alternatively, create a notebook in [Kaggle Code](https://www.kaggle.com/code)
   and import the `.ipynb` by GitHub URL or file upload.
2. Enable **Internet** in notebook settings and choose a **GPU** accelerator
   for training, then **Run all**.
3. The first code cell detects Kaggle and downloads missing source into
   `/kaggle/working/Continual-Learning-with-Diffusion-Vision-Transformers`.
   It installs missing Python dependencies while using Kaggle's managed CUDA.
4. Check the next cell's `GPUs` count before starting a long training run.
   Export or save the complete results directory in the notebook's outputs.

Use `/kaggle/working` for writable files; attached input datasets live separately.
GPU access and quotas depend on the account and service availability. See
[Kaggle's notebook documentation](https://www.kaggle.com/docs/notebooks).
The badge uses Kaggle's documented
[external-notebook import link](https://www.kaggle.com/product-feedback/152480).

Kaggle detection takes precedence over Colab because current Kaggle images
inherit the Colab base. The shared requirements accept IPykernel 6 and 7 to
preserve either provider's compatible live kernel. This follows Kaggle's
[image source](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl)
and [release metadata](https://github.com/Kaggle/docker-python/releases).

### Binder: small CPU checks

[![Launch development notebook in Binder](https://img.shields.io/badge/launch-binder-F5793A?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/hosein-fanai/Continual-Learning-with-Diffusion-Vision-Transformers/main?urlpath=lab%2Ftree%2Fnotebooks%2Fthesis%2F00_Development.ipynb)

The [Binder configuration](../../.binder) selects Python 3.11 and installs the
same `requirements.txt` through the bootstrap with CUDA pip dependencies omitted.
It does not maintain another requirements list. Binder starts with the repository
already present; its first build may take time.

Use public Binder to inspect code or run small CPU checks. Full CIFAR training
and the benchmark campaign need more resources: public Binder provides only
1-2 GB RAM and limits computational session time. Its files are temporary.
See [Binder's usage limits](https://mybinder.readthedocs.io/en/latest/about/user-guidelines.html).

### Other hosted Jupyter services

In the first code cell, set:

```python
RUNTIME = "hosted"
CUDA = False  # CPU, or a provider that already supplies compatible CUDA libraries.
```

Use `CUDA = True` for an NVIDIA Linux x86_64 runtime that needs TensorFlow's
CUDA pip dependencies. The host must already expose a compatible NVIDIA driver;
pip cannot allocate a GPU. Run all from a fresh compatible Python kernel.
The checkout is downloaded into the current directory when neither Kaggle's
working directory nor Colab's `/content` exists. Use a writable project folder.

Existing **SageMaker Studio Lab** accounts can select `RUNTIME = "studiolab"`.
Choose a Python 3.11-3.13 kernel; `CUDA = None` retains the Linux CUDA extra,
while `CUDA = False` works for CPU or an already provisioned CUDA environment.
Studio Lab is no longer open to new customers; see its
[service notice](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html)
and [environment guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-use-manage.html).

[Open development notebook in Studio Lab (existing accounts)](https://studiolab.sagemaker.aws/import/github/hosein-fanai/Continual-Learning-with-Diffusion-Vision-Transformers/blob/main/notebooks/thesis/00_Development.ipynb)

The Studio Lab link opens a preview. Start a runtime, copy the notebook to
your project, select a compatible kernel, then set the first code cell's runtime
options before running it. See AWS's
[GitHub import and badge instructions](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-use-external.html).

`RUNTIME = "local"` disables automatic installation even on a detected service.
The `CONTINUAL_RUNTIME` environment variable supplies the same selection when
`RUNTIME = "auto"`. Generic JupyterHub markers and merely installed provider SDKs
do not enable installation. The generic hosted mode is configurable support,
not a claim that every provider image has been tested.

### Checkout and runtime behavior

`CHECKOUT_NAME = "Continual-Learning-with-Diffusion-Vision-Transformers"` matches
the canonical name verified through GitHub. `REPOSITORY` is constructed as
`f"https://github.com/hosein-fanai/{CHECKOUT_NAME}.git"`. A notebook first loads
its local `notebooks/init.py`; if absent, it downloads just that initializer,
which then downloads the missing repository. See the
[initializer guide](../INITIALIZATION.md) for the shared entry points.

The first code cell reuses a checkout found in the current directory, its parents,
or the named download directory. Rerunning it does not pull updates or overwrite
local edits. To run a different Git branch, set `REVISION` before the first
download; it does not switch an existing checkout. To update an existing
checkout, save your work and update it explicitly before starting a new run.

The single [`requirements.txt`](../../requirements.txt) is shared by hosted,
Docker, and local environments. It pins TensorFlow 2.20.0 and Keras 3.11.2 and
uses compatible ranges for scientific and notebook packages, including managed
IPython/kernel versions. Missing tools such as JupyterLab are installed
from the same file. On Linux x86_64 (including WSL2 and Docker), the file requests
`tensorflow[and-cuda]`; other platforms use plain TensorFlow. The startup helper
removes only that extra for Colab/Kaggle managed CUDA, Binder CPU, or an explicit
`CUDA = False`. Run the startup cell on these services instead of installing the
file directly: pip's environment markers cannot detect notebook providers.
Local notebooks verify dependencies without installing into your selected kernel,
and accept CUDA libraries supplied by a GPU image. Explicit local installation
with `prepare_runtime(ROOT, install=True)` also checks the extra's pip dependencies.

Always run the bootstrap cell before importing the model. If an incompatible
TensorFlow/Keras version is already loaded, restart the session and **Run all**.
Before installing, setup also checks pip's proposed changes and stops if they
would replace an already imported package. For an incompatible hosted runtime,
select a supported runtime and start fresh; Colab's **2026.07** runtime includes
TensorFlow 2.20 and Python 3.12. Runtime choices and installed packages change;
see [Colab's runtime version guide](https://research.google.com/colaboratory/runtime-version-faq.html).

### Results and campaign prerequisites

Notebook **00** can start a development experiment in a fresh session. Notebooks
**11 and 12** can also start their reference runs independently. Notebook **01**
prepares the fixed benchmark campaign; notebooks **02 through 09** require
that campaign, and notebook **10** requires its completed streams. Restore the
same campaign artifacts when moving those stages to another hosted runtime.
Downloading the source alone does not recreate a frozen campaign or its results.

Colab virtual-machine files are temporary. Preserve the whole relevant
`results/thesis_route_one/` campaign or development directory, including
checkpoints and `frozen_design.json`, outside the runtime before deleting it.
Keeping only the notebook in Drive does not save the VM's other files. See the
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

Finish source and notebook changes before freezing a benchmark campaign.
The bootstrap and maintained helper code participate in its source identity;
an existing frozen campaign must use its retained source, or a new campaign
must be prepared. Setup does not relax that provenance check.

## Run sequence

| Notebook | Purpose |
|---|---|
| [00 Development](00_Development.ipynb) | Seed 17, validation only. Check platform and learned on each dataset before freezing. |
| [01 Freeze](01_Freeze_Experiment.ipynb) | Save the exact recipe, source identities and randomized execution checklist. No training. |
| [02 CIFAR-10 platform](02_CIFAR10_platform.ipynb) | Ordinary joint diffusion/classification with generated replay and distillation. |
| [03 CIFAR-10 extra joint](03_CIFAR10_extra_joint.ipynb) | Extra ordinary updates matching the semantic optimizer-update allowance. |
| [04 CIFAR-10 learned](04_CIFAR10_learned.ipynb) | Learned class modulation followed by semantic consolidation. |
| [05 CIFAR-100 platform](05_CIFAR100_platform.ipynb) | Full 100-class platform reference. |
| [06 CIFAR-100 extra joint](06_CIFAR100_extra_joint.ipynb) | Primary additional-training control. |
| [07 CIFAR-100 learned](07_CIFAR100_learned.ipynb) | Full proposed procedure. |
| [08 CIFAR-100 random](08_CIFAR100_random.ipynb) | Random gates; supporting mechanism comparison. |
| [09 CIFAR-100 CE only](09_CIFAR100_ce_only.ipynb) | Acquisition plus replacement CE updates, without alignment gradients. |
| [10 Collect](10_Collect_Thesis_Results.ipynb) | Authenticate saved outcomes and export the compact writing package. No new predictions. |
| [11 Offline joint reference](11_Offline_Joint_Reference.ipynb) | Train on all classes together, with no replay, distillation or semantic phases. |
| [12 Naive sequential reference](12_Naive_Sequential_Reference.ipynb) | Train the same DiT platform task by task using current-task examples only, with no CL retention mechanism. |
| [13 CIFAR-10 joint HPO](13_CIFAR10_Joint_HPO.ipynb) | Tune ordinary all-class diffusion/classification with a learned class token, no distillation, and persistent TensorBoard/Optuna results. |
| [14 CIFAR-100 joint HPO](14_CIFAR100_Joint_HPO.ipynb) | The same V1 search on CIFAR-100. |

Notebooks **13 and 14** are independent development searches, outside the frozen
campaign. They use V1, raw weights without EMA, separate ordinary accuracy and
noise-loss objectives, and 50 epochs without early stopping or fit validation.
Version-13 notebooks select on a stratified 20% holdout of official training
data; the other 80% supplies gradients. Earlier test-selected HPO studies remain
exploratory and cannot support independent confirmation on those same test rows.
Keep HPO selection metadata when transferring settings: strict confirmation
preparation rejects recorded official-test selection. The approved campaign
instead uses native `phase="benchmark"` with bound selection provenance
`test_informed=true` and `independent_confirmation=false`. Manual copying can
discard provenance, so absence of a recorded warning does not establish an
untouched test set. The explicit benchmark designation preserves the known
selection history.
See the [joint-classifier HPO guide](JOINT_CLASSIFIER_HPO.md)
for the search space, ordinary scoring rules, hardware/budget guidance, and
recommendations from the reports. Their results do not automatically replace
the continual-learning recipes.

## Offline and naive reference benchmarks

Notebooks **11 and 12** each support CIFAR-10 and CIFAR-100 through `DATASET`.
They retain the shared DiT diffusion/classification objective, use matching seeds,
class orders and train/validation partitions, and remove replay, distillation and
semantic consolidation. Offline joint training sees every class from the start;
naive sequential training retains the learned model while introducing one task at
a time. The existing **platform**, **extra joint** and **CE-only** conditions are
different controls and do not implement these two references.

These are empirical upper/lower reference comparisons, **not guaranteed maximum
or minimum accuracy**. Offline training has no task-transition trajectory, so its
forgetting, backward transfer and average incremental accuracy are unavailable.
Both notebooks default to validation evaluation; optional test evaluation is
explicitly labeled as a supplemental reference. Use a fresh kernel for each run.

Each run saves a separate directory under
`results/thesis_route_one/reference_benchmarks`, including the resolved recipe,
source provenance, final per-task accuracy, summary and native training outputs.
Naive runs also save their task accuracy matrix and completed-task checkpoints.
These additional runs are separate from the frozen 24-stream campaign and the
notebook 10 collector. See [the reference protocol](BENCHMARK_REFERENCES.md) for
budget interpretation, artifacts and primary sources.

## Small, fixed scientific design

The benchmark plans three paired seeds: **1103, 2207, 3301**. CIFAR-10 has five two-class tasks and three methods (9 streams). CIFAR-100 has ten ten-class tasks and five methods (15 streams). Once prepared, follow the saved **24-row checklist**, one fresh kernel per row. Every method in a block shares the seed and class order. A failed or unfavorable stream cannot be skipped.

The primary comparison is **learned minus extra joint in final ensemble task-average test accuracy**. The registered raw timestep ensemble uses `max_t=256`, `t_range_drop_rate=0.75` (64 retained timesteps), `clf_acc_coef=1.0` and `clf_distil_acc_coef=0.0`. It averages primary-head predictions over timesteps; the distillation head retains its training loss but contributes no inference vote. This keeps the endpoint on a supervised head at the first task and in the KD-disabled supplemental references. Ordinary clean accuracy remains a separately labeled diagnostic. Extra joint matches the number of additional optimizer updates. Different phases update different parameters and use different batches; this does not match examples, FLOPs or wall time. Random and CE-only are supporting comparisons.

The common platform uses a float32, nonvariational DiTClassifier/V1, dimension 128, backbone and classifier depth 6, CNN patchification, patch size 4, four heads, raw inference and no EMA. Joint classifier training uses noisy inputs, null conditioning and no null-row mask; all rows contribute to both classifier and denoising objectives. Common APIs own splitting, class-balanced generated replay, soft distillation, losses and class growth. Acquisition freezes the backbone and learns class gates. Consolidation trains the classifier projection/head and temporary predictor against a frozen acquired target. The maintained recipes use TMCL's published image policy: acquisition flips and four independently cropped/colored consolidation views. `noise_levels=[0]` disables diffusion noise in the semantic objective, not image augmentation or ensemble inference. CE uses its separate clean input. Inference uses all seen classes without task identity, gates or predictor. This does not test noise-dependent consolidation or reproduce TMCL's full view-invariance objective.

Learned versus extra joint compares the complete added procedure, including its
image-view policy and losses. Equal optimizer-update allowances do not match
images, target passes, FLOPs or wall time. Learned versus random shares the image
policy and tests gate acquisition within that procedure; the minimum campaign
has no identity-gate or augmentation-only arm, so it cannot isolate every cause.

| Budget | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Joint batch / epochs per task | 128 / 50 | 128 / 50 |
| Acquisition / consolidation updates per task | 200 / 400 | 1000 / 2000 |
| Semantic batch / Adam learning rate | 32 / 0.0003 | Same |
| Joint Adam / weight decay | Initial LR 0.001, cosine / 0 | Same |
| Whole-stream cosine horizon | 49,950 updates | 116,150 updates |
| Current / generated-old pool | All current rows; 4,000 generated rows per old class | All current rows; 400 generated rows per old class |
| Replay sampling | 1,000 steps, eta 1, CFG scale 3 | Same |
| Fixed validation probe | 8 examples per class | Same |
| Recovery interval | 200 completed updates, plus native phase boundaries | Same |

These user-selected settings are **not a measured convergence or runtime result**. The approved LR reduction from 0.005 to 0.001 is a conservative choice, not a demonstrated optimum for this full recipe. With the maintained 20% validation split, expected ordinary updates are 46,950/86,150 per stream; learned and extra-joint total optimizer applications are 49,950/116,150. Semantic optimizers have their own clocks. The shared joint cosine horizon includes the maximum fixed extra-joint allowance and never restarts per task; extra-joint updates therefore advance later tasks farther along this global schedule. Each final-task training pool contains 40,000 rows with equal class counts. Measure replay usefulness, full-stream learning and cost in notebook 00. The [recipe rationale](HYPERPARAMETER_RATIONALE.md) records the settings and budget assumptions.

## Validation, test and metrics

Current development seed 17 selects on validation outcomes. The configured 20% validation split is excluded from training gradients; retained historical validation probes are disclosed observation access, not rehearsal data. The fixed benchmark evaluates the official test split with its known prior exposure from historical HPO. A freeze and fresh seeds do not remove that exposure or establish independent confirmation. After freezing, benchmark test outcomes must not choose settings, stopping rules, seeds or which streams are reported.

For `R[i,j]`, accuracy on task `j` after learning task `i`, using all seen classes:

- **Final accuracy:** mean of the final row.
- **Incremental accuracy:** mean of each learned-prefix row mean.
- **Signed forgetting:** mean, over old tasks, of the best earlier accuracy minus final accuracy. The maximum excludes the final row and starts when the task was learned. Negative values are preserved.
- **Backward transfer:** mean final-minus-acquisition accuracy over old tasks.

This local backward-transfer definition differs from TMCL's single-task-reference BT; the notebooks do not calculate the paper's FT. Do not compare those numbers directly.

Compute each metric within a complete stream first. Report individual values, mean, sample SD (`ddof=1`) and actual independent-stream `n`. Accuracy uses percent; forgetting and accuracy differences use percentage points. The native primary paired 95% t interval is separate from SD and describes run-to-run uncertainty under this fixed, test-informed recipe; it does not correct selection bias. **Three pairs give limited precision**; tasks, images and gates do not increase independent `n`.

Phase changes require identical hashed validation examples and positive actual counts. Unavailable endpoints cannot yield numeric differences. Small-cohort CKA is descriptive; invalid two-example CKA stays unavailable. Replay agreement measures classifier self-consistency, and saved grids are qualitative. Sum only complete, disjoint active-task timers. Resumed committed segments are included; measured checkpoint writes and downtime are separate. Uncommitted lost work and interrupted unfinished write timers are unavailable, so these observations are not complete restart-inclusive wall time. Nested phase timers, notebook pauses, sampled RSS, allocator peaks and tensor payloads have distinct scopes.

## Freeze, interruption and rerun

Finish all source and recipe changes before notebook 01. The configured campaign path is `results/thesis_route_one/minimum_v5_tf220/`. Prepare with `phase="benchmark"` and the declared selection provenance; the bound `benchmark_selection.json` preserves the designation and reason alongside the planned 24 streams. Keep an independent unchanged copy of `frozen_design.json`. Existing campaigns are preserved and cannot be resealed silently. Source or scientific-setting changes require a new campaign.

Checkpoint saving is sealed into the recipe. Each stream has its own `checkpoints/<run_id>/` directory. Native recovery preserves completed-task boundaries and the latest two intermediate progress snapshots. The notebook uses the native authenticated selector; it never reconstructs model state itself. After an interruption, restart the kernel and **Run All**: the same unfinished repeat resumes from its latest valid commit, and work after that commit is repeated. Before the first published commit, the same stream restarts while unpublished temporary evidence is retained. Damaged published evidence is reported, not overwritten.

A nonblocking OS lock prevents simultaneous training of one stream in multiple kernels. Kernel exit releases the lock; the remaining lock file does not mean the stream is active. A failed training cell also releases ownership. A valid completed stream is never trained again. If completion publication was interrupted, saved native evidence can repair the index without training or re-evaluation. If only a plot failed after completion, rerun its view or collection.

Runtime checkpoint and resume locators remain separate from the immutable YAML. Do not alter hashes to accept changed code, configuration or evidence. Checkpoint boundaries preserve supported computation state; GPU execution is not a blanket promise of bitwise determinism across devices or software versions.

Development checkpoint identities include the resolved recipe, inherited settings and executable source. Changing any of them starts a separate pilot. This audit changes the development identity once; older checkpoints remain preserved and are not adopted into the revised pilot. Use the retained original source/recipe to continue an old pilot.

## Compact final output

Benchmark training notebooks show a small scalar outcome table by default. Choose `SHOW_DIAGNOSTICS=True` before freezing if those training notebooks should also show saved validation, replay and learning plots. Their cell sources are frozen, so do not edit that flag during the campaign; later saved-only diagnostics remain available through notebook 10. Notebook 10 requires all 24 complete streams and displays one treatment summary plus the primary paired effect/interval. Its ZIP retains numeric source rows, exact source/configuration identities and interpretation limits.

`DETAILS=True` adds extended diagnostic tables and figures; choose a separate `OUTPUT` to preserve previous exports. `PROGRESS=True` labels a partial saved-results snapshot and omits final paired inference. Identical exports are authenticated and reused. Collection never loads datasets, trains, predicts or samples new images. Write from the saved observations and preserve uncertainty, missing values and negative findings.

[VALIDATION.md](VALIDATION.md) records software checks and their limits. No synthetic fixture or successful software test is a thesis accuracy result.
