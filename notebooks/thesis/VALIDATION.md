# Thesis workflow validation — audit of 15 September 2026

## Shared initializer and notebook preservation — 16 September 2026

GitHub's repository metadata confirmed the public repository's canonical name:
`hosein-fanai/Continual-Learning-with-Diffusion-Vision-Transformers`, default branch
`main`. Both the old underscore URL and the lowercase hyphen URL resolved to
that name. Notebook repository URLs now use an f-string based on `CHECKOUT_NAME`.

The former `notebooks/init/` package is now `notebooks/init.py`. It owns checkout
discovery/download and shared path setup, and delegates dependency preparation
to the existing bootstrap. Root and thesis `import init` entry points retain
their root/path setup; the subsequent cleanup removed their redundant scientific
imports. New frozen campaigns bind both root and shared initializers.

Before notebook rollout, **13 prototype checks passed**. After rollout,
**42 initializer tests passed**, covering all **96 notebooks** outside automatic
checkpoint directories. The preservation audit compared every original notebook
to a local pre-edit snapshot and confirmed:

- All **1,695 existing cells** retained their non-source fields, including output
  data, execution counts, IDs, attachments and metadata.
- All **14,158 saved output records** were preserved.
- **1,421 code cells** were unchanged; **44 setup code cells** changed only for
  the shared loader or removal of duplicate/relative directory initialization.
- Notebook metadata and format versions were preserved. The 13 maintained
  bootstrap cells were replaced; the other 83 notebooks gained a setup cell.

Commands used the mount-verified `tf_env_220` container, Python **3.11.13**,
TensorFlow **2.20.0**, and Keras **3.11.2**:

```text
docker exec -w /workspace tf_env_220 /usr/bin/python -m unittest notebooks.thesis.tests.test_bootstrap notebooks.thesis.tests.test_notebook_init -v
docker exec -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=3 -w /workspace tf_env_220 /usr/bin/python -m unittest notebooks.thesis.tests.test_recipe.NotebookContractTests notebooks.thesis.tests.test_recipe.PreparedRecipeTests.test_all_24_native_configs_preserve_paired_full_streams_and_recipe
```

The second command passed **4 tests**, including rejection of a frozen campaign
when the shared initializer's digest changes: **46 post-rollout tests passed**
in total, with no failures or skips.

The actual first two code cells of all 13 maintained notebooks also executed in
a clean source snapshot using a separate Jupyter kernel. Both dataset recipes
and a fresh 24-stream campaign/checklist passed. No notebook was re-executed in
place, and no training, dataset download or package installation was performed.
Historical notebook training code and its optional dependencies were preserved,
not validated as complete TensorFlow 2.20 experiments.

## Hosted runtime support checks — 16 September 2026

Startup now detects Kaggle, Binder and Colab, with explicit Studio Lab and generic
hosted modes. Kaggle's writable download directory, provider precedence, CUDA
selection and installation overrides are covered by simulated runtime tests.
IPykernel 6 and 7 are accepted by the shared requirements. Binder's build hook
uses that same manifest with CUDA pip dependencies omitted.

The mount-verified `tf_env_220` container supplied Python **3.11.13**, TensorFlow
**2.20.0** and Keras **3.11.2**. Checks ran in separate processes:

```text
docker exec -w /workspace tf_env_220 /usr/bin/python -m unittest notebooks.thesis.tests.test_bootstrap
docker exec -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=3 -w /workspace tf_env_220 /usr/bin/python -m unittest notebooks.thesis.tests.test_recipe.NotebookContractTests
```

- **34 bootstrap tests passed**, with clone and installation actions mocked.
- **3 existing notebook contract tests passed**.
- The actual first two code cells of all **13 maintained notebooks** executed in
  a fresh Jupyter kernel using a clean publishable-source snapshot. Notebook
  schemas, both recipes, source identity and a fresh 24-stream checklist passed.
- Binder's shell syntax and embedded Python setup passed. All six runtime
  policies reused the installed compatible packages with subprocess installation
  explicitly blocked during this check.

No dependencies were installed, datasets downloaded or models trained. Live
Kaggle/Colab/Studio Lab sessions and a complete Binder image build were not run.
These checks establish local startup behavior, not compatibility with every
current or future provider image. Run a hosted smoke check after publishing.

## Colab bootstrap check — 16 September 2026

The maintained notebooks **00 through 12** now share a checkout/dependency
bootstrap and use portable Python kernel metadata. Required artifact and tensor
inventory helpers live in `common`, so a GitHub checkout does not need the ignored
`allocation_study` or `gist_memory` packages.

Checks used the existing, mount-verified `tf_env_220` container: Python **3.11.13**,
TensorFlow **2.20.0**, Keras **3.11.2**. Scientific checks ran in separate CPU
processes; the startup execution used a new Jupyter kernel.

| Check | Result |
|---|---|
| New bootstrap contracts, including clone/reuse and safe dependency installation | **12 passed**; clone and pip actions mocked. |
| Updated existing notebook contracts | **3 passed**; all 13 canonical notebooks validated. |
| Shared artifact/tensor helpers, semantic study/runtime and thesis completion | **59 distinct cases passed**, no skips, following a stable rerun of the 17 study cases. |
| Clean publishable-file snapshot with both ignored packages absent | **Passed**: actual first two code cells of all 13 notebooks, notebook schemas, both dataset recipes, source identity, and a freshly prepared 24-stream campaign/checklist. |

One initial study check observed a changed source fingerprint while notebook edits
were still in progress. After those edits stopped, all 17 study checks passed.

Commands ran from `/workspace`, using `/usr/bin/python`:

```text
-m unittest notebooks.thesis.tests.test_bootstrap
-m unittest notebooks.thesis.tests.test_recipe.NotebookContractTests
-m unittest common.tests.test_study_artifacts common.tests.test_tensor_inventory semantic_consolidation.tests.test_study semantic_consolidation.tests.test_experimental_runtime notebooks.thesis.tests.test_completion
-m unittest semantic_consolidation.tests.test_study
.tmp/colab-check-20260916/validate_checkout.py
```

The clean snapshot used the current contents of Git-tracked files plus the new
bootstrap, hosted requirements and bootstrap tests. It could not import helpers
from the original workspace. Existing compatible packages were reused; no data
downloads, package installations, full training or actual hosted Colab execution
were performed. Colab's package/GPU environment therefore still needs a hosted
smoke check after publishing these changes. These are software checks, not thesis
results. The historical audit below describes its own earlier scope.

## Environment and scope

Used the existing running `tf_env_220` container, image
`tensorflow/full-tensorflow:2.20.0-gpu-jupyter`. Docker inspection verified that
`/workspace` maps to this exact Windows checkout. Installed versions were checked:
Python **3.11.13**, TensorFlow **2.20.0**, native Keras **3.11.2**, NumPy **2.3.2**,
SciPy **1.17.1**, pandas **3.0.5**, PyYAML **6.0.2**, nbformat **5.10.4**.
The GPU was an **NVIDIA GeForce RTX 2060**. No container rebuild, package downgrade
or live notebook kernel was used. Assertions remained enabled.

The 24-stream, three-seed design, model settings and training budgets are unchanged.
Changes address pilot checkpoint identity, complete-result validation, compact BWT
reporting, notebook readability and one shared classifier argument-order bug.
See [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md) for the scientific verdict and fixes.

## Actual checks

| Check | Result |
|---|---|
| Existing complete thesis suite, before edits | **53 passed**, no failures/skips; 452.278 s. Includes actual staged interruption/resume and completion recovery. |
| Revised notebook/recipe and saved-results suites | **29 passed**, no failures/skips; 70.155 s; CPU. Includes all eleven notebook schemas/cell syntax, 24 materialized recipes, inherited-setting/source identity changes and rejection of incomplete scalar displays. |
| Selected semantic objective, memory, phase, boundary, evaluation, diagnostic, integration and recovery suites | **83 passed**, no failures/skips; 225.830 s; CPU. Actual small synthetic fitting, controls and inference reload included. |
| Shared allocation artifact contracts | **6 passed**, no failures/skips; CPU. |
| Shared gist tensor inventory | **6 passed**, no failures/skips; CPU. |
| Full common suite, before classifier repair | **380 tests**, 1 failure and 1 error within the same positional-argument test; 2 skips; 1464.602 s. |
| Full architecture regression module after classifier repair | **12 passed**, no failures/errors/skips; 36.842 s; CPU. |
| Static source contracts after repair | **Passed**: 170 Git-discovered Python files, 209 classes, 1960 functions, 3675 branches. |
| Complete model self-test registry after repair, `test.py` | **37 module suites / 40 classes passed**, no exclusions or skips; 464.646 s; exit 0. Includes the 170-file static checks. |
| Selected-capacity GPU check after repair | **Passed**: two joint updates at batch 64 and two updates per semantic phase at batch 32, with 100-class support. |

Counts overlap and must not be added into a count of independent experiments.
The complete common run exposed `compute_class(..., training)` binding its
positional training flag to `return_logits`. Both its failure and error came from
that one bug. The repair restores the original argument position. The full
12-case architecture module was rerun; the entire 380-test suite was not repeated
for this argument-order change. The two skips were inherited TensorFlow
`test_session` helpers, not unrun thesis experiments.

Static checking initially found six missing branch comments in the same classifier
file; these were corrected. Static Git discovery excludes ignored packages,
including `allocation_study` and `gist_memory`; their relevant artifact and
resource helpers were audited and tested separately.

## Configured-capacity check

The separate GPU process retained the actual CIFAR-100 geometry: dimension 128,
depth 4, RGB 32-by-32 inputs, 1000 diffusion timesteps, 100-class student and
100 retained gates. It created an independent 90-class previous teacher and a
100-class acquired target. It verified actual optimizer applications, finite
objectives, positive measured classifier/noise distillation terms, changing
intended parameters, unchanged frozen parameters and normalized all-seen inference.

The short process recorded 716,019,456 bytes (about 683 MiB) peak TensorFlow
allocator usage. This excludes full-stream observers, checkpoints, complete replay
generation, process/device overhead and the user's other workloads. The 33.72 s
process duration includes construction/tracing and is not a campaign-time estimate.
Synthetic pixels and initialized old gates represent tensor shapes only; the
check does not simulate nine already learned tasks or establish efficacy.

## Commands

All Python commands used `/usr/bin/python` from `/workspace` inside the verified
container. From Windows, prefix each command with:

```powershell
docker exec -w /workspace tf_env_220 /usr/bin/python
```

CPU checks additionally used `-e CUDA_VISIBLE_DEVICES=-1` before `-w`. The selected
semantic checks also limited intra/inter-op threads to two. Exact Python arguments:

```text
-m unittest discover -s notebooks/thesis/tests -v
-m unittest notebooks.thesis.tests.test_recipe notebooks.thesis.tests.test_results_package -v
-m unittest discover -s common/tests -t .
-m unittest common.tests.test_architecture_verified_repairs -v
-m unittest semantic_consolidation.tests.test_objectives semantic_consolidation.tests.test_memory semantic_consolidation.tests.test_phases semantic_consolidation.tests.test_boundary_diagnostics semantic_consolidation.tests.test_evaluation semantic_consolidation.tests.test_experimental_diagnostics semantic_consolidation.tests.test_integration semantic_consolidation.tests.test_recovery -v
-m unittest discover -s allocation_study/tests -p test_artifacts.py
-m unittest discover -s gist_memory/tests -p test_resource_inventory.py
test.py
.tmp/thesis-audit-20260915/validate_configured_capacity.py
```

The pure static call was `import test; test.assert_static_contracts()`.
The final registry output is retained at
`.tmp/thesis-audit-20260915/model_registry.log`.
Original notebook/helper files were retained under
`.tmp/thesis-audit-20260915-originals/thesis/` before editing.

## Scientific and execution limits

Full-budget CIFAR development and the 24-stream confirmation campaign were **not
run**. Notebook cells were validated structurally and syntactically; native calls
were exercised with bounded synthetic fixtures. The eleven production notebooks
were not executed top-to-bottom with their full data and budgets. No synthetic
accuracy, smoke result, allocator observation or successful software test is a
thesis treatment result.

Use notebook 00 to assess useful learning, late-task retention, replay and measured
cost. Freeze only after validation-driven choices are settled. Keep all declared
confirmation streams and report individual values, sample SD, the paired interval,
negative findings and unavailable measurements. No result in this record promises
convergence, a positive treatment effect, original-paper reproduction or thesis
acceptance. Earlier audit records such as `repair_validation.md` describe their
own historical snapshots, not additional results from this run.
