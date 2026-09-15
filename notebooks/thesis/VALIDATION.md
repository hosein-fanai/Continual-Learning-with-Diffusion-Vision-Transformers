# Thesis workflow validation — audit of 15 September 2026

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
