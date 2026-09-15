# Minimum Route One thesis experiments

These notebooks use TensorFlow 2.20 and native Keras 3. Select the project's TensorFlow kernel, restart it, and run one notebook from top to bottom. The setup cell finds the repository automatically. Each training notebook runs one complete seed/class-order stream.

**Defense scope:** this is a TMCL-inspired supervised classifier experiment. It can produce the accuracy, forgetting, local backward-transfer and paired-effect values for that claim after complete runs. It does not reproduce TMCL's published protocol or test noise-dependent consolidation. The [scientific audit](SCIENTIFIC_AUDIT.md) explains the comparison, required evidence and limits. No authenticated confirmation campaign was present in this checkout on 15 September 2026.

Training notebooks keep four main steps: select a stream, load data/create the model, train, and save/read results. Development then reviews diagnostics; confirmation diagnostics are optional.

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

## Small, fixed scientific design

Confirmation uses three paired seeds: **1103, 2207, 3301**. CIFAR-10 has five two-class tasks and three methods (9 streams). CIFAR-100 has ten ten-class tasks and five methods (15 streams). Follow the saved **24-row checklist**, one fresh kernel per row. Every method in a block shares the seed and class order. A failed or unfavorable stream cannot be skipped.

The primary comparison is **learned minus extra joint in final clean task-average test accuracy**. Extra joint matches the number of additional optimizer updates. Different phases update different parameters and use different batches; this does not match examples, FLOPs or wall time. Random and CE-only are supporting comparisons. This deliberately reduced design does not reproduce every study or ablation proposed in the thesis guides.

The common platform uses a float32, nonvariational DiTClassifier/V1, dimension 128, depth 4, patch size 4, four heads, raw inference and no EMA. Common APIs own splitting, replay, losses and class growth. Acquisition freezes the backbone and learns class gates. Consolidation trains the classifier projection/head and temporary predictor against a frozen acquired target. Deployed inference uses the clean, unmodulated all-seen classifier, without task identity, gates or predictor. Clean semantic views and uniform reliability do not test noise-dependent reliability claims.

| Budget | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Joint batch / epochs per task | 64 / 40 | 64 / 60 |
| Acquisition / consolidation updates per task | 200 / 400 | 1000 / 2000 |
| Semantic batch / Adam learning rate | 32 / 0.0003 | Same |
| Joint Adam learning rate / decay | 0.0002 / 0 | Same |
| Current / generated-old pool | All permitted current rows / fixed 2048 old rows | Same policy |
| Fixed validation probe | 8 examples per class | Same |
| Recovery interval | 200 completed updates, plus native phase boundaries | Same |

These are reduced starting budgets, **not a promise of short runtime or convergence**. Expected ordinary updates are 30,120/55,080 per stream; learned and extra-joint totals are 33,120/85,080. Measure full-stream learning, replay usefulness, task time, checkpoint overhead and storage in notebook 00, including the last CIFAR-100 task. Adjust only from validation evidence before freezing. The [recipe rationale](HYPERPARAMETER_RATIONALE.md) distinguishes source settings from local choices.

## Validation, test and metrics

Development seed 17 uses only validation outcomes. The configured 20% validation split is excluded from training gradients; retained historical validation probes are disclosed observation access, not rehearsal data. Confirmation evaluates the untouched test split under the frozen procedure. Test outcomes cannot choose settings, stopping rules or seeds.

For `R[i,j]`, accuracy on task `j` after learning task `i`, using all seen classes:

- **Final accuracy:** mean of the final row.
- **Incremental accuracy:** mean of each learned-prefix row mean.
- **Signed forgetting:** mean, over old tasks, of the best earlier accuracy minus final accuracy. The maximum excludes the final row and starts when the task was learned. Negative values are preserved.
- **Backward transfer:** mean final-minus-acquisition accuracy over old tasks.

This local backward-transfer definition differs from TMCL's single-task-reference BT; the notebooks do not calculate the paper's FT. Do not compare those numbers directly.

Compute each metric within a complete stream first. Report individual values, mean, sample SD (`ddof=1`) and actual independent-stream `n`. Accuracy uses percent; forgetting and accuracy differences use percentage points. The native primary paired 95% t interval is separate from SD. **Three pairs give limited precision**; tasks, images and gates do not increase independent `n`.

Phase changes require identical hashed validation examples and positive actual counts. Unavailable endpoints cannot yield numeric differences. Small-cohort CKA is descriptive; invalid two-example CKA stays unavailable. Replay agreement measures classifier self-consistency, and saved grids are qualitative. Sum only complete, disjoint active-task timers. Resumed committed segments are included; measured checkpoint writes and downtime are separate. Uncommitted lost work and interrupted unfinished write timers are unavailable, so these observations are not complete restart-inclusive wall time. Nested phase timers, notebook pauses, sampled RSS, allocator peaks and tensor payloads have distinct scopes.

## Freeze, interruption and rerun

Finish all source and recipe changes before notebook 01. The configured campaign path is `results/thesis_route_one/minimum_v4_tf220/`. Keep an independent unchanged copy of `frozen_design.json`. Existing campaigns are preserved and cannot be resealed silently. Source or scientific-setting changes require a new campaign.

Checkpoint saving is sealed into the recipe. Each stream has its own `checkpoints/<run_id>/` directory. Native recovery preserves completed-task boundaries and the latest two intermediate progress snapshots. The notebook uses the native authenticated selector; it never reconstructs model state itself. After an interruption, restart the kernel and **Run All**: the same unfinished repeat resumes from its latest valid commit, and work after that commit is repeated. Before the first published commit, the same stream restarts while unpublished temporary evidence is retained. Damaged published evidence is reported, not overwritten.

A nonblocking OS lock prevents simultaneous training of one stream in multiple kernels. Kernel exit releases the lock; the remaining lock file does not mean the stream is active. A failed training cell also releases ownership. A valid completed stream is never trained again. If completion publication was interrupted, saved native evidence can repair the index without training or re-evaluation. If only a plot failed after completion, rerun its view or collection.

Runtime checkpoint and resume locators remain separate from the immutable YAML. Do not alter hashes to accept changed code, configuration or evidence. Checkpoint boundaries preserve supported computation state; GPU execution is not a blanket promise of bitwise determinism across devices or software versions.

Development checkpoint identities include the resolved recipe, inherited settings and executable source. Changing any of them starts a separate pilot. This audit changes the development identity once; older checkpoints remain preserved and are not adopted into the revised pilot. Use the retained original source/recipe to continue an old pilot.

## Compact final output

Confirmation notebooks show a small scalar outcome table by default. Choose `SHOW_DIAGNOSTICS=True` before freezing if those training notebooks should also show saved validation, replay and learning plots. Their cell sources are frozen, so do not edit that flag during the campaign; later saved-only diagnostics remain available through notebook 10. Notebook 10 requires all 24 complete streams and displays one treatment summary plus the primary paired effect/interval. Its ZIP retains numeric source rows, exact source/configuration identities and interpretation limits.

`DETAILS=True` adds extended diagnostic tables and figures; choose a separate `OUTPUT` to preserve previous exports. `PROGRESS=True` labels a partial saved-results snapshot and omits final paired inference. Identical exports are authenticated and reused. Collection never loads datasets, trains, predicts or samples new images. Write from the saved observations and preserve uncertainty, missing values and negative findings.

[VALIDATION.md](VALIDATION.md) records software checks and their limits. No synthetic fixture or successful software test is a thesis accuracy result.
