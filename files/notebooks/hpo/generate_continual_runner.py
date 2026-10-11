"""Author the two-container continual HPO notebook on an online container."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _cell(kind: str, cell_id: str, source: str) -> dict:
    """Build one unexecuted notebook cell from maintained source."""

    cell = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}
    # Only executable cells carry execution state.
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def make_notebook() -> dict:
    """Compose a shared launch notebook for two independently admitted GPU hosts."""

    cells = [
        _cell("markdown", "overview", """# Continual DiT HPO: two 8-A100 containers, twelve hours

Open this notebook on either prepared container using **Python (Continual TF2.20)**.
One launch starts both waiting host services through the shared workspace.

- **16 A100 80 GB GPUs:** 24 search workers per host, three per GPU.
- **One persistent 12-hour clock:** eight hours for search, four for confirmations.
- **96 completed search trials is the planning target**, 48 per host. Search keeps
  refilling free slots up to 96 completions per host or the eight-hour cutoff.
  This higher ceiling permits extra trials when cheaper recipes finish early.
- **Four global finalists, three paired seeds:** twelve confirmations in total,
  eight on host A and four on host B. Selection ranks both searches together.

The two Optuna studies have independent databases and adaptive histories, using
proposal/model seeds 42 and 43. Their task seed remains 42. This is not a single
distributed study. Parallel startup suggestions precede most observations;
completed trial count is not sequential TPE update count. Deadlines take priority
over targets; neither a fixed completion count nor convergence is guaranteed.
"""), 
        _cell("markdown", "protocol", """## Fixed scientific protocol

Every trial has **five two-class CIFAR10 tasks**, with original-label groups
`[7,3], [2,8], [5,6], [9,4], [0,1]`, fixed across both hosts and confirmations.
Optuna maximizes task-balanced final average accuracy.

**Every evaluation uses official test rows.** No training holdout is created.
The native validation metric namespace contains official-test scores. These are
test-guided selection results, not untouched generalization estimates. All models
start fresh; raw networks are used with EMA disabled.

The categorical catalog retains **four students, three classifier specialists
and three noise specialists**: joint165/joint155 clean/noisy students; joint165
clean/noisy and joint135 transfer classifier specialists; UNet23, CIFAR10 DiT1351
and joint165 noisy epsilon specialists. Previous teachers are student snapshots.
Clean/noisy models retain paired training and testing settings. Noisy evaluation
uses one sampled view below timestep64, unlike the historical32-view ensemble.

This deployment uses **10 epochs per task, batch128, generative replay only**.
Active current specialists share the task budget with native optimizer recipes.
The unmeasured cumulative-data workload and batch64 throughput are excluded from
this bounded recipe. It changes fidelity from the earlier20-epoch proposal and
requires a fresh campaign. Equal epochs do not imply equal compute across routes.
"""), 
        _cell("code", "remote-setup", '''"""Launch the prepared two-container continual HPO campaign."""

import os
import platform
import sys

from pathlib import Path


# The project prohibits task computation on a laptop.
if platform.system() != "Linux" or not Path("/.dockerenv").exists():
    raise RuntimeError("Open this notebook on a prepared online GPU container.")
# The coordinator must not reserve a GPU context before worker admission.
if "tensorflow" in sys.modules or "keras" in sys.modules:
    raise RuntimeError("Restart the kernel and run setup before importing TensorFlow.")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("KERAS_HOME", "/workspace/.keras")
CHECKOUT_ROOT = Path("/workspace/dit_continual_twohost_20261011/checkout")
CAMPAIGN_ROOT = Path("/workspace/dit_continual_twohost_20261011/campaign_three_seeds")
# Only the isolated deployed checkout contains the matching cluster implementation.
if not (CHECKOUT_ROOT / "common/dit_continual_cluster_runtime.py").is_file():
    raise RuntimeError("The prepared two-container checkout is missing.")
os.chdir(CHECKOUT_ROOT)
# Keep the prepared source ahead of unrelated notebook checkouts.
if str(CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHECKOUT_ROOT))

from common.dit_continual_candidates import CATALOG_EVIDENCE_NOTES, candidate_profile
from common.dit_continual_cluster_runtime import (
    campaign_settings, prepare_cluster, launch_cluster, cluster_status
)


RUN_HPO = False  # Set True after reviewing the ready hosts and recipe below.
settings = campaign_settings()
display(settings)
'''), 
        _cell("markdown", "capacity", """## Capacity and timing evidence

Three simultaneous workers completed all five tasks at two epochs per task on
one A10080GB in36-40minutes, using about73.3GiB together. Each worker has a
24,576MiB TensorFlow cap plus allocator overhead. Both new hosts need independent
runtime, GPU identity, ownership and reservation checks.

The pilot-based work model projects about **three hours per concurrent10-epoch
trial** at maximum revised replay. Two waves of48 trials take about six hours.
A20%planning margin gives a3.2-hour per-trial threshold for two search waves in
eight hours and one confirmation wave in four hours. This is a projection, not a
complete measured run or an upper bound; other candidate combinations and host
contention remain uncertain.

The services continuously refill24 search slots per host. There is no blocking
32-trial review stage or small-batch drain. At the cutoff, owned incomplete
workers are stopped and reaped, retaining partial checkpoints. Confirmation
starts after both exports are frozen; transfer/wait time remains on the same
clock. Rerunning cells cannot extend that clock. Initialization, evaluation and
completion tails can temporarily leave GPUs idle; constant100%utilization is
not promised.
"""), 
        _cell("code", "prepare", """request = prepare_cluster(CAMPAIGN_ROOT, CHECKOUT_ROOT, settings=settings)
display(request)
profile = candidate_profile(task_seed=settings["task_seed"])
display([
    {"role": role, "candidate": candidate_id, **item["provenance"]}
    for role in ("student_candidates", "classifier_teacher_candidates", "noise_teacher_candidates")
    for candidate_id, item in profile[role].items()
])
display(CATALOG_EVIDENCE_NOTES)
display(cluster_status(CAMPAIGN_ROOT))
"""), 
        _cell("markdown", "launch-notes", """## Start both containers

Each prepared host service waits without training for the shared start request.
Launch requires fresh readiness from both expected hosts,16distinct GPUs, and
matching source/runtime/recipe identities. The services then search, select four
global finalists and run confirmations automatically; the notebook need not stay
connected. No inter-container SSH or private key is needed.

Each host keeps SQLite transactions on its own local disk with persistent
snapshots in a distinct shared output directory. The shared volume carries
control artifacts, configuration identities and results, never one database
written by both hosts. Do not start a second coordinator for a shard.
"""), 
        _cell("code", "launch", """# Explicit launch publishes one idempotent start request for both waiting services.
if RUN_HPO:
    display(launch_cluster(CAMPAIGN_ROOT))
# Setup leaves all training disabled until the launch switch is enabled.
else:
    print("Ready for review. Set RUN_HPO=True and run this cell to start both hosts.")
"""), 
        _cell("code", "status", """# This read-only status cell cannot extend the campaign clock.
display(cluster_status(CAMPAIGN_ROOT))
"""), 
        _cell("markdown", "evidence", """## Results

Read `global_confirmation_summary.json` in the campaign directory for paired
three-seed means, standard deviations and completeness. Partial seed sets are
reported without being labeled comparable. The global manifest preserves source
host/trial, exact YAML hashes and unique global candidate IDs.

Retain both study histories, task checkpoints, official-test matrices, TensorBoard
logs, shared clock, host identities and completion receipts. Candidate evidence
and historical pilot details are in`DIT_CONTINUAL_HPO_DESIGN.md`beside this notebook.
""")
    ]
    metadata = {
        "kernelspec": {"display_name": "Python (Continual TF2.20)", "language": "python", "name": "continual-tf220"}, 
        "language_info": {"name": "python", "version": "3.12.3"}, 
        "protocol_version": 3, "execution": "remote_only", "unexecuted": True
    }
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    """Write only the maintained continual runner notebook."""

    destination = ROOT / "DiT_Continual_HPO_Runner.ipynb"
    destination.write_text(json.dumps(make_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


# Importing the builder never rewrites the notebook.
if __name__ == "__main__":
    main()
