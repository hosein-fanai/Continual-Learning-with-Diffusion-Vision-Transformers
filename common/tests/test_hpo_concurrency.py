"""Real Optuna persistence around a controlled, asynchronous training boundary.

These checks exercise scheduling and bookkeeping without launching expensive HPO
fits. The separate transport tests exercise actual subprocesses and TensorFlow.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from collections.abc import Iterator
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import time as wall_time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd

from common.config import Config, load_config, save_config
from common.hpo import run_hpo


def _training_result(config: Config, **kwargs: object) -> dict[str, object]:
    """Produce distinctive trial scores and a normal, isolated report directory."""
    del kwargs
    number = config.hpo["trial_number"]
    directory = Path(config.training.results_path) / f"trial-{number:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "results_path": str(directory),
        "history": {"classifier_accuracy": [0.99]},
        "evaluations": {
            "valset_network_eval": {
                "classifier_accuracy": 0.4 + number / 1000,
                "noise_loss": 0.2 + number / 2000,
            },
            "testset_network_eval": {"classifier_accuracy": 0.98},
            "valset_ema_eval": {"classifier_accuracy": 0.97, "noise_loss": 0.01},
        },
    }


class _Workers:
    """Deterministic fake children that finish independently when polled."""

    def __init__(self, *, delays: dict | None = None, outcomes: dict | None = None) -> None:
        """Track worker launches, scripted poll delays, and result envelopes."""
        self.delays = delays or {}
        self.outcomes = outcomes or {}
        self.handles = []
        self.events = []
        self.active = set()
        self.max_active = 0
        self.stopped = []
        self.after_start = None

    def start(self, config_path: Path, output_path: Path, log_path: Path, **kwargs: object) -> SimpleNamespace:
        """Load the coordinator's actual YAML and expose a pollable child handle."""
        config = load_config(config_path)
        number = config.hpo["trial_number"]
        handle = SimpleNamespace(
            config=config, number=number, output_path=Path(output_path),
            log_path=Path(log_path), launch_options=kwargs,
            remaining=self.delays.get(number, 0), exit_code=None,
        )

        def poll() -> int | None:
            """Consume one scripted delay and report a stable terminal status."""
            # Repeated polling never changes a terminal result.
            if handle.exit_code is not None:
                return handle.exit_code
            # Different delays model children that complete out of launch order.
            if handle.remaining > 0:
                handle.remaining -= 1
                return None
            handle.exit_code = 0
            self.active.discard(number)
            self.events.append(("exit", number))
            return 0

        handle.process = SimpleNamespace(poll=poll, pid=1000 + number)
        self.handles.append(handle)
        self.active.add(number)
        self.events.append(("start", number))
        self.max_active = max(self.max_active, len(self.active))
        # Hooks can expire the scheduler clock or probe coordinator locking.
        if self.after_start is not None:
            self.after_start(handle)
        return handle

    def finish(self, handle: SimpleNamespace) -> dict[str, object]:
        """Publish either a configured outcome or normal trial metrics."""
        self.events.append(("finish", handle.number))
        outcome = self.outcomes.get(handle.number)
        # Exceptions model transport crashes and notebook interruption.
        if isinstance(outcome, BaseException):
            raise outcome
        # Callable outcomes can corrupt a real saved handoff for negative cases.
        if callable(outcome):
            return outcome(handle)
        # Explicit envelopes model pruning, OOM, and programming failures.
        if outcome is not None:
            return outcome
        return self.success(handle)

    @staticmethod
    def success(handle: SimpleNamespace) -> dict[str, object]:
        """Return the same resolved config and flat metric envelope as a worker."""
        result = _training_result(handle.config)
        config_path = Path(result["results_path"]) / "config.yaml"
        handle.config.training.results_path = result["results_path"]
        save_config(handle.config, config_path)
        return dict(result, status="complete", config_path=str(config_path))

    def stop(self, handles: list[SimpleNamespace]) -> None:
        """Record cleanup and make every supplied child terminal."""
        for handle in handles:
            self.stopped.append(handle.number)
            self.active.discard(handle.number)
            handle.exit_code = -15
            self.events.append(("stop", handle.number))

    @contextmanager
    def installed(self) -> Iterator[_Workers]:
        """Replace only the subprocess boundary; keep persistence and scoring real."""
        with ExitStack() as stack:
            stack.enter_context(patch("common.hpo.start_worker", side_effect=self.start))
            stack.enter_context(patch("common.hpo.finish_worker", side_effect=self.finish))
            stack.enter_context(patch("common.hpo.stop_workers", side_effect=self.stop))
            stack.enter_context(patch("common.hpo.time.sleep"))
            stack.enter_context(patch("common.hpo.main", side_effect=AssertionError(
                "Concurrent training must not execute in the coordinator."
            )))
            stack.enter_context(patch("common.hpo.tf.keras.backend.clear_session", side_effect=AssertionError(
                "The coordinator must not reset the notebook's Keras session."
            )))
            yield self


class HpoConcurrencyTests(unittest.TestCase):
    """Validate concurrent orchestration using real SQLite and YAML artifacts."""

    def setUp(self) -> None:
        """Give each regression its own disposable studies and artifacts."""
        temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def _space() -> dict[str, list]:
        """Keep valid small architectures while retaining all classifier recipes."""
        return {
            "optimizer": ["adam"], "dim": [32], "mha_num_heads": [4],
            "depth": [3], "clf_depth": [1], "patch_size": [4],
            "feature_aggregation": ["last"],
            "clf_train_batch_fraction": [0.0, 0.25, 0.5],
            "clf_train_noisy_input_type": ["noisy", "clean"],
            "clf_train_class_input_type": ["null_class_only", "all_classes"],
        }

    def _options(self, **changes: object) -> dict[str, object]:
        """Return a reproducible joint profile request with optional replacements."""
        options = {
            "task": "joint", "model_name": "dit_classifier",
            "dataset_name": "cifar10", "search_profile": "joint_dit_classifier",
            "trial_budget_mode": "total", "n_trials": 3, "epochs": 1,
            "seed": 17, "n_startup_trials": 1, "results_path": str(self.root),
            "search_space_overrides": self._space(), "validation_source": "test",
            "validation_ratio": 0.0, "concurrent_trials": 2,
        }
        options.update(changes)
        return options

    @staticmethod
    def _study_root(study: optuna.study.Study) -> Path:
        """Resolve the authoritative artifact root from an allocated config."""
        config = load_config(study.trials[0].user_attrs["config_path"])
        return Path(config.hpo["study_root"])

    def _load_study(self) -> optuna.study.Study:
        """Reload the sole local study after an intentionally aborted call."""
        database, = self.root.rglob("study.db")
        storage = "sqlite:///" + database.resolve().as_posix()
        summaries = optuna.get_all_study_summaries(storage)
        self.assertEqual(len(summaries), 1)
        return optuna.load_study(study_name=summaries[0].study_name, storage=storage)

    def test_parallel_controls_are_validated_before_storage(self) -> None:
        """Reject unsupported runtime controls before a directory or study exists."""
        cases = [
            {"concurrent_trials": value}
            for value in (0, -1, True, np.bool_(True), 1.5, "2", None)
        ] + [
            {"worker_gpu_memory_limit_mb": value}
            for value in (0, -1, True, np.bool_(True), float("nan"), float("inf"), "1024")
        ] + [
            {"concurrent_trials": 1, "worker_gpu_memory_limit_mb": 4096},
            {"search_profile": None},
            {"task": "classification", "model_name": "cnn", "search_profile": None},
            {"fit_method": "fit_progressively"},
            {"teacher_network": object()},
            {"use_distillation": True},
        ]
        for index, changes in enumerate(cases):
            destination = self.root / str(index)
            with self.subTest(changes=changes), patch("optuna.create_study") as create:
                with self.assertRaises(ValueError):
                    run_hpo(**self._options(results_path=str(destination), **changes))
                create.assert_not_called()
                self.assertFalse(destination.exists())

    def test_one_worker_preserves_default_sequential_training(self) -> None:
        """The default count continues to use ordinary in-process training."""
        with patch("common.hpo.main", side_effect=_training_result) as training, \
                patch("common.hpo.start_worker") as start:
            study = run_hpo(**self._options(concurrent_trials=1, n_trials=1))
        training.assert_called_once()
        start.assert_not_called()
        self.assertEqual(study.trials[0].values, [0.4, 0.2])

    def test_slots_refill_before_a_slower_trial_finishes(self) -> None:
        """A free slot starts fresh work while an older trial is still running."""
        workers = _Workers(delays={0: 15})
        with workers.installed():
            study = run_hpo(**self._options(n_trials=4))
        self.assertEqual(workers.max_active, 2)
        self.assertLess(workers.events.index(("start", 2)), workers.events.index(("exit", 0)))
        self.assertFalse(workers.active)
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE"] * 4)
        for trial in study.trials:
            self.assertEqual(trial.values, [0.4 + trial.number / 1000, 0.2 + trial.number / 2000])

    def test_serial_parallel_serial_resume_preserves_study_and_per_trial_runtime(self) -> None:
        """Changing execution mode reuses the study and records each trial's mode."""
        with patch("common.hpo.main", side_effect=_training_result):
            first = run_hpo(**self._options(n_trials=1, concurrent_trials=1))
        root = self._study_root(first)
        identity = dict(first.user_attrs["study_spec"])
        workers = _Workers()
        with workers.installed():
            expanded = run_hpo(**self._options(n_trials=3, resume_from=root))
        with patch("common.hpo.main", side_effect=_training_result) as training:
            reopened = run_hpo(**self._options(n_trials=3, concurrent_trials=1, resume_from=root))
        training.assert_not_called()
        self.assertEqual(expanded.study_name, first.study_name)
        self.assertEqual(reopened.user_attrs["study_spec"], identity)
        self.assertEqual(reopened.user_attrs["execution"]["concurrent_trials"], 1)
        self.assertEqual(len(reopened.trials), 3)
        self.assertEqual(len(workers.handles), 2)
        for trial, expected in zip(reopened.trials, (1, 2, 2)):
            self.assertEqual(trial.user_attrs["execution"]["concurrent_trials"], expected)
            config = load_config(trial.user_attrs["resolved_config_path"])
            self.assertEqual(config.hpo["execution"]["concurrent_trials"], expected)

    def test_fewer_trials_than_workers_never_overallocates(self) -> None:
        """The allocation budget dominates the maximum worker count."""
        for budget in (1, 2):
            with self.subTest(budget=budget):
                workers = _Workers(delays={0: 3, 1: 3})
                with workers.installed():
                    study = run_hpo(**self._options(
                        results_path=str(self.root / str(budget)),
                        n_trials=budget, concurrent_trials=4,
                        worker_gpu_memory_limit_mb=4096.0,
                    ))
                self.assertEqual(len(study.trials), budget)
                self.assertEqual(len(workers.handles), budget)
                self.assertLessEqual(workers.max_active, budget)
                for handle in workers.handles:
                    self.assertEqual(handle.launch_options["gpu_memory_limit_mb"], 4096.0)

    def test_total_additional_and_changed_concurrency_preserve_identity(self) -> None:
        """Runtime count changes preserve scientific identity and budget semantics."""
        workers = _Workers()
        with workers.installed():
            first = run_hpo(**self._options(n_trials=2))
            study_root = self._study_root(first)
            identity = dict(first.user_attrs["study_spec"])
            original = [trial.params for trial in first.trials]
            repeated = run_hpo(**self._options(n_trials=2, concurrent_trials=4))
            self.assertEqual(len(repeated.trials), 2)
            smaller = run_hpo(**self._options(n_trials=1, concurrent_trials=3))
            self.assertEqual(len(smaller.trials), 2)
            expanded = run_hpo(**self._options(
                n_trials=3, concurrent_trials=3, resume_from=study_root,
                results_path=str(self.root / "unused-root"),
            ))
            self.assertEqual(len(expanded.trials), 3)
            appended = run_hpo(**self._options(
                n_trials=2, trial_budget_mode="additional", concurrent_trials=4,
                resume_from=study_root,
            ))
        self.assertEqual(len(appended.trials), 5)
        self.assertEqual(len(workers.handles), 5)
        self.assertEqual(appended.user_attrs["study_spec"], identity)
        self.assertEqual([trial.params for trial in appended.trials[:2]], original)
        self.assertFalse((self.root / "unused-root").exists())
        for trial in appended.trials:
            for key in ("config_path", "resolved_config_path", "results_path", "checkpoint_dir"):
                self.assertTrue(Path(trial.user_attrs[key]).is_relative_to(study_root))
        self.assertEqual(pd.read_csv(study_root / "trials.csv")["number"].tolist(), list(range(5)))

    def test_waiting_trials_obey_total_budget_and_keep_their_parameters(self) -> None:
        """Only queued trial numbers within a requested total allowance execute."""
        workers = _Workers()
        with workers.installed():
            study = run_hpo(**self._options(n_trials=1))
            params = dict(study.trials[0].params)
            for fraction in (0.25, 0.5, 0.0):
                study.enqueue_trial(dict(params, clf_train_batch_fraction=fraction))
            resumed = run_hpo(**self._options(n_trials=3, resume_from=self._study_root(study)))
        self.assertEqual(len(workers.handles), 3)
        self.assertEqual([trial.state.name for trial in resumed.trials],
                         ["COMPLETE", "COMPLETE", "COMPLETE", "WAITING"])
        self.assertEqual([trial.params["clf_train_batch_fraction"] for trial in resumed.trials[1:3]],
                         [0.25, 0.5])

    def test_abandoned_trial_recovery_is_once_and_within_total_budget(self) -> None:
        """Interrupted allocations retain parameters without duplicating retries."""
        workers = _Workers()
        with workers.installed():
            study = run_hpo(**self._options(n_trials=1))
            original = study.trials[0]
            abandoned = study.ask(fixed_distributions=original.distributions)
            root = self._study_root(study)
            exhausted = run_hpo(**self._options(n_trials=2, resume_from=root))
            self.assertEqual(len(workers.handles), 1)
            self.assertEqual([trial.state.name for trial in exhausted.trials], ["COMPLETE", "RUNNING"])
            resumed = run_hpo(**self._options(n_trials=3, resume_from=root))
            repeated = run_hpo(**self._options(n_trials=3, resume_from=root))
        self.assertEqual(len(workers.handles), 2)
        self.assertEqual(len(repeated.trials), 3)
        retry = resumed.trials[2]
        self.assertEqual(retry.params, abandoned.params)
        self.assertEqual(retry.user_attrs["resume_source_trial_number"], abandoned.number)
        self.assertEqual(retry.user_attrs["resume_original_trial_number"], abandoned.number)

    def test_all_twelve_recipes_for_both_datasets_keep_scores_and_artifacts_paired(self) -> None:
        """Exercise all 24 dataset/recipe combinations with out-of-order finishes."""
        dimensions = tuple(product((0.0, 0.25, 0.5), ("noisy", "clean"),
                                   ("null_class_only", "all_classes")))
        keys = ("clf_train_batch_fraction", "clf_train_noisy_input_type", "clf_train_class_input_type")
        for dataset in ("cifar10", "cifar100"):
            with self.subTest(dataset=dataset):
                workers = _Workers(delays={1: 8, 3: 4})
                options = self._options(dataset_name=dataset, results_path=str(self.root / dataset))
                with workers.installed():
                    study = run_hpo(**dict(options, n_trials=1))
                    params = dict(study.trials[0].params)
                    for values in dimensions:
                        study.enqueue_trial(dict(params, **dict(zip(keys, values))))
                    study = run_hpo(**dict(options, n_trials=13, concurrent_trials=3))
                self.assertEqual(len(study.trials), 13)
                self.assertEqual(len({trial.user_attrs["results_path"] for trial in study.trials}), 13)
                for trial, expected in zip(study.trials[1:], dimensions):
                    config = load_config(trial.user_attrs["resolved_config_path"])
                    self.assertEqual(config.dataset.name, dataset)
                    self.assertEqual(config.hpo["trial_number"], trial.number)
                    self.assertEqual(tuple(trial.params[key] for key in keys), expected)
                    self.assertEqual(tuple(config.model.wrapper_kwargs[key] for key in keys), expected)
                    self.assertEqual(config.training.fit_kwargs, {"validation_freq": []})
                    self.assertEqual(config.hpo["objectives"], trial.values)
                    self.assertEqual(trial.values, [0.4 + trial.number / 1000, 0.2 + trial.number / 2000])
                    self.assertTrue((Path(trial.user_attrs["results_path"]) / "objectives.csv").is_file())

    def test_timeout_stops_new_allocations_and_drains_active_trials(self) -> None:
        """Crossing the wall-time budget preserves work already in flight."""
        clock = SimpleNamespace(now=0.0)
        workers = _Workers(delays={0: 5, 1: 2})

        def after_start(handle: SimpleNamespace) -> None:
            """Expire the fake clock after the two initial slots fill."""
            # Both initial workers start before the timeout is crossed.
            if handle.number == 1:
                clock.now = 2.0

        workers.after_start = after_start
        timer = SimpleNamespace(monotonic=lambda: clock.now, sleep=lambda seconds: None, time=wall_time)
        with workers.installed(), patch("common.hpo.time", timer):
            study = run_hpo(**self._options(n_trials=6, timeout=1.0))
        self.assertEqual(len(workers.handles), 2)
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "COMPLETE"])
        self.assertFalse(workers.active)
        self.assertFalse(workers.stopped)

    def test_pruning_oom_and_nonfinite_final_score_allow_later_trials(self) -> None:
        """Scientific pruning and resource failures do not abort later candidates."""
        evidence = {"reason": "nonfinite_loss", "phase": "joint", "metric": "loss", "value": "nan"}
        workers = _Workers(outcomes={
            0: {"status": "pruned", "error": "Diverged", "divergence": evidence,
                "divergence_path": None, "results_path": str(self.root / "partial")},
            1: {"status": "oom", "error": "Synthetic out of memory"},
        })

        def nonfinite(handle: SimpleNamespace) -> dict[str, object]:
            """Model a finite fit followed by divergent final noise evaluation."""
            result = workers.success(handle)
            result["evaluations"]["valset_network_eval"]["noise_loss"] = float("inf")
            return result

        workers.outcomes[2] = nonfinite
        with workers.installed():
            study = run_hpo(**self._options(n_trials=4))
        self.assertEqual([trial.state.name for trial in study.trials], ["PRUNED", "FAIL", "PRUNED", "COMPLETE"])
        self.assertEqual(study.trials[0].user_attrs["divergence"], evidence)
        self.assertIn("out of memory", study.trials[1].user_attrs["worker_error"])
        self.assertEqual(study.trials[2].user_attrs["divergence"]["reason"], "nonfinite_objective")
        for trial in study.trials[:3]:
            self.assertIsNone(trial.values)
        root = self._study_root(study)
        self.assertEqual(pd.read_csv(root / "trials.csv")["state"].tolist(),
                         ["PRUNED", "FAIL", "PRUNED", "COMPLETE"])
        self.assertEqual([trial.number for trial in study.best_trials], [3])

    def test_unexpected_worker_and_transport_errors_abort_and_stop_other_children(self) -> None:
        """Programming and protocol errors fail visibly and stop surviving workers."""
        for index, outcome in enumerate((
            {"status": "error", "error": "Synthetic programming error"},
            RuntimeError("Worker crashed before publishing valid JSON"),
        )):
            with self.subTest(outcome=outcome):
                workers = _Workers(delays={1: 100}, outcomes={0: outcome})
                with workers.installed(), self.assertRaises(RuntimeError):
                    run_hpo(**self._options(n_trials=5, results_path=str(self.root / str(index))))
                self.assertEqual(len(workers.handles), 2)
                self.assertFalse(workers.active)
                self.assertIn(1, workers.stopped)
                database, = (self.root / str(index)).rglob("study.db")
                storage = "sqlite:///" + database.resolve().as_posix()
                summary, = optuna.get_all_study_summaries(storage)
                study = optuna.load_study(study_name=summary.study_name, storage=storage)
                self.assertEqual([trial.state.name for trial in study.trials], ["FAIL", "RUNNING"])
                self.assertIn("worker_error", study.trials[0].user_attrs)

    def test_invalid_worker_trial_identity_aborts_without_assigning_another_score(self) -> None:
        """A mismatched child config cannot contribute another trial's metrics."""
        workers = _Workers(delays={1: 100})

        def mismatched(handle: SimpleNamespace) -> dict[str, object]:
            """Write the wrong trial number into an otherwise valid handoff."""
            result = workers.success(handle)
            config = load_config(result["config_path"])
            config.hpo["trial_number"] = 99
            save_config(config, result["config_path"])
            return result

        workers.outcomes[0] = mismatched
        with workers.installed(), self.assertRaisesRegex(ValueError, "does not match"):
            run_hpo(**self._options(n_trials=4))
        study = self._load_study()
        self.assertEqual([trial.state.name for trial in study.trials], ["FAIL", "RUNNING"])
        self.assertTrue(all(trial.values is None for trial in study.trials))
        self.assertFalse(workers.active)

    def test_missing_final_evaluation_aborts_instead_of_scoring_training_history(self) -> None:
        """No fallback may replace a missing final validation objective."""
        workers = _Workers(delays={1: 100})

        def missing_validation(handle: SimpleNamespace) -> dict[str, object]:
            """Remove only the ordinary validation report from a valid envelope."""
            result = workers.success(handle)
            result["evaluations"].pop("valset_network_eval")
            return result

        workers.outcomes[0] = missing_validation
        with workers.installed(), self.assertRaises(KeyError):
            run_hpo(**self._options(n_trials=4))
        study = self._load_study()
        self.assertEqual([trial.state.name for trial in study.trials], ["FAIL", "RUNNING"])
        self.assertFalse(workers.active)

    def test_keyboard_interrupt_stops_children_and_permits_parameter_identical_retry(self) -> None:
        """Notebook cancellation stops every child and leaves retriable records."""
        workers = _Workers(delays={1: 100}, outcomes={0: KeyboardInterrupt()})
        with workers.installed(), self.assertRaises(KeyboardInterrupt):
            run_hpo(**self._options(n_trials=4))
        interrupted = self._load_study()
        self.assertEqual([trial.state.name for trial in interrupted.trials], ["RUNNING", "RUNNING"])
        self.assertEqual(set(workers.stopped), {0, 1})
        self.assertFalse(workers.active)
        root = self._study_root(interrupted)
        retries = _Workers()
        with retries.installed():
            study = run_hpo(**self._options(n_trials=4, resume_from=root))
        self.assertEqual([trial.state.name for trial in study.trials], ["RUNNING", "RUNNING", "COMPLETE", "COMPLETE"])
        for source, retry in zip(study.trials[:2], study.trials[2:]):
            self.assertEqual(retry.params, source.params)
            self.assertEqual(retry.user_attrs["resume_source_trial_number"], source.number)

    def test_configuration_error_persists_sampler_and_stops_launched_children(self) -> None:
        """A configuration failure must not orphan an already launched trial."""
        from common.hpo import _build_trial_config

        workers = _Workers(delays={0: 100})

        def builder(trial: optuna.trial.Trial, *args: object, **kwargs: object) -> Config:
            """Let one child start before the next configuration fails."""
            # The second allocation fails after the first worker is active.
            if trial.number == 1:
                raise ValueError("Synthetic configuration failure")
            return _build_trial_config(trial, *args, **kwargs)

        with workers.installed(), patch("common.hpo._build_trial_config", side_effect=builder), \
                self.assertRaisesRegex(ValueError, "configuration failure"):
            run_hpo(**self._options(n_trials=4))
        study = self._load_study()
        self.assertEqual([trial.state.name for trial in study.trials], ["RUNNING", "FAIL"])
        self.assertIn("sampler_rng_state", study.user_attrs)
        self.assertEqual(workers.stopped, [0])

    def test_launch_failure_records_failed_allocation_and_cleans_existing_worker(self) -> None:
        """An operating-system launch error releases every already running child."""
        workers = _Workers(delays={0: 100})

        def start(config_path: Path, *args: object, **kwargs: object) -> SimpleNamespace:
            """Fail the second launch after the initial worker occupies its slot."""
            # The first child starts normally; the second launch cannot create one.
            if load_config(config_path).hpo["trial_number"] == 1:
                raise OSError("Synthetic process launch failure")
            return workers.start(config_path, *args, **kwargs)

        with workers.installed(), patch("common.hpo.start_worker", side_effect=start), \
                self.assertRaisesRegex(OSError, "process launch failure"):
            run_hpo(**self._options(n_trials=4))
        study = self._load_study()
        self.assertEqual([trial.state.name for trial in study.trials], ["RUNNING", "FAIL"])
        self.assertEqual(workers.stopped, [0])
        self.assertFalse(workers.active)

    def test_active_coordinator_blocks_reentry_before_allocating_another_trial(self) -> None:
        """A second coordinator cannot mark active work abandoned or resample it."""
        workers = _Workers()
        rejected = []

        def reenter(handle: SimpleNamespace) -> None:
            """Try sequential reentry while the concurrent owner holds its lock."""
            with self.assertRaisesRegex(RuntimeError, "coordinator.*lock"):
                run_hpo(**self._options(n_trials=1, concurrent_trials=1))
            rejected.append(handle.number)

        workers.after_start = reenter
        with workers.installed():
            study = run_hpo(**self._options(n_trials=2))
        self.assertEqual(rejected, [0, 1])
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "COMPLETE"])
        self.assertEqual(len(workers.handles), 2)

    def test_two_real_cpu_workers_train_and_publish_through_the_coordinator(self) -> None:
        """Run the actual scheduler, two isolated fits, evaluation, and Optuna storage."""
        from common.hpo import _build_trial_config

        processes = []
        overlap = []
        original_popen = subprocess.Popen
        fixture = (
            "import sys,runpy,numpy as np\n"
            "from unittest.mock import patch\n"
            "rng=np.random.default_rng(17)\n"
            "data=((rng.integers(0,256,(8,32,32,3),dtype=np.uint8),(np.arange(8)%2).reshape(-1,1)),"
            "(rng.integers(0,256,(4,32,32,3),dtype=np.uint8),(np.arange(4)%2).reshape(-1,1)))\n"
            "sys.argv=['common.hpo_worker',*sys.argv[1:]]\n"
            "with patch('tensorflow.keras.datasets.cifar10.load_data',return_value=data):\n"
            " runpy.run_module('common.hpo_worker',run_name='__main__')\n"
        )

        def reduced_config(*args: object, **kwargs: object) -> Config:
            """Shrink only compute/reporting after the real profile is constructed."""
            config = _build_trial_config(*args, **kwargs)
            config.dataset.batch_size = 4
            config.model.show_network_summary = False
            config.model.kwargs.update(dim=8, depth=1, mha_num_heads=1,
                                       clf_mha_num_heads=1, timesteps=4)
            config.model.wrapper_kwargs.update(test_steps=2)
            config.training.verbose = 0
            config.training.tensorboard = False
            config.reporting.save_history_plot = False
            config.reporting.save_final_images = False
            config.reporting.save_final_gifs = False
            config.hpo["software_check"] = "Synthetic CIFAR rows; reduced architecture and budget"
            return config

        def synthetic_worker(command: list[str], **options: object) -> subprocess.Popen:
            """Keep the real launcher and substitute only child dataset loading."""
            overlap.append(any(process.poll() is None for process in processes))
            process = original_popen([command[0], "-u", "-c", fixture, *command[4:]], **options)
            processes.append(process)
            return process

        space = self._space()
        space.update(clf_train_batch_fraction=[0.5], clf_train_noisy_input_type=["clean"],
                     clf_train_class_input_type=["null_class_only"])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3"}), \
                patch("common.hpo._build_trial_config", side_effect=reduced_config), \
                patch("common.hpo_process.subprocess.Popen", side_effect=synthetic_worker), \
                patch("common.hpo.main", side_effect=AssertionError("Training belongs in the children")):
            try:
                study = run_hpo(**self._options(n_trials=2, search_space_overrides=space,
                                               max_train_samples=4, max_val_samples=2))
            except Exception as error:
                details = "\n".join(path.read_text() for path in self.root.rglob("trial-*.log"))
                self.fail(f"Real concurrent training failed: {error}\n{details}")
        self.assertEqual(len(processes), 2)
        self.assertEqual(overlap, [False, True])
        self.assertEqual(len({process.pid for process in processes}), 2)
        self.assertNotIn(os.getpid(), {process.pid for process in processes})
        self.assertEqual([process.poll() for process in processes], [0, 0])
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "COMPLETE"])
        self.assertEqual({trial.user_attrs["worker_pid"] for trial in study.trials},
                         {process.pid for process in processes})
        for trial in study.trials:
            self.assertTrue(all(np.isfinite(value) for value in trial.values))
            config = load_config(trial.user_attrs["resolved_config_path"])
            self.assertEqual(config.hpo["objectives"], trial.values)
            self.assertEqual(config.hpo["trial_number"], trial.number)
            self.assertEqual(config.dataset.split_metadata["training_rows_per_epoch"], 4)
            payload = json.loads(Path(trial.user_attrs["worker_result_path"]).read_text())
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(len(payload["history"]["loss"]), 1)
            self.assertFalse(any(key.startswith("val_") for key in payload["history"]))
            self.assertIn("classifier_accuracy", payload["evaluations"]["valset_network_eval"])
            self.assertIn("noise_loss", payload["evaluations"]["valset_network_eval"])
            self.assertTrue((Path(trial.user_attrs["results_path"]) / "objectives.csv").is_file())


class HpoDatasetLockTests(unittest.TestCase):
    """Keep shared CIFAR cache extraction locked without serializing preprocessing."""

    def _check_loader(self, dataset_name: str, *, fail: bool) -> None:
        """Observe the cache lock around a synthetic Keras loader call."""
        from common import dataloader
        from tensorflow.keras import datasets

        active = []
        transitions = []
        arrays = ((np.zeros((2, 32, 32, 3), dtype=np.uint8), np.array([[0], [1]])),) * 2
        sentinel = object()

        @contextmanager
        def cache_lock(name: str) -> Iterator[None]:
            """Record lock ownership and guarantee release on loader exceptions."""
            self.assertEqual(name, dataset_name)
            active.append(name)
            transitions.append("locked")
            try:
                yield
            finally:
                active.pop()
                transitions.append("unlocked")

        def load() -> tuple:
            """Check that reading and extracting the shared cache owns its lock."""
            self.assertEqual(active, [dataset_name])
            transitions.append("load")
            # Failed downloads or corrupt archives must still release ownership.
            if fail:
                raise OSError("Synthetic CIFAR cache failure")
            return arrays

        def preprocess(*args: object, **kwargs: object) -> object:
            """Verify per-trial preprocessing is outside the cache critical section."""
            del args, kwargs
            self.assertFalse(active)
            transitions.append("preprocess")
            return sentinel

        with patch.object(dataloader, "dataset_load_lock", side_effect=cache_lock), \
                patch.object(getattr(datasets, dataset_name), "load_data", side_effect=load), \
                patch.object(dataloader, "preprocess_dataset", side_effect=preprocess) as preprocessing:
            loader = getattr(dataloader, "load_" + dataset_name)
            # Failure exits before preprocessing and retains the source exception.
            if fail:
                with self.assertRaisesRegex(OSError, "CIFAR cache failure"):
                    loader(verbose=0)
                preprocessing.assert_not_called()
            # Successful loading passes raw rows to ordinary preprocessing unchanged.
            else:
                self.assertIs(loader(verbose=0), sentinel)
                preprocessing.assert_called_once()
        self.assertFalse(active)
        self.assertEqual(transitions, ["locked", "load", "unlocked"] + ([] if fail else ["preprocess"]))

    def test_cifar_cache_reads_are_locked_but_preprocessing_is_not(self) -> None:
        """Both CIFAR loaders protect extraction and release the lock promptly."""
        for dataset_name in ("cifar10", "cifar100"):
            with self.subTest(dataset=dataset_name):
                self._check_loader(dataset_name, fail=False)

    def test_cifar_cache_lock_is_released_when_loading_fails(self) -> None:
        """A failed cache load cannot leave subsequent trials blocked."""
        for dataset_name in ("cifar10", "cifar100"):
            with self.subTest(dataset=dataset_name):
                self._check_loader(dataset_name, fail=True)


# Direct execution uses the same test cases as repository discovery.
if __name__ == "__main__":
    unittest.main()
