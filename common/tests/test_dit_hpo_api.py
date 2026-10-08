"""Real DiT Optuna persistence and TensorBoard events with mocked training."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import Mock, patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from common.config import load_config
from common.hpo import run_hpo, summarize_hpo
from common.tests.test_hpo_concurrency import _Workers, _training_result


def _nonfinite_result(handle: object) -> dict:
    """Retain a real worker handoff but inject a nonfinite final EMA score."""

    result = _Workers.success(handle)
    result["evaluations"]["valset_ema_eval"]["noise_loss"] = float("nan")
    return result


class _PruningWorkers(_Workers):
    """Model epoch handshakes around real Optuna percentile decisions."""

    def __init__(self, reports: dict[int, list[float]]) -> None:
        """Keep deterministic epoch values and all coordinator decisions."""

        super().__init__()
        self.reports = reports
        self.decisions = []
        self.finished_training = []

    def start(self, config_path: Path, output_path: Path, log_path: Path, **kwargs: object) -> SimpleNamespace:
        """Keep each child active until its epoch reports have been answered."""

        handle = super().start(config_path, output_path, log_path, **kwargs)
        handle.epoch_reports = [
            {"step": step, "value": value, "monitor": "val_noise_loss"}
            for step, value in enumerate(self.reports.get(handle.number, [1.0]))
        ]
        original_poll = handle.process.poll

        def poll() -> int | None:
            """A child waiting for a coordinator decision cannot complete."""

            # Epoch callbacks block before final evaluation and result publication.
            if handle.epoch_reports:
                return None
            return original_poll()

        handle.process.poll = poll
        return handle

    def read_report(self, handle: SimpleNamespace) -> dict | None:
        """Return epoch reports while modeling a faster initial reference trial."""

        # Later workers can train slowly while the first reference finishes.
        if handle.number > 0 and not self.finished_training:
            return None
        return handle.epoch_reports[0] if handle.epoch_reports else None

    def answer_report(self, handle: SimpleNamespace, report: dict, prune: bool) -> None:
        """Resume one epoch or stop before the mocked final training report."""

        self.decisions.append((handle.number, report["step"], prune))
        handle.epoch_reports.pop(0)
        # A pruning decision exits training without entering final evaluation.
        if prune:
            evidence = {"reason": "percentile", **report}
            self.outcomes[handle.number] = {
                "status": "pruned", "pruning": evidence, "pruning_path": None, 
                "error": "Validation loss exceeded the reference percentile."
            }
            handle.epoch_reports.clear()

    def finish(self, handle: SimpleNamespace) -> dict:
        """Track which workers reached their final report instead of pruning."""

        # Only uninterrupted workers can supply a post-training objective.
        if handle.number not in self.outcomes:
            self.finished_training.append(handle.number)
        return super().finish(handle)

    @contextmanager
    def installed(self) -> Iterator[_PruningWorkers]:
        """Keep study persistence real while replacing the epoch transport."""

        with super().installed(), \
                patch("common.hpo.read_pruning_report", side_effect=self.read_report), \
                patch("common.hpo.answer_pruning_report", side_effect=self.answer_report):
            yield self


class DitHpoApiTests(TestCase):
    """Protect shared-API concurrency, exact budgets, objectives, and event paths."""

    def setUp(self) -> None:
        """Create a disposable experiment root without touching real study state."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def options(self, **changes: object) -> dict:
        """Build ordinary generation API arguments with the notebook's protocol."""

        options = {
            "task": "generation", "model_name": "diffusion_transformer", 
            "dataset_name": "CIFAR10", "results_path": str(self.root), 
            "n_trials": 3, "epochs": 50, "n_startup_trials": 40, 
            "trial_budget_mode": "total", "validation_source": "split", 
            "validation_ratio": 0.2, "concurrent_trials": 2, 
            "objective_metrics": ["generation_loss"], 
            "objective_directions": ["minimize"], "seed": 42
        }
        options.update(changes)
        return options

    @staticmethod
    def study_root(study: object) -> Path:
        """Locate the authoritative study root from an actual allocated YAML."""

        config = load_config(study.trials[0].user_attrs["config_path"])
        return Path(config.hpo["study_root"])

    @staticmethod
    def event_value(root: Path, number: int, tag: str) -> float:
        """Read one actual TensorBoard scalar tensor from the shared outcome writer."""

        events = EventAccumulator(str(root / "tensorboard" / f"trial-{number:04d}" / "outcome"))
        events.Reload()
        return events.Tensors(tag)[-1].tensor_proto.float_val[0]

    def test_parallel_cifar_studies_keep_ema_objectives_and_tensorboard(self) -> None:
        """Both supported datasets use bounded workers and real final-outcome events."""

        for dataset in ("CIFAR10", "CIFAR100"):
            with self.subTest(dataset=dataset):
                workers = _Workers(delays={0: 8})
                with workers.installed():
                    study = run_hpo(**self.options(
                        dataset_name=dataset, results_path=str(self.root / dataset), 
                        worker_gpu_memory_limit_mb=12288
                    ))
                root = self.study_root(study)
                self.assertEqual(workers.max_active, 2)
                self.assertEqual(len(study.trials), 3)
                for trial in study.trials:
                    config = load_config(root / "configs" / f"trial-{trial.number:04d}.yaml")
                    self.assertAlmostEqual(trial.value, 0.01)
                    self.assertEqual(config.training.epochs, 50)
                    self.assertEqual(config.training.patience, 5)
                    self.assertTrue(config.training.tensorboard)
                    self.assertEqual(config.training.tensorboard_path, str(root / "tensorboard"))
                    self.assertEqual(config.dataset.validation_source, "split")
                    self.assertEqual(config.dataset.validation_ratio, 0.2)
                    self.assertEqual(config.training.seed, 42)
                    self.assertAlmostEqual(self.event_value(root, trial.number, "hpo/generation_loss"), 0.01)
                    self.assertEqual(self.event_value(root, trial.number, "hpo/completed"), 1.0)
                self.assertFalse((root / "pareto_trials.csv").exists())
                for handle in workers.handles:
                    self.assertEqual(handle.launch_options["gpu_memory_limit_mb"], 12288.0)

    def test_serial_parallel_resume_keeps_identity_paths_and_total_budget(self) -> None:
        """Switching execution modes preserves EMA scoring and all original trial paths."""

        with patch("common.hpo.main", side_effect=_training_result):
            first = run_hpo(**self.options(n_trials=1, concurrent_trials=1))
        root = self.study_root(first)
        identity = dict(first.user_attrs["study_spec"])
        workers = _Workers()
        with workers.installed():
            expanded = run_hpo(**self.options(
                n_trials=3, resume_from=root, results_path=str(self.root / "unused")
            ))
        with patch("common.hpo.main", side_effect=_training_result) as train:
            repeated = run_hpo(**self.options(
                n_trials=3, resume_from=root, concurrent_trials=1
            ))
        train.assert_not_called()
        self.assertEqual(len(workers.handles), 2)
        self.assertEqual(len(repeated.trials), 3)
        self.assertEqual(expanded.study_name, first.study_name)
        self.assertEqual(repeated.user_attrs["study_spec"], identity)
        self.assertFalse((self.root / "unused").exists())
        for trial in repeated.trials:
            config = load_config(root / "configs" / f"trial-{trial.number:04d}.yaml")
            self.assertEqual(config.training.tensorboard_path, str(root / "tensorboard"))
            self.assertAlmostEqual(trial.value, 0.01)
            self.assertAlmostEqual(self.event_value(root, trial.number, "hpo/generation_loss"), 0.01)

    def test_runtime_context_is_forwarded_without_serializing_or_sealing_it(self) -> None:
        """Each child receives the caller's admission factory while study identity stays scientific."""

        factory = Mock()
        workers = _Workers()
        with workers.installed():
            first = run_hpo(**self.options(n_trials=2, worker_context=factory))
            identity = dict(first.user_attrs["study_spec"])
            resumed = run_hpo(**self.options(n_trials=3, resume_from=self.study_root(first)))
        self.assertIs(workers.handles[0].launch_options["worker_context"], factory)
        self.assertIs(workers.handles[1].launch_options["worker_context"], factory)
        self.assertIsNone(workers.handles[2].launch_options["worker_context"])
        self.assertEqual(resumed.user_attrs["study_spec"], identity)
        self.assertNotIn("worker_context", resumed.user_attrs["execution"])
        factory.assert_not_called()

    def test_gpu_routing_resume_keeps_scientific_identity_and_tensorboard(self) -> None:
        """DiT switches from one GPU to several while preserving existing outcomes."""

        factory = Mock()
        workers = _Workers(delays={1: 6})
        with workers.installed():
            first = run_hpo(**self.options(
                n_trials=1, concurrent_trials=1, worker_gpu_ids=["GPU-first"], 
                gpu_worker_context=factory, worker_gpu_memory_limit_mb=4096
            ))
            identity = dict(first.user_attrs["study_spec"])
            root = self.study_root(first)
            resumed = run_hpo(**self.options(
                n_trials=5, concurrent_trials=4, resume_from=root, 
                worker_gpu_ids=["GPU-first", "GPU-second"], gpu_worker_context=factory
            ))
        self.assertEqual(resumed.user_attrs["study_spec"], identity)
        self.assertEqual(len(resumed.trials), 5)
        self.assertEqual(
            [handle.launch_options["gpu_id"] for handle in workers.handles], 
            ["GPU-first", "GPU-first", "GPU-second", "GPU-first", "GPU-second"]
        )
        self.assertNotIn("gpu_worker_context", resumed.user_attrs["execution"])
        for handle in workers.handles:
            self.assertIs(handle.launch_options["gpu_worker_context"], factory)
            self.assertIsNone(handle.launch_options["worker_context"])
        for trial in resumed.trials:
            self.assertAlmostEqual(trial.value, 0.01)
            self.assertAlmostEqual(self.event_value(root, trial.number, "hpo/generation_loss"), 0.01)
        factory.assert_not_called()

    def test_invalid_parallel_modes_and_contexts_fail_before_storage(self) -> None:
        """Keep unsupported teachers, datasets, modes, and ignored admission hooks closed."""

        cases = [
            {"dataset_name": "MNIST"}, {"dataset_name": "FMNIST"}, 
            {"model_name": "unet"}, {"fit_method": "fit_progressively"}, 
            {"teacher_network": object()}, {"use_distillation": True}, 
            {"concurrent_trials": 1, "worker_context": Mock()}, 
            {"worker_context": object()}, 
            {"concurrent_trials": 1, "worker_gpu_memory_limit_mb": 4096}
        ]
        for index, changes in enumerate(cases):
            destination = self.root / str(index)
            with self.subTest(changes=changes), patch("optuna.create_study") as create:
                with self.assertRaises(ValueError):
                    run_hpo(**self.options(results_path=str(destination), **changes))
                create.assert_not_called()
                self.assertFalse(destination.exists())

    def test_failed_and_pruned_trials_keep_distinct_outcome_events(self) -> None:
        """OOM and nonfinite final loss consume attempts without recording a successful score."""

        workers = _Workers(outcomes={
            1: {"status": "oom", "error": "controlled memory exhaustion"}, 
            2: _nonfinite_result
        })
        with workers.installed():
            study = run_hpo(**self.options())
        root = self.study_root(study)
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "FAIL", "PRUNED"])
        for number, tag in enumerate(("hpo/completed", "hpo/failed", "hpo/pruned")):
            self.assertEqual(self.event_value(root, number, tag), 1.0)
        for number in (1, 2):
            events = EventAccumulator(str(root / "tensorboard" / f"trial-{number:04d}" / "outcome"))
            events.Reload()
            self.assertNotIn("hpo/generation_loss", events.Tags()["tensors"])

    def test_epoch_pruning_uses_completed_references_and_preserves_final_objectives(self) -> None:
        """Warmup protects candidates and a poor curve exits before final scoring."""

        workers = _PruningWorkers({0: [1.0] * 5, 1: [2.0] * 5, 2: [10.0] * 5, 3: [0.1, 5.0, 5.0]})
        policy = {
            "n_startup_trials": 2, "n_warmup_steps": 2, "interval_steps": 2, 
            "n_min_trials": 2
        }
        with workers.installed():
            study = run_hpo(**self.options(n_trials=4, concurrent_trials=1, pruning=policy))
        root = self.study_root(study)
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "COMPLETE", "PRUNED", "COMPLETE"])
        self.assertEqual(workers.finished_training, [0, 1, 3])
        self.assertEqual([entry for entry in workers.decisions if entry[0] == 2], [
            (2, 0, False), (2, 1, False), (2, 2, True)
        ])
        self.assertEqual(study.trials[2].intermediate_values, {0: 10.0, 1: 10.0, 2: 10.0})
        self.assertEqual(study.trials[2].user_attrs["pruning"]["step"], 2)
        self.assertAlmostEqual(study.trials[3].value, 0.01)
        self.assertNotIn(2, [trial.number for trial in study.best_trials])
        self.assertEqual(self.event_value(root, 2, "hpo/pruned"), 1.0)
        events = EventAccumulator(str(root / "tensorboard" / "trial-0002" / "outcome"))
        events.Reload()
        self.assertIn("hpo/pruning", events.Tags()["tensors"])
        self.assertNotIn("hpo/generation_loss", events.Tags()["tensors"])
        for handle in workers.handles:
            self.assertEqual(handle.launch_options["pruning_monitor"], "val_noise_loss")
            self.assertEqual(handle.launch_options["pruning_trial_number"], handle.number)
            self.assertEqual(handle.config.hpo["pruning"], study.user_attrs["study_spec"]["pruning"])
        self.assertEqual(workers.max_active, 1)
        summary = summarize_hpo(study, pareto_only=False).set_index("trial")
        self.assertTrue(summary.loc[2, ["generation_loss"]].isna().all())
        self.assertAlmostEqual(summary.loc[3, "generation_loss"], 0.01)

    def test_pruning_operates_across_single_and_multiple_gpu_slots(self) -> None:
        """Independent epoch decisions preserve fixed routing and total allocations."""

        for count, gpu_ids in ((1, None), (3, [0]), (4, [0, 1])):
            with self.subTest(count=count, gpu_ids=gpu_ids):
                workers = _PruningWorkers({0: [1.0, 1.0], 1: [10.0] * 4, 2: [10.0] * 4, 3: [10.0] * 4})
                with workers.installed():
                    study = run_hpo(**self.options(
                        n_trials=4, concurrent_trials=count, worker_gpu_ids=gpu_ids, 
                        results_path=str(self.root / str(count)), pruning={
                            "n_startup_trials": 1, "n_warmup_steps": 1, 
                            "interval_steps": 1, "n_min_trials": 1
                        }
                    ))
                self.assertEqual(len(study.trials), 4)
                self.assertTrue(any(trial.state.name == "PRUNED" for trial in study.trials))
                self.assertFalse(workers.active)
                self.assertEqual(workers.max_active, count)
                for handle in workers.handles:
                    expected = None if gpu_ids is None else str(gpu_ids[handle.number % len(gpu_ids)])
                    self.assertEqual(handle.launch_options["gpu_id"], expected)

    def test_pruning_policy_is_sealed_and_disabled_studies_keep_legacy_identity(self) -> None:
        """Resumes retain reference curves and reject changed or removed policies."""

        workers = _PruningWorkers({0: [1.0], 1: [2.0]})
        with workers.installed():
            first = run_hpo(**self.options(n_trials=1, pruning={}))
            root = self.study_root(first)
            resumed = run_hpo(**self.options(n_trials=2, concurrent_trials=1, resume_from=root, pruning={}))
            for policy in (None, {"percentile": 50.0}, {"n_warmup_steps": 4}):
                with self.subTest(policy=policy), patch("optuna.load_study") as load:
                    with self.assertRaisesRegex(ValueError, "specification differs"):
                        run_hpo(**self.options(n_trials=3, resume_from=root, pruning=policy))
                    load.assert_not_called()
        self.assertEqual(resumed.user_attrs["study_spec"], first.user_attrs["study_spec"])
        self.assertEqual(resumed.trials[0].intermediate_values, {0: 1.0})
        self.assertEqual(first.user_attrs["study_spec"]["pruning"]["percentile"], 75.0)
        plain = _Workers()
        with plain.installed():
            legacy = run_hpo(**self.options(n_trials=1, results_path=str(self.root / "legacy")))
        self.assertNotIn("pruning", legacy.user_attrs["study_spec"])
        self.assertNotIn("pruning", plain.handles[0].config.hpo)
        self.assertIsNone(plain.handles[0].launch_options["pruning_monitor"])

    def test_pruning_rejects_incompatible_metrics_and_policies_before_storage(self) -> None:
        """A percentile decision cannot compare a different branch or objective."""

        cases = [
            {"pruning": {"monitor": "val_loss"}}, {"pruning": {"type": "median"}}, 
            {"pruning": {"unknown": 1}}, {"pruning": {"percentile": 101}}, 
            {"objective_metrics": ["generation_loss", "noise_loss"], "objective_directions": ["minimize", "minimize"]}, 
            {"objective_directions": ["maximize"]}, 
            {"wrapper_overrides": {"test_network_name": "raw"}}, 
            {"fit_kwargs": {"validation_freq": 2}}, {"fit_kwargs": {"callbacks": []}}, 
            {"model_name": "unet"}, {"teacher_network": object()}
        ]
        for index, changes in enumerate(cases):
            options = self.options(results_path=str(self.root / str(index)), pruning={})
            options.update(changes)
            with self.subTest(changes=changes), patch("optuna.create_study") as create:
                with self.assertRaises(ValueError):
                    run_hpo(**options)
                create.assert_not_called()
                self.assertFalse(Path(options["results_path"]).exists())

    def test_minimum_reference_count_protects_a_poor_candidate(self) -> None:
        """Startup alone cannot prune when too few completed curves report a step."""

        workers = _PruningWorkers({0: [1.0] * 3, 1: [10.0] * 3})
        with workers.installed():
            study = run_hpo(**self.options(n_trials=2, concurrent_trials=1, pruning={
                "n_startup_trials": 1, "n_warmup_steps": 0, "interval_steps": 1, 
                "n_min_trials": 2
            }))
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "COMPLETE"])
        self.assertTrue(all(not prune for _, _, prune in workers.decisions))

    def test_recovery_retains_parameters_once_within_the_total_ceiling(self) -> None:
        """A resumed abandoned DiT allocation uses the existing retry protocol without extra attempts."""

        workers = _Workers()
        with workers.installed():
            first = run_hpo(**self.options(n_trials=1))
            original = first.trials[0]
            first.enqueue_trial(dict(original.params))
            abandoned = first.ask(fixed_distributions=original.distributions)
            root = self.study_root(first)
            exhausted = run_hpo(**self.options(n_trials=2, resume_from=root))
            self.assertEqual(len(workers.handles), 1)
            self.assertEqual([trial.state.name for trial in exhausted.trials], ["COMPLETE", "RUNNING"])
            resumed = run_hpo(**self.options(n_trials=3, resume_from=root))
            repeated = run_hpo(**self.options(n_trials=3, resume_from=root))
        self.assertEqual(len(workers.handles), 2)
        self.assertEqual(len(repeated.trials), 3)
        retry = resumed.trials[2]
        self.assertEqual(retry.params, abandoned.params)
        self.assertEqual(retry.user_attrs["resume_original_trial_number"], abandoned.number)


# This focused entry point uses process doubles and never launches model training.
if __name__ == "__main__":
    main()
