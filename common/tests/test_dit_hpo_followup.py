"""Fresh-study hint scheduling, queue recovery and runner transfer integration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import optuna

from common import dit_hpo_runner as runner
from common import hpo
from common.dit_hpo_backbones import FOLLOWUP_BRANCHES, MISSING_CAPACITY_CHOICES
from common.dit_hpo_followup import followup_initial_trials, validate_followup_protocol
from common.dit_hpo_transfer import freeze_transfer, _fingerprint
from common.hpo_initial_trials import (
    enqueue_initial_trials, initial_trials_digest, normalize_initial_trials
)
from common.tests import test_dit_hpo_transfer as transfer_tests
from common.tests.test_hpo_concurrency import _Workers


class InitialTrialQueueTests(unittest.TestCase):
    """Exercise actual Optuna queue semantics without model or storage mocks."""

    def setUp(self) -> None:
        """Use an isolated Optuna study and three distinguishable partial hints."""

        self.study = optuna.create_study(direction="minimize")
        self.hints = [{"learning_rate": 0.001}, {"learning_rate": 0.002}, {"optimizer": "adamw"}]

    def test_queue_is_idempotent_and_contains_no_objectives(self) -> None:
        """Hint allocation creates fresh WAITING trials without source observations."""

        self.assertEqual(enqueue_initial_trials(self.study, self.hints), 3)
        self.assertEqual(enqueue_initial_trials(self.study, self.hints), 0)
        for index, trial in enumerate(self.study.trials):
            self.assertEqual(trial.state.name, "WAITING")
            self.assertIsNone(trial.values)
            self.assertEqual(trial.params, {})
            self.assertEqual(trial.system_attrs["fixed_params"], self.hints[index])
            self.assertEqual(set(trial.user_attrs), {"initial_trial"})
            self.assertEqual(trial.user_attrs["initial_trial"]["index"], index)

    def test_partial_allowance_completes_remaining_queue_once(self) -> None:
        """Small launch budgets preserve the original order without over-allocation."""

        self.assertEqual(enqueue_initial_trials(self.study, self.hints, max_new_trials=1), 1)
        self.assertEqual(enqueue_initial_trials(self.study, self.hints, max_new_trials=0), 0)
        self.assertEqual(enqueue_initial_trials(self.study, self.hints, max_new_trials=1), 1)
        self.assertEqual(enqueue_initial_trials(self.study, self.hints, max_new_trials=5), 1)
        self.assertEqual([trial.system_attrs["fixed_params"] for trial in self.study.trials], self.hints)

    def test_crash_after_atomic_enqueue_does_not_duplicate_hint(self) -> None:
        """Persisted marker remains authoritative if the caller dies after enqueue."""

        original = self.study.enqueue_trial

        def interrupted(params: dict, **kwargs: object) -> None:
            """Commit the real enqueue before simulating coordinator interruption."""

            original(params, **kwargs)
            raise KeyboardInterrupt("after durable enqueue")

        with patch.object(self.study, "enqueue_trial", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                enqueue_initial_trials(self.study, self.hints)
        self.assertEqual(len(self.study.trials), 1)
        self.assertEqual(enqueue_initial_trials(self.study, self.hints), 2)
        self.assertEqual(len(self.study.trials), 3)

    def test_incompatible_marker_fails_before_more_allocations(self) -> None:
        """A changed hint list or corrupt index cannot silently mix seeded studies."""

        enqueue_initial_trials(self.study, self.hints)
        changed = copy.deepcopy(self.hints)
        changed[0]["learning_rate"] = 0.004
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            enqueue_initial_trials(self.study, changed)
        other = optuna.create_study()
        other.enqueue_trial({}, user_attrs={"initial_trial": {
            "sha256": initial_trials_digest(self.hints), "index": True
        }})
        with self.assertRaisesRegex(ValueError, "invalid index"):
            enqueue_initial_trials(other, self.hints)
        self.assertEqual(len(other.trials), 1)

    def test_normalization_accepts_scalars_and_detaches_inputs(self) -> None:
        """Every supported scalar retains its type and caller mutation is isolated."""

        points = [{"none": None, "integer": 4, "float": 0.2, "bool": False, "text": "adam"}]
        result = normalize_initial_trials(points)
        self.assertEqual(result, points)
        self.assertIsNot(result, points)
        self.assertIsNot(result[0], points[0])
        points[0]["integer"] = 8
        self.assertEqual(result[0]["integer"], 4)
        self.assertIsNone(normalize_initial_trials(None))
        self.assertEqual(normalize_initial_trials([]), [])

    def test_normalization_rejects_nested_or_nonfinite_metadata(self) -> None:
        """Checkpoint dictionaries, arrays and nonfinite values cannot enter hints."""

        for value in ["text", {}, [{1: 2}], [{"": 2}], [None], [{"x": []}], [{"x": {}}]]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                normalize_initial_trials(value)
        for value in [float("inf"), float("-inf"), float("nan")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_initial_trials([{"x": value}])
        self.assertNotEqual(initial_trials_digest(self.hints), initial_trials_digest(list(reversed(self.hints))))


class FollowupPlanTests(unittest.TestCase):
    """Seal authentic source snapshots through the public runner plan API."""

    def setUp(self) -> None:
        """Freeze two real completed source configs and mock remote inventory only."""

        fixture = transfer_tests.DitHpoTransferTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture._candidate(0.2)
        fixture._candidate(0.3, learning_rate=0.002)
        self.manifest = freeze_transfer(fixture.results, fixture.manifest, top_k=2)
        self.root = fixture.root
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.results = self.root / "destination"
        identity = {"source_sha256": {"common/hpo.py": "current-source"}, "versions": {}, 
                    "python": "test-python", "worker_policy": {"tf_memory_mib": 24576}}
        self.remote = patch.dict(sys.modules, {
            "common.dit_hpo_remote": SimpleNamespace(inspect_remote=Mock(return_value=identity))
        })
        self.remote.start()
        self.addCleanup(self.remote.stop)

    def _plan(self, **changes: object) -> dict:
        """Build one target plan with the scientific protocol of the source."""

        options = {
            "checkout_root": self.checkout, "results_path": self.results, 
            "validation_source": "test", "validation_ratio": 0.0, 
            "search_space_overrides": {"dit_followup_branch": list(FOLLOWUP_BRANCHES)}, 
            "transfer_manifest": self.manifest
        }
        options.update(changes)
        return runner.make_plan(**options)

    def test_branch_schedule_covers_every_missing_axis_per_source(self) -> None:
        """Each top source setting initializes all topologies and excluded plain axes."""

        hints = followup_initial_trials(self.manifest, list(FOLLOWUP_BRANCHES))
        per_source = len(FOLLOWUP_BRANCHES) - 1 + len(MISSING_CAPACITY_CHOICES)
        self.assertEqual(len(hints), 2 * per_source)
        first = hints[:per_source]
        self.assertEqual({hint["followup_missing_axis"] for hint in first
                          if hint["dit_followup_branch"] == "plain_missing"}, set(MISSING_CAPACITY_CHOICES))
        self.assertEqual({hint["dit_followup_branch"] for hint in first}, set(FOLLOWUP_BRANCHES))
        self.assertTrue(all(hint["learning_rate"] == 0.001 for hint in first))
        self.assertTrue(all(hint["learning_rate"] == 0.002 for hint in hints[per_source:]))
        self.assertTrue(all("value" not in hint and "weights_path" not in hint for hint in hints))
        self.assertTrue(all("dim" not in hint and "depth" not in hint for hint in hints))

    def test_branch_schedule_rejects_unknown_duplicate_or_empty_selection(self) -> None:
        """Only explicit distinct supported follow-up branch lists are meaningful."""

        for branches in [[], ["plain"], ["u_skip", "u_skip"]]:
            with self.subTest(branches=branches), self.assertRaisesRegex(ValueError, "distinct supported"):
                followup_initial_trials(self.manifest, branches)

    def test_runner_seals_manifest_and_derived_hints_without_starting_clock(self) -> None:
        """Setup retains both provenance and actual queue contents in its recipe."""

        plan = self._plan(experiment_hours=20)
        recipe = json.loads((Path(plan["control_root"]) / "recipe.json").read_text())
        self.assertEqual(recipe["transfer_manifest"], self.manifest)
        self.assertEqual(recipe["hpo"]["initial_trials"], plan["hpo"]["initial_trials"])
        self.assertEqual(plan["transfer_manifest"], self.manifest)
        self.assertFalse((Path(plan["control_root"]) / "budget.json").exists())
        self.assertFalse((Path(plan["study_root"]) / "study.db").exists())
        self.assertEqual(self._plan()["hpo"], plan["hpo"])
        self.manifest["initial_trials"][0]["learning_rate"] = 0.5
        self.assertNotEqual(plan["transfer_manifest"], self.manifest)

    def test_changed_valid_snapshot_rejected_on_recipe_resume(self) -> None:
        """A second upstream snapshot requires a separate destination study."""

        self._plan()
        changed = copy.deepcopy(self.manifest)
        changed["source"]["sqlite_snapshot_sha256"] = "another-snapshot"
        changed["transfer_sha256"] = _fingerprint({
            key: value for key, value in changed.items() if key != "transfer_sha256"
        })
        with self.assertRaisesRegex(ValueError, "recipe changed"):
            self._plan(transfer_manifest=changed)

    def test_target_protocol_and_source_overlap_fail_before_recipe_write(self) -> None:
        """Transferred setup cannot change evaluation units or write into its source."""

        for changes in [
            {"epochs": 2}, {"dataset_name": "CIFAR100"}, {"validation_source": "split", "validation_ratio": 0.2}, 
            {"results_path": self.manifest["request"]["source_results_path"]}, 
            {"results_path": Path(self.manifest["request"]["source_results_path"]) / "nested"}, 
            {"search_space_overrides": {}}, {"search_space_overrides": {"dit_followup_branch": ["unknown"]}}
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._plan(**changes)
        self.assertFalse(self.results.exists())

    def test_protocol_validator_checks_loss_objective_and_returns_detached_manifest(self) -> None:
        """Direct callers cannot switch objective names or reuse mutable provenance."""

        hpo_options = self._plan()["hpo"]
        for key, value in [
            ("task", "joint"), ("model_name", "unet"), ("dtype_policy", "mixed_float16"), 
            ("objective_metrics", ["noise_loss"]), ("objective_directions", ["maximize"]), 
            ("max_train_samples", 128), ("max_val_samples", 128), 
            ("fit_kwargs", {"steps_per_epoch": 2}), 
            ("model_overrides", {"compile_args": {"evaluation_loss": "mae"}}), 
            ("model_overrides", {"depth": 2}), 
            ("wrapper_overrides", {"test_network_name": "network"}), 
            ("wrapper_overrides", {"swap_noise_image": True}), 
            ("wrapper_overrides", {"use_ema": False}), 
            ("use_ensemble_accuracy", True), ("ensemble_accuracy_kwargs", {"max_t": 100}), 
            ("use_distillation", True), ("teacher_network", object()), 
            ("feature_archive_path", "different-data.npz"), ("search_profile", "joint_dit_classifier")
        ]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "protocol differs"):
                validate_followup_protocol(self.manifest, dict(hpo_options, **{key: value}))
        validated = validate_followup_protocol(self.manifest, hpo_options)
        self.assertEqual(validated, self.manifest)
        self.assertIsNot(validated, self.manifest)
        explicit_defaults = {
            "max_train_samples": None, "max_val_samples": None, "teacher_network": None, 
            "feature_archive_path": None, "search_profile": None, "fit_kwargs": {}, 
            "model_overrides": {}, "wrapper_overrides": {}, "ensemble_accuracy_kwargs": {}, 
            "use_ensemble_accuracy": False, "use_distillation": False
        }
        self.assertEqual(validate_followup_protocol(
            self.manifest, dict(hpo_options, **explicit_defaults)
        ), self.manifest)
        absent_mapping_defaults = {
            key: None for key in ("fit_kwargs", "model_overrides", "wrapper_overrides", "ensemble_accuracy_kwargs")
        }
        self.assertEqual(validate_followup_protocol(
            self.manifest, dict(hpo_options, **absent_mapping_defaults)
        ), self.manifest)
        training_loss_choices = copy.deepcopy(hpo_options)
        training_loss_choices["search_space_overrides"]["loss_function"] = ["mse", "mae"]
        self.assertEqual(validate_followup_protocol(self.manifest, training_loss_choices), self.manifest)


class InitialTrialPublicApiTests(unittest.TestCase):
    """Exercise real HPO storage and sampler recovery across controlled worker boundaries."""

    def setUp(self) -> None:
        """Use bounded small configuration suggestions without creating a model."""

        temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.hints = [{"learning_rate": 0.001}, {"learning_rate": 0.002}, {"learning_rate": 0.003}]
        self.options = {
            "task": "generation", "model_name": "diffusion_transformer", "dataset_name": "cifar10", 
            "n_trials": 2, "epochs": 1, "concurrent_trials": 2, "results_path": str(self.root), 
            "trial_budget_mode": "total", "n_startup_trials": 1, "initial_trials": self.hints, 
            "search_space_overrides": {
                "dim": [32], "mha_num_heads": [4], "depth": [3], "patch_size": [4], 
                "dit_architecture_grid4": ["plain"]
            }
        }

    def _stored(self) -> optuna.study.Study:
        """Reload the actual study after an intentionally interrupted coordinator."""

        database = self.root / "generation/diffusion_transformer/cifar10/study.db"
        return optuna.load_study(
            study_name="generation-diffusion_transformer-cifar10", storage="sqlite:///" + database.as_posix()
        )

    def test_public_api_seals_hints_and_expands_total_allowance_without_duplicates(self) -> None:
        """New trials train from hints while each launch respects the total allocation target."""

        with _Workers().installed():
            first = hpo.run_hpo(**self.options)
        self.assertEqual(len(first.trials), 2)
        self.assertEqual([trial.params["learning_rate"] for trial in first.trials], [0.001, 0.002])
        self.assertTrue(all(trial.state.name == "COMPLETE" for trial in first.trials))
        self.assertTrue(all(trial.value == 0.01 for trial in first.trials))
        self.assertEqual(first.user_attrs["study_spec"]["initial_trials"], self.hints)
        with _Workers().installed():
            resumed = hpo.run_hpo(**dict(self.options, n_trials=3))
        self.assertEqual([trial.params["learning_rate"] for trial in resumed.trials], [0.001, 0.002, 0.003])
        self.assertEqual(len(resumed.trials), 3)
        with self.assertRaisesRegex(ValueError, "specification differs"):
            hpo.run_hpo(**dict(self.options, initial_trials=list(reversed(self.hints))))

    def test_public_api_restores_sampler_after_pretraining_enqueue_crash(self) -> None:
        """The sampler cursor is durable before a queued hint makes the study nonempty."""

        original = hpo.enqueue_initial_trials

        def interrupted(study: object, hints: list, max_new_trials: int | None = None) -> None:
            """Publish one waiting hint and interrupt before scheduling workers."""

            original(study, hints, max_new_trials=1)
            raise KeyboardInterrupt("queued before training")

        with patch.object(hpo, "enqueue_initial_trials", side_effect=interrupted), \
                patch.object(hpo, "start_worker") as launch:
            with self.assertRaises(KeyboardInterrupt):
                hpo.run_hpo(**self.options)
        launch.assert_not_called()
        pending = self._stored()
        self.assertEqual([trial.state.name for trial in pending.trials], ["WAITING"])
        self.assertIn(hpo._SAMPLER_RNG_STATE_ATTR, pending.user_attrs)
        self.assertIsNone(pending.trials[0].value)
        with _Workers().installed():
            resumed = hpo.run_hpo(**self.options)
        self.assertEqual(len(resumed.trials), 2)
        self.assertEqual([trial.params["learning_rate"] for trial in resumed.trials], [0.001, 0.002])
        self.assertTrue(all(trial.state.name == "COMPLETE" for trial in resumed.trials))

    def test_invalid_hints_rejected_before_storage_creation(self) -> None:
        """Nested metadata is rejected before a study or worker can be created."""

        with patch("optuna.create_study") as create:
            with self.assertRaises(TypeError):
                hpo.run_hpo(**dict(self.options, initial_trials=[{"checkpoint": {"path": "old"}}]))
        create.assert_not_called()
        self.assertFalse((self.root / "generation").exists())


# Execute this focused suite only when explicitly invoked as a script.
if __name__ == "__main__":
    unittest.main()
