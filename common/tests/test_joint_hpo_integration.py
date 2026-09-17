"""Exercise joint DiT HPO persistence without constructing or training a model.

Real Optuna SQLite studies, Config YAML round trips, objective extraction, and
TensorBoard event files run normally. Only the training entry point is replaced
with deliberately conflicting train, validation, test, and EMA scores.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import optuna
import pandas as pd
import tensorflow as tf

from common.config import load_config, save_config
from common.hpo import run_hpo
from common import hpo_profiles


class JointHpoIntegrationTests(unittest.TestCase):
    """Keep the expensive training boundary mocked and storage authoritative."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @staticmethod
    def _space(aggregation: str = "last") -> dict:
        """Select a valid, small architecture while retaining real Optuna draws."""
        choices = {
            "optimizer": ["adam"],
            "dim": [32],
            "mha_num_heads": [4],
            "depth": [3],
            "clf_depth": [1],
            "patch_size": [4],
            "feature_aggregation": [aggregation],
        }
        return choices

    def _options(self, **changes: object) -> dict:
        options = {
            "task": "joint",
            "model_name": "dit_classifier",
            "dataset_name": "cifar10",
            "search_profile": "joint_dit_classifier",
            "trial_budget_mode": "total",
            "n_trials": 1,
            "epochs": 50,
            "seed": 17,
            "n_startup_trials": 1,
            "results_path": str(self.root),
            "search_space_overrides": self._space(),
            "validation_source": "test",
            "validation_ratio": 0.0,
        }
        options.update(changes)
        return options

    @staticmethod
    def _fake_training(config, **kwargs: object) -> dict:
        """Persist validation fixtures as a training report normally would."""
        del kwargs
        output = Path(config.training.results_path) / f"trial-{config.hpo['trial_number']:04d}"
        output.mkdir(parents=True, exist_ok=True)
        evaluations = {
            "valset_network_eval": {
                "classifier_accuracy": 0.21,
                "ensemble_accuracy": 0.73,
                "noise_loss": 0.04,
            },
            "valset_ema_eval": {
                "classifier_accuracy": 0.82,
                "ensemble_accuracy": 0.83,
                "noise_loss": 0.07,
            },
            "trainset_network_eval": {
                "classifier_accuracy": 0.91,
                "ensemble_accuracy": 0.92,
            },
            "testset_network_eval": {
                "classifier_accuracy": 0.97,
                "ensemble_accuracy": 0.98,
            },
        }
        (output / "evaluations.json").write_text(json.dumps(evaluations), encoding="utf-8")
        return {
            "results_path": str(output),
            "history": {"val_classifier_accuracy": [0.99], "val_ensemble_accuracy": [1.0]},
            "evaluations": evaluations,
        }

    @staticmethod
    def _study_root(config) -> Path:
        return Path(config.hpo["study_root"])

    @staticmethod
    def _outcome_scalars(study_root: Path, trial_number: int) -> dict[str, float]:
        """Read actual TensorBoard scalar events, including TF2 tensor encoding."""
        files = list((study_root / "tensorboard" / f"trial-{trial_number:04d}" / "outcome").glob(
            "events.out.tfevents.*"
        ))
        if not files:
            raise AssertionError("No TensorBoard outcome event file was written.")
        scalars = {}
        for path in files:
            for event in tf.compat.v1.train.summary_iterator(str(path)):
                for value in event.summary.value:
                    if value.metadata.plugin_data.plugin_name == "scalars":
                        scalars[value.tag] = float(tf.make_ndarray(value.tensor).item())
        return scalars

    def test_selected_validation_score_and_configs_survive_real_storage(self) -> None:
        """Every CIFAR joint objective uses raw ordinary scores despite EMA fixtures."""
        cases = (
            ("cifar10", "last"),
            ("cifar100", "last"),
            ("cifar10", "all"),
        )
        selected, expected = "classification_accuracy", 0.21
        for index, (dataset, aggregation) in enumerate(cases):
            with self.subTest(dataset=dataset, aggregation=aggregation), patch(
                "common.hpo.main", side_effect=self._fake_training,
            ) as training:
                study = run_hpo(**self._options(
                    results_path=str(self.root / str(index)),
                    dataset_name=dataset,
                    search_space_overrides=self._space(aggregation),
                ))
                training.assert_called_once()
                trial = study.trials[0]
                self.assertEqual(trial.state, optuna.trial.TrialState.COMPLETE)
                self.assertEqual([direction.name for direction in study.directions], ["MAXIMIZE", "MINIMIZE"])
                self.assertEqual(trial.values, [expected, 0.04])
                self.assertEqual(trial.user_attrs["accuracy_metric"], selected)

                resolved = load_config(trial.user_attrs["resolved_config_path"])
                source = load_config(resolved.hpo["input_config_path"])
                self.assertEqual(resolved.model.wrapper_name, "diffusion_classifier")
                self.assertEqual(resolved.dataset.name, dataset)
                self.assertEqual(resolved.optimizer.schedule, "cosine")
                self.assertTrue(resolved.model.kwargs["patchify_with_cnn"])
                self.assertNotIn("modify_first_t", resolved.model.wrapper_kwargs)
                self.assertEqual(source.hpo["accuracy_metric"], selected)
                self.assertEqual(resolved.hpo["objective_metrics"], ["classification_accuracy", "noise_loss"])
                self.assertEqual(resolved.hpo["objectives"], [expected, 0.04])
                self.assertFalse(resolved.reporting.evaluate_ensemble_accuracy)
                self.assertFalse(resolved.training.ensemble_monitor)
                self.assertFalse(resolved.hpo["use_ensemble_accuracy"])
                self.assertEqual(resolved.reporting.ensemble_accuracy_kwargs, {})
                self.assertEqual(resolved.hpo["ensemble_accuracy_kwargs"], {})
                self.assertEqual(resolved.model.wrapper_kwargs["test_network_name"], "raw")
                self.assertFalse(resolved.model.wrapper_kwargs["use_ema"])
                self.assertEqual(resolved.dataset.validation_source, "test")
                self.assertEqual(resolved.dataset.validation_ratio, 0.0)
                self.assertFalse(resolved.dataset.drop_remainder)
                selection = study.user_attrs["study_spec"]["data_selection"]
                self.assertEqual(selection, resolved.hpo["data_selection"])
                self.assertEqual(selection["requested"], {
                    "validation_source": "test", "validation_ratio": 0.0,
                })
                self.assertEqual(selection["resolved"]["effective_validation_ratio"], 0.0)
                self.assertEqual(resolved.training.epochs, 50)
                self.assertEqual(resolved.training.patience, 0)
                self.assertEqual(resolved.model.wrapper_kwargs["clf_train_noisy_input_type"], "clean")
                self.assertEqual(resolved.model.wrapper_kwargs["clf_train_class_input_type"], "null_class_only")
                self.assertEqual(resolved.model.wrapper_kwargs["clf_train_type"], "uncond")
                self.assertEqual(resolved.model.wrapper_kwargs["clf_loss_coef"], 1.0)
                self.assertFalse(resolved.model.wrapper_kwargs["mask_by_nulls"])
                self.assertFalse(resolved.model.kwargs["aggregate_from_noises"])
                self.assertEqual(resolved.training.reduce_lr_patience, 0)
                self.assertEqual(resolved.dataset.batch_size, 128)
                self.assertFalse(resolved.reporting.run_trainset_eval)
                self.assertTrue(resolved.reporting.run_valset_eval)
                self.assertNotIn("clf_train_noisified_max_timesteps", resolved.model.wrapper_kwargs)
                self.assertNotIn("clf_test_noisified_max_timesteps", resolved.model.wrapper_kwargs)
                round_trip = self.root / f"round-trip-{index}.yaml"
                save_config(resolved, round_trip)
                self.assertEqual(load_config(round_trip), resolved)

                report = Path(trial.user_attrs["results_path"])
                saved = json.loads((report / "evaluations.json").read_text(encoding="utf-8"))
                self.assertEqual(trial.values[0], saved["valset_network_eval"]["classifier_accuracy"])
                self.assertNotEqual(trial.values[0], saved["valset_ema_eval"]["classifier_accuracy"])
                self.assertNotEqual(trial.values[0], saved["valset_ema_eval"]["ensemble_accuracy"])
                self.assertEqual(trial.values[1], saved["valset_network_eval"]["noise_loss"])
                objectives = pd.read_csv(report / "objectives.csv")
                self.assertEqual(objectives["name"].tolist(), ["classification_accuracy", "noise_loss"])
                self.assertEqual(objectives["direction"].tolist(), ["maximize", "minimize"])
                self.assertAlmostEqual(objectives["value"].iloc[0], expected)
                self.assertAlmostEqual(objectives["value"].iloc[1], 0.04)
                scalars = self._outcome_scalars(self._study_root(resolved), trial.number)
                self.assertEqual(scalars["hpo/completed"], 1.0)
                self.assertEqual(scalars["hpo/failed"], 0.0)
                self.assertAlmostEqual(scalars["hpo/classification_accuracy"], expected, places=6)
                self.assertAlmostEqual(scalars["hpo/noise_loss"], 0.04, places=6)
                self.assertAlmostEqual(scalars["validation/noise_loss"], 0.04, places=6)
                self.assertAlmostEqual(scalars["validation/classifier_accuracy"], 0.21, places=6)

    def test_total_budget_reopens_without_duplicate_trials_and_can_increase(self) -> None:
        """Both explicit resume and normal reopening honor allocated trial count."""
        with patch("common.hpo.main", side_effect=self._fake_training) as training:
            first = run_hpo(**self._options(n_trials=2))
            original_params = [trial.params for trial in first.trials]
            config = load_config(first.trials[0].user_attrs["resolved_config_path"])
            study_root = self._study_root(config)
            repeated = run_hpo(**self._options(n_trials=2, resume_from=study_root))
            self.assertEqual(len(repeated.trials), 2)
            self.assertEqual(training.call_count, 2)
            expanded = run_hpo(**self._options(n_trials=3, resume_from=study_root))
            self.assertEqual([trial.number for trial in expanded.trials], [0, 1, 2])
            self.assertEqual(training.call_count, 3)
            smaller = run_hpo(**self._options(n_trials=1))
            self.assertEqual(len(smaller.trials), 3)
            self.assertEqual(training.call_count, 3)
            self.assertEqual([trial.params for trial in smaller.trials[:2]],
                             original_params)
            persisted = pd.read_csv(study_root / "trials.csv")
            self.assertEqual(persisted["number"].tolist(), [0, 1, 2])

    def test_total_budget_limits_interrupted_trial_retries(self) -> None:
        """An abandoned trial is retried only within a newly available allocation."""
        with patch("common.hpo.main", side_effect=self._fake_training) as training:
            study = run_hpo(**self._options())
            original = study.trials[0]
            config = load_config(original.user_attrs["resolved_config_path"])
            study_root = self._study_root(config)
            abandoned = study.ask(fixed_distributions=original.distributions)
            self.assertEqual(abandoned.number, 1)
            abandoned_params = dict(abandoned.params)

            exhausted = run_hpo(**self._options(n_trials=2, resume_from=study_root))
            self.assertEqual(len(exhausted.trials), 2)
            self.assertEqual(training.call_count, 1)
            self.assertEqual([trial.state.name for trial in exhausted.trials],
                             ["COMPLETE", "RUNNING"])

            resumed = run_hpo(**self._options(n_trials=3, resume_from=study_root))
            self.assertEqual(len(resumed.trials), 3)
            self.assertEqual(training.call_count, 2)
            self.assertEqual([trial.state.name for trial in resumed.trials],
                             ["COMPLETE", "RUNNING", "COMPLETE"])
            retry = resumed.trials[2]
            self.assertEqual(retry.params, abandoned_params)
            self.assertEqual(retry.user_attrs["resume_source_trial_number"], 1)
            self.assertEqual(retry.user_attrs["resume_original_trial_number"], 1)
            self.assertEqual(retry.values, [0.21, 0.04])

            repeated = run_hpo(**self._options(n_trials=3, resume_from=study_root))
            self.assertEqual(len(repeated.trials), 3)
            self.assertEqual(training.call_count, 2)
            self.assertEqual(pd.read_csv(study_root / "trials.csv")["state"].tolist(),
                             ["COMPLETE", "RUNNING", "COMPLETE"])

    def test_total_budget_runs_preallocated_waiting_trials(self) -> None:
        """A queued trial already occupies its slot and must still be executed."""
        with patch("common.hpo.main", side_effect=self._fake_training) as training:
            study = run_hpo(**self._options())
            config = load_config(study.trials[0].user_attrs["resolved_config_path"])
            study_root = self._study_root(config)
            queued_params = dict(study.trials[0].params)
            study.enqueue_trial(queued_params)
            self.assertEqual([trial.state.name for trial in study.trials],
                             ["COMPLETE", "WAITING"])

            resumed = run_hpo(**self._options(n_trials=2, resume_from=study_root))
            self.assertEqual(len(resumed.trials), 2)
            self.assertEqual(training.call_count, 2)
            self.assertEqual([trial.state.name for trial in resumed.trials],
                             ["COMPLETE", "COMPLETE"])
            self.assertEqual(resumed.trials[1].params, queued_params)
            repeated = run_hpo(**self._options(n_trials=2, resume_from=study_root))
            self.assertEqual(len(repeated.trials), 2)
            self.assertEqual(training.call_count, 2)

    def test_resume_rejects_changed_space_profile_and_profile_version(self) -> None:
        """A different scientific search cannot reuse the persistent study."""
        with patch("common.hpo.main", side_effect=self._fake_training) as training:
            study = run_hpo(**self._options())
            config = load_config(study.trials[0].user_attrs["resolved_config_path"])
            study_root = self._study_root(config)
            original_attrs = deepcopy(study.user_attrs)
            changed_space = self._space()
            changed_space["dim"] = [64]
            for changes in (
                {"search_space_overrides": changed_space},
                {"search_profile": None, "objective_metrics": "classification_accuracy"},
                {"validation_source": "split", "validation_ratio": 0.1},
                {"validation_ratio": 0.2},
            ):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "specification differs"):
                    run_hpo(**self._options(resume_from=study_root, **changes))
            with patch.object(hpo_profiles, "JOINT_CLASSIFIER_PROFILE_VERSION",
                              hpo_profiles.JOINT_CLASSIFIER_PROFILE_VERSION + 1):
                with self.assertRaisesRegex(ValueError, "specification differs"):
                    run_hpo(**self._options(resume_from=study_root))
            self.assertEqual(training.call_count, 1)
            self.assertEqual(len(study.trials), 1)
            self.assertEqual(study.user_attrs, original_attrs)

    def test_oom_is_failed_and_logged_while_later_trial_completes(self) -> None:
        """OOM must not become a made-up finite score or escape the total budget."""
        def training(config, **kwargs: object) -> dict:
            if config.hpo["trial_number"] == 0:
                raise tf.errors.ResourceExhaustedError(None, None, "synthetic OOM")
            return self._fake_training(config, **kwargs)

        with patch("common.hpo.main", side_effect=training) as mocked:
            study = run_hpo(**self._options(n_trials=2))
            failed, complete = study.trials
            self.assertEqual(failed.state, optuna.trial.TrialState.FAIL)
            self.assertIsNone(failed.values)
            self.assertEqual(complete.state, optuna.trial.TrialState.COMPLETE)
            self.assertEqual(complete.values, [0.21, 0.04])
            config = load_config(complete.user_attrs["resolved_config_path"])
            study_root = self._study_root(config)
            scalars = self._outcome_scalars(study_root, failed.number)
            self.assertEqual(scalars["hpo/completed"], 0.0)
            self.assertEqual(scalars["hpo/failed"], 1.0)
            self.assertNotIn("hpo/classification_accuracy", scalars)
            states = pd.read_csv(study_root / "trials.csv")["state"].tolist()
            self.assertEqual(states, ["FAIL", "COMPLETE"])
            repeated = run_hpo(**self._options(n_trials=2, resume_from=study_root))
            self.assertEqual(len(repeated.trials), 2)
            self.assertEqual(mocked.call_count, 2)

    def test_pareto_front_retains_accuracy_noise_tradeoffs(self) -> None:
        """A lower-noise candidate remains alongside the highest-accuracy one."""
        pairs = ((0.9, 0.2), (0.8, 0.1), (0.7, 0.3))

        def training(config, **kwargs):
            result = self._fake_training(config, **kwargs)
            accuracy, noise = pairs[config.hpo["trial_number"]]
            result["evaluations"]["valset_network_eval"].update(
                classifier_accuracy=accuracy, noise_loss=noise,
            )
            return result

        with patch("common.hpo.main", side_effect=training):
            study = run_hpo(**self._options(n_trials=3))
        self.assertEqual([trial.values for trial in study.trials], [list(pair) for pair in pairs])
        self.assertEqual({trial.number for trial in study.best_trials}, {0, 1})

    def test_nonfinite_final_objectives_are_pruned_and_search_continues(self) -> None:
        """A finite fit does not guarantee valid final ordinary accuracy/noise."""
        pairs = ((0.8, float("inf")), (float("inf"), 0.2),
                 (float("nan"), 0.2), (0.8, float("nan")), (0.7, 0.3))

        def training(config, **kwargs):
            result = self._fake_training(config, **kwargs)
            accuracy, noise = pairs[config.hpo["trial_number"]]
            result["evaluations"]["valset_network_eval"].update(
                classifier_accuracy=accuracy, noise_loss=noise,
            )
            return result

        with patch("common.hpo.main", side_effect=training):
            study = run_hpo(**self._options(n_trials=len(pairs)))
        self.assertEqual([trial.state.name for trial in study.trials],
                         ["PRUNED"] * 4 + ["COMPLETE"])
        self.assertEqual([trial.number for trial in study.best_trials], [4])
        for trial in study.trials[:4]:
            self.assertIsNone(trial.values)
            evidence = json.loads(Path(trial.user_attrs["divergence_path"]).read_text())
            self.assertEqual(evidence["reason"], "nonfinite_objective")
            self.assertEqual(evidence["phase"], "final_evaluation")
            self.assertEqual(evidence, trial.user_attrs["divergence"])
            self.assertFalse((Path(trial.user_attrs["results_path"]) / "objectives.csv").exists())
            scalars = self._outcome_scalars(
                Path(trial.user_attrs["config_path"]).parent.parent, trial.number,
            )
            self.assertEqual(scalars["hpo/pruned"], 1.0)
            self.assertNotIn("hpo/classification_accuracy", scalars)

    def test_validation_options_reach_configs_and_study_identity(self) -> None:
        """Public choices preserve configured ratios and record effective selection."""
        cases = (
            (None, None, "split", 0.2, 0.2),
            ("split", 0.1, "split", 0.1, 0.1),
            ("test", None, "test", 0.0, 0.0),
            ("test", 0.3, "test", 0.3, 0.0),
        )
        for index, (source, ratio, expected_source, expected_ratio, effective_ratio) in enumerate(cases):
            with self.subTest(source=source, ratio=ratio), patch(
                "common.hpo.main", side_effect=self._fake_training,
            ):
                study = run_hpo(**self._options(
                    results_path=str(self.root / str(index)),
                    validation_source=source, validation_ratio=ratio,
                ))
                config = load_config(study.trials[0].user_attrs["resolved_config_path"])
                source_config = load_config(config.hpo["input_config_path"])
                for saved in (source_config, config):
                    self.assertEqual(saved.dataset.validation_source, expected_source)
                    self.assertEqual(saved.dataset.validation_ratio, expected_ratio)
                    self.assertFalse(saved.dataset.drop_remainder)
                selection = study.user_attrs["study_spec"]["data_selection"]
                self.assertEqual(selection["requested"], {
                    "validation_source": source, "validation_ratio": ratio,
                })
                self.assertEqual(selection["resolved"]["effective_validation_ratio"], effective_ratio)
                self.assertEqual(selection, config.hpo["data_selection"])

    def test_generic_ordinary_hpo_preserves_defaults_and_accepts_explicit_test_source(self) -> None:
        """Omitted generic options retain legacy identity; explicit choices are sealed."""
        def training(config, **kwargs):
            result = self._fake_training(config, **kwargs)
            result["evaluations"]["valset_eval"] = {"accuracy": 0.55}
            return result

        for index, source in enumerate((None, "test")):
            with self.subTest(source=source), patch("common.hpo.main", side_effect=training):
                study = run_hpo(**self._options(
                    task="classification", model_name="cnn", search_profile=None,
                    search_space_overrides={"batch_size": [4]},
                    results_path=str(self.root / str(index)),
                    validation_source=source, validation_ratio=None,
                ))
                config = load_config(study.trials[0].user_attrs["resolved_config_path"])
                self.assertAlmostEqual(study.trials[0].value, 0.55)
                if source is None:
                    self.assertEqual(config.dataset.validation_source, "split")
                    self.assertEqual(config.dataset.validation_ratio, 0.2)
                    self.assertNotIn("data_selection", study.user_attrs["study_spec"])
                    self.assertNotIn("data_selection", config.hpo)
                else:
                    self.assertEqual(config.dataset.validation_source, "test")
                    self.assertEqual(config.dataset.validation_ratio, 0.0)
                    self.assertFalse(config.dataset.drop_remainder)
                    self.assertEqual(config.hpo["data_selection"],
                                     study.user_attrs["study_spec"]["data_selection"])

    def test_invalid_validation_options_fail_before_allocating_a_study(self) -> None:
        for changes in (
            {"validation_source": "unknown"},
            {"validation_source": "split", "validation_ratio": 0.0},
            {"validation_ratio": float("nan")},
            {"validation_ratio": True},
            {"validation_ratio": -0.1},
            {"validation_ratio": 1.0},
            {"task": "continual"},
        ):
            with self.subTest(changes=changes), patch("common.hpo.main") as training:
                destination = self.root / "invalid"
                with self.assertRaises(ValueError):
                    run_hpo(**self._options(results_path=str(destination), **changes))
                training.assert_not_called()
                self.assertFalse(destination.exists())

    def test_profile_rejects_explicit_ensemble_accuracy_before_allocating_a_study(self) -> None:
        """The ordinary-accuracy profile must not silently re-enable ensembles."""
        for changes in ({"use_ensemble_accuracy": True},
                        {"ensemble_accuracy_kwargs": {"max_t": 128}}):
            with self.subTest(changes=changes), patch("common.hpo.main") as training:
                destination = self.root / "invalid-ensemble"
                with self.assertRaisesRegex(ValueError, "ordinary.*accuracy"):
                    run_hpo(**self._options(results_path=str(destination), **changes))
                training.assert_not_called()
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
