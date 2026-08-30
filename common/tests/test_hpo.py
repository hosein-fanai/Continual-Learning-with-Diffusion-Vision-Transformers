"""Focused tests for validation-safe and resumable HPO orchestration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import optuna

from common.config import Config
from common.hpo import (
    _build_trial_config,
    _capture_sampler_rng_state,
    _enqueue_recovery_trials,
    _has_committed_task_checkpoint,
    _make_study_spec,
    _normalize_objective_spec,
    _objective_values,
    _restore_sampler_rng_state,
    _write_study_spec,
    run_hpo,
)
from common.recovery import fingerprint_state
from common.recovery import save_task_checkpoint


class _SuggestionTrial:
    """Deterministic first-choice trial sufficient for config construction."""

    number = 3

    def __init__(self) -> None:
        """Exercise the test helper named __init__.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        self.params: dict[str, object] = {}

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        """Exercise the test helper named suggest_categorical.

        Args:
            name (str): Test input named name.
            choices (list[object]): Test input named choices.

        Returns:
            object: Result produced by the test helper.
        """
        value = choices[0]
        self.params[name] = value
        return value

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        **kwargs: object,
    ) -> float:
        """Exercise the test helper named suggest_float.

        Args:
            name (str): Test input named name.
            low (float): Test input named low.
            high (float): Test input named high.
            kwargs (object): Test input named kwargs.

        Returns:
            float: Result produced by the test helper.
        """
        del high, kwargs
        self.params[name] = low
        return low

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        **kwargs: object,
    ) -> int:
        """Exercise the test helper named suggest_int.

        Args:
            name (str): Test input named name.
            low (int): Test input named low.
            high (int): Test input named high.
            kwargs (object): Test input named kwargs.

        Returns:
            int: Result produced by the test helper.
        """
        del high, kwargs
        self.params[name] = low
        return low


class _RuntimeTrial:
    """Small Optuna trial double that records persistent user attributes."""

    def __init__(self, number: int = 2) -> None:
        """Exercise the test helper named __init__.

        Args:
            number (int): Test input named number.

        Returns:
            None: Result produced by the test helper.
        """
        self.number = number
        self.params: dict[str, object] = {}
        self.user_attrs: dict[str, object] = {}

    def set_user_attr(self, name: str, value: object) -> None:
        """Exercise the test helper named set_user_attr.

        Args:
            name (str): Test input named name.
            value (object): Test input named value.

        Returns:
            None: Result produced by the test helper.
        """
        self.user_attrs[name] = value


class _TrialState:
    """Named state double compatible with Optuna's enum interface."""

    def __init__(self, name: str) -> None:
        """Exercise the test helper named __init__.

        Args:
            name (str): Test input named name.

        Returns:
            None: Result produced by the test helper.
        """
        self.name = name


class _FrozenTrial:
    """Persistent trial double used by automatic recovery discovery."""

    def __init__(
        self,
        number: int,
        params: dict[str, object],
        state: str = "RUNNING",
        user_attrs: dict[str, object] | None = None,
    ) -> None:
        """Exercise the test helper named __init__.

        Args:
            number (int): Test input named number.
            params (dict[str, object]): Test input named params.
            state (str): Test input named state.
            user_attrs (dict[str, object] | None): Test input named user_attrs.

        Returns:
            None: Result produced by the test helper.
        """
        self.number = number
        self.params = params
        self.state = _TrialState(state)
        self.user_attrs = dict(user_attrs or {})


class _Study:
    """Study double that executes exactly one objective synchronously."""

    def __init__(
        self,
        trial: _RuntimeTrial,
        frozen_trials: list[_FrozenTrial] | None = None,
    ) -> None:
        """Exercise the test helper named __init__.

        Args:
            trial (_RuntimeTrial): Test input named trial.
            frozen_trials (list[_FrozenTrial] | None): Test input named frozen_trials.

        Returns:
            None: Result produced by the test helper.
        """
        self.trial = trial
        self.frozen_trials = list(frozen_trials or [])
        self.user_attrs: dict[str, object] = {}
        self.enqueued: list[tuple[dict[str, object], dict[str, object]]] = []
        self.value: float | tuple[float, ...] | None = None

    def get_trials(self, deepcopy: bool = False) -> list[_FrozenTrial]:
        """Exercise the test helper named get_trials.

        Args:
            deepcopy (bool): Test input named deepcopy.

        Returns:
            list[_FrozenTrial]: Result produced by the test helper.
        """
        del deepcopy
        return self.frozen_trials

    def enqueue_trial(
        self,
        params: dict[str, object],
        user_attrs: dict[str, object] | None = None,
    ) -> None:
        """Exercise the test helper named enqueue_trial.

        Args:
            params (dict[str, object]): Test input named params.
            user_attrs (dict[str, object] | None): Test input named user_attrs.

        Returns:
            None: Result produced by the test helper.
        """
        attrs = dict(user_attrs or {})
        self.enqueued.append((dict(params), attrs))
        self.trial.params = dict(params)
        self.trial.user_attrs.update(attrs)

    def set_user_attr(self, name: str, value: object) -> None:
        """Exercise the test helper named set_user_attr.

        Args:
            name (str): Test input named name.
            value (object): Test input named value.

        Returns:
            None: Result produced by the test helper.
        """
        self.user_attrs[name] = value

    def optimize(self, objective: object, **kwargs: object) -> None:
        """Exercise the test helper named optimize.

        Args:
            objective (object): Test input named objective.
            kwargs (object): Test input named kwargs.

        Returns:
            None: Result produced by the test helper.
        """
        self.value = objective(self.trial)
        for callback in kwargs.get("callbacks", []):
            callback(self, self.trial)

    def trials_dataframe(self) -> pd.DataFrame:
        """Exercise the test helper named trials_dataframe.

        Args:
            None.

        Returns:
            pd.DataFrame: Result produced by the test helper.
        """
        return pd.DataFrame([{"number": self.trial.number}])


class HpoObjectiveTests(unittest.TestCase):
    """Verify objective selection and its no-leakage continual contract."""

    def test_objective_spec_defaults_and_multiple_metric_inference(self) -> None:
        """Exercise the test helper named test_objective_spec_defaults_and_multiple_metric_inference.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        self.assertEqual(
            _normalize_objective_spec("continual"),
            (("final_average_accuracy",), ("maximize",)),
        )
        self.assertEqual(
            _normalize_objective_spec(
                "continual",
                ["final_average_accuracy", "average_forgetting"],
            ),
            (
                ("final_average_accuracy", "average_forgetting"),
                ("maximize", "minimize"),
            ),
        )
        with self.assertRaises(ValueError):
            _normalize_objective_spec(
                "continual",
                ["final_average_accuracy", "average_forgetting"],
                ["maximize"],
            )

    def test_continual_multiple_objectives_are_validation_only(self) -> None:
        """Exercise the test helper named test_continual_multiple_objectives_are_validation_only.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        values = _objective_values(
            "continual",
            "dit_classifier",
            {
                "task_val_accuracy": [0.99],
                "continual_accuracy": [0.98],
            },
            evaluations={
                "validation_continual_metrics": {
                    "final_average_accuracy": 0.63,
                    "average_forgetting": 0.08,
                },
                "final_average_accuracy": 0.97,
            },
            objective_metrics=[
                "final_average_accuracy",
                "average_forgetting",
            ],
        )
        self.assertEqual(values, (0.63, 0.08))

    def test_continual_objective_never_falls_back_to_test_or_history(self) -> None:
        """Exercise the test helper named test_continual_objective_never_falls_back_to_test_or_history.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        with self.assertRaises(KeyError):
            _objective_values(
                "continual",
                "dit_classifier",
                {"task_val_accuracy": [0.91], "continual_accuracy": [0.95]},
                evaluations={
                    "continual_metrics": {"final_average_accuracy": 0.96},
                    "final_average_accuracy": 0.97,
                },
            )

    def test_continual_ensemble_uses_selected_validation_mapping(self) -> None:
        """Exercise the test helper named test_continual_ensemble_uses_selected_validation_mapping.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        value = _objective_values(
            "continual",
            "dit_classifier",
            {},
            evaluations={
                "validation_continual_metrics": {
                    "final_average_accuracy": 0.71,
                },
            },
            use_ensemble_accuracy=True,
        )
        self.assertEqual(value, 0.71)

    def test_noncontinual_legacy_defaults_remain_unchanged(self) -> None:
        """Exercise the test helper named test_noncontinual_legacy_defaults_remain_unchanged.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        self.assertEqual(
            _objective_values(
                "generation",
                "diffusion_transformer",
                {"val_noise_loss": [0.8, 0.4, 0.6]},
            ),
            0.4,
        )
        self.assertEqual(
            _objective_values(
                "classification",
                "cnn",
                {"val_accuracy": [0.4, 0.8, 0.7]},
            ),
            0.8,
        )
        self.assertEqual(
            _objective_values(
                "joint",
                "dit_classifier",
                {
                    "val_noise_loss": [0.2, 0.3],
                    "val_classifier_accuracy": [0.8, 0.7],
                },
            ),
            (0.3, 0.7),
        )


class HpoConfigTests(unittest.TestCase):
    """Verify trial config and persistent study integration fields."""

    def test_builder_sets_runtime_continual_and_objective_metadata(self) -> None:
        """Exercise the test helper named test_builder_sets_runtime_continual_and_objective_metadata.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        config = _build_trial_config(
            _SuggestionTrial(),
            "continual",
            "dit_classifier",
            "mnist",
            epochs=1,
            seed=17,
            results_path="results/hpo",
            use_ensemble_accuracy=True,
            objective_metrics=[
                "final_average_accuracy",
                "average_forgetting",
            ],
            dtype_policy="mixed_float16",
            deterministic_ops=True,
            snapshot_network_name="ema",
            class_num=4,
            class_order=[0, 1, 2, 3],
            task_groups=[[0, 1], [2, 3]],
            task_size=2,
            class_order_mode="fixed",
            task_order_mode="random",
        )
        self.assertEqual(config.training.dtype_policy, "mixed_float16")
        self.assertTrue(config.training.deterministic_ops)
        self.assertFalse(Config().continually_learn.save_task_checkpoints)
        self.assertTrue(config.continually_learn.save_task_checkpoints)
        self.assertEqual(config.continually_learn.seed, 17)
        self.assertTrue(config.continually_learn.use_ensemble_accuracy)
        self.assertTrue(config.continually_learn.evaluate_ensemble_accuracy)
        self.assertEqual(config.continually_learn.snapshot_network_name, "ema")
        self.assertEqual(config.continually_learn.class_num, 4)
        self.assertEqual(config.continually_learn.class_order, [0, 1, 2, 3])
        self.assertEqual(
            config.continually_learn.task_groups,
            [[0, 1], [2, 3]],
        )
        self.assertEqual(config.continually_learn.task_size, 2)
        self.assertEqual(config.continually_learn.task_order_mode, "random")
        self.assertEqual(
            config.hpo["objective_metrics"],
            ["final_average_accuracy", "average_forgetting"],
        )
        self.assertEqual(
            config.hpo["objective_directions"],
            ["maximize", "minimize"],
        )

    def test_real_optuna_recovery_queue_is_persistent_and_idempotent(self) -> None:
        """Exercise the test helper named test_real_optuna_recovery_queue_is_persistent_and_idempotent.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        with tempfile.TemporaryDirectory() as temporary:
            study_root = Path(temporary)
            study = optuna.create_study(
                study_name="recovery-test",
                storage=optuna.storages.InMemoryStorage(),
                direction="maximize",
            )
            source = study.ask()
            source_depth = source.suggest_int("depth", 1, 4)
            committed = (
                study_root
                / "checkpoints"
                / f"trial-{source.number:04d}"
            )
            save_task_checkpoint(
                committed,
                completed_task_index=0,
                state={
                    "class_order": [0],
                    "task_groups": [[0]],
                },
            )

            self.assertEqual(
                _enqueue_recovery_trials(study, study_root),
                (source.number,),
            )
            self.assertEqual(_enqueue_recovery_trials(study, study_root), ())
            queued = [
                trial
                for trial in study.get_trials(deepcopy=False)
                if trial.state.name == "WAITING"
            ]
            self.assertEqual(len(queued), 1)
            self.assertEqual(
                queued[0].system_attrs["fixed_params"],
                {"depth": source_depth},
            )
            self.assertEqual(
                queued[0].user_attrs["resume_original_trial_number"],
                source.number,
            )
            self.assertTrue(
                queued[0].user_attrs["resume_has_task_checkpoint"]
            )
            self.assertEqual(
                study.user_attrs["recovery_enqueued_trial_numbers"],
                [source.number],
            )

    def test_abandoned_trial_without_boundary_is_retried_from_task_zero(self) -> None:
        """An empty root retries parameters but is not a model resume path.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        with tempfile.TemporaryDirectory() as temporary:
            study_root = Path(temporary)
            checkpoint_dir = study_root / "checkpoints" / "trial-0002"
            checkpoint_dir.mkdir(parents=True)
            # A fake marker cannot upgrade an incomplete directory.
            fake = checkpoint_dir / "task-0000"
            fake.mkdir()
            (fake / "COMMITTED").touch()
            self.assertFalse(_has_committed_task_checkpoint(checkpoint_dir))

            trial = _RuntimeTrial(number=3)
            source = _FrozenTrial(2, {"depth": 4})
            study = _Study(trial, [source])
            self.assertEqual(
                _enqueue_recovery_trials(study, study_root),
                (2,),
            )
            params, attrs = study.enqueued[0]
            self.assertEqual(params, {"depth": 4})
            self.assertFalse(attrs["resume_has_task_checkpoint"])
            self.assertEqual(
                Path(attrs["resume_checkpoint_dir"]),
                checkpoint_dir.resolve(),
            )

    def test_failed_trial_is_requeued_only_with_a_committed_boundary(self) -> None:
        """Retry failed trials only when task-level recovery state is valid.

        Args:
            None.

        Returns:
            None: Only the recoverable failed trial is enqueued.
        """

        with tempfile.TemporaryDirectory() as temporary:
            study_root = Path(temporary)
            committed_root = (
                study_root / "checkpoints" / "trial-0003"
            )
            save_task_checkpoint(
                committed_root,
                completed_task_index=0,
                state={
                    "class_order": [0, 1],
                    "task_groups": [[0], [1]],
                },
            )

            runtime_trial = _RuntimeTrial(number=5)
            recoverable = _FrozenTrial(
                3,
                {"depth": 2},
                state="FAIL",
            )
            unrecoverable = _FrozenTrial(
                4,
                {"depth": 3},
                state="FAIL",
            )
            study = _Study(
                runtime_trial,
                [recoverable, unrecoverable],
            )

            self.assertEqual(
                _enqueue_recovery_trials(study, study_root),
                (3,),
            )
            self.assertEqual(len(study.enqueued), 1)
            params, attrs = study.enqueued[0]
            self.assertEqual(params, {"depth": 2})
            self.assertTrue(attrs["resume_has_task_checkpoint"])
            self.assertEqual(attrs["resume_source_trial_number"], 3)
            self.assertEqual(
                study.user_attrs["recovery_enqueued_trial_numbers"],
                [3],
            )

    def test_sampler_rng_round_trip_preserves_both_tpe_streams(self) -> None:
        """Reopened TPE sampling continues after the persisted draw cursor.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        source = optuna.samplers.TPESampler(seed=29)
        source_tpe = source._rng.rng
        source_random = source._random_sampler._rng.rng
        source_tpe.randint(0, 1000, size=7)
        source_random.randint(0, 1000, size=5)
        state = _capture_sampler_rng_state(source)
        expected_tpe = source_tpe.randint(0, 2**31 - 1, size=16)
        expected_random = source_random.randint(0, 2**31 - 1, size=16)

        reopened = optuna.samplers.TPESampler(seed=999)
        _restore_sampler_rng_state(reopened, state)
        np.testing.assert_array_equal(
            expected_tpe,
            reopened._rng.rng.randint(0, 2**31 - 1, size=16),
        )
        np.testing.assert_array_equal(
            expected_random,
            reopened._random_sampler._rng.rng.randint(
                0, 2**31 - 1, size=16
            ),
        )

    def test_resume_spec_mismatch_fails_before_optuna_load(self) -> None:
        """A changed seed/identity cannot create or load another DB study.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        with tempfile.TemporaryDirectory() as temporary:
            study_root = Path(temporary)
            (study_root / "study.db").touch()
            original = _make_study_spec(
                study_name="continual-dit_classifier-cifar10",
                task="continual",
                model_name="dit_classifier",
                dataset_name="CIFAR10",
                epochs=1,
                seed=11,
                use_ensemble_accuracy=False,
                ensemble_accuracy_kwargs=None,
                fit_method="fit",
                fit_kwargs={},
                teacher_network=None,
                effective_distillation=False,
                objective_metrics=("final_average_accuracy",),
                objective_directions=("maximize",),
                dtype_policy="float32",
                deterministic_ops=False,
                snapshot_network_name="raw",
                class_num=None,
                class_order=None,
                task_groups=None,
                task_size=1,
                class_order_mode="fixed",
                task_order_mode="fixed",
            )
            _write_study_spec(study_root, original)

            with patch("optuna.load_study") as load_study:
                with self.assertRaisesRegex(
                    ValueError,
                    "specification differs",
                ):
                    run_hpo(
                        "continual",
                        "dit_classifier",
                        n_trials=1,
                        epochs=1,
                        seed=12,
                        resume_from=study_root,
                    )
            load_study.assert_not_called()

    def test_run_hpo_loads_root_and_publishes_trial_recovery_before_main(
        self,
    ) -> None:
        """Exercise the test helper named test_run_hpo_loads_root_and_publishes_trial_recovery_before_main.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        with tempfile.TemporaryDirectory() as temporary:
            study_root = Path(temporary) / "existing-study"
            study_root.mkdir()
            (study_root / "study.db").touch()
            checkpoint_dir = study_root / "checkpoints" / "trial-0004"
            save_task_checkpoint(
                checkpoint_dir,
                completed_task_index=0,
                state={
                    "class_order": [0],
                    "task_groups": [[0]],
                },
            )
            result_dir = study_root / "completed-run"
            trial = _RuntimeTrial(number=5)
            source = _FrozenTrial(4, {"depth": 3})
            study = _Study(trial, [source])
            sampler = optuna.samplers.TPESampler(seed=100)
            study_spec = _make_study_spec(
                study_name=(
                    "continual-dit_classifier-cifar10-ensemble-accuracy"
                ),
                task="continual",
                model_name="dit_classifier",
                dataset_name="CIFAR10",
                epochs=1,
                seed=100,
                use_ensemble_accuracy=True,
                ensemble_accuracy_kwargs=None,
                fit_method="fit",
                fit_kwargs={},
                teacher_network=None,
                effective_distillation=False,
                objective_metrics=(
                    "final_average_accuracy",
                    "average_forgetting",
                ),
                objective_directions=("maximize", "minimize"),
                dtype_policy="mixed_float16",
                deterministic_ops=True,
                snapshot_network_name="ema",
                class_num=None,
                class_order=None,
                task_groups=None,
                task_size=1,
                class_order_mode="fixed",
                task_order_mode="fixed",
            )
            _write_study_spec(study_root, study_spec)
            study.user_attrs.update({
                "study_spec": study_spec,
                "study_spec_fingerprint": fingerprint_state(study_spec),
                "sampler_rng_state": _capture_sampler_rng_state(sampler),
            })
            load_kwargs: dict[str, object] = {}
            main_observation: dict[str, object] = {}

            def fake_load_study(**kwargs: object) -> _Study:
                """Exercise the test helper named fake_load_study.

                Args:
                    kwargs (object): Test input named kwargs.

                Returns:
                    _Study: Result produced by the test helper.
                """
                load_kwargs.update(kwargs)
                return study

            def fake_builder(*args: object, **kwargs: object) -> Config:
                """Exercise the test helper named fake_builder.

                Args:
                    args (object): Test input named args.
                    kwargs (object): Test input named kwargs.

                Returns:
                    Config: Result produced by the test helper.
                """
                main_observation["builder_kwargs"] = kwargs
                return Config(
                    training={"task": "continual"},
                    hpo={
                        "use_ensemble_accuracy": True,
                        "objective_metrics": [
                            "final_average_accuracy",
                            "average_forgetting",
                        ],
                        "objective_directions": ["maximize", "minimize"],
                    },
                )

            def fake_main(config: Config, **kwargs: object) -> dict[str, object]:
                """Exercise the test helper named fake_main.

                Args:
                    config (Config): Test input named config.
                    kwargs (object): Test input named kwargs.

                Returns:
                    dict[str, object]: Result produced by the test helper.
                """
                del kwargs
                main_observation["attrs_before_main"] = dict(trial.user_attrs)
                main_observation["config"] = config
                result_dir.mkdir(exist_ok=True)
                return {
                    "history": {"task_val_accuracy": [0.99]},
                    "evaluations": {
                        "validation_continual_metrics": {
                            "final_average_accuracy": 0.67,
                            "average_forgetting": 0.11,
                        },
                    },
                    "results_path": str(result_dir),
                }

            with patch("optuna.load_study", side_effect=fake_load_study), \
                    patch("optuna.samplers.TPESampler", return_value=sampler), \
                    patch("common.hpo._build_trial_config", side_effect=fake_builder), \
                    patch("common.hpo.main", side_effect=fake_main), \
                    patch("common.hpo.tf.keras.backend.clear_session"), \
                    patch("common.hpo.gc.collect"):
                returned = run_hpo(
                    "continual",
                    "dit_classifier",
                    n_trials=1,
                    epochs=1,
                    seed=100,
                    use_ensemble_accuracy=True,
                    objective_metrics=[
                        "final_average_accuracy",
                        "average_forgetting",
                    ],
                    dtype_policy="mixed_float16",
                    deterministic_ops=True,
                    resume_from=study_root,
                    snapshot_network_name="ema",
                )

            self.assertIs(returned, study)
            self.assertEqual(study.value, (0.67, 0.11))
            self.assertIn(str((study_root / "study.db").resolve().as_posix()),
                          load_kwargs["storage"])
            self.assertEqual(
                load_kwargs["study_name"],
                "continual-dit_classifier-cifar10-ensemble-accuracy",
            )
            attrs = main_observation["attrs_before_main"]
            self.assertEqual(attrs["seed"], 100)
            self.assertEqual(attrs["checkpoint_dir"], str(checkpoint_dir))
            self.assertTrue(str(attrs["config_path"]).endswith("trial-0005.yaml"))
            self.assertEqual(attrs["resume_original_trial_number"], 4)
            self.assertEqual(attrs["resume_source_trial_number"], 4)
            self.assertEqual(study.enqueued, [
                (
                    {"depth": 3},
                    {
                        "resume_checkpoint_dir": str(checkpoint_dir.resolve()),
                        "resume_has_task_checkpoint": True,
                        "resume_original_trial_number": 4,
                        "resume_source_trial_number": 4,
                    },
                ),
            ])
            self.assertEqual(
                study.user_attrs["recovery_enqueued_trial_numbers"], [4]
            )
            self.assertEqual(_enqueue_recovery_trials(study, study_root), ())
            self.assertEqual(len(study.enqueued), 1)
            config = main_observation["config"]
            self.assertEqual(config.continually_learn.checkpoint_dir,
                             str(checkpoint_dir))
            self.assertEqual(config.continually_learn.resume_from,
                             str(checkpoint_dir))
            builder_kwargs = main_observation["builder_kwargs"]
            self.assertEqual(builder_kwargs["dtype_policy"], "mixed_float16")
            self.assertTrue(builder_kwargs["deterministic_ops"])
            self.assertEqual(builder_kwargs["snapshot_network_name"], "ema")


# Select the test action required by this condition.
if __name__ == "__main__":
    unittest.main()
