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
    SEARCH_SPACE_VERSION,
    _build_trial_config,
    _capture_sampler_rng_state,
    _enqueue_recovery_trials,
    _feature_archive_signature,
    _has_committed_task_checkpoint,
    _make_study_spec,
    _normalize_objective_spec,
    _objective_values,
    _restore_sampler_rng_state,
    _suggest_joint,
    _tensorboard_name,
    _write_study_spec,
    run_hpo,
)
from common.learner import _continual_metrics
from common.recovery import fingerprint_state
from common.recovery import save_task_checkpoint
from common.utils import save_feature_split_metadata, save_samples


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
        self.assertEqual(
            _normalize_objective_spec("generation", "rmse")[1],
            ("minimize",),
        )
        with self.assertRaisesRegex(ValueError, "provide objective_directions"):
            _normalize_objective_spec("generation", "custom_score")

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
        with self.assertRaisesRegex(ValueError, "must be finite"):
            _objective_values(
                "continual",
                "dit_classifier",
                {},
                evaluations={
                    "validation_continual_metrics": {
                        "final_average_accuracy": float("nan"),
                    },
                },
            )

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

    def test_noncontinual_defaults_use_validation_metrics(self) -> None:
        """Tie scalar and joint objectives to the final saved model state.

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
                evaluations={
                    "valset_ema_eval": {"noise_loss": 0.55},
                    "valset_network_eval": {"noise_loss": 0.65},
                },
            ),
            0.55,
        )
        self.assertEqual(
            _objective_values(
                "classification",
                "cnn",
                {"val_accuracy": [0.4, 0.8, 0.7]},
                evaluations={"valset_eval": {"accuracy": 0.75}},
            ),
            0.75,
        )
        self.assertEqual(
            _objective_values(
                "joint",
                "dit_classifier",
                {
                    "val_noise_loss": [0.2, 0.3],
                    "val_classifier_accuracy": [0.8, 0.7],
                },
                evaluations={
                    "valset_ema_eval": {
                        "noise_loss": 0.25,
                        "classifier_accuracy": 0.72,
                    },
                },
            ),
            (0.25, 0.72),
        )
        self.assertEqual(
            _objective_values(
                "generation",
                "diffusion_transformer",
                {},
                evaluations={
                    "valset_ema_eval": {"noise_loss": 0.55},
                    "valset_network_eval": {"noise_loss": 0.65},
                },
                diffusion_network_name="raw",
            ),
            0.65,
        )

    def test_explicit_multiobjective_values_share_one_epoch(self) -> None:
        """Keep a Pareto vector tied to one attainable checkpoint state.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        values = _objective_values(
            "joint",
            "dit_classifier",
            {
                "val_noise_loss": [0.2, 0.3],
                "val_classifier_accuracy": [0.8, 0.7],
            },
            evaluations={
                "valset_ema_eval": {
                    "noise_loss": 0.27,
                    "classifier_accuracy": 0.73,
                },
            },
            objective_metrics=(
                "val_noise_loss",
                "val_classifier_accuracy",
            ),
            objective_directions=("minimize", "maximize"),
        )
        self.assertEqual(values, (0.27, 0.73))

    def test_vae_classifier_objective_excludes_classifier_loss(self) -> None:
        """Prefer the beta-VAE objective over joint total loss."""

        evaluations = {
            "valset_eval": {
                "loss": 4.2,
                "generative_loss": 0.35,
                "recon_loss": 0.3,
                "clf_accuracy": 0.8,
            },
        }
        self.assertEqual(
            _objective_values(
                "generation",
                "vae_classifier",
                {},
                evaluations=evaluations,
            ),
            0.35,
        )
        self.assertEqual(
            _objective_values(
                "joint",
                "vae_classifier",
                {},
                evaluations=evaluations,
            ),
            (0.35, 0.8),
        )
        without_generative_loss = {
            "valset_eval": {
                "loss": 4.2,
                "recon_loss": 0.3,
                "clf_accuracy": 0.8,
            },
        }
        with self.assertRaisesRegex(KeyError, "generative_loss"):
            _objective_values(
                "joint",
                "vae_classifier",
                {},
                evaluations=without_generative_loss,
            )
        self.assertEqual(
            _objective_values(
                "joint",
                "vae_classifier",
                {},
                evaluations=evaluations,
                objective_metrics=(
                    "generation_loss",
                    "classification_accuracy",
                ),
                objective_directions=("minimize", "maximize"),
            ),
            (0.35, 0.8),
        )

    def test_x0_objective_includes_weighted_main_kl(self) -> None:
        """Score x0 reconstruction and KL without classifier contamination."""

        evaluations = {
            "valset_ema_eval": {
                "loss": 7.0,
                "noise_loss": 0.3,
                "kl_loss": 0.4,
                "classifier_accuracy": 0.8,
            },
        }
        kwargs = {
            "evaluations": evaluations,
            "swap_noise_image": True,
            "kl_loss_coef": 0.25,
        }
        self.assertAlmostEqual(
            _objective_values(
                "generation",
                "diffusion_transformer",
                {},
                **kwargs,
            ),
            0.4,
        )
        self.assertEqual(
            _objective_values(
                "joint",
                "dit_encoder_decoder_classifier",
                {},
                **kwargs,
            ),
            (0.4, 0.8),
        )
        self.assertAlmostEqual(
            _objective_values(
                "generation",
                "diffusion_transformer",
                {},
                objective_metrics="generation_loss",
                objective_directions="minimize",
                **kwargs,
            ),
            0.4,
        )

    def test_noncontinual_objectives_do_not_fall_back_to_training(self) -> None:
        """Reject training metrics when validation metrics are unavailable.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """
        for task, model_name, history in (
            ("generation", "diffusion_transformer", {"noise_loss": [0.1]}),
            ("classification", "cnn", {"accuracy": [0.9]}),
            (
                "joint",
                "dit_classifier",
                {"noise_loss": [0.1], "classifier_accuracy": [0.9]},
            ),
        ):
            with self.subTest(task=task), self.assertRaises(KeyError):
                _objective_values(task, model_name, history)


class HpoConfigTests(unittest.TestCase):
    """Verify HPO configuration, search spaces, and objective contracts."""

    def test_tensorboard_name_hashes_wide_conditional_spaces(self) -> None:
        """Keep Windows event paths short without losing run identity."""

        trial = _SuggestionTrial()
        trial.params = {
            f"family.parameter_{index}": "conditional_value_" * 3
            for index in range(20)
        }
        first = _tensorboard_name(trial)
        self.assertRegex(first, r"^t0003-h[0-9a-f]{16}$")
        self.assertEqual(first, _tensorboard_name(trial))

        trial.params["family.parameter_0"] = "changed"
        self.assertNotEqual(first, _tensorboard_name(trial))

    def test_umbrella_overrides_cannot_escape_valid_domains(self) -> None:
        """Reject non-classifier families and impossible sampling horizons."""

        common = {
            "trial": _SuggestionTrial(),
            "task": "continual",
            "model_name": "diffusion_classifier",
            "dataset_name": "mnist",
            "epochs": 1,
            "seed": 19,
            "results_path": "results/hpo",
            "class_num": 4,
            "task_groups": [[0, 1], [2, 3]],
            "task_size": 2,
        }
        with self.assertRaisesRegex(ValueError, "model_family"):
            _build_trial_config(
                **common,
                search_space_overrides={
                    "model_family": ["diffusion_transformer"],
                },
            )
        with self.assertRaisesRegex(ValueError, "test_steps"):
            _build_trial_config(
                **common,
                search_space_overrides={
                    "model_family": ["dit_classifier"],
                    "timesteps": [250],
                    "test_steps": [1000],
                },
            )

    def test_umbrella_requires_family_safe_fixed_overrides(self) -> None:
        """Reserve raw/wrapper overrides for an exact architecture family."""

        with self.assertRaisesRegex(ValueError, "exact diffusion classifier"):
            _build_trial_config(
                _SuggestionTrial(),
                "continual",
                "diffusion_classifier",
                "mnist",
                epochs=1,
                seed=19,
                results_path="results/hpo",
                class_num=4,
                task_size=2,
                model_overrides={"drop_prob": 0.1},
            )

    def test_feature_archive_signature_binds_dataset_shape_and_metadata(
        self,
    ) -> None:
        """Reject cross-dataset bundles and fingerprint label alignment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "cifar10_xception_features_safe"
            bundle = np.empty(3, dtype=object)
            bundle[:] = [
                np.zeros((2, 2048), np.float32),
                np.zeros((1, 2048), np.float32),
                np.zeros((2, 2048), np.float32),
            ]
            save_samples(bundle, base, ".npy")
            save_feature_split_metadata(base, 7, 0.2)
            first = _feature_archive_signature(base, "cifar10")
            save_feature_split_metadata(base, 11, 0.2)
            second = _feature_archive_signature(base, "cifar10")

            self.assertEqual(first["sha256"], second["sha256"])
            self.assertNotEqual(first["metadata"], second["metadata"])
            self.assertEqual(first["split_shapes"][0], [2, 2048])
            with self.assertRaisesRegex(ValueError, "dataset marker"):
                _feature_archive_signature(base, "cifar100")

            bad_base = Path(temp_dir) / "cifar10_bad_features_safe"
            bad_bundle = np.empty(3, dtype=object)
            bad_bundle[:] = [
                np.zeros((2, 32), np.float32),
                np.zeros((1, 32), np.float32),
                np.zeros((2, 32), np.float32),
            ]
            save_samples(bad_bundle, bad_base, ".npy")
            with self.assertRaisesRegex(ValueError, "2048"):
                _feature_archive_signature(bad_base, "cifar10")

            for name, dtype in (("text", "U1"), ("boolean", np.bool_)):
                invalid_base = Path(temp_dir) / f"cifar10_{name}_features_safe"
                invalid_bundle = np.empty(3, dtype=object)
                invalid_bundle[:] = [
                    np.zeros((2, 2048), dtype=dtype),
                    np.zeros((1, 2048), dtype=dtype),
                    np.zeros((2, 2048), dtype=dtype),
                ]
                save_samples(invalid_bundle, invalid_base, ".npy")
                with self.subTest(dtype=name), self.assertRaisesRegex(
                    ValueError,
                    "real numeric",
                ):
                    _feature_archive_signature(invalid_base, "cifar10")

            no_metadata = Path(temp_dir) / "cifar10_no_metadata_safe"
            save_samples(bundle, no_metadata, ".npy")
            with self.assertRaisesRegex(FileNotFoundError, "split metadata"):
                _feature_archive_signature(no_metadata, "cifar10")

    def test_singleton_classifier_schedule_excludes_sequential_protocol(
        self,
    ) -> None:
        """Keep the first one-way head from becoming a permanent baseline."""
        singleton_trial = _SuggestionTrial()
        singleton = _build_trial_config(
            singleton_trial,
            "continual",
            "cnn",
            "mnist",
            epochs=1,
            seed=3,
            results_path="results/hpo",
            class_num=4,
            task_size=1,
        )
        self.assertEqual(singleton.continually_learn.baseline, "cumulative")
        self.assertNotIn("continual_protocol", singleton_trial.params)
        self.assertEqual(
            singleton_trial.params["continual_protocol_singleton"],
            "cumulative",
        )

        multiclass_trial = _SuggestionTrial()
        multiclass = _build_trial_config(
            multiclass_trial,
            "continual",
            "cnn",
            "mnist",
            epochs=1,
            seed=3,
            results_path="results/hpo",
            class_num=4,
            task_size=2,
        )
        self.assertEqual(multiclass.continually_learn.baseline, "sequential")
        self.assertEqual(
            multiclass_trial.params["continual_protocol_multiclass"],
            "sequential",
        )

        partial_trial = _SuggestionTrial()
        partial = _build_trial_config(
            partial_trial,
            "continual",
            "cnn",
            "mnist",
            epochs=1,
            seed=4,
            results_path="results/hpo",
            class_num=5,
            task_size=2,
            task_order_mode="random",
        )
        self.assertEqual(partial.continually_learn.baseline, "sequential")
        self.assertIn("continual_protocol_multiclass", partial_trial.params)
        for metric in ("average_forgetting", "backward_transfer"):
            singleton_metric = _build_trial_config(
                _SuggestionTrial(),
                "continual",
                "cnn",
                "mnist",
                epochs=1,
                seed=3,
                results_path="results/hpo",
                class_num=3,
                task_size=1,
                objective_metrics=metric,
            )
            self.assertEqual(singleton_metric.hpo["objective_metrics"], [metric])

    def test_reverse_sampling_is_tuned_only_when_it_affects_replay(self) -> None:
        """Exclude unused reverse-process dimensions from loss objectives."""

        generation_trial = _SuggestionTrial()
        generation = _build_trial_config(
            generation_trial,
            "generation",
            "diffusion_transformer",
            "mnist",
            epochs=1,
            seed=3,
            results_path="results/hpo",
        )
        self.assertEqual(generation.model.wrapper_kwargs["test_steps"], 50)
        self.assertEqual(generation.model.wrapper_kwargs["test_cfg_scale"], 4.)
        self.assertEqual(generation.model.wrapper_kwargs["test_eta"], 0.)
        self.assertEqual(generation.training.monitor, "val_noise_loss")
        self.assertNotIn("test_cfg_scale", generation_trial.params)
        self.assertNotIn("test_eta", generation_trial.params)
        self.assertFalse(any(
            name.startswith("test_steps_t") for name in generation_trial.params
        ))

        continual_trial = _SuggestionTrial()
        continual = _build_trial_config(
            continual_trial,
            "continual",
            "diffusion_transformer",
            "mnist",
            epochs=1,
            seed=3,
            results_path="results/hpo",
            class_num=4,
            task_size=2,
        )
        self.assertEqual(continual.model.wrapper_kwargs["test_steps"], 10)
        self.assertIn("test_steps_t250", continual_trial.params)
        self.assertIn("test_cfg_scale", continual_trial.params)
        self.assertIn("test_eta", continual_trial.params)

        vae = _build_trial_config(
            _SuggestionTrial(),
            "generation",
            "vae",
            "mnist",
            epochs=1,
            seed=3,
            results_path="results/hpo",
        )
        self.assertEqual(vae.training.monitor, "val_loss")

    def test_swap_noise_override_removes_incompatible_hpo_dimensions(
        self,
    ) -> None:
        """Keep x0 HPO variational, sampleable, and trajectory-free."""

        dit_vae_overrides = {
            "vit_block_ids": [1],
            "reshaper_ids_dict": {1: "flatten", 2: "unflatten"},
            "reshaper_kwargs": {"add_kl": True},
        }

        with self.assertRaisesRegex(ValueError, "reshaper_kwargs"):
            _build_trial_config(
                _SuggestionTrial(),
                "continual",
                "diffusion_transformer",
                "mnist",
                epochs=1,
                seed=3,
                results_path="results/hpo",
                class_num=4,
                task_size=2,
                wrapper_overrides={"swap_noise_image": True},
            )

        trial = _SuggestionTrial()
        config = _build_trial_config(
            trial,
            "continual",
            "diffusion_transformer",
            "mnist",
            epochs=1,
            seed=3,
            results_path="results/hpo",
            class_num=4,
            task_size=2,
            model_overrides=dit_vae_overrides,
            wrapper_overrides={"swap_noise_image": True},
        )
        self.assertTrue(config.model.wrapper_kwargs["swap_noise_image"])
        self.assertEqual(config.model.wrapper_kwargs["image_loss_coef"], 0.)
        self.assertGreater(config.model.wrapper_kwargs["kl_loss_coef"], 0.)
        self.assertNotIn("image_loss_coef", trial.params)
        self.assertIn("kl_loss_coef", trial.params)
        self.assertNotIn("test_cfg_scale", trial.params)
        self.assertNotIn("test_eta", trial.params)
        self.assertFalse(any(
            name.startswith("test_steps_t") for name in trial.params
        ))

        generation_trial = _SuggestionTrial()
        generation = _build_trial_config(
            generation_trial,
            "generation",
            "diffusion_transformer",
            "mnist",
            epochs=1,
            seed=3,
            results_path="results/hpo",
            model_overrides=dit_vae_overrides,
            wrapper_overrides={
                "swap_noise_image": True,
                "kl_loss_coef": 0.02,
            },
        )
        self.assertEqual(generation.model.wrapper_kwargs["kl_loss_coef"], 0.02)
        self.assertNotIn("kl_loss_coef", generation_trial.params)
        self.assertEqual(generation.training.monitor, "val_loss")
        self.assertFalse(generation.reporting.save_final_gifs)

        unet = _build_trial_config(
            _SuggestionTrial(),
            "generation",
            "unet",
            "mnist",
            epochs=1,
            seed=3,
            results_path="results/hpo",
            model_overrides={
                "reshaper_kwargs": {"add_kl": True},
                "use_skip_connections": True,
            },
            wrapper_overrides={"swap_noise_image": True},
        )
        self.assertTrue(unet.model.wrapper_kwargs["swap_noise_image"])

        classifier = _build_trial_config(
            _SuggestionTrial(),
            "joint",
            "dit_classifier",
            "mnist",
            epochs=1,
            seed=3,
            results_path="results/hpo",
            model_overrides=dit_vae_overrides,
            wrapper_overrides={"swap_noise_image": True},
        )
        self.assertTrue(classifier.model.wrapper_kwargs["swap_noise_image"])

        with self.assertRaisesRegex(ValueError, "noise_loss_coef"):
            _build_trial_config(
                _SuggestionTrial(),
                "generation",
                "diffusion_transformer",
                "mnist",
                epochs=1,
                seed=3,
                results_path="results/hpo",
                model_overrides=dit_vae_overrides,
                wrapper_overrides={
                    "swap_noise_image": True,
                    "noise_loss_coef": 0.5,
                },
            )

        with self.assertRaisesRegex(ValueError, "swap_noise_image"):
            _build_trial_config(
                _SuggestionTrial(),
                "continual",
                "diffusion_transformer",
                "mnist",
                epochs=1,
                seed=3,
                results_path="results/hpo",
                use_distillation=True,
                class_num=4,
                task_size=2,
                wrapper_overrides={"swap_noise_image": True},
            )

        with tempfile.TemporaryDirectory() as temporary:
            results_path = Path(temporary) / "hpo"
            with self.assertRaisesRegex(ValueError, "swap_noise_image"):
                run_hpo(
                    "continual",
                    "diffusion_transformer",
                    dataset_name="mnist",
                    n_trials=1,
                    epochs=1,
                    results_path=str(results_path),
                    use_distillation=True,
                    class_num=4,
                    task_size=2,
                    wrapper_overrides={"swap_noise_image": True},
                )
            self.assertFalse(results_path.exists())

        with tempfile.TemporaryDirectory() as temporary:
            results_path = Path(temporary) / "hpo"
            with self.assertRaisesRegex(ValueError, "reshaper_kwargs"):
                run_hpo(
                    "generation",
                    "diffusion_transformer",
                    dataset_name="mnist",
                    n_trials=1,
                    epochs=1,
                    results_path=str(results_path),
                    wrapper_overrides={"swap_noise_image": True},
                )
            self.assertFalse(results_path.exists())

    def test_stochastic_u_vae_hpo_templates_construct(self) -> None:
        """Keep every U-VAE classifier skip behind a sampled latent."""

        import tensorflow as tf
        from diffusion.models.transformer.di_t_classifier import DiTClassifier

        class ArchitectureTrial(_SuggestionTrial):
            """Force one classifier architecture while accepting other choices."""

            def __init__(self, architecture: str) -> None:
                super().__init__()
                self.architecture = architecture

            def suggest_categorical(
                self, name: str, choices: list[object]
            ) -> object:
                """Return the requested architecture or a valid default choice."""

                if name.startswith("classifier_architecture"):
                    value = self.architecture
                elif name == "feature_aggregation":
                    value = "last"
                elif name == "classifier_only_cls_token":
                    value = False
                elif name == "clf_latent_dim_ratio":
                    value = 0.125
                else:
                    value = choices[0]
                self.params[name] = value
                return value

        inputs = (
            tf.zeros((2, 8, 8, 1)),
            tf.zeros((2,), tf.int32),
            tf.constant([1, 2], tf.uint8),
        )
        for architecture, latent_count in (
            ("u_vae", 2),
            ("u_multilevel_vae", 3),
        ):
            network_kwargs = {
                "num_classes": 2,
                "use_cfg": True,
                "timesteps": 4,
                "image_size": 8,
                "channels": 1,
                "patch_size": 2,
                "dim": 8,
                "depth": 1,
                "mha_num_heads": 1,
                "vit_block_mlp_ratio": 1.,
            }
            _suggest_joint(
                ArchitectureTrial(architecture),
                "dit_classifier",
                network_kwargs,
                {},
                image_size=8,
            )
            network = DiTClassifier(**network_kwargs)
            outputs = network(inputs, full_return=True, training=False)
            self.assertEqual(len(outputs["clf_z_vals_list"]), latent_count)
            unflatten_ids = {
                depth for depth, reshape_type in
                network.clf_reshaper_ids_dict.items()
                if reshape_type == "unflatten"
            }
            for depth, sources in network.clf_connection_ids_dict.items():
                if depth <= network.clf_depth:
                    self.assertTrue(set(sources) & unflatten_ids)

    def test_run_hpo_accepts_singleton_transfer_objectives(self) -> None:
        """Later task cells make singleton-first CL objectives computable."""

        accuracy_matrix = [
            [np.nan, np.nan, np.nan],
            [0.80, 0.75, np.nan],
            [0.65, 0.70, 0.85],
        ]
        metrics = _continual_metrics(accuracy_matrix)
        self.assertAlmostEqual(metrics["average_forgetting"], 0.10)
        self.assertAlmostEqual(metrics["backward_transfer"], -0.05)

        with tempfile.TemporaryDirectory() as temporary:
            results_path = Path(temporary) / "hpo"
            trial_results = Path(temporary) / "trial-results"
            study = _Study(_RuntimeTrial(number=0))
            builder_kwargs: dict[str, object] = {}

            def fake_builder(*args: object, **kwargs: object) -> Config:
                """Return the minimal configured trial used by this regression."""

                del args
                builder_kwargs.update(kwargs)
                return Config(
                    model={"name": "cnn"},
                    training={"task": "continual"},
                    hpo={"use_ensemble_accuracy": False},
                )

            def fake_main(*args: object, **kwargs: object) -> dict[str, object]:
                """Return staged continual metrics without running training."""

                del args, kwargs
                trial_results.mkdir(parents=True, exist_ok=True)
                return {
                    "history": {},
                    "evaluations": {
                        "validation_continual_metrics": metrics,
                    },
                    "results_path": str(trial_results),
                }

            with patch("optuna.create_study", return_value=study), \
                    patch("common.hpo._build_trial_config", side_effect=fake_builder), \
                    patch("common.hpo.main", side_effect=fake_main), \
                    patch("common.hpo.tf.keras.backend.clear_session"), \
                    patch("common.hpo.gc.collect"):
                returned = run_hpo(
                    "continual",
                    "cnn",
                    dataset_name="mnist",
                    n_trials=1,
                    epochs=1,
                    results_path=str(results_path),
                    class_num=3,
                    task_size=1,
                    objective_metrics=(
                        "average_forgetting",
                        "backward_transfer",
                    ),
                )

            self.assertIs(returned, study)
            self.assertEqual(builder_kwargs["class_num"], 3)
            self.assertEqual(builder_kwargs["task_size"], 1)
            np.testing.assert_allclose(study.value, [0.10, -0.05])

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
            self.assertEqual(SEARCH_SPACE_VERSION, 7)
            self.assertEqual(original["search_space_version"], 7)
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
                snapshot_network_name="raw",
                class_num=4,
                class_order=None,
                task_groups=None,
                task_size=2,
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
                    model={"name": "dit_classifier"},
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
                    snapshot_network_name="raw",
                    class_num=4,
                    task_size=2,
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
            self.assertEqual(builder_kwargs["snapshot_network_name"], "raw")

    def test_umbrella_classifier_uses_prefixed_bounded_space(self) -> None:
        """Resolve one umbrella family with sealed bounds and KD controls."""

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=19),
        )
        configs: list[Config] = []
        space = {
            "model_family": ["unet_classifier"],
            "learning_rate": {
                "low": 2e-4,
                "high": 3e-4,
                "log": True,
            },
            "batch_size": [32],
            "optimizer": ["adam"],
            "clipnorm": [1.],
            "timesteps": [250],
            "test_steps": [10],
            "snapshot_network_name": ["ema"],
            "continual_strategy_multiclass": ["generative_replay"],
            "clf_distil_scope_generative_replay": ["replay_only"],
            "use_noise_distillation": [True],
            "noise_distil_loss_coef": {
                "low": 0.1,
                "high": 0.2,
                "log": True,
            },
            "wrapper_name": ["diffusion_classifier"],
            "clf_distil_type": ["soft"],
            "clf_distil_temperature": {
                "low": 2.,
                "high": 2.1,
                "log": True,
            },
            "replay_budget_mode": ["fixed_total"],
            "replay_old_examples": [100],
            "replay_current_examples": [100],
            "replay_selection": ["all"],
        }

        def objective(trial: optuna.trial.Trial) -> float:
            """Build, but do not train, one bounded umbrella candidate."""
            configs.append(_build_trial_config(
                trial,
                "continual",
                "diffusion_classifier",
                "mnist",
                epochs=1,
                seed=19,
                results_path="results/hpo",
                use_distillation=True,
                class_num=4,
                task_size=2,
                max_train_samples=64,
                max_val_samples=32,
                search_space_overrides=space,
            ))
            return 0.

        study.optimize(objective, n_trials=1)
        config = configs[0]
        trial = study.trials[0]

        self.assertEqual(config.hpo["study_model"], "diffusion_classifier")
        self.assertEqual(config.hpo["model_family"], "unet_classifier")
        self.assertEqual(config.model.name, "unet_classifier")
        self.assertEqual(config.dataset.max_train_samples, 64)
        self.assertEqual(config.dataset.max_val_samples, 32)
        self.assertGreaterEqual(config.optimizer.initial_learning_rate, 2e-4)
        self.assertLessEqual(config.optimizer.initial_learning_rate, 3e-4)
        self.assertEqual(config.optimizer.clipnorm, 1.)
        self.assertTrue(config.continually_learn.use_generative_replay)
        self.assertEqual(config.continually_learn.replay_budget_mode,
                         "fixed_total")
        self.assertEqual(config.continually_learn.snapshot_network_name, "ema")
        self.assertEqual(config.model.wrapper_kwargs["clf_distil_type"], "soft")
        self.assertGreaterEqual(
            config.model.wrapper_kwargs["clf_distil_temperature"], 2.
        )
        self.assertLessEqual(
            config.model.wrapper_kwargs["clf_distil_temperature"], 2.1
        )
        self.assertEqual(
            config.model.wrapper_kwargs["clf_distil_scope"], "replay_only"
        )
        self.assertGreater(
            config.model.wrapper_kwargs["noise_distil_loss_coef"], 0.
        )
        self.assertEqual(trial.params["model_family"], "unet_classifier")
        self.assertIn(
            "unet_classifier.snapshot_network_name",
            trial.params,
        )
        self.assertIn(
            "unet_classifier.clf_distil_temperature",
            trial.params,
        )
        self.assertIn(
            "unet_classifier.clf_distil_scope_generative_replay",
            trial.params,
        )
        self.assertTrue(all(
            name == "model_family" or name.startswith("unet_classifier.")
            for name in trial.params
        ))
        self.assertEqual(config.hpo["params"], trial.params)

    def test_run_hpo_forwards_tpe_startup_and_selects_best_trial(self) -> None:
        """Use real Optuna storage around a mocked, ordered objective."""

        with tempfile.TemporaryDirectory() as temporary:
            current: dict[str, Config] = {}
            scores = [0.2, 0.8, 0.4]

            def fake_builder(
                trial: optuna.trial.Trial,
                *args: object,
                **kwargs: object,
            ) -> Config:
                """Return the smallest config required by the objective."""
                del args, kwargs
                config = Config(
                    model={
                        "name": "dit_classifier",
                        "wrapper_kwargs": {"test_network_name": "raw"},
                    },
                    training={"task": "continual"},
                    hpo={
                        "trial_number": trial.number,
                        "use_ensemble_accuracy": False,
                    },
                )
                current["config"] = config
                return config

            def fake_main(
                config: Config,
                **kwargs: object,
            ) -> dict[str, object]:
                """Emit a deterministic validation score for each trial."""
                del kwargs
                trial_number = config.hpo["trial_number"]
                result_path = Path(temporary) / f"run-{trial_number}"
                result_path.mkdir()
                return {
                    "history": {},
                    "evaluations": {
                        "validation_continual_metrics": {
                            "final_average_accuracy": scores[trial_number],
                        },
                    },
                    "results_path": str(result_path),
                }

            original_sampler = optuna.samplers.TPESampler
            original_create_study = optuna.create_study

            def create_in_memory_study(**kwargs: object) -> object:
                """Keep this plumbing check independent of Windows DB locks."""
                kwargs.pop("storage", None)
                return original_create_study(**kwargs)

            with patch(
                "optuna.samplers.TPESampler",
                wraps=original_sampler,
            ) as sampler_factory, patch(
                "optuna.create_study",
                side_effect=create_in_memory_study,
            ), patch(
                "common.hpo._build_trial_config",
                side_effect=fake_builder,
            ), patch(
                "common.hpo.main",
                side_effect=fake_main,
            ), patch(
                "common.hpo.save_config",
            ), patch(
                "common.hpo.load_config",
                side_effect=lambda path: current["config"],
            ), patch(
                "common.hpo.tf.keras.backend.clear_session",
            ), patch("common.hpo.gc.collect"):
                study = run_hpo(
                    "continual",
                    "dit_classifier",
                    dataset_name="mnist",
                    n_trials=3,
                    epochs=1,
                    seed=23,
                    results_path=str(Path(temporary) / "hpo"),
                    class_num=4,
                    task_size=2,
                    n_startup_trials=1,
                )

            self.assertEqual(sampler_factory.call_args.kwargs["seed"], 23)
            self.assertEqual(
                sampler_factory.call_args.kwargs["n_startup_trials"], 1
            )
            self.assertEqual(len(study.trials), 3)
            self.assertTrue(all(
                trial.state == optuna.trial.TrialState.COMPLETE
                for trial in study.trials
            ))
            self.assertEqual(study.best_trial.number, 1)
            self.assertAlmostEqual(study.best_value, 0.8)
            self.assertEqual(
                study.user_attrs["study_spec"]["n_startup_trials"], 1
            )


    def test_grid_specific_architecture_spaces_are_optuna_static(self) -> None:
        """Run odd and even patch grids in one persistent search contract.

        Args:
            None.

        Returns:
            None: Both conditional trials complete without dynamic-space errors.
        """
        study = optuna.create_study(direction="minimize")
        study.enqueue_trial({"patch_size": 2})
        study.enqueue_trial({"patch_size": 4})

        def objective(trial: optuna.trial.Trial) -> float:
            """Build one classifier route for the queued patch grid."""
            patch_size = trial.suggest_categorical("patch_size", [2, 4])
            _suggest_joint(
                trial,
                "dit_classifier",
                {"patch_size": patch_size, "dim": 32},
                {},
                image_size=28,
            )
            return 0.

        study.optimize(objective, n_trials=2)
        self.assertTrue(all(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        ))
        self.assertIn("classifier_architecture_grid2", study.trials[0].params)
        self.assertIn("classifier_architecture_flat", study.trials[1].params)

    def test_v2_timestep_spaces_are_optuna_static(self) -> None:
        """Run two diffusion horizons in one persistent V2 study."""
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.RandomSampler(seed=7),
        )
        study.enqueue_trial({
            "timesteps": 250,
            "wrapper_name": "diffusion_classifier_v2",
        })
        study.enqueue_trial({
            "timesteps": 500,
            "wrapper_name": "diffusion_classifier_v2",
        })
        configs: list[Config] = []

        def objective(trial: optuna.trial.Trial) -> float:
            """Build a complete V2 route for the queued diffusion horizon."""
            configs.append(_build_trial_config(
                trial,
                "joint",
                "dit_classifier",
                "mnist",
                epochs=1,
                seed=7,
                results_path="results/hpo",
            ))
            return 0.

        study.optimize(objective, n_trials=2)
        self.assertTrue(all(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        ))
        self.assertIn(
            "clf_train_noisified_max_timesteps_t250",
            study.trials[0].params,
        )
        self.assertIn(
            "clf_train_noisified_max_timesteps_t500",
            study.trials[1].params,
        )
        self.assertIn(
            "clf_train_noisified_max_timesteps",
            study.trials[0].user_attrs,
        )
        self.assertTrue(all(
            config.model.wrapper_kwargs["mask_by_nulls"] is False
            for config in configs
        ))


# Select the test action required by this condition.
if __name__ == "__main__":
    unittest.main()
