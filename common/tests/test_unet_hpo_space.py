"""Check native denoiser coverage, active dimensions, and worker eligibility."""

from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import optuna

from common.hpo import _TrialView, _build_trial_config, _suggest_unet, run_hpo
from common.tests.test_hpo import _SuggestionTrial
from common.unet_hpo import ARCHITECTURE_CHOICES, SEARCH_SPACE_OVERRIDES


class UnetDenoiserSpaceTests(unittest.TestCase):
    """The opt-in space is broad while legacy studies retain their recipe."""

    def _config(
        self, overrides: dict[str, object] | None = None, 
        trial: object | None = None, model_name: str = "unet"
    ) -> object:
        """Construct a typed study input without datasets, networks, or workers."""

        return _build_trial_config(
            trial if trial is not None else _SuggestionTrial(), 
            "generation", model_name, "cifar10", 1, 
            results_path="unused", search_space_overrides=overrides, 
            seed=17
        )

    def test_legacy_choices_remain_unchanged(self) -> None:
        """No opt-in marker means no new sampled names or constructor controls."""

        config = self._config()
        self.assertEqual(config.model.kwargs["widths"], (32, 64))
        self.assertEqual(config.model.kwargs["bottleneck_width"], 96)
        self.assertEqual(config.model.kwargs["block_depth"], 1)
        self.assertEqual(config.model.kwargs["time_embedding_dim"], 22)
        self.assertEqual(config.hpo["params"]["resampling"], "pool")
        for key in ("unet_architecture_space", "embedding_layout", "activation_func", "upsampling_interpolation"):
            self.assertNotIn(key, config.hpo["params"])

    def test_every_width_and_embedding_layout_preserves_geometry(self) -> None:
        """Hierarchy widths and conditioning budgets map exactly to native kwargs."""

        for widths in ARCHITECTURE_CHOICES["widths"]:
            for total in ARCHITECTURE_CHOICES["embedding_dim"]:
                for layout in ARCHITECTURE_CHOICES["embedding_layout"]:
                    with self.subTest(widths=widths, total=total, layout=layout):
                        config = self._config({
                            **SEARCH_SPACE_OVERRIDES, "widths": [widths], 
                            "embedding_dim": [total], "embedding_layout": [layout]
                        })
                        kwargs = config.model.kwargs
                        self.assertEqual(kwargs["widths"], tuple(int(value) for value in widths.split("-")))
                        self.assertEqual(sum(kwargs[key] for key in (
                            "image_embedding_dim", "time_embedding_dim", "label_embedding_dim"
                        )), total)
                        self.assertEqual(kwargs["image_embedding_dim"], kwargs["label_embedding_dim"])
                        # The time-rich allocation reserves exactly half for timesteps.
                        if layout == "time_rich":
                            self.assertEqual(kwargs["time_embedding_dim"], total // 2)
                        self.assertTrue(kwargs["use_skip_connections"])
                        self.assertEqual(kwargs["final_activation_func"], "linear")

    def test_native_resampling_is_independent_and_interpolation_conditional(self) -> None:
        """All nine scaler pairs are forwarded without inactive transpose interpolation."""

        for down in ARCHITECTURE_CHOICES["downsampling_method"]:
            for up in ARCHITECTURE_CHOICES["upsampling_method"]:
                with self.subTest(down=down, up=up):
                    config = self._config({
                        **SEARCH_SPACE_OVERRIDES, "downsampling_method": [down], 
                        "upsampling_method": [up], "upsampling_interpolation": ["nearest"]
                    })
                    self.assertEqual(config.model.kwargs["downsampling_method"], down)
                    self.assertEqual(config.model.kwargs["upsampling_method"], up)
                    self.assertEqual("upsampling_interpolation" in config.hpo["params"], up != "cnn_transpose")
                    # Only a resize-based upsampler consumes the interpolation choice.
                    if up != "cnn_transpose":
                        self.assertEqual(config.model.kwargs["upsampling_interpolation"], "nearest")

    def test_architectural_extremes_reach_the_public_model(self) -> None:
        """Larger depth, dropout, activation and bottleneck are real constructor settings."""

        config = self._config({
            **SEARCH_SPACE_OVERRIDES, "widths": ["64-128-256"], 
            "block_depth": [3], "bottleneck_mult": [2.0], "bottleneck_depth": [3], 
            "embedding_dim": [192], "embedding_layout": ["time_rich"], 
            "batch_norm": [True], "dropout": [0.2], "activation_func": ["gelu"]
        })
        for key, expected in {
            "block_depth": 3, "bottleneck_width": 512, "bottleneck_depth": 3, 
            "image_embedding_dim": 48, "time_embedding_dim": 96, "label_embedding_dim": 48, 
            "use_batch_norm": True, "dropout_rate": 0.2, "activation_func": "gelu"
        }.items():
            self.assertEqual(config.model.kwargs[key], expected)

    def test_notebook_recipe_has_common_denoising_units_and_no_auxiliary_loss(self) -> None:
        """Every candidate uses the same noising process and pure MSE objective."""

        before = deepcopy(SEARCH_SPACE_OVERRIDES)
        config = self._config(SEARCH_SPACE_OVERRIDES)
        self.assertEqual(config.model.loss_function, "mse")
        self.assertEqual(config.model.kwargs["timesteps"], 1000)
        self.assertTrue(config.model.kwargs["use_cfg"])
        self.assertEqual(config.model.wrapper_kwargs["scheduler_name"], "clipped_cosine")
        self.assertEqual(config.model.wrapper_kwargs["image_loss_coef"], 0.0)
        self.assertEqual(config.model.wrapper_kwargs["test_network_name"], "ema")
        self.assertEqual(config.training.monitor, "val_noise_loss")
        for key in ("test_cfg_scale", "test_eta", "test_steps", "weight_decay"):
            self.assertNotIn(key, config.hpo["params"])
        self.assertEqual(SEARCH_SPACE_OVERRIDES, before)

    def test_adamw_alone_samples_weight_decay(self) -> None:
        """Optimizer branches do not spend trials on inactive regularization."""

        for optimizer in ("adam", "adamw"):
            with self.subTest(optimizer=optimizer):
                config = self._config({**SEARCH_SPACE_OVERRIDES, "optimizer": [optimizer]})
                self.assertEqual("weight_decay" in config.hpo["params"], optimizer == "adamw")

    def test_optuna_distributions_remain_stable_across_conditional_branches(self) -> None:
        """One persistent study accepts scaler, embedding and optimizer transitions."""

        study = optuna.create_study()
        branches = [
            {"widths": "32-64", "upsampling_method": "interpolate", "embedding_layout": "balanced", "optimizer": "adam"}, 
            {"widths": "64-128-256", "upsampling_method": "cnn_transpose", "embedding_layout": "time_rich", "optimizer": "adamw"}, 
            {"widths": "32-64-128-256", "upsampling_method": "cnn_interpolate", "embedding_layout": "balanced", "optimizer": "adamw"}
        ]
        for branch in branches:
            study.enqueue_trial(branch)
        for branch in branches:
            trial = study.ask()
            self._config(SEARCH_SPACE_OVERRIDES, trial=trial)
            self.assertEqual("upsampling_interpolation" in trial.params, branch["upsampling_method"] != "cnn_transpose")
            self.assertEqual(list(trial.distributions["widths"].choices), ARCHITECTURE_CHOICES["widths"])
            study.tell(trial, 1.0)
        self.assertEqual(len(study.trials), len(branches))

    def test_conflicting_or_unknown_space_is_rejected(self) -> None:
        """A profile marker cannot silently combine incompatible sampling semantics."""

        with self.assertRaisesRegex(ValueError, "omit resampling"):
            self._config({**SEARCH_SPACE_OVERRIDES, "resampling": ["pool"]})
        with self.assertRaisesRegex(ValueError, "must use choices"):
            self._config({"unet_architecture_space": ["unknown"]})

    def test_denoiser_space_cannot_replace_a_classifier_architecture(self) -> None:
        """The opt-in dispatch rejects classifier use before sampling any settings."""

        trial = _SuggestionTrial()
        view = _TrialView(trial, overrides=SEARCH_SPACE_OVERRIDES)
        with self.assertRaisesRegex(ValueError, "ordinary unet denoiser"):
            _suggest_unet(view, classifier=True)
        self.assertEqual(trial.params, {})

    def test_isolated_and_pruned_cifar_unet_reaches_study_identity(self) -> None:
        """The public worker gate accepts the ordinary serialized U-Net recipe."""

        for dataset in ("CIFAR10", "CIFAR100"):
            with self.subTest(dataset=dataset), patch(
                "common.hpo._make_study_spec", side_effect=RuntimeError("validated worker gate")
            ) as seal:
                with self.assertRaisesRegex(RuntimeError, "validated worker gate"):
                    run_hpo(
                        "generation", "unet", dataset, n_trials=1, epochs=1, 
                        results_path="unused", concurrent_trials=1, worker_gpu_ids=[0], 
                        pruning={"n_startup_trials": 2, "n_min_trials": 2}, 
                        search_space_overrides=SEARCH_SPACE_OVERRIDES
                    )
                seal.assert_called_once()

    def test_worker_gate_still_rejects_unet_teachers_and_non_cifar(self) -> None:
        """Broadening a model family does not broaden teacher or dataset ownership."""

        for options in ({"teacher_network": object()}, {"dataset_name": "MNIST"}):
            with self.subTest(options=options), patch("common.hpo._make_study_spec") as seal:
                with self.assertRaisesRegex(ValueError, "Isolated workers"):
                    run_hpo(
                        "generation", "unet", n_trials=1, epochs=1, 
                        results_path="unused", worker_gpu_ids=[0], **options
                    )
                seal.assert_not_called()


# Execute this focused suite in a separate remote Python process.
if __name__ == "__main__":
    unittest.main()
