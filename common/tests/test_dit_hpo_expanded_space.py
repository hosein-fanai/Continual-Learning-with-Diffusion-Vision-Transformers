"""Configuration and Optuna distribution checks for opt-in expanded DiT search."""

from __future__ import annotations

import unittest

import optuna

from common.hpo import _build_trial_config
from common.tests.test_hpo import _SuggestionTrial


class ExpandedDitSearchTests(unittest.TestCase):
    """Expanded categorical choices preserve native model and study semantics."""

    def _config(self, overrides: dict[str, object], trial: object | None = None) -> object:
        """Build a generation config without loading data or constructing a network."""

        return _build_trial_config(
            trial if trial is not None else _SuggestionTrial(), 
            "generation", "diffusion_transformer", "cifar10", 1, 
            results_path="unused", search_space_overrides=overrides, 
            seed=17
        )

    def test_default_configuration_keeps_legacy_dimensions(self) -> None:
        """Omitting new overrides keeps the existing paired width/head template."""

        config = self._config({})
        self.assertEqual(config.model.kwargs["dim"], 32)
        self.assertEqual(config.model.kwargs["mha_num_heads"], 4)
        self.assertEqual(config.model.kwargs["depth"], 2)
        self.assertTrue(config.model.kwargs["use_cfg"])
        self.assertEqual(config.model.wrapper_kwargs["p_uncond"], 0.05)
        self.assertEqual(config.model.wrapper_kwargs["test_cfg_scale"], 4.)
        self.assertEqual(config.reporting.final_images_cfg_scale, 3.)
        self.assertIn("capacity", config.hpo["params"])
        for parameter in ("dim", "mha_num_heads", "mha_key_dim", "use_cfg", "time_freq_dim", "conds_merger_type", "final_activation_func"):
            self.assertNotIn(parameter, config.hpo["params"])

    def test_expanded_settings_reach_raw_model_and_reporting(self) -> None:
        """Requested extremes and CFG-off behavior survive typed config creation."""

        choices = {
            "use_cfg": [False], "dim": [256], "depth": [10], 
            "mha_num_heads": [6], "mha_key_dim": ["dim"], "ln_mlp_ratio": [4], 
            "final_activation_func": ["tanh"], 
            "patches_pos_embed_type": ["1d_sincos"], 
            "patches_pos_merger_type": ["concat"], "conds_merger_type": ["concat"], 
            "time_freq_dim": [1], "time_embed_trainable": [True], "time_mlp_ratio": [4], 
            "label_embed_type": ["1d_sincos"], "label_freq_dim": [8], 
            "label_mlp_ratio": [2], "use_refiner_cnn": [True]
        }
        config = self._config(choices)
        for parameter, values in choices.items():
            expected = 256 if parameter == "mha_key_dim" else values[0]
            self.assertEqual(config.model.kwargs[parameter], expected)
        self.assertNotIn("capacity", config.hpo["params"])
        self.assertNotIn("p_uncond", config.hpo["params"])
        self.assertEqual(config.model.wrapper_kwargs["p_uncond"], 0.)
        self.assertEqual(config.model.wrapper_kwargs["test_cfg_scale"], 1.)
        self.assertEqual(config.reporting.final_images_cfg_scale, 1.)

    def test_requested_width_head_depth_choices_are_accepted(self) -> None:
        """Nondivisible head counts remain valid under the model's projection API."""

        for dimension in (16, 32, 64, 128, 256):
            for heads in (4, 6, 8):
                with self.subTest(dimension=dimension, heads=heads):
                    config = self._config({"dim": [dimension], "mha_num_heads": [heads]})
                    self.assertEqual(config.model.kwargs["dim"], dimension)
                    self.assertEqual(config.model.kwargs["mha_num_heads"], heads)
        for depth in range(3, 11):
            with self.subTest(depth=depth):
                self.assertEqual(self._config({"depth": [depth]}).model.kwargs["depth"], depth)

    def test_final_activation_choices_are_forwarded(self) -> None:
        """Both requested output activations reach the raw DiT constructor."""

        for activation in ("linear", "tanh"):
            with self.subTest(activation=activation):
                config = self._config({"final_activation_func": [activation]})
                self.assertEqual(config.model.kwargs["final_activation_func"], activation)
                self.assertEqual(config.hpo["params"]["final_activation_func"], activation)

    def test_training_loss_choices_keep_a_fixed_validation_objective(self) -> None:
        """MSE and MAE training compile with the same MSE validation contract."""

        for loss in ("mse", "mae"):
            with self.subTest(loss=loss):
                config = self._config({"loss_function": [loss]})
                self.assertEqual(config.model.loss_function, loss)
                self.assertEqual(config.model.kwargs["compile_args"], {"evaluation_loss": "mse"})
                self.assertEqual(config.training.monitor, "val_noise_loss")
                self.assertEqual(config.hpo["params"]["loss_function"], loss)
        legacy = self._config({})
        self.assertEqual(legacy.model.loss_function, "mse")
        self.assertNotIn("compile_args", legacy.model.kwargs)

    def test_extended_diffusion_horizons_keep_report_steps(self) -> None:
        """Longer noising horizons preserve the existing 50-step reporting budget."""

        for timesteps in (500, 1000, 2500, 5000):
            with self.subTest(timesteps=timesteps):
                config = self._config({"timesteps": [timesteps]})
                self.assertEqual(config.model.kwargs["timesteps"], timesteps)
                self.assertEqual(config.model.wrapper_kwargs["test_steps"], 50)
                self.assertEqual(config.reporting.final_images_steps, 50)

    def test_none_frequency_excludes_inactive_mlp_draws(self) -> None:
        """No frequency projection means no MLP, including an explicit ratio override."""

        config = self._config({
            "time_freq_dim": [None], "time_mlp_ratio": [4], 
            "label_freq_dim": [None], "label_mlp_ratio": [2]
        })
        for condition in ("time", "label"):
            parameter = condition + "_mlp_ratio"
            self.assertIsNone(config.model.kwargs[parameter])
            self.assertNotIn(parameter, config.hpo["params"])

    def test_frequency_widths_and_active_mlp_choices(self) -> None:
        """Both embedding types retain every requested active frequency/MLP choice."""

        for frequency in (1, 2, 4, 8):
            for ratio in (None, 1, 2, 4):
                with self.subTest(frequency=frequency, ratio=ratio):
                    config = self._config({
                        "time_freq_dim": [frequency], "time_mlp_ratio": [ratio], 
                        "label_freq_dim": [frequency], "label_mlp_ratio": [ratio], 
                        "label_embed_type": ["new_weight"], "time_embed_trainable": [False], 
                        "ln_mlp_ratio": [ratio]
                    })
                    self.assertEqual(config.model.kwargs["ln_mlp_ratio"], ratio)
                    for condition in ("time", "label"):
                        self.assertEqual(config.model.kwargs[condition + "_freq_dim"], frequency)
                        self.assertEqual(config.hpo["params"][condition + "_mlp_ratio"], ratio)

    def test_u_shape_keeps_fixed_topology_and_base_key_dimension(self) -> None:
        """The sampled plain depth never overwrites U-DiT's nine-stage graph."""

        config = self._config({
            "dit_architecture_grid4": ["u_skip"], "dim": [16], "mha_num_heads": [6], 
            "mha_key_dim": ["dim"], "depth": [10]
        })
        self.assertEqual(config.model.kwargs["depth"], 9)
        self.assertNotIn("depth", config.hpo["params"])
        self.assertEqual(config.model.kwargs["mha_key_dim"], 16)
        self.assertEqual(config.model.kwargs["vit_block_mlp_output_dims"][3], 32)

    def test_stable_distributions_across_dimensions_and_condition_branches(self) -> None:
        """One real Optuna study accepts changing widths with a symbolic key choice."""

        overrides = {
            "dim": [16, 256], "mha_num_heads": [4, 6, 8], "mha_key_dim": [None, "dim"], 
            "use_cfg": [True, False], "time_freq_dim": [None, 1, 8], 
            "time_mlp_ratio": [None, 1, 2, 4], "dit_architecture_grid4": ["plain"]
        }
        study = optuna.create_study()
        expected = [
            {"dim": 16, "mha_key_dim": "dim", "use_cfg": False, "time_freq_dim": None}, 
            {"dim": 256, "mha_key_dim": "dim", "use_cfg": True, "time_freq_dim": 1}, 
            {"dim": 16, "mha_key_dim": None, "use_cfg": False, "time_freq_dim": 8}
        ]
        for parameters in expected:
            study.enqueue_trial(parameters)
        for parameters in expected:
            trial = study.ask()
            config = self._config(overrides, trial=trial)
            expected_key = parameters["dim"] if parameters["mha_key_dim"] == "dim" else None
            self.assertEqual(config.model.kwargs["mha_key_dim"], expected_key)
            self.assertEqual(list(trial.distributions["mha_key_dim"].choices), [None, "dim"])
            self.assertEqual("p_uncond" in trial.params, parameters["use_cfg"])
            self.assertEqual("time_mlp_ratio" in trial.params, parameters["time_freq_dim"] is not None)
            study.tell(trial, 0.)
        self.assertEqual(len(study.trials), 3)

    def test_independent_capacity_rejects_conflicting_template(self) -> None:
        """Two overlapping width/head selection mechanisms cannot silently compete."""

        with self.assertRaisesRegex(ValueError, "either capacity"):
            self._config({"capacity": ["32x4"], "dim": [16]})


# Run only when this focused regression module is invoked directly.
if __name__ == "__main__":
    unittest.main()
