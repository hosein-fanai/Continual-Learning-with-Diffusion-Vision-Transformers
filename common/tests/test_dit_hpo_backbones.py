"""Follow-up DiT capacity coverage, stable sampling and native topology checks."""

from __future__ import annotations

import unittest

import optuna
import tensorflow as tf

from common.dit_hpo_backbones import (
    CAPACITY_CHOICES, FOLLOWUP_BRANCHES, MISSING_CAPACITY_CHOICES, build_followup_backbone
)
from common.hpo import _build_trial_config
from common.tests.test_hpo import _SuggestionTrial
from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer


class FollowupBackboneTests(unittest.TestCase):
    """The follow-up explores omitted capacities and new directed topologies."""

    def _config(self, overrides: dict, trial: object | None = None) -> object:
        """Resolve a standalone generation config with deterministic defaults."""

        return _build_trial_config(
            trial if trial is not None else _SuggestionTrial(), 
            "generation", "diffusion_transformer", "cifar10", 1, 
            results_path="unused", search_space_overrides=overrides, seed=19
        )

    def test_plain_branches_always_leave_original_capacity_domain(self) -> None:
        """Every plain route forces one missing axis while preserving other choices."""

        for axis, choices in MISSING_CAPACITY_CHOICES.items():
            for choice in choices:
                with self.subTest(axis=axis, choice=choice):
                    overrides = {
                        **CAPACITY_CHOICES, "dit_followup_branch": ["plain_missing"], 
                        "followup_missing_axis": [axis], "followup_missing_" + axis: [choice]
                    }
                    config = self._config(overrides)
                    kwargs = config.model.kwargs
                    expected = kwargs["dim"] if axis == "mha_key_dim" else choice
                    self.assertEqual(kwargs[axis], expected)
                    self.assertEqual(config.hpo["params"]["followup_missing_" + axis], choice)
                    self.assertNotIn(axis, config.hpo["params"])
                    self.assertTrue(
                        kwargs["dim"] == 256 or kwargs["depth"] >= 7
                        or kwargs["mha_num_heads"] in (6, 8) or kwargs["mha_key_dim"] is not None
                    )

    def test_all_native_branches_keep_requested_expanded_capacity(self) -> None:
        """Novel topology does not silently restore the small v10 restrictions."""

        for branch in FOLLOWUP_BRANCHES[1:]:
            with self.subTest(branch=branch):
                config = self._config({
                    "dit_followup_branch": [branch], "dim": [256], 
                    "depth": [10], "mha_num_heads": [8], "mha_key_dim": ["dim"], 
                    "batch_size": [128]
                })
                self.assertEqual(config.model.kwargs["dim"], 256)
                self.assertEqual(config.model.kwargs["mha_num_heads"], 8)
                self.assertEqual(config.model.kwargs["mha_key_dim"], 256)
                expected_depth = 9 if branch in ("u_skip", "u_cross") else 10
                self.assertEqual(config.model.kwargs["depth"], expected_depth)
                self.assertEqual(config.dataset.batch_size, 128)

    def test_directed_routes_preserve_stream_and_spatial_levels(self) -> None:
        """Routes are acyclic, and feature fusion retains the previous stage."""

        for branch in FOLLOWUP_BRANCHES:
            kwargs = build_followup_backbone(branch, 16, 6)
            for field in ("connection_ids_dict", "cross_attention_ids_dict"):
                for target, sources in kwargs.get(field, {}).items():
                    self.assertTrue(sources)
                    self.assertEqual(len(sources), len(set(sources)))
                    self.assertTrue(all(0 <= source < target <= kwargs["depth"] for source in sources))
                    # Feature routes must preserve the immediate predecessor.
                    if field == "connection_ids_dict":
                        self.assertIn(target - 1, sources)
            # Dense feature fusion includes every preceding transformer output.
            if branch == "feature_dense":
                self.assertEqual(kwargs["connection_ids_dict"][6], [1, 2, 3, 4, 5])
            # Dense attention excludes its current query stream from external sources.
            if branch == "cross_dense":
                self.assertEqual(kwargs["cross_attention_ids_dict"][6], [1, 2, 3, 4])
                self.assertEqual(kwargs["use_decoder_ids"], [3, 4, 5, 6])
            # U attention routes connect matching spatial resolutions.
            if branch == "u_cross":
                self.assertEqual(kwargs["cross_attention_ids_dict"], {7: [3], 9: [1]})

    def test_local_variants_are_inside_selected_vit_blocks(self) -> None:
        """Every kernel and placement selects the native within-block API."""

        for variant in ("depthwise", "separable", "expanded_pointwise"):
            for kernel in (3, 5, 7):
                for placement in ("every", "alternating", "late"):
                    kwargs = build_followup_backbone(
                        "local_hybrid", 16, 5, local_mixer_variant=variant, 
                        local_mixer_kernel_size=kernel, local_mixer_placement=placement
                    )
                    self.assertNotIn("local_mixer_ids", kwargs)
                    self.assertTrue(kwargs["vit_block_local_mixer_ids"])
                    mixer = kwargs["vit_block_local_mixer_kwargs"]
                    self.assertEqual(mixer["kernel_size"], kernel)
                    self.assertEqual(mixer["use_pointwise"], variant != "depthwise")
                    self.assertEqual(mixer["pointwise_dim_ratio"], 2 if variant == "expanded_pointwise" else 1)
                    self.assertNotIn("mlp_output_dim", mixer)

    def test_optuna_distributions_remain_stable_across_all_branches(self) -> None:
        """One persistent study accepts every conditional topology and missing axis."""

        study = optuna.create_study()
        cases = [
            {"dit_followup_branch": "plain_missing", "followup_missing_axis": axis}
            for axis in MISSING_CAPACITY_CHOICES
        ] + [{"dit_followup_branch": branch} for branch in FOLLOWUP_BRANCHES[1:]]
        overrides = {**CAPACITY_CHOICES, "dit_followup_branch": FOLLOWUP_BRANCHES}
        for params in cases:
            study.enqueue_trial(params)
        for params in cases:
            trial = study.ask()
            config = self._config(overrides, trial=trial)
            self.assertEqual(trial.params["dit_followup_branch"], params["dit_followup_branch"])
            self.assertEqual(trial.user_attrs["dit_architecture"], params["dit_followup_branch"])
            self.assertNotIn("dit_architecture_grid4", trial.params)
            self.assertTrue(config.model.kwargs["depth"] >= 3)
            study.tell(trial, 0.)
        self.assertEqual(len(study.trials), len(cases))

    def test_followup_does_not_change_legacy_draws(self) -> None:
        """Existing studies keep the old capacity and architecture distributions."""

        config = self._config({})
        self.assertEqual(config.model.kwargs["dim"], 32)
        self.assertEqual(config.model.kwargs["depth"], 2)
        self.assertEqual(config.hpo["params"]["capacity"], "32x4")
        self.assertEqual(config.hpo["params"]["dit_architecture_grid4"], "plain")
        self.assertNotIn("dit_followup_branch", config.hpo["params"])


class NativeFollowupTopologyTests(unittest.TestCase):
    """Tiny raw models prove native shape, forward and differentiation contracts."""

    def test_all_topologies_forward_and_backpropagate(self) -> None:
        """Every backbone returns image-shaped finite output and finite gradients."""

        cases = [(branch, "concat", "values") for branch in FOLLOWUP_BRANCHES]
        cases += [(branch, "add", "queries") for branch in (
            "feature_ladder", "feature_dense", "cross_ladder", "cross_dense", "u_cross"
        )]
        for branch, merge, plug in cases:
            with self.subTest(branch=branch, merge=merge, plug=plug):
                tf.keras.backend.clear_session()
                kwargs = build_followup_backbone(
                    branch, 8, 4, feature_merge=merge, cross_merge=merge, cross_plug_type=plug
                )
                model = DiffusionTransformer(
                    num_classes=10, use_cfg=True, timesteps=8, 
                    image_size=8, channels=3, patch_size=2, dim=8, 
                    patches_pos_merger_type="concat", mha_num_heads=2, 
                    vit_block_mlp_ratio=1., seed=23, **kwargs
                )
                inputs = (
                    tf.ones((1, 8, 8, 3)), tf.constant([2], tf.int32), tf.constant([1], tf.int32)
                )
                with tf.GradientTape() as tape:
                    output = model(inputs, training=True)
                    loss = tf.reduce_mean(tf.square(output - 1.))
                gradients = tape.gradient(loss, model.trainable_variables)
                self.assertEqual(tuple(output.shape), (1, 8, 8, 3))
                self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(output))))
                retained = [gradient for gradient in gradients if gradient is not None]
                self.assertTrue(retained)
                for gradient in retained:
                    values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
                    self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(values))))


# Run this focused regression module directly when requested.
if __name__ == "__main__":
    unittest.main()
