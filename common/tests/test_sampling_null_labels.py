"""Regress optional CFG null labels through real diffusion and VAE sampling.

Tiny transformer fixtures cover fixed and restored dynamic vocabularies, label
ordering, explicit overrides, swapped VAE routing, and positional argument order.
"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.wrapper.diffusion_model import DiffusionModel
from common.utils import plot_images


def _make_model(use_cfg: bool = True, dynamic: bool = False) -> DiffusionModel:
    """Build a tiny variational transformer supporting both sampling methods."""

    network = DiffusionTransformer(
        num_classes=None if dynamic else 2,
        use_cfg=use_cfg,
        timesteps=4,
        image_size=4,
        channels=1,
        patch_size=2,
        dim=4,
        depth=4,
        mha_num_heads=1,
        vit_block_mlp_ratio=1.0,
        vit_block_ids=[1, 4],
        cls_token_type="new_weight",
        cls_token_regularizer_ids=[None],
        reshaper_ids_dict={2: "flatten", 3: "unflatten"},
        reshaper_kwargs={"add_kl": True, "latent_dim_ratio": [1.0]},
        connection_ids_dict={},
    )
    return DiffusionModel(
        network=network,
        use_ema=False,
        test_network_name="raw",
        scheduler_name="linear",
        test_steps=2,
        test_eta=0.0,
        seen_classes={19: 1, 7: 0} if dynamic else {},
        seed=17,
    )


class SamplingNullLabelTests(unittest.TestCase):
    """Keep optional null labels consistent across both public sampling paths."""

    def tearDown(self) -> None:
        """Release Keras fixture state after each regression case."""

        tf.keras.backend.clear_session()
        super().tearDown()

    def _assert_sample_labels(
        self,
        model: DiffusionModel,
        method_name: str,
        expected: list[int],
        **kwargs: object,
    ) -> None:
        """Run real sampling and inspect the normalized labels it consumes."""

        prepared_labels: list[tf.Tensor] = []
        prepare = model._prepare_sampling_labels

        def capture_labels(
            network: DiffusionTransformer,
            labels: tf.Tensor | list[int],
            samples_per_label: int,
        ) -> tf.Tensor:
            """Record prepared labels while retaining real validation and expansion."""

            result = prepare(network, labels, samples_per_label)
            prepared_labels.append(result)
            return result

        with patch.object(model, "_prepare_sampling_labels", side_effect=capture_labels):
            images = getattr(model, method_name)(network_name="raw", **kwargs)
        self.assertEqual(len(prepared_labels), 1)
        np.testing.assert_array_equal(prepared_labels[0].numpy(), expected)
        self.assertEqual(images.shape, (len(expected), 4, 4, 1))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(images))))

    def test_defaults_and_optional_null_preserve_class_order(self) -> None:
        """Prepend null only under CFG, including restored dynamic class ordering."""

        # Dynamic transformer construction requires CFG; fixed vocabularies support either mode.
        for dynamic, use_cfg in ((False, False), (False, True), (True, True)):
            model = _make_model(use_cfg=use_cfg, dynamic=dynamic)
            targets = [1, 0] if dynamic else [0, 1]
            defaults = [value + int(use_cfg) for value in targets]
            with_null = [0] + defaults if use_cfg else defaults
            for method_name in ("sample", "sample_vae"):
                with self.subTest(dynamic=dynamic, cfg=use_cfg, method=method_name):
                    self._assert_sample_labels(model, method_name, defaults)
                    self._assert_sample_labels(
                        model,
                        method_name,
                        [value for value in with_null for _ in range(2)],
                        add_null_label=True,
                        samples_per_label=2,
                    )

    def test_explicit_labels_override_optional_null(self) -> None:
        """Keep explicit network IDs and their order unchanged by the flag."""

        for dynamic in (False, True):
            model = _make_model(dynamic=dynamic)
            for method_name in ("sample", "sample_vae"):
                with self.subTest(dynamic=dynamic, method=method_name):
                    self._assert_sample_labels(
                        model, method_name, [2, 1], labels=[2, 1], add_null_label=True,
                    )
                    self._assert_sample_labels(
                        model, method_name, [0, 2], labels=tf.constant([0, 2]),
                        add_null_label=True,
                    )

    def test_explicit_empty_labels_do_not_select_defaults(self) -> None:
        """Isolate empty-list precedence before unrelated zero-batch model execution."""

        model = _make_model()
        for method_name in ("sample", "sample_vae"):
            with self.subTest(method=method_name):
                with patch.object(
                    model, "_prepare_sampling_labels", side_effect=RuntimeError("labels captured"),
                ) as prepare:
                    with self.assertRaisesRegex(RuntimeError, "labels captured"):
                        getattr(model, method_name)(labels=[], add_null_label=True)
                self.assertEqual(prepare.call_args.args[1], [])

    def test_swap_forwards_null_flag_and_sample_count(self) -> None:
        """Swapped sampling reaches the real VAE decoder with both options intact."""

        model = _make_model()
        model.swap_noise_image = True
        with patch.object(model, "sample_vae", wraps=model.sample_vae) as sample_vae:
            self._assert_sample_labels(
                model, "sample", [0, 0, 1, 1, 2, 2],
                add_null_label=True, samples_per_label=2,
            )
        self.assertTrue(sample_vae.call_args.kwargs["add_null_label"])
        self.assertEqual(sample_vae.call_args.kwargs["samples_per_label"], 2)

    def test_null_option_follows_labels_in_positional_calls(self) -> None:
        """Accept the null flag directly after labels in both sampler signatures."""

        model = _make_model()
        images = model.sample(
            "raw", [1], False, 2, tf.zeros((2, 4, 4, 1)),
            2, 1.0, 0.0, False, False, 53, False,
        )
        self.assertEqual(images.shape, (2, 4, 4, 1))
        reshaper = model.network.layers_dicts[1][model.network.R]
        latent_width = int(reshaper.output_shape[1][-1])
        vae_images = model.sample_vae(
            "raw", [1], False, 2, tf.zeros((2, latent_width)), 53,
        )
        self.assertEqual(vae_images.shape, (2, 4, 4, 1))

    def test_plot_titles_and_positional_file_output(self) -> None:
        """Accept the null flag after column count in positional file-only calls."""

        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        images = np.zeros((2, 4, 4, 1), dtype=np.float32)
        cases = (
            (False, ["0", "1"]),
            (True, ["-1", "0"]),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (has_null_label, titles) in enumerate(cases):
                with self.subTest(has_null_label=has_null_label):
                    destination = Path(directory) / f"grid_{index}.png"
                    with patch.object(plt, "close", wraps=plt.close) as close:
                        with patch.object(plt, "show") as show:
                            plot_images(images, 1, 2, has_null_label, False, destination)
                    show.assert_not_called()
                    figure = close.call_args.args[0]
                    self.assertEqual([axis.get_title() for axis in figure.axes], titles)
                    self.assertGreater(destination.stat().st_size, 0)


# Support standalone execution and unittest discovery.
if __name__ == "__main__":
    unittest.main()
