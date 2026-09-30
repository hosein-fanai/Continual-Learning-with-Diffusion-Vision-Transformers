"""Final sampler-mode routing, artifact, configuration, and reproducibility checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config, load_config, save_config
from common.train import _report_final_visuals, _resolve_reporting_options, report
from common.utils import plot_images
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier


MODES = [
    {"name": "full_stochastic_scale3", "steps": 1000, "scale": 3.0, "eta": 1.0}, 
    {"name": "full_default_eta_scale3", "steps": 1000, "scale": 3.0, "eta": None}, 
    {"name": "default_scale3", "steps": None, "scale": 3.0, "eta": None}, 
    {"name": "default_scale4", "steps": None, "scale": 4.0, "eta": None}
]


class _Sampler:
    """Record sampler arguments and emit deterministic float32 image/trajectory fixtures."""
    swap_noise_image = False
    test_network_name = "raw"
    test_steps = 50
    test_eta = 0.0
    timesteps = 1000
    use_ema = True
    dtype_policy = tf.keras.mixed_precision.Policy("float32")
    preprocess_type = "standardize"
    preprocess = DiffusionClassifier.preprocess

    def __init__(self) -> None:
        """Initialize an empty list of captured sampling keyword mappings."""

        self.calls = []

    def get_network(self, name: str) -> SimpleNamespace:
        """Return a CFG-enabled fixture for a recognized network branch.

        Args:
            name (str): Exactly raw or ema; other values raise ValueError.
        Returns:
            SimpleNamespace: Network metadata with use_cfg=True; no weights are created."""

        # Keep the fixture strict about the network branches exercised by reporting.
        if name not in ("ema", "raw"):
            raise ValueError(name)
        return SimpleNamespace(use_cfg=True)

    def sample(self, **kwargs: object) -> np.ndarray | tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Capture one call and return distinguishable constant images.

        Args:
            **kwargs (object): Recorded sampler options. A truthy return_x_ts requests
                both single-frame trajectory lists alongside the final images.
        Returns:
            np.ndarray | tuple: Float32 (11, 2, 2, 1) images, optionally followed by two
            lists containing the same array; each call raises raw pixel intensity by 25.5."""

        self.calls.append(kwargs)
        images = np.full((11, 2, 2, 1), len(self.calls) * 25.5, np.float32)
        return (images, [images], [images]) if kwargs.get("return_x_ts") else images


class GenerationModeTests(unittest.TestCase):
    """Check sampler-mode resolution, artifact pairing, and report RNG preservation."""
    def test_four_modes_resolve_defaults_and_share_png_gif_draws(self) -> None:
        """Resolve four mode specifications and reuse each image batch for its PNG and GIF."""

        model = _Sampler()
        with tempfile.TemporaryDirectory() as directory, patch("common.train.DiffusionModel", _Sampler), \
                patch("common.train.plot_images") as plot, patch("common.train.create_gif") as gif:
            _report_final_visuals(
                model, "mnist", directory, False, True, True, 9, 9.0, 17, 
                final_generation_modes=MODES, final_generation_network_name="ema"
            )
            manifest = json.loads((Path(directory) / "final-generation-modes.json").read_text())
        self.assertEqual(len(model.calls), 4)
        self.assertEqual([call["steps"] for call in model.calls], [1000, 1000, 50, 50])
        self.assertEqual([call["eta"] for call in model.calls], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual([call["scale"] for call in model.calls], [3.0, 3.0, 3.0, 4.0])
        self.assertEqual(plot.call_count, 4)
        self.assertEqual(gif.call_count, 4)
        for index, call in enumerate(model.calls):
            self.assertTrue(call["add_null_label"])
            self.assertEqual(call["network_name"], "ema")
            self.assertTrue(plot.call_args_list[index].kwargs["has_null_label"])
            np.testing.assert_allclose(plot.call_args_list[index].args[0], (index + 1) / 10)
            np.testing.assert_allclose(gif.call_args_list[index].args[1][0], (index + 1) / 10)
            np.testing.assert_allclose(gif.call_args_list[index].args[2][0], (index + 1) / 10)
        self.assertEqual(len({row["image"] for row in manifest}), 4)
        self.assertEqual(len({row["seed"] for row in manifest}), 4)
        self.assertTrue(all(row["sample_count"] == 11 for row in manifest))

    def test_resolves_nonstandard_wrapper_defaults_instead_of_hardcoding(self) -> None:
        """Honor the trained wrapper default steps, eta, and selected network."""

        model = _Sampler()
        model.test_steps, model.test_eta = 17, 0.25
        with patch("common.train.DiffusionModel", _Sampler), patch("common.train.plot_images"):
            _report_final_visuals(model, "mnist", None, True, False, False, 9, 9.0, 17, 
                                  final_generation_modes=[MODES[2]])
        self.assertEqual(model.calls[0]["steps"], 17)
        self.assertEqual(model.calls[0]["eta"], 0.25)
        self.assertEqual(model.calls[0]["network_name"], "raw")

    def test_ema_request_records_raw_when_wrapper_has_no_ema(self) -> None:
        """Record both requested EMA and effective raw generation when EMA is disabled."""

        model = _Sampler()
        model.use_ema = False
        with tempfile.TemporaryDirectory() as directory, patch("common.train.DiffusionModel", _Sampler), \
                patch("common.train.plot_images"):
            _report_final_visuals(model, "mnist", directory, False, True, False, 2, 3.0, 17, 
                                  final_generation_modes=[MODES[2]], final_generation_network_name="ema")
            manifest = json.loads((Path(directory) / "final-generation-modes.json").read_text())
        self.assertEqual(model.calls[0]["network_name"], "raw")
        self.assertEqual(manifest[0]["network_name"], "raw")
        self.assertEqual(manifest[0]["requested_network_name"], "ema")

    def test_sampling_failure_restores_report_rng_state(self) -> None:
        """Restore the saved sampling counter even when the sampler raises."""

        model = _Sampler()
        state = tf.Variable([512, 0, 43], dtype=tf.int64)
        model._random_streams = {"sampling": SimpleNamespace(state=state)}
        before = state.numpy().copy()

        def failed_sample(**kwargs: object) -> None:
            """Advance the fixture RNG once and raise an injected sampling failure.

            Args:
                **kwargs (object): Sampler arguments accepted without interpretation.
            Returns:
                None: Never returns normally.
            Raises:
                RuntimeError: Always, after advancing the tracked PHILOX counter."""

            state.assign_add([256, 0, 0])
            raise RuntimeError("sampling interrupted")

        model.sample = failed_sample
        with patch("common.train.DiffusionModel", _Sampler):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                _report_final_visuals(model, "mnist", None, True, False, False, 2, 3.0, 17, 
                                      final_generation_modes=[MODES[2]])
        np.testing.assert_array_equal(state.numpy(), before)

    def test_mode_validation_finishes_before_any_sampling(self) -> None:
        """Reject malformed later modes before producing any earlier-mode samples."""

        invalid_modes = [
            [MODES[0], {"name": "bad", "steps": 1001, "scale": 3.0}], 
            [MODES[0], {"name": "../bad", "steps": 2, "scale": 3.0}], 
            [MODES[0], MODES[0]], 
            [{"name": "bad", "steps": True, "scale": 3.0}], 
            [{"name": "bad", "scale": float("nan")}], 
            [{"name": "bad", "scale": 3.0, "eta": -1}], 
            [{"name": "bad", "scale": 3.0, "unknown": 1}]
        ]
        for modes in invalid_modes:
            model = _Sampler()
            with self.subTest(modes=modes), patch("common.train.DiffusionModel", _Sampler):
                with self.assertRaises(ValueError):
                    _report_final_visuals(model, "mnist", None, True, False, False, 2, 3.0, 17, 
                                          final_generation_modes=modes)
            self.assertEqual(model.calls, [])

    def test_config_roundtrip_and_resolution_detach_modes(self) -> None:
        """Preserve mode settings through YAML and detach resolved mutable options."""

        config = Config()
        config.reporting.final_generation_modes = [dict(mode) for mode in MODES]
        config.reporting.final_generation_network_name = "ema"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            save_config(config, path)
            restored = load_config(path)
        self.assertEqual(restored.reporting.final_generation_modes, MODES)
        self.assertEqual(restored.reporting.final_generation_network_name, "ema")
        options = _resolve_reporting_options(restored, {})
        options["final_generation_modes"][0]["scale"] = 2.0
        self.assertEqual(restored.reporting.final_generation_modes[0]["scale"], 3.0)
        self.assertEqual(Config().reporting.final_generation_modes, [])

    def test_report_routes_modes_and_network_from_typed_config(self) -> None:
        """Forward typed report mode and network settings to final visual generation."""

        config = Config()
        config.reporting.final_generation_modes = MODES
        config.reporting.final_generation_network_name = "ema"
        config.reporting.run_trainset_eval = config.reporting.run_valset_eval = False
        config.reporting.save_history_plot = config.reporting.save_csv = False
        model = _Sampler()
        with tempfile.TemporaryDirectory() as directory, patch("common.train.DiffusionModel", _Sampler), \
                patch("common.train._report_final_visuals") as visuals:
            config.training.results_path = directory
            report(config=config, model=model, history={})
        self.assertEqual(visuals.call_args.kwargs["final_generation_modes"], MODES)
        self.assertEqual(visuals.call_args.kwargs["final_generation_network_name"], "ema")
        self.assertTrue(visuals.call_args.kwargs["final_generation_add_null_label"])

    def test_tiny_real_sampler_writes_four_modes_and_preserves_rng(self) -> None:
        """Verify real PNG/GIF files and deterministic repeated samples without RNG consumption."""

        from PIL import Image


        network = DiTClassifier(
            num_classes=2, use_cfg=True, timesteps=4, image_size=4, channels=1, 
            patch_size=2, dim=4, depth=1, mha_num_heads=1, 
            vit_block_mlp_ratio=1.0, clf_mha_num_heads=1, 
            clf_vit_block_mlp_ratio=1.0, 
            feature_aggregation_ids_dict={1: tuple([-1])}, clf_connection_ids_dict={-1: tuple([-1])}
        )
        model = DiffusionClassifier(network=network, use_ema=True, 
                                    scheduler_name="linear", test_steps=2, seed=43)
        modes = [{**mode, "steps": 3 if mode["steps"] is not None else None} for mode in MODES]
        before = model._random_streams["sampling"].state.numpy().copy()
        first_images, repeated_images = [], []

        def render(images: np.ndarray | tf.Tensor, **kwargs: object) -> None:
            """Capture generated pixels before invoking the real plotting helper.

            Args:
                images (np.ndarray | tf.Tensor): Image batch copied into first_images.
                **kwargs (object): Plot display/save options forwarded unchanged.
            Returns:
                None: The plotting helper displays or saves its image grid."""

            first_images.append(np.asarray(images).copy())
            return plot_images(images, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch("common.train.plot_images", side_effect=render):
            _report_final_visuals(model, "mnist", directory, False, True, True, 2, 3.0, 17, 
                                  final_generation_modes=modes, final_generation_network_name="ema")
            np.testing.assert_array_equal(model._random_streams["sampling"].state.numpy(), before)
            manifest = json.loads((Path(directory) / "final-generation-modes.json").read_text())
            self.assertEqual(len(manifest), 4)
            for row in manifest:
                self.assertEqual(row["sample_count"], 3)  # Two real classes plus the null condition.
                for key in ("image", "gif"):
                    with Image.open(Path(directory) / row[key]) as artifact:
                        artifact.verify()
        with patch("common.train.plot_images", side_effect=lambda images, **kw: repeated_images.append(np.asarray(images).copy())):
            _report_final_visuals(model, "mnist", None, True, False, False, 2, 3.0, 17, 
                                  final_generation_modes=modes, final_generation_network_name="ema")
        for expected, actual in zip(first_images, repeated_images):
            np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(model._random_streams["sampling"].state.numpy(), before)



# Run these regression cases when the module is executed directly.
if __name__ == "__main__":
    unittest.main()
