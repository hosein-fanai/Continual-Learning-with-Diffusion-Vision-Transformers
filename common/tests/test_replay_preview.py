"""Generated previews preserve replay data, label identity, and sampling state."""

import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from common.replay_preview import _sample_null_preview, show_generated_replay
from diffusion import DiffusionModel, DiffusionTransformer


class ReplayPreviewTests(unittest.TestCase):
    """Exercise real plotting and diffusion sampling on small fixed fixtures."""

    def tearDown(self) -> None:
        """Release figures and Keras state after each isolated check."""
        plt.close("all")
        tf.keras.backend.clear_session()

    def test_preview_uses_original_labels_and_local_random_selection(self) -> None:
        """Pick actual pool rows reproducibly without changing inputs or NumPy state."""
        images = np.linspace(-1., 1., 24).reshape(6, 2, 2, 1)
        labels = np.array([0, 0, 0, 1, 1, 1])
        before = images.copy()
        random_before = np.random.get_state()
        observed = []

        def collect() -> None:
            """Retain displayed arrays and titles before the figure is closed."""
            axes = plt.gcf().axes
            observed.append([(axis.get_title(), np.asarray(axis.images[0].get_array()).copy())
                             for axis in axes if axis.images])

        with patch.object(plt, "show", side_effect=collect):
            for _ in range(2):
                show_generated_replay(images, labels, {0: 4, 1: 9},
                                      data_min=-1., data_range=2., seed=17)
        self.assertEqual([title for title, _ in observed[0]], ["Class 4", "Class 9"])
        for index, ((_, first), (_, second)) in enumerate(zip(*observed)):
            np.testing.assert_array_equal(first, second)
            self.assertTrue(any(np.array_equal(first, (image[..., 0] + 1.) / 2.)
                                for image in images[labels == index]))
        np.testing.assert_array_equal(images, before)
        random_after = np.random.get_state()
        np.testing.assert_array_equal(random_before[1], random_after[1])
        self.assertEqual(random_before[2:], random_after[2:])
        self.assertEqual(plt.get_fignums(), [])

    def test_null_sample_preserves_model_and_bypasses_replay_capture(self) -> None:
        """A display-only null sample changes neither weights nor subsequent samples."""
        network = DiffusionTransformer(
            num_classes=None, use_cfg=True, timesteps=4,
            image_size=2, channels=1, patch_size=1,
            dim=4, depth=1, mha_num_heads=1, vit_block_mlp_ratio=1., seed=17,
        )
        model = DiffusionModel(network, use_ema=False, test_network_name="raw",
                               test_steps=2, test_eta=0.5, scheduler_name="linear",
                               seed=17)
        model._check_new_labels(y=np.array([0, 1]), verbose=False)
        before = model.get_weights()
        expected = model.sample(network_name="raw", labels=[1, 2], seed=31).numpy()
        model.set_weights(before)
        with patch.object(model, "sample", side_effect=AssertionError("replay capture called")):
            null = _sample_null_preview(model, seed=53, verbose=False)
        self.assertEqual(null.shape, (2, 2, 1))
        self.assertTrue(np.all(np.isfinite(null)))
        for actual, original in zip(model.get_weights(), before):
            np.testing.assert_array_equal(actual, original)
        actual = model.sample(network_name="raw", labels=[1, 2], seed=31).numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_null_state_is_restored_when_sampling_fails(self) -> None:
        """The finally path restores an advanced seed even when generation raises."""
        from common.random import SeedStream

        stream = SeedStream(17)
        before = stream.get_weights()
        from types import SimpleNamespace
        model = SimpleNamespace(test_network_name="raw", _flatten_layers=lambda: [stream])

        def fail(*args: object, **kwargs: object) -> None:
            """Advance the sampling stream before simulating a sampler failure."""
            stream.next_seed()
            raise RuntimeError("sample failure")

        with patch.object(DiffusionModel, "sample", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "sample failure"):
                _sample_null_preview(model, seed=53, verbose=False)
        for actual, original in zip(stream.get_weights(), before):
            np.testing.assert_array_equal(actual, original)

    def test_empty_pool_does_not_plot_or_sample(self) -> None:
        """An explicit zero replay budget adds no image-generation work."""
        with patch.object(plt, "show") as show, patch(
            "common.replay_preview._sample_null_preview"
        ) as sample:
            show_generated_replay(np.empty((0, 2, 2, 1)), np.empty(0, dtype=int), {})
        show.assert_not_called()
        sample.assert_not_called()
