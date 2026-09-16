"""Architecture display belongs to orchestration, not diffusion model state."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.learner import _run_continual_tasks
from common.model import get_model
from common.tests.test_wrapper_verified_repairs import _network
from diffusion import DiffusionClassifier, DiffusionModel, DiffusionTransformer


class _ReachedTraining(RuntimeError):
    """Stop after the real continual expansion boundary, before optimization."""


def _generator() -> DiffusionModel:
    """Construct a small dynamic wrapper with independently expandable depth."""
    network = DiffusionTransformer(
        num_classes=None, use_cfg=True, timesteps=4, image_size=2, channels=1,
        patch_size=1, dim=4, depth=1, mha_num_heads=1,
        vit_block_mlp_ratio=1., seed=37,
    )
    model = DiffusionModel(network, use_ema=False, test_network_name="raw",
                           test_steps=2, scheduler_name="linear", seed=37)
    model.compile(optimizer="adam", loss="mse", run_eagerly=True)
    return model


def _loader(indices: list[int], **kwargs: object) -> tuple:
    """Return independent tiny image splits for a real continual boundary."""
    del kwargs
    labels = np.repeat(np.asarray(indices, dtype="int32"), 2)
    images = np.zeros((len(labels), 4, 4, 1), dtype="float32")
    return images, labels, images.copy(), labels.copy(), images.copy(), labels.copy()


class NetworkSummaryOwnershipTests(unittest.TestCase):
    """Exercise actual growth while recording only architecture display calls."""

    def tearDown(self) -> None:
        """Release independent Keras state after each focused check."""
        tf.keras.backend.clear_session()

    def test_direct_wrapper_growth_never_summarizes(self) -> None:
        """Class replacement and in-place depth growth remain display-free."""
        model = _generator()
        with patch.object(tf.keras.Model, "summary") as summary:
            model._check_new_labels(y=np.array([0, 1]), verbose=False)
            growth = model._add_depths("vision_transformer_block")
            model._check_new_labels(y=np.array([0, 1, 2]), verbose=False)
        self.assertEqual(len(model.seen_classes), 3)
        self.assertEqual(growth["network"]["added"], 1)
        summary.assert_not_called()

    def test_wrapper_configuration_does_not_own_summary_preference(self) -> None:
        """Current configs omit display state; old saved configs still restore."""
        model = _generator()
        self.assertNotIn("show_network_summary", inspect.signature(DiffusionModel).parameters)
        self.assertFalse(hasattr(model, "show_network_summary"))
        config = model.get_config()
        self.assertNotIn("show_network_summary", config)
        config["show_network_summary"] = True
        with patch.object(tf.keras.Model, "summary") as summary:
            restored = DiffusionModel.from_config(config)
        self.assertFalse(hasattr(restored, "show_network_summary"))
        self.assertNotIn("show_network_summary", restored.get_config())
        summary.assert_not_called()

    def test_factory_controls_display_without_injecting_wrapper_state(self) -> None:
        """Construction summaries keep their API setting outside the model config."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.object(tf.keras.Model, "summary") as summary:
                model = get_model(
                    model_name="diffusion_transformer", dataset_name="mnist",
                    image_shape=(2, 2, 1), class_num=2, show_network_summary=enabled,
                    model_kwargs={"timesteps": 4, "patch_size": 1, "dim": 4,
                                  "depth": 1, "mha_num_heads": 1,
                                  "vit_block_mlp_ratio": 1.},
                    wrapper_kwargs={"use_ema": False, "test_network_name": "raw",
                                    "test_steps": 2, "scheduler_name": "linear"},
                    seed=37,
                )
                self.assertIsInstance(model, DiffusionModel)
                self.assertEqual(summary.call_count, int(enabled))
                self.assertFalse(hasattr(model, "show_network_summary"))
                self.assertNotIn("show_network_summary", model.get_config())

    def test_continual_boundary_summarizes_only_actual_enabled_expansion(self) -> None:
        """The learner displays the expanded network once, independently of verbose."""
        for enabled, already_known in ((True, False), (False, False),
                                       (None, False), (True, True)):
            with self.subTest(enabled=enabled, already_known=already_known):
                model = DiffusionClassifier(
                    network=_network(classes=None, distil=False), use_ema=False,
                    test_network_name="raw", scheduler_name="linear", test_steps=2,
                    seed=37,
                )
                model.compile(optimizer="adam", loss="mse", run_eagerly=True)
                if already_known:
                    model._check_new_labels(y=np.array([0, 1]), verbose=False)
                displayed = []

                def record_summary(network: tf.keras.Model, *args: object, **kwargs: object) -> None:
                    """Retain the actual network that was displayed after growth."""
                    del args, kwargs
                    displayed.append(network)

                with patch.object(tf.keras.Model, "summary", autospec=True,
                                  side_effect=record_summary), patch(
                    "common.train.train_model", side_effect=_ReachedTraining
                ) as train:
                    with self.assertRaises(_ReachedTraining):
                        _run_continual_tasks(
                            class_num=4, task_size=2, load_dataset_fn=_loader,
                            load_dataset_fn_kwargs={"preprocess": "fixed-min-max"},
                            generative_model=model, use_generative_model_classifier=True,
                            use_generative_replay=False, epochs=1, batch_size=2,
                            callback_patience=0, plot_results=False, verbose=False,
                            show_network_summary=enabled, seed=37,
                        )
                train.assert_called_once()
                self.assertEqual(len(model.seen_classes), 2)
                expected = int(enabled is not False and not already_known)
                self.assertEqual(len(displayed), expected)
                if expected:
                    self.assertIs(displayed[0], model.network)
                self.assertFalse(hasattr(model, "show_network_summary"))
                self.assertNotIn("show_network_summary", model.get_config())
