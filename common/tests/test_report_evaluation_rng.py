"""Check final-report corruption pairing, RNG isolation, and EMA restoration."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import tensorflow as tf

from common.config import Config
from common.random import SeedStream
from common.train import (
    _fit_control_callbacks, _report_evaluation_random_streams,
    _resolve_training_options, report,
)
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


def _tiny_wrapper(wrapper_type):
    network = DiTClassifier(
        num_classes=2, use_cfg=True, timesteps=4, image_size=4, channels=1,
        patch_size=2, dim=4, depth=1, mha_num_heads=1,
        vit_block_mlp_ratio=1.0, clf_mha_num_heads=1,
        clf_vit_block_mlp_ratio=1.0,
        feature_aggregation_ids_dict={1: (-1,)}, clf_connection_ids_dict={-1: (-1,)},
    )
    model = wrapper_type(network=network, use_ema=True, test_network_name="ema",
                         scheduler_name="linear", test_steps=2, seed=43)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


class ReportEvaluationRngTests(unittest.TestCase):
    def test_traced_draws_repeat_and_restore_state_even_after_failure(self):
        stream = SeedStream(43)
        model = SimpleNamespace(_random_streams={"noise": stream})

        @tf.function
        def draw():
            return tf.random.stateless_normal((3,), seed=stream.next_seed(43))

        draw()  # Cache the graph before the report temporarily changes state.
        before = stream.state.numpy().copy()
        with _report_evaluation_random_streams(model, 17):
            first = draw().numpy()
        np.testing.assert_array_equal(stream.state.numpy(), before)
        draw()
        advanced = stream.state.numpy().copy()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with _report_evaluation_random_streams(model, 17):
                np.testing.assert_array_equal(draw().numpy(), first)
                raise RuntimeError("injected")
        np.testing.assert_array_equal(stream.state.numpy(), advanced)
        with _report_evaluation_random_streams(model, 18):
            self.assertFalse(np.array_equal(draw().numpy(), first))
        self.assertEqual(draw.experimental_get_tracing_count(), 1)

    def test_unseeded_reporting_preserves_advancing_behavior(self):
        stream = SeedStream(43)
        model = SimpleNamespace(_random_streams={"noise": stream})
        before = stream.state.numpy().copy()
        with _report_evaluation_random_streams(model, None):
            stream.next_seed()
        self.assertFalse(np.array_equal(before, stream.state.numpy()))

    def test_real_v1_v2_reports_pair_ema_and_raw_after_training_draws(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.reshape(tf.linspace(-1.0, 1.0, 48), (3, 4, 4, 1)),
            tf.constant([0, 1, 0], dtype=tf.int32),
        )).batch(2)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        dataset = dataset.with_options(options)
        for wrapper_type in (DiffusionClassifier, DiffusionClassifierV2):
            with self.subTest(wrapper=wrapper_type.__name__):
                model = _tiny_wrapper(wrapper_type)
                kwargs = dict(model=model, valset=dataset, run_trainset_eval=False,
                              show_history_plot=False, show_final_images=False,
                              save_final_images=False, verbose=0, seed=17)
                before = {name: stream.state.numpy().copy()
                          for name, stream in model._random_streams.items()}
                first = report(**kwargs)
                for name, stream in model._random_streams.items():
                    np.testing.assert_array_equal(stream.state.numpy(), before[name])
                self.assertEqual(first["valset_ema_eval"]["noise_loss"],
                                 first["valset_network_eval"]["noise_loss"])
                # Simulate additional training's corruption draws without changing weights.
                for stream in model._random_streams.values():
                    for _ in range(7):
                        stream.next_seed()
                repeated = report(**kwargs)
                self.assertEqual(first, repeated)

    def test_early_stopping_and_saved_weights_keep_paired_raw_ema_snapshot(self):
        model = _tiny_wrapper(DiffusionClassifier)
        raw = model.network.weights[0]
        ema = model.ema_network.weights[0]
        raw.assign(tf.ones_like(raw) * 2)
        ema.assign(tf.ones_like(ema) * 3)
        config = Config()
        config.training.patience = 1
        config.training.reduce_lr_patience = 1
        config.training.monitor = "val_classifier_accuracy"
        config.training.monitor_mode = "max"
        stop = _fit_control_callbacks(_resolve_training_options(config, None, {}), object())[0]
        stop.set_model(model)
        stop.on_train_begin()
        stop.on_epoch_end(0, {"val_classifier_accuracy": 0.8})
        raw.assign(tf.ones_like(raw) * 7)
        ema.assign(tf.ones_like(ema) * 9)
        model.optimizer.iterations.assign(4)
        stop.on_epoch_end(1, {"val_classifier_accuracy": 0.7})
        stop.on_train_end()
        np.testing.assert_array_equal(raw.numpy(), np.ones(raw.shape) * 2)
        np.testing.assert_array_equal(ema.numpy(), np.ones(ema.shape) * 3)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 4)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.weights.h5")
            model.save_weights(path)
            raw.assign(tf.zeros_like(raw))
            ema.assign(tf.zeros_like(ema))
            model.load_weights(path)
        np.testing.assert_array_equal(raw.numpy(), np.ones(raw.shape) * 2)
        np.testing.assert_array_equal(ema.numpy(), np.ones(ema.shape) * 3)


if __name__ == "__main__":
    unittest.main()
