"""Round-trip and failure-safety tests for :mod:`common.recovery`."""

from __future__ import annotations

import json
import random
import tempfile
import unittest

from pathlib import Path

import numpy as np
import tensorflow as tf

from common.recovery import (
    SCHEMA_VERSION,
    TaskCheckpoint,
    capture_rng_state,
    find_latest_task_checkpoint,
    fingerprint_state,
    load_task_checkpoint,
    restore_replay_buffer,
    restore_rng_state,
    save_task_checkpoint,
)
from common.replay_buffer import ReplayBuffer
from diffusion import (
    DiTClassifier,
    DiffusionClassifier,
    DiffusionClassifierV2,
    UNetClassifier,
)


def _make_dynamic_diffusion_classifier(seed: int) -> DiffusionClassifier:
    """Build a tiny dynamic raw/EMA classifier with deferred distillation.

    Args:
        seed (int): Wrapper and TensorFlow initialization seed.

    Returns:
        DiffusionClassifier: Compiled dynamic classifier wrapper.
    """

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    network = DiTClassifier(
        num_classes=None,
        use_cfg=True,
        timesteps=4,
        image_size=4,
        channels=1,
        patch_size=2,
        dim=4,
        depth=1,
        mha_num_heads=1,
        vit_block_mlp_ratio=1.0,
        clf_mha_num_heads=1,
        clf_vit_block_mlp_ratio=1.0,
        feature_aggregation_ids_dict={1: (-1,)},
        clf_connection_ids_dict={-1: (-1,)},
        clf_distil_token_type="new_weight",
    )
    wrapper = DiffusionClassifier(
        network=network,
        use_ema=True,
        test_network_name="ema",
        scheduler_name="linear",
        test_steps=2,
        p_uncond=0.0,
        mask_by_nulls=False,
        defer_teacher=True,
        clf_distil_loss_coef=1.0,
        seed=seed,
    )
    wrapper.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        run_eagerly=True,
    )
    return wrapper


class RecoveryTests(unittest.TestCase):
    """Exercise committed discovery, state, replay, and TF optimizer recovery."""

    def test_task_checkpoint_state_exposes_recovery_cursor(self) -> None:
        """The integration mapping must combine state, schedule, and cursor."""

        checkpoint = TaskCheckpoint(
            task_dir=Path("task-0000"),
            completed_task_index=0,
            next_task_index=1,
            class_order=(4, 2),
            task_groups=((4,), (2,)),
            experiment_state={"accuracy": 0.75},
            rng_state={"schema_version": SCHEMA_VERSION},
            replay_state=None,
            fingerprint="test-fingerprint",
        )

        self.assertEqual(checkpoint.state, {
            "accuracy": 0.75,
            "completed_task_index": 0,
            "next_task_index": 1,
            "class_order": [4, 2],
            "task_groups": [[4], [2]],
            "rng_state": {"schema_version": SCHEMA_VERSION},
            "fingerprint": "test-fingerprint",
        })

    def test_literal_recovery_type_mapping_round_trips(self) -> None:
        """Keep user mappings distinct from the recovery serializer's tags."""

        literal = {
            "__recovery_type__": "path",
            "value": "literal-not-a-path",
        }
        with tempfile.TemporaryDirectory() as temporary:
            save_task_checkpoint(
                temporary,
                completed_task_index=0,
                state={
                    "class_order": [0, 1],
                    "task_groups": [[0], [1]],
                    "literal": literal,
                },
            )
            restored = load_task_checkpoint(temporary)

        self.assertEqual(restored.experiment_state["literal"], literal)
        self.assertNotEqual(fingerprint_state(literal), fingerprint_state(Path(
            "literal-not-a-path"
        )))

    @staticmethod
    def _create_optimizer_slots(
        optimizer: tf.keras.optimizers.Optimizer,
        variables: list[tf.Variable],
    ) -> None:
        """Create slots through the API available in TF 2.10 or later.

        Args:
            optimizer (tf.keras.optimizers.Optimizer): Test input named optimizer.
            variables (list[tf.Variable]): Test input named variables.

        Returns:
            None: Result produced by the test helper.
        """

        # Select the test action required by this condition.
        if hasattr(optimizer, "_create_all_weights"):
            optimizer._create_all_weights(variables)
        # Select the test action required by this condition.
        elif hasattr(optimizer, "build"):
            optimizer.build(variables)
        # Handle the complementary test case.
        else:
            raise AssertionError("Optimizer exposes no slot-registration API.")

    @staticmethod
    def _optimizer_step(
        variable: tf.Variable,
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> float:
        """Apply one deterministic scalar gradient step and return its loss.

        Args:
            variable (tf.Variable): Test input named variable.
            optimizer (tf.keras.optimizers.Optimizer): Test input named optimizer.

        Returns:
            float: Result produced by the test helper.
        """

        with tf.GradientTape() as tape:
            loss = tf.square(variable - 3.0)
        gradient = tape.gradient(loss, variable)
        optimizer.apply_gradients([(gradient, variable)])
        return float(loss.numpy())

    def test_tensorflow_model_and_optimizer_round_trip(self) -> None:
        """Weights, optimizer slots, and the next update survive recovery.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        with tempfile.TemporaryDirectory() as temporary:
            source_variable = tf.Variable(1.0, name="weight")
            source_optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
            self._optimizer_step(source_variable, source_optimizer)

            save_task_checkpoint(
                temporary,
                completed_task_index=0,
                state={
                    "class_order": [7, 3],
                    "task_groups": [[7], [3]],
                    "accuracy_matrix": np.asarray([[0.75, np.nan]]),
                    "fingerprint": "probe-run",
                },
                trackables={
                    "model_weight": source_variable,
                    "optimizer": source_optimizer,
                },
            )

            restored_variable = tf.Variable(-4.0, name="weight")
            restored_optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
            self._create_optimizer_slots(restored_optimizer, [restored_variable])
            restored = load_task_checkpoint(
                temporary,
                trackables={
                    "model_weight": restored_variable,
                    "optimizer": restored_optimizer,
                },
                expected_class_order=[7, 3],
                expected_task_groups=[[7], [3]],
                expected_fingerprint="probe-run",
            )

            self.assertEqual(restored.completed_task_index, 0)
            self.assertEqual(restored.next_task_index, 1)
            self.assertEqual(restored.class_order, (7, 3))
            self.assertEqual(restored.state["class_order"], [7, 3])
            self.assertEqual(restored.state["next_task_index"], 1)
            self.assertTrue(np.isnan(
                restored.experiment_state["accuracy_matrix"][0, 1]
            ))
            self.assertEqual(
                float(restored_variable.numpy()),
                float(source_variable.numpy()),
            )
            self.assertEqual(
                int(restored_optimizer.iterations.numpy()),
                int(source_optimizer.iterations.numpy()),
            )
            for source, target in zip(
                source_optimizer.variables(),
                restored_optimizer.variables(),
            ):
                np.testing.assert_array_equal(source.numpy(), target.numpy())

            source_loss = self._optimizer_step(source_variable, source_optimizer)
            target_loss = self._optimizer_step(
                restored_variable,
                restored_optimizer,
            )
            self.assertEqual(source_loss, target_loss)
            np.testing.assert_array_equal(
                source_variable.numpy(),
                restored_variable.numpy(),
            )

    def test_raw_ema_teacher_and_optimizer_restore_strictly(self) -> None:
        """Dynamic raw/EMA/teacher topology consumes a strict checkpoint.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        with tempfile.TemporaryDirectory() as temporary:
            source = _make_dynamic_diffusion_classifier(41)
            labels = tf.constant([0, 1], dtype=tf.uint8)
            images = tf.reshape(
                tf.linspace(-1.0, 1.0, 32),
                (2, 4, 4, 1),
            )
            source._check_new_labels(y=labels, verbose=False)
            source._add_depths({
                "classifier": "vision_transformer_block"
            })
            source.train_step((images, labels))
            source_teacher = source.snapshot_teacher_network("ema")
            source.set_teacher_network(source_teacher)
            source_trackables = {
                "classifier": source.network.classifier,
                "replay_model": source,
                "replay_optimizer": source.optimizer,
                "teacher": source_teacher,
            }
            save_task_checkpoint(
                temporary,
                completed_task_index=0,
                state={
                    "class_order": [0, 1],
                    "task_groups": [[0], [1]],
                },
                trackables=source_trackables,
            )

            target = _make_dynamic_diffusion_classifier(41)
            target._check_new_labels(y=labels, verbose=False)
            target._add_depths({
                "classifier": "vision_transformer_block"
            })
            target_teacher = target.snapshot_teacher_network("ema")
            target.set_teacher_network(target_teacher)
            target._register_optimizer_variables()
            target_trackables = {
                "classifier": target.network.classifier,
                "replay_model": target,
                "replay_optimizer": target.optimizer,
                "teacher": target_teacher,
            }
            loaded = load_task_checkpoint(
                temporary,
                trackables=target_trackables,
                assert_consumed=True,
            )
            self.assertIsNotNone(loaded.restore_status)

            for source_model, target_model in (
                (source.network, target.network),
                (source.ema_network, target.ema_network),
                (source_teacher, target_teacher),
            ):
                for expected, actual in zip(
                    source_model.get_weights(),
                    target_model.get_weights(),
                ):
                    np.testing.assert_array_equal(expected, actual)
            for expected, actual in zip(
                source.optimizer.variables(),
                target.optimizer.variables(),
            ):
                np.testing.assert_array_equal(
                    expected.numpy(),
                    actual.numpy(),
                )

    def test_skip_connected_unet_teacher_is_checkpoint_safe(self) -> None:
        """Integer-keyed U-Net routes stay outside TensorFlow checkpoints."""

        tf.keras.backend.clear_session()
        network = UNetClassifier(
            num_classes=2,
            use_cfg=True,
            timesteps=4,
            image_size=4,
            channels=1,
            widths=(2,),
            block_depth=1,
            bottleneck_width=3,
            bottleneck_depth=1,
            image_embedding_dim=2,
            time_embedding_dim=3,
            label_embedding_dim=2,
            feature_aggregation_ids_dict={1: (-1,)},
            build=True,
        )
        wrapper = DiffusionClassifierV2(
            network=network,
            use_ema=True,
            test_network_name="ema",
            test_steps=2,
            defer_teacher=True,
            clf_distil_loss_coef=1.0,
            seed=47,
        )
        teacher = wrapper.snapshot_teacher_network("ema")
        wrapper.set_teacher_network(teacher)

        for routed_network in (
            wrapper.network,
            wrapper.ema_network,
            teacher,
        ):
            self.assertTrue(routed_network.connection_ids_dict)
            self.assertTrue(all(
                isinstance(key, int)
                for key in routed_network.connection_ids_dict
            ))
            self.assertIs(type(routed_network.connection_ids_dict), dict)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = tf.train.Checkpoint(
                replay_model=wrapper,
                teacher=teacher,
            ).write(str(Path(temporary) / "ckpt"))
            self.assertTrue(Path(checkpoint_path + ".index").is_file())

    def test_dynamic_dual_heads_save_with_unique_hdf5_weight_names(self) -> None:
        """Expanded classifier and distillation heads remain HDF5-safe."""

        dit_wrapper = _make_dynamic_diffusion_classifier(43)
        dit_wrapper._check_new_labels(
            y=tf.constant([0, 1], dtype=tf.uint8),
            verbose=False,
        )
        dit_network = dit_wrapper.network

        unet_network = UNetClassifier(
            num_classes=None,
            use_cfg=True,
            timesteps=4,
            image_size=4,
            channels=1,
            widths=(2,),
            block_depth=1,
            bottleneck_width=3,
            bottleneck_depth=1,
            image_embedding_dim=2,
            time_embedding_dim=3,
            label_embedding_dim=2,
            feature_aggregation_ids_dict={1: (-1,)},
            classifier_only_distil_token=True,
            build=True,
        )
        unet_network.add_class()
        unet_network.add_class()

        inputs = (
            tf.zeros((2, 4, 4, 1), dtype=tf.float32),
            tf.zeros((2,), dtype=tf.int32),
            tf.constant([1, 2], dtype=tf.int32),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for network in (dit_network, unet_network):
                with self.subTest(network=type(network).__name__):
                    outputs = network(
                        inputs,
                        full_return=True,
                        training=False,
                    )
                    self.assertEqual(outputs["classes"].shape, (2, 2))
                    self.assertEqual(outputs["distil_classes"].shape, (2, 2))

                    classifier_names = {
                        weight.name for weight in network.classifier.weights
                    }
                    distillation_names = {
                        weight.name
                        for weight in network.distil_classifier.weights
                    }
                    all_names = classifier_names | distillation_names
                    self.assertEqual(
                        len(all_names),
                        len(network.classifier.weights)
                        + len(network.distil_classifier.weights),
                    )
                    self.assertTrue(classifier_names.isdisjoint(
                        distillation_names
                    ))
                    self.assertTrue(all(
                        "/" in name and "distil" not in name
                        for name in classifier_names
                    ))
                    self.assertTrue(all(
                        "/" in name and "distil" in name
                        for name in distillation_names
                    ))

                    weights_path = Path(temporary) / (
                        type(network).__name__ + ".weights.h5"
                    )
                    network.save_weights(weights_path)
                    self.assertGreater(weights_path.stat().st_size, 0)

    def test_replay_and_local_rng_round_trip(self) -> None:
        """Replay order/private RNG and local Python/NumPy RNGs continue exactly.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        with tempfile.TemporaryDirectory() as temporary:
            replay = ReplayBuffer(maxlen=4, seed=91)
            replay.extend([
                (np.asarray([1.0, 2.0], dtype=np.float32), np.uint8(0)),
                (np.asarray([3.0, 4.0], dtype=np.float32), np.uint8(1)),
                (np.asarray([5.0, 6.0], dtype=np.float32), np.uint8(2)),
            ])
            replay.sample_buffer(2)  # Advance the private generator.
            numpy_generator = np.random.default_rng(123)
            python_rng = random.Random(456)
            numpy_generator.integers(0, 100, size=3)
            python_rng.random()
            rng_state = capture_rng_state(
                numpy_generator=numpy_generator,
                python_rng=python_rng,
                include_globals=False,
            )

            save_task_checkpoint(
                temporary,
                completed_task_index=0,
                state={
                    "class_order": [0, 1, 2],
                    "task_groups": [[0], [1], [2]],
                },
                replay_buffer=replay,
                rng_state=rng_state,
            )
            expected_replay_sample = replay.sample_buffer(2)
            expected_numpy = numpy_generator.integers(0, 10_000, size=8)
            expected_python = [python_rng.random() for _ in range(8)]

            loaded = load_task_checkpoint(temporary)
            restored_replay = ReplayBuffer(maxlen=4, seed=0)
            restore_replay_buffer(restored_replay, loaded.replay_state)
            restored_rngs = restore_rng_state(
                loaded.rng_state,
                restore_globals=False,
            )

            actual_replay_sample = restored_replay.sample_buffer(2)
            for expected_pair, actual_pair in zip(
                expected_replay_sample,
                actual_replay_sample,
            ):
                np.testing.assert_array_equal(expected_pair[0], actual_pair[0])
                np.testing.assert_array_equal(expected_pair[1], actual_pair[1])
            np.testing.assert_array_equal(
                expected_numpy,
                restored_rngs["numpy_generator"].integers(
                    0,
                    10_000,
                    size=8,
                ),
            )
            self.assertEqual(
                expected_python,
                [restored_rngs["python_rng"].random() for _ in range(8)],
            )

    def test_latest_falls_back_from_corrupt_or_incomplete_task(self) -> None:
        """Only a valid COMMITTED directory can replace the recovery boundary.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_zero = save_task_checkpoint(
                root,
                completed_task_index=0,
                state={
                    "class_order": [0, 1, 2],
                    "task_groups": [[0], [1], [2]],
                },
            )

            incomplete = root / "task-0001"
            incomplete.mkdir()
            (incomplete / "state.json").write_text("{}", encoding="utf-8")
            (root / "latest.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "completed_task_index": 1,
                    "task_dir": "task-0001",
                    "state_sha256": "invalid",
                }),
                encoding="utf-8",
            )
            self.assertEqual(find_latest_task_checkpoint(root), task_zero)

            # A higher directory with a fake marker is also ignored.
            corrupt = root / "task-0002"
            corrupt.mkdir()
            (corrupt / "state.json").write_text("{}", encoding="utf-8")
            (corrupt / "COMMITTED").write_text("{}", encoding="utf-8")
            self.assertEqual(find_latest_task_checkpoint(root), task_zero)

    def test_explicit_tensorflow_generator_round_trip(self) -> None:
        """An explicit trackable TensorFlow generator resumes its next draw.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        source = tf.random.Generator.from_seed(808)
        source.normal((3,))
        state = capture_rng_state(
            include_globals=False,
            tensorflow_generator=source,
        )
        expected = source.normal((8,)).numpy()
        restored = restore_rng_state(
            state,
            restore_globals=False,
        )["tensorflow_generator"]
        np.testing.assert_array_equal(expected, restored.normal((8,)).numpy())

    def test_schedule_and_fingerprint_mismatches_are_rejected(self) -> None:
        """Resume cannot silently change the materialized experimental stream.

        Args:
            None.

        Returns:
            None: Result produced by the test helper.
        """

        with tempfile.TemporaryDirectory() as temporary:
            run_fingerprint = fingerprint_state({
                "seed": 17,
                "model": "dit_classifier",
            })
            save_task_checkpoint(
                temporary,
                completed_task_index=0,
                state={
                    "class_order": [4, 2, 9],
                    "task_groups": [[4], [2, 9]],
                    "fingerprint": run_fingerprint,
                },
            )

            with self.assertRaisesRegex(ValueError, "schedule differs"):
                load_task_checkpoint(
                    temporary,
                    expected_class_order=[2, 4, 9],
                    expected_task_groups=[[2], [4, 9]],
                )
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                load_task_checkpoint(
                    temporary,
                    expected_fingerprint="another-run",
                )
            with self.assertRaises(FileExistsError):
                save_task_checkpoint(
                    temporary,
                    completed_task_index=0,
                    state={
                        "class_order": [4, 2, 9],
                        "task_groups": [[4], [2, 9]],
                    },
                )


# Select the test action required by this condition.
if __name__ == "__main__":
    unittest.main()
