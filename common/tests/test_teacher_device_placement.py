"""Placement and numerical coverage for every frozen teacher inference protocol."""

from types import SimpleNamespace
import unittest

import numpy as np
import tensorflow as tf

from common.tests import test_clean_classifier_training as fixtures
from common.tests import test_dual_teachers as dual_fixtures
from diffusion import DiffusionClassifier, DiffusionClassifierV2, DiffusionModel


class TeacherDevicePlacementTests(unittest.TestCase):
    """Keep mapped teacher compute beside its weights without changing targets."""

    setUp = fixtures.CleanClassifierTrainingTests.setUp
    tearDown = fixtures.CleanClassifierTrainingTests.tearDown
    make_network = fixtures.CleanClassifierTrainingTests.make_network
    noise_network = staticmethod(dual_fixtures.DualTeacherTests.noise_network)

    @staticmethod
    def devices() -> list[str]:
        """Always check CPU ownership and exercise GPU ownership when available."""

        return ["/CPU:0"] + (["/GPU:0"] if tf.config.list_logical_devices("GPU") else [])

    def make_wrapper(self, wrapper_cls: type, teacher: tf.keras.Model, role: str, head: str) -> DiffusionModel:
        """Attach one independent role with only its relevant objective enabled."""

        options = dict(
            network=self.noise_network() if wrapper_cls is DiffusionModel else self.make_network(), 
            use_ema=False, preprocess_type=None, scheduler_name="clipped_cosine", 
            test_steps=4, p_uncond=0., noise_loss_coef=0., 
            noise_distil_loss_coef=1. if head == "noise" else 0., train_cfg_scale=1.5, 
            dual_teacher_scope="all", seed=811
        )
        options["teacher_network" if role == "previous" else role + "_teacher_network"] = teacher
        # Classifier wrappers add an independently selected classification objective.
        if wrapper_cls is not DiffusionModel:
            options.update(
                mask_by_nulls=False, mask_by_t_threshold=False, clf_loss_coef=0., 
                clf_distil_loss_coef=1. if head == "classifier" else 0., 
                clf_distil_type="soft", 
                clf_train_noisy_input_type="noisy" if wrapper_cls is DiffusionClassifierV2 else "clean", 
                clf_train_class_input_type="null_class_only", 
                clf_train_noisified_max_timesteps=-1, clf_test_noisified_max_timesteps=-1
            )
        model = wrapper_cls(**options)
        model.compile(optimizer=tf.keras.optimizers.SGD(.001), loss="mse", 
                      run_eagerly=False, jit_compile=False)
        # V2 preparation depends on the explicitly selected training/test phase.
        if wrapper_cls is DiffusionClassifierV2:
            phase = "discriminator" if head == "classifier" else "generator"
            model._switch_train_part(phase)
            model._test_part = phase
        return model

    @staticmethod
    def activate_noise_head(teacher: tf.keras.Model) -> None:
        """Give the native zero-initialized epsilon head a nontrivial reference."""

        head = teacher.unpatchifier.get_layer(teacher.name_prefix + "unpatchifier__ffn")
        values = tf.linspace(-.2, .2, int(np.prod(head.kernel.shape)))
        head.kernel.assign(tf.reshape(values, head.kernel.shape))

    def mapped(self, model: DiffusionModel) -> tuple:
        """Trace preparation under the CPU worker placement that caused the slowdown."""

        def prepare(images: tf.Tensor, labels: tf.Tensor) -> tuple:
            """Make the caller placement explicit within the traced map function."""

            with tf.device("/CPU:0"):
                return model.prep_inputs_map(images, labels)

        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        dataset = tf.data.Dataset.from_tensors((self.images, self.labels)).with_options(options)
        dataset = dataset.map(prepare, num_parallel_calls=1)
        return dataset._map_func.function.graph.as_graph_def(), next(iter(dataset))

    def assert_compute_device(self, graph: object, device: str) -> None:
        """Inspect heavyweight operations, including nested concrete functions."""

        nodes = list(graph.node)
        for function in graph.library.function:
            nodes.extend(function.node_def)
        compute = [node for node in nodes if node.op in {
            "Conv2D", "MatMul", "BatchMatMulV2", "Einsum"
        }]
        self.assertTrue(compute, "A placement test must execute actual teacher compute")
        expected = tf.DeviceSpec.from_string(device)
        for node in compute:
            actual = tf.DeviceSpec.from_string(node.device)
            self.assertEqual(actual.device_type, expected.device_type, node.name)
            self.assertEqual(actual.device_index, expected.device_index, node.name)

    def assert_unchanged(self, teacher: tf.keras.Model, before: list) -> None:
        """Require every frozen teacher variable to retain its exact value."""

        self.assertFalse(teacher.trainable)
        self.assertEqual(len(before), len(teacher.weights))
        for expected, actual in zip(before, teacher.get_weights()):
            np.testing.assert_array_equal(actual, expected)

    def test_native_classifier_roles_keep_weight_placement_and_detached_targets(self) -> None:
        """Exercise native predict_class for previous, shared-current and specialist roles."""

        for device in self.devices():
            for wrapper_cls in (DiffusionClassifier, DiffusionClassifierV2):
                for role in ("previous", "current", "classifier"):
                    with self.subTest(device=device, wrapper=wrapper_cls.__name__, role=role):
                        with tf.device(device):
                            teacher = self.make_network()
                        model = self.make_wrapper(wrapper_cls, teacher, role, "classifier")
                        before = teacher.get_weights()
                        graph, prepared = self.mapped(model)
                        self.assert_compute_device(graph, device)
                        # V2 discriminator batches have their own public tensor ordering.
                        if wrapper_cls is DiffusionClassifierV2:
                            times, images, labels = prepared[:3]
                            clean = prepared[4]
                        # V1 retains the general diffusion preparation prefix.
                        else:
                            images, times = model._classifier_inputs(prepared, training=True)
                            labels, clean = prepared[5], prepared[0]
                        with tf.device(device):
                            expected = teacher.predict_class(
                                (images, times, labels), max_encoder_num=None, training=False
                            )
                        for target in tf.nest.flatten(prepared[-1]):
                            np.testing.assert_allclose(target, expected, rtol=2e-5, atol=2e-6)
                        sources = [images] + [
                            weight.value for weight in teacher.weights
                            if tf.as_dtype(weight.dtype).is_floating or tf.as_dtype(weight.dtype).is_complex
                        ]
                        with tf.GradientTape() as tape:
                            tape.watch(sources)
                            target = model._predict_single_teacher_labels(
                                images, times, labels, clean_images=clean, teacher_network=teacher
                            )
                            objective = tf.reduce_sum(target[:, 0])
                        self.assertTrue(all(gradient is None for gradient in tape.gradient(objective, sources)))
                        self.assert_unchanged(teacher, before)

    def test_noise_roles_keep_weight_placement_and_guided_detached_targets(self) -> None:
        """Exercise native/callable epsilon teachers through the common mapped boundary."""

        for device in self.devices():
            for native in (True, False):
                for role in ("previous", "current", "noise"):
                    with self.subTest(device=device, native=native, role=role):
                        with tf.device(device):
                            teacher = self.noise_network() if native else tf.keras.Sequential([
                                tf.keras.Input((4, 4, 1)), tf.keras.layers.Conv2D(1, 1)
                            ])
                            # Nonzero native outputs make parity and guidance checks meaningful.
                            if native:
                                self.activate_noise_head(teacher)
                        model = self.make_wrapper(DiffusionModel, teacher, role, "noise")
                        # Ordinary callable noise teachers receive only the configured inputs.
                        if not native:
                            model.teacher_noise_input_type = "images"
                        before = teacher.get_weights()
                        graph, prepared = self.mapped(model)
                        self.assert_compute_device(graph, device)
                        images, times = prepared[3], prepared[2]
                        conditional, null = prepared[4], prepared[5]
                        with tf.device(device):
                            # Native epsilon teachers consume timestep and condition tensors.
                            if native:
                                expected_cond = teacher((images, times, conditional), training=False)
                                expected_null = teacher((images, times, null), training=False)
                                expected = expected_null + 1.5 * (expected_cond - expected_null)
                            # Image-only callable teachers have identical conditional/null outputs.
                            else:
                                expected = teacher(images, training=False)
                        for target in tf.nest.flatten(prepared[-2]):
                            np.testing.assert_allclose(target, expected, rtol=2e-5, atol=2e-6)
                        sources = [images] + [
                            weight.value for weight in teacher.weights
                            if tf.as_dtype(weight.dtype).is_floating or tf.as_dtype(weight.dtype).is_complex
                        ]
                        with tf.GradientTape() as tape:
                            tape.watch(sources)
                            target = model._predict_teacher_noise(
                                images, times, conditional, null, scale=1.5, teacher_network=teacher
                            )
                            objective = tf.reduce_sum(target)
                        self.assertTrue(all(gradient is None for gradient in tape.gradient(objective, sources)))
                        self.assert_unchanged(teacher, before)

    def test_fused_previous_teacher_keeps_both_cfg_passes_with_its_weights(self) -> None:
        """Cover the joint forward path that bypasses separate target predictors."""

        for device in self.devices():
            with self.subTest(device=device):
                with tf.device(device):
                    teacher = self.make_network()
                    self.activate_noise_head(teacher)
                model = self.make_wrapper(DiffusionClassifier, teacher, "previous", "classifier")
                model.noise_distil_loss_coef = 1.
                model.clf_train_noisy_input_type = "noisy"
                model.clf_train_class_input_type = "all_classes"
                model._refresh_loss_flags()
                before = teacher.get_weights()
                graph, prepared = self.mapped(model)
                self.assert_compute_device(graph, device)
                with tf.device(device):
                    conditional = teacher(
                        (prepared[3], prepared[2], prepared[4]), full_return=True, training=False
                    )
                    null = teacher(
                        (prepared[3], prepared[2], prepared[5]), full_return=True, training=False
                    )
                expected_noise = null["noises"] + 1.5 * (conditional["noises"] - null["noises"])
                np.testing.assert_allclose(prepared[-3], expected_noise, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(prepared[-1], conditional["classes"], rtol=2e-5, atol=2e-6)
                self.assert_unchanged(teacher, before)

    def test_v2_native_predict_noise_keeps_teacher_compute_with_its_weights(self) -> None:
        """Exercise the V2 direct forward protocol as well as mapped base dispatch."""

        for device in self.devices():
            with self.subTest(device=device):
                with tf.device(device):
                    teacher = self.make_network()
                    self.activate_noise_head(teacher)
                model = self.make_wrapper(DiffusionClassifierV2, teacher, "previous", "noise")
                times = tf.ones_like(self.labels)
                conditional, null = self.labels + 1, tf.zeros_like(self.labels)

                @tf.function
                def predict(images: tf.Tensor) -> tuple:
                    """Trace V2 native noise predictions inside a CPU caller."""

                    with tf.device("/CPU:0"):
                        return model.call_network(
                            images, times, conditional, null, scale=1.5, 
                            network_name="teacher", training=False
                        )[0]

                graph = predict.get_concrete_function(self.images).graph.as_graph_def()
                self.assert_compute_device(graph, device)
                actual_cond, actual_null = predict(self.images)
                with tf.device(device):
                    expected_cond = teacher.predict_noise((self.images, times, conditional), training=False)
                    expected_null = teacher.predict_noise((self.images, times, null), training=False)
                np.testing.assert_allclose(actual_cond, expected_cond, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(actual_null, expected_null, rtol=2e-5, atol=2e-6)

    def test_distributed_teacher_scope_preserves_callers_device(self) -> None:
        """Avoid reading a distributed variable's handle outside replica context."""

        strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])
        with strategy.scope():
            weight = tf.Variable(1.)
        teacher = SimpleNamespace(weights=[weight])
        with tf.device("/CPU:0"):
            with DiffusionModel._teacher_inference_scope(teacher):
                result = tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        self.assertEqual(tf.DeviceSpec.from_string(result.device).device_type, "CPU")

    def test_mixed_weight_devices_preserve_callers_device(self) -> None:
        """Do not select one device when the teacher deliberately spans devices."""

        # A single-device host cannot construct a genuinely mixed-device teacher.
        if not tf.config.list_logical_devices("GPU"):
            self.skipTest("Mixed CPU/GPU ownership needs an available GPU")
        with tf.device("/CPU:0"):
            cpu_weight = tf.Variable(1.)
        with tf.device("/GPU:0"):
            gpu_weight = tf.Variable(1.)
        teacher = SimpleNamespace(weights=[cpu_weight, gpu_weight])
        with tf.device("/CPU:0"):
            with DiffusionModel._teacher_inference_scope(teacher):
                result = tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        self.assertEqual(tf.DeviceSpec.from_string(result.device).device_type, "CPU")


# Keep focused execution available without starting tests during imports.
if __name__ == "__main__":
    unittest.main()