"""Bounded runtime audit of the joint HPO profile's architecture interactions.

The route/condition matrix covers every pair with the fixed CNN stem and both
patch sizes. A separate width/head matrix covers all requested combinations,
including nondivisible six-head widths. These checks use synthetic CIFAR-size
inputs and ordinary forward/backward passes, not dataset training or HPO.
"""

from __future__ import annotations

from copy import deepcopy
import itertools
import unittest

import numpy as np
import tensorflow as tf

from common.config import Config
from common.hpo_profiles import build_joint_classifier_config
from diffusion.models.transformer.di_t_classifier import DiTClassifier


class _FirstTrial:
    """Select the first permitted value through the actual HPO adapter."""

    number = 0

    def __init__(self) -> None:
        """Initialize the deterministic suggestion recorder with an empty parameter mapping."""

        self.params = {}

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        """Record and return the first categorical choice.

        Args:
            name (str): Parameter name stored in params.
            choices (list[object]): Nonempty permitted values; no random draw is made.

        Returns:
            object: The first supplied value, also recorded under name."""

        self.params[name] = choices[0]
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, **kwargs: object) -> float:
        """Record the lower endpoint of a floating suggestion interval.

        Args:
            name (str): Parameter name stored in params.
            low (float): Inclusive lower bound used as the deterministic suggestion.
            high (float): Upper bound accepted for the Optuna-compatible signature.
            kwargs (object): Extra distribution options, ignored by this fixture.

        Returns:
            float: The selected endpoint, also recorded under name."""

        self.params[name] = low
        return low


class JointHpoArchitectureTests(unittest.TestCase):
    """Verify executable shapes, active routes, gradients and raw reconstruction."""

    def tearDown(self) -> None:
        """Clear Keras global model and naming state after the architecture case."""

        tf.keras.backend.clear_session()

    def network(self, route: str='last', condition: str | None='time_label', patch: int=4, dim: int=32, heads: int=6, depth: int=3, clf_depth: int=1, mlp: int=1, dropout: float=0.0, drop_path: float=0.0, classes: int=10, policy: str='float32') -> tuple[DiTClassifier, Config]:
        """Construct a real DiT from the production profile with the requested architecture.

        Args:
            route (str): "all" aggregates every encoder stage; other values select "last".
            condition (str | None): Classifier conditioning: time, label, time_label, or None.
            patch (int): Image patch side length for 32-by-32 RGB inputs.
            dim (int): Encoder and classifier hidden width.
            heads (int): Encoder attention head count; classifier heads remain profile controlled.
            depth (int): Encoder block count.
            clf_depth (int): Classifier block count.
            mlp (int): Classifier hidden projection width multiplier.
            dropout (float): Classifier Dense-head dropout probability.
            drop_path (float): Classifier stochastic-depth probability.
            classes (int): Ten selects CIFAR-10; other fixture values select CIFAR-100.
            policy (str): TensorFlow precision policy passed to the builder and network.

        Returns:
            tuple[DiTClassifier, Config]: Built network and resolved profile. Resets Keras state and seed to 1729."""

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(1729)
        overrides = {
            "feature_aggregation": ["all" if route == "all" else "last"], 
            "clf_cond_type": [condition], 
            "patch_size": [patch], "dim": [dim], "mha_num_heads": [heads], 
            "depth": [depth], "clf_depth": [clf_depth], 
            "classifier_mlp_ratio": [mlp], "classifier_dropout_rate": [dropout], 
            "clf_droppath_rate": [drop_path]
        }
        config = build_joint_classifier_config(
            _FirstTrial(), dataset_name="cifar10" if classes == 10 else "cifar100", 
            epochs=1, seed=1729, results_path="unused-architecture-audit", 
            dtype_policy=policy, search_space_overrides=overrides
        )
        network = DiTClassifier(**config.model.kwargs, seed=1729, dtype=policy)
        return network, config

    @staticmethod
    def inputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Create deterministic CIFAR-shaped network inputs.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Float32 images [2,32,32,3] in [-1,1),
                int32 timesteps [0,127], and int32 CFG-null conditions [0,0]."""

        images = tf.random.stateless_uniform((2, 32, 32, 3), (21, 34), -1.0, 1.0)
        return images, tf.constant([0, 127]), tf.constant([0, 0])

    @staticmethod
    def open_zero_initializers(network: DiTClassifier) -> None:
        """Expose structural gradient paths hidden by native zero-initialized gates.

        DiT intentionally starts with zero adaptive gates and a zero denoising
        head. Perturb only all-zero trainable tensors for connectivity checks;
        the ordinary-initialization checks run before this helper.
        """

        rng = np.random.default_rng(1729)
        for variable in network.trainable_variables:
            # Open only zero-initialized tensors so structural gradients can be measured.
            if not np.any(variable.numpy()):
                variable.assign(rng.normal(0.0, 0.02, variable.shape))

    def check_shapes_and_routes(self, network: DiTClassifier, config: Config, inputs: tuple[tf.Tensor, ...]) -> None:
        """Assert denoiser/classifier shapes, stage routing, token ownership and public API parity.

        Args:
            network (DiTClassifier): Built real network under test, evaluated without training state updates.
            config (Config): Resolved profile containing the expected classes and architecture.
            inputs (tuple[tf.Tensor, ...]): Float32 RGB [2,32,32,3], int32 times [2], and conditions [2].

        Returns:
            None: All expected output shapes, finite probabilities, and routing identities match."""

        result = network(inputs, full_return=True, training=False)
        classes = config.model.kwargs["num_classes"]
        self.assertEqual(tuple(result["noises"].shape), (2, 32, 32, 3))
        self.assertEqual(tuple(result["classes"].shape), (2, classes))
        self.assertTrue(np.isfinite(result["noises"].numpy()).all())
        self.assertTrue(np.isfinite(result["classes"].numpy()).all())
        np.testing.assert_allclose(result["classes"].numpy().sum(-1), 1.0, atol=1e-6)
        self.assertEqual(result["classes"].dtype, tf.float32)
        self.assertEqual(network.prepended_tokens_num, 0)
        self.assertEqual(network.clf_prepended_tokens_num, 1)
        self.assertIsNone(network.distil_classifier)
        self.assertIsNone(network.distil_token)
        self.assertIsNotNone(network.cls_token.token)
        tokens = (32 // network.patch_size) ** 2
        self.assertEqual(tuple(result["features_list"][0].shape), (2, tokens, network.dim))
        self.assertEqual(tuple(result["clf_features_list"][0].shape), 
                         (2, tokens + 1, network.dim))
        self.assertEqual(len(result["features_list"]), network.depth + 1)
        self.assertEqual(len(network.clf_layers_dicts), network.clf_depth + 1)
        self.assertEqual(result["clf_cond"] is None, network.clf_cond_type is None)
        self.assertEqual(network.clf_dim, network.dim)
        self.assertFalse(network.aggregate_from_noises)
        self.assertTrue(network.patchify_with_cnn)
        expected = list(range(network.depth + 1)) if network.clf_dim_forced else [network.depth]
        self.assertEqual(network.feature_aggregation_ids_dict[1], expected)
        for stage in network.layers_dicts:
            block = stage.get(network.VTB)
            # Only transformer stages own attention-head and stochastic-depth settings.
            if block is not None:
                self.assertEqual(block.num_heads, config.model.kwargs["mha_num_heads"])
                self.assertEqual(block.key_dim, network.dim // block.num_heads)
        self.assertEqual(sum(network.VTB in stage for stage in network.layers_dicts), 
                         network.depth)
        for stage in network.clf_layers_dicts:
            block = stage.get(network.VTB)
            # Only transformer stages own attention-head and stochastic-depth settings.
            if block is not None:
                self.assertEqual(block.num_heads, 4)
                self.assertEqual(block.droppath_rate, config.model.kwargs["clf_droppath_rate"])
        self.assertEqual(sum(network.VTB in stage for stage in network.clf_layers_dicts), 
                         network.clf_depth)
        dense = [layer for layer in network.classifier.layers
                 if isinstance(layer, tf.keras.layers.Dense)]
        self.assertEqual(len(dense), 2)
        self.assertGreater(network.classifier_mlp_ratio, 0)
        self.assertEqual(dense[0].units, network.dim * network.classifier_mlp_ratio)
        self.assertEqual(any(isinstance(layer, tf.keras.layers.Dropout)
                             for layer in network.classifier.layers), 
                         network.classifier_dropout_rate > 0)
        np.testing.assert_allclose(network.predict_class(inputs, training=False), 
                                   result["classes"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(network.predict_noise(inputs, training=False), 
                                   result["noises"], rtol=1e-5, atol=1e-6)

    def check_joint_gradients_and_image_dependence(self, network: DiTClassifier, inputs: tuple[tf.Tensor, ...]) -> None:
        """Check every parameter and image path using CE plus denoising MSE.

        Args:
            network (DiTClassifier): Built network; zero initializers are opened and one SGD step mutates weights.
            inputs (tuple[tf.Tensor, ...]): Float32 RGB [2,32,32,3], int32 times [2], and conditions [2].

        Returns:
            None: Parameter gradients are connected and finite, and classifier scores depend on image pixels."""

        self.open_zero_initializers(network)
        images, times, labels = inputs
        target_noise = tf.random.stateless_normal(tf.shape(images), (55, 89))
        targets = tf.constant([0, network.num_classes - 1])
        with tf.GradientTape() as tape:
            result = network((images, times, labels), training=True)
            loss = tf.reduce_mean(tf.square(tf.cast(result["noises"], tf.float32) - target_noise))
            loss += tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                targets, result["classes"]))
        gradients = tape.gradient(loss, network.trainable_variables)
        disconnected = [variable.path for variable, grad in
                        zip(network.trainable_variables, gradients) if grad is None]
        self.assertEqual(disconnected, [])
        for variable, grad in zip(network.trainable_variables, gradients):
            self.assertTrue(np.isfinite(tf.convert_to_tensor(grad).numpy()).all(), variable.path)
        token_grads = [grad for variable, grad in zip(network.trainable_variables, gradients)
                       if variable is network.cls_token.token]
        self.assertEqual(len(token_grads), 1)
        self.assertGreater(float(tf.reduce_max(tf.abs(token_grads[0]))), 0.0)
        tf.keras.optimizers.SGD(1e-5).apply_gradients(zip(gradients, network.trainable_variables))
        with tf.GradientTape() as tape:
            tape.watch(images)
            probabilities = network.predict_class((images, times, labels), training=False)
            score = tf.reduce_sum(tf.math.log(probabilities[:, 0]))
        image_grad = tape.gradient(score, images)
        self.assertIsNotNone(image_grad)
        self.assertTrue(np.isfinite(image_grad.numpy()).all())
        self.assertGreater(float(tf.reduce_max(tf.abs(image_grad))), 0.0)

    def test_route_condition_matrix_has_trainable_image_connected_cnn_classifier(self) -> None:
        """Cover eight route/condition pairs, both patches, and all dropout choices."""

        conditions = (None, "time", "label", "time_label")
        for route_index, route in enumerate(("last", "all")):
            for condition_index, condition in enumerate(conditions):
                case = route_index * 4 + condition_index
                options = dict(
                    route=route, condition=condition, 
                    patch=2 if case % 2 else 4, 
                    heads=(4, 6, 8)[case % 3], depth=2 + case % 6, 
                    clf_depth=1 + case % 5, 
                    mlp=(1, 2, 4)[case % 3], 
                    dropout=(0.0, 0.15, 0.25)[case % 3], 
                    drop_path=(0.0, 0.15, 0.25)[(case + 1) % 3], 
                    classes=100 if case % 2 else 10
                )
                with self.subTest(**options):
                    network, config = self.network(**options)
                    inputs = self.inputs()
                    self.check_shapes_and_routes(network, config, inputs)
                    self.check_joint_gradients_and_image_dependence(network, inputs)

    def test_every_requested_width_and_head_count_preserves_output_width(self) -> None:
        """Cover all 12 width/head pairs, including floor-sized six-head keys."""

        for case, (dim, heads) in enumerate(itertools.product((32, 64, 128, 256), (4, 6, 8))):
            with self.subTest(dim=dim, heads=heads):
                network, config = self.network(
                    dim=dim, heads=heads, route=("last", "all")[case % 2], 
                    patch=2 if case % 2 else 4, 
                    condition=(None, "time", "label", "time_label")[case % 4], 
                    depth=7 if dim == 256 else 3, clf_depth=5 if dim == 256 else 1, 
                    mlp=4 if dim == 256 else 1, classes=100
                )
                self.check_shapes_and_routes(network, config, self.inputs())

    def test_float32_graph_and_serialization_clone_preserve_each_route(self) -> None:
        """Cover four route/patch pairs with CNN stems and 0.15 regularization."""

        for route, patch in itertools.product(("last", "all"), (2, 4)):
            with self.subTest(route=route, patch=patch):
                network, config = self.network(
                    route=route, condition="time_label" if patch == 2 else None, 
                    patch=patch, policy="float32", classes=100, 
                    dropout=0.15, drop_path=0.15, mlp=2
                )
                inputs = self.inputs()
                self.check_shapes_and_routes(network, config, inputs)
                self.check_joint_gradients_and_image_dependence(network, inputs)
                graph = tf.function(
                    lambda x, t, y: network((x, t, y), training=False), 
                    input_signature=[tf.TensorSpec((None, 32, 32, 3), tf.float32), 
                                     tf.TensorSpec(tuple([None]), tf.int32), 
                                     tf.TensorSpec(tuple([None]), tf.int32)]
                )
                expected = network(inputs, training=False)
                actual = graph(*inputs)
                for name in ("noises", "classes"):
                    np.testing.assert_allclose(actual[name], expected[name], rtol=1e-5, atol=1e-6)
                tail = graph(*(value[:1] for value in inputs))
                for name in ("noises", "classes"):
                    np.testing.assert_allclose(tail[name], actual[name][:1], rtol=1e-5, atol=1e-6)
                self.assertEqual(graph.experimental_get_tracing_count(), 1)
                clone = DiTClassifier.from_config(deepcopy(network.get_config()))
                clone.set_weights(network.get_weights())
                self.assertEqual(clone.dtype_policy.name, "float32")
                self.assertEqual([tuple(v.shape) for v in clone.weights], 
                                 [tuple(v.shape) for v in network.weights])
                cloned = clone(inputs, training=False)
                for name in ("noises", "classes"):
                    np.testing.assert_array_equal(cloned[name], expected[name])

    def test_classifier_condition_selection_changes_only_requested_inputs(self) -> None:
        """Hold encoder features fixed to isolate classifier time/label conditioning."""

        for condition in (None, "time", "label", "time_label"):
            with self.subTest(condition=condition):
                network, _ = self.network(condition=condition)
                self.open_zero_initializers(network)
                images, times, labels = self.inputs()
                full = network((images, times, labels), full_return=True, training=False)

                def classify(t: tf.Tensor, y: tf.Tensor) -> np.ndarray:
                    """Predict from captured encoder features while replacing classifier conditions.

                    Args:
                        t (tf.Tensor): Int32 timestep vector [2].
                        y (tf.Tensor): Int32 class-condition vector [2].

                    Returns:
                        np.ndarray: Float32 primary probabilities [2,C], with encoder features held fixed."""

                    return network.compute_class(full["features_list"], full["noises"], 
                                                 t, y, training=False)[0].numpy()

                baseline = classify(times, labels)
                changed_time = classify(tf.constant([333, 777]), labels)
                changed_label = classify(times, tf.constant([1, 2]))
                for input_name, changed in (("time", changed_time), ("label", changed_label)):
                    # Active classifier conditions must influence the fixed-feature prediction.
                    if condition is not None and input_name in condition:
                        self.assertGreater(float(np.max(np.abs(changed - baseline))), 0.0)
                    # Inactive conditions must leave the fixed-feature prediction identical.
                    else:
                        np.testing.assert_array_equal(changed, baseline)

    def test_head_dropout_and_classifier_drop_path_are_independently_active(self) -> None:
        """Each searched regularizer changes training outputs, never inference."""

        for dropout, drop_path in ((0.15, 0.0), (0.0, 0.15), (0.25, 0.0), (0.0, 0.25)):
            with self.subTest(dropout=dropout, drop_path=drop_path):
                network, _ = self.network(condition=None, dropout=dropout, drop_path=drop_path)
                inputs = self.inputs()
                self.open_zero_initializers(network)
                before = network(inputs, training=False)["classes"].numpy()
                training = [network(inputs, training=True)["classes"].numpy() for _ in range(6)]
                self.assertTrue(any(np.any(output != training[0]) for output in training[1:]))
                np.testing.assert_array_equal(network(inputs, training=False)["classes"], before)


# Execute this focused test module only when invoked directly.
if __name__ == "__main__":
    unittest.main()
