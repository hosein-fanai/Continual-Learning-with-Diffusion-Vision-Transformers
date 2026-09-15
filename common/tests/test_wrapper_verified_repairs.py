"""Counterexamples and real wrapper lifecycles for verified repairs F01 through F05."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from diffusion import DiTClassifier, DiffusionClassifier, DiffusionClassifierV2
from diffusion.models.convolution.unet_classifier import UNetClassifier
from diffusion.models.wrapper.diffusion_model import DiffusionModel
from diffusion.callbacks.batch_loss_plateau import BatchLossPlateau
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


def _network(classes: int | None = 3, distil: bool = True,
             auxiliary: str | None = None, convolution: bool = False) -> tf.keras.Model:
    """Build a tiny public raw classifier with optional independent/auxiliary KD heads.

    Args:
        classes (int | None): Fixed head width or None for a dynamic vocabulary.
        distil (bool): Enable the independent distillation head.
        auxiliary (str | None): Auxiliary token training mode, or None to omit it.
        convolution (bool): False selects the transformer; True selects the matching UNet fixture.

    Returns:
        model (tf.keras.Model): Built float32 raw classifier with 4x4 single-channel image inputs.
    """

    common = dict(num_classes=classes, use_cfg=True, timesteps=4, image_size=4,
                  channels=1, seed=37)
    # Cover the compatible convolutional family with its own supported shape parameters.
    if convolution:
        return UNetClassifier(**common, widths=[4], block_depth=1,
                              bottleneck_depth=1, image_embedding_dim=2,
                              time_embedding_dim=3, label_embedding_dim=2,
                              classifier_only_distil_token=distil)
    options = dict(patch_size=2, dim=4, depth=1, mha_num_heads=1,
                   vit_block_mlp_ratio=1., clf_mha_num_heads=1,
                   clf_vit_block_mlp_ratio=1., feature_aggregation_ids_dict={1: [-1]},
                   clf_connection_ids_dict={-1: [-1]})
    # Distillation token ownership enables the independent softmax head.
    if distil:
        options["clf_distil_token_type"] = "new_weight"
    # Auxiliary regularizers expose an independently weighted probability mixture.
    if auxiliary is not None:
        options.update(clf_cls_token_regularizer_ids=[1],
                       clf_cls_token_regularizer_kwargs={"train_type": auxiliary,
                                                        "distil_type": "soft", "start": 0, "end": 1})
    return DiTClassifier(**common, **options)


def _constant_head(head: tf.keras.layers.Layer, logits: list[float]) -> None:
    """Make a real classifier emit fixed scores while retaining trainable bias gradients.

    Args:
        head (tf.keras.layers.Layer): Built head whose final weight is its bias.
        logits (list[float]): Constant pre-softmax scores, one per output class.

    Returns:
        result (None): Zero all head weights and assign its trainable bias in the destination dtype.

    Raises:
        ValueError: If the supplied logit width does not match the head bias.
    """

    for variable in head.weights:
        variable.assign(tf.zeros_like(variable))
    head.weights[-1].assign(logits)


def _dataset(tensors: tuple[tf.Tensor, ...], batch: int = 2) -> tf.data.Dataset:
    """Bound input threads and preserve uneven final batches.

    Args:
        tensors (tuple[tf.Tensor, ...]): Aligned tensors with matching leading dimensions.
        batch (int): Positive batch size, default two; the final partial batch is retained.

    Returns:
        data (tf.data.Dataset): Batches with original tensor dtypes and one private input thread.

    Raises:
        ValueError: If tensor lengths or batch settings are incompatible.
    """

    data = tf.data.Dataset.from_tensor_slices(tensors).batch(batch)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return data.with_options(options)


def _wrapper(version: int = 1, classes: int | None = 3, temperature: float = 1.,
             scope: str = "old_classes", auxiliary: str | None = None,
             convolution: bool = False, eager: bool = True) -> DiffusionClassifier:
    """Compile the actual V1/V2 update path with a frozen two-class teacher.

    Args:
        version (int): One selects joint V1; other fixture values select alternating V2.
        classes (int | None): Student head width, or None for dynamic growth.
        temperature (float): Positive soft-distillation temperature.
        scope (str): Old/current/replay class selection policy.
        auxiliary (str | None): Optional auxiliary token training mode.
        convolution (bool): Select UNet instead of transformer for both student and teacher.
        eager (bool): Whether Keras custom training runs eagerly.

    Returns:
        model (DiffusionClassifier): SGD-compiled float32 wrapper with a frozen two-class teacher, raw inference, and soft KD enabled.

    Raises:
        ValueError: If a distillation or architecture option is invalid.
    """

    raw = _network(classes, auxiliary=auxiliary, convolution=convolution)
    teacher = _network(2, distil=False, convolution=convolution)
    # Select the advertised joint or alternating wrapper without altering either implementation.
    wrapper_class = DiffusionClassifier if version == 1 else DiffusionClassifierV2
    model = wrapper_class(network=raw, teacher_network=teacher, use_ema=False,
                          test_network_name="raw", scheduler_name="linear", test_steps=2,
                          train_noisified_max_timesteps=None, test_noisified_max_timesteps=None,
                          p_uncond=1., mask_by_nulls=False, train_cfg_scale=None,
                          clf_distil_loss_coef=1., clf_distil_type="soft",
                          clf_distil_temperature=temperature, clf_distil_scope=scope,
                          clf_loss_coef=0., noise_loss_coef=0., seed=37,
                          ctr_loss_coef=1. if auxiliary is not None else 0.)
    model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="mse", run_eagerly=eager)
    _constant_head(model.teacher_network.classifier, np.log([.0001, .9999]).tolist())
    return model


class WrapperVerifiedRepairTests(unittest.TestCase):
    """Check numerical formulas, deployed predictions, and actual Keras wrapper consumers."""

    def tearDown(self) -> None:
        """Release model graphs between independent fixtures.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.
        """

        tf.keras.backend.clear_session()

    def test_soft_kd_extreme_gradients_support_weights_and_frozen_targets(self) -> None:
        """Compare gradients to T*(softmax(z/T)-q) including saturation and empty scopes.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        model = _wrapper()
        teacher = tf.Variable([[.0001, .9999], [.8, .2], [.3, .7]])
        labels = tf.constant([0, 2, 1])
        row_weights = tf.constant([1., 9., 3.])
        for temperature in (1., 2.):
            for gap in (5., 30., 120.):
                logits = tf.Variable([[gap, 0., -3.], [0., gap, -1.], [gap, 0., 1.]])
                with tf.GradientTape(persistent=True) as tape:
                    loss, _ = model.compute_clf_distil_loss(
                        teacher, tf.nn.softmax(logits), classes=labels,
                        clf_distil_loss_mask=row_weights,
                        clf_distil_temperature=temperature, student_logits=logits,
                    )
                gradient = tape.gradient(loss, logits)
                self.assertIsNone(tape.gradient(loss, teacher))
                softened = tf.nn.softmax(tf.math.log(teacher) / temperature)
                target = tf.pad(softened, [[0, 0], [0, 1]])
                expected = temperature * (tf.nn.softmax(logits / temperature) - target)
                expected *= tf.constant([[.25], [0.], [.75]])
                np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=2e-7)
                self.assertTrue(np.isfinite(loss))
                with tf.GradientTape() as empty_tape:
                    empty, _ = model.compute_clf_distil_loss(
                        teacher, tf.nn.softmax(logits), student_logits=logits,
                        replay_mask=tf.zeros(3, tf.bool), clf_distil_scope="replay_only",
                        clf_distil_temperature=temperature,
                    )
                self.assertEqual(float(empty), 0.)
                np.testing.assert_array_equal(empty_tape.gradient(empty, logits), np.zeros((3, 3)))

    def test_legacy_positional_forward_preserves_actual_training_mode(self) -> None:
        """Legacy positional calls and ordinary V1 updates keep the raw student in training mode.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        model = _wrapper(classes=2)
        model.clf_distil_type = "hard"
        model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="mse", run_eagerly=True)
        images = tf.zeros((2, 4, 4, 1))
        labels = tf.constant([0, 1])
        times = tf.zeros(2, tf.int32)
        with patch.object(model.network, "call", wraps=model.network.call) as calls:
            model.call_network(images, times, labels, None, None, "raw", True)
            model.forward("raw", images, times, times, labels, None, None, True)
            model.train_step((images, labels))
        self.assertEqual(calls.call_count, 3)
        self.assertEqual([call.kwargs["training"] for call in calls.call_args_list], [True] * 3)
        self.assertTrue(all("return_logits" not in call.kwargs for call in calls.call_args_list))

    def test_legacy_positional_losses_and_unmasked_noise_metric(self) -> None:
        """Existing loss/metric positional arguments retain their meaning beside additive metadata.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        model = _wrapper(classes=2)
        labels = tf.constant([0, 1])
        images = tf.zeros((2, 4, 4, 1))
        teacher = tf.constant([[.8, .2], [.1, .9]])
        logits = tf.constant([[2., -1.], [-1., 1.]])
        probabilities = tf.nn.softmax(logits)
        mask = tf.constant([1., 3.])
        replay = tf.constant([True, True])
        actual = model.compute_clf_kl_ctr_distil_loss(
            labels, None, None, None, None, probabilities, [], [], probabilities,
            mask, "uncond", "uncond", "uncond", teacher, images, replay, False,
            logits_u={"distil_logits": logits},
        )
        expected, _ = model.compute_clf_distil_loss(
            teacher, probabilities, classes=labels, clf_distil_loss_mask=mask,
            replay_mask=replay, student_logits=logits,
        )
        np.testing.assert_allclose(actual[0], expected)
        np.testing.assert_array_equal(actual[5], probabilities)
        result = model.get_results_dict(
            tf.constant(0.), None, None, tf.constant(3.), None, None, None, None, None,
            labels, None, False, True, False, False, False,
        )
        self.assertEqual(float(result["noise_distil_loss"]), 3.)
        self.assertEqual(float(model.noise_distil_loss_tracker.count), 2.)
        result = model.get_results_dict(
            tf.constant(0.), None, None, tf.constant(5.), None, None, None, None, None,
            tf.constant([0, 1, 0]), None, False, True, False, False, False,
        )
        self.assertAlmostEqual(float(result["noise_distil_loss"]), 4.2, places=6)
        self.assertEqual(float(model.noise_distil_loss_tracker.count), 5.)

    def test_recompile_refreshes_same_pass_logits_before_real_updates(self) -> None:
        """Switching hard/soft objectives at compile refreshes V1/V2 cached raw-network kwargs.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        images = tf.zeros((2, 4, 4, 1))
        labels = tf.constant([0, 1])
        for version in (1, 2):
            model = _wrapper(version=version, classes=2)
            self.assertEqual(dict(model.use_logits_instead), {"return_logits": True})
            model.clf_distil_type = "hard"
            model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="mse", run_eagerly=True)
            self.assertEqual(dict(model.use_logits_instead), {})
            model.clf_distil_type = "soft"
            model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="mse", run_eagerly=True)
            self.assertEqual(dict(model.use_logits_instead), {"return_logits": True})
            _constant_head(model.network.distil_classifier, [120., 0.])
            with patch.object(model.network, "compute_class", wraps=model.network.compute_class) as calls:
                # Alternating updates use the public phase's classifier variable selection.
                if version == 2:
                    model._set_clf_variables()
                    result = model.discriminator_train_step((images, labels))
                # Joint updates request the same metadata through the diffusion forward path.
                else:
                    result = model.train_step((images, labels))
            self.assertEqual(calls.call_count, 1)
            self.assertTrue(np.isfinite(result["clf_distil_loss"]))
            self.assertAlmostEqual(float(model.network.distil_classifier.weights[-1][1]), .09999, places=6)

    def test_auxiliary_logits_cache_reconstructs_with_teacher_and_weights(self) -> None:
        """Auxiliary-only soft KD keeps its renamed metadata through real updates and reconstruction.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        images = tf.zeros((2, 4, 4, 1))
        labels = tf.constant([0, 1])
        nulls = tf.zeros(2, tf.int32)
        for version in (1, 2):
            model = _wrapper(version=version, auxiliary="distil")
            model.clf_distil_loss_coef = 0.
            model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="mse", run_eagerly=True)
            self.assertEqual(dict(model.use_logits_instead), {"return_logits": True})
            auxiliary_heads = [stage[model.network.CTR] for stage in model.network.clf_layers_dicts
                               if model.network.CTR in stage]
            self.assertTrue(auxiliary_heads)
            for head in auxiliary_heads:
                _constant_head(head, [120., 0., -3.])
            # Select the classifier phase for the alternating wrapper's actual update.
            if version == 2:
                model._set_clf_variables()
                result = model.discriminator_train_step((images, labels))
            # Joint training must carry the same auxiliary metadata through its forward outputs.
            else:
                result = model.train_step((images, labels))
            self.assertTrue(np.isfinite(result["clf_ctr_loss"]))
            self.assertGreater(float(auxiliary_heads[0].weights[-1][1]), .09)
            config = model.get_config()
            config["clf_distil_loss_coef"] = 0.
            restored = model.__class__.from_config(config)
            self.assertEqual(dict(restored.use_logits_instead), {})
            restored.set_teacher_network(_network(2, distil=False))
            restored.compile(optimizer="sgd", loss="mse", run_eagerly=True)
            self.assertEqual(dict(restored.use_logits_instead), {"return_logits": True})
            # Legacy HDF5 includes a constructor-supplied teacher; reconstruct that same topology.
            config["teacher_network"] = _network(2, distil=False)
            restored = model.__class__.from_config(config)
            restored.compile(optimizer="sgd", loss="mse", run_eagerly=True)
            self.assertEqual(dict(restored.use_logits_instead), {"return_logits": True})
            with tempfile.TemporaryDirectory() as temporary:
                checkpoint = str(Path(temporary) / "auxiliary.weights.h5")
                model.save_weights(checkpoint)
                restored.load_weights(checkpoint)
                before = model.network.predict_class((images, nulls, nulls), full_return=True,
                                                     **model.use_logits_instead)
                after = restored.network.predict_class((images, nulls, nulls), full_return=True,
                                                       **restored.use_logits_instead)
                for first, second in zip(before[-1]["clf_regs_logits_list"],
                                         after[-1]["clf_regs_logits_list"]):
                    # Unregularized stages deliberately retain their None placeholder.
                    if first is None:
                        self.assertIsNone(second)
                    # Check the actual auxiliary logits from the native HDF5 reload.
                    else:
                        np.testing.assert_allclose(first, second)

    def test_v1_v2_real_updates_use_same_pass_logits_and_preserve_allocation(self) -> None:
        """Actual train steps correct saturated biases once and leave all teacher weights frozen.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        images = tf.zeros((3, 4, 4, 1))
        labels = tf.constant([0, 2, 1])
        provenance = tf.constant([True, False, True])
        for version in (1, 2):
            for temperature in (1., 2.):
                model = _wrapper(version=version, temperature=temperature)
                _constant_head(model.network.distil_classifier, [30., 0., -3.])
                teacher_before = [value.numpy().copy() for value in model.teacher_network.weights]
                seen = []

                def allocate(wrapper: object, losses: tf.Tensor, mask: tf.Tensor,
                             x0: tf.Tensor, classes: tf.Tensor,
                             replay: tf.Tensor) -> tf.Tensor:
                    """Record the real allocation hook inputs and preserve its weighted reduction.

                    Args:
                        wrapper (object): Wrapper invoking the hook, retained for interface compatibility.
                        losses (tf.Tensor): Per-row floating distillation losses.
                        mask (tf.Tensor): Floating eligible-row weights.
                        x0 (tf.Tensor): Clean image batch.
                        classes (tf.Tensor): Integer class IDs.
                        replay (tf.Tensor): Boolean replay-origin flags.

                    Returns:
                        loss (tf.Tensor): Mask-weighted floating scalar mean, zero for no eligible rows; hook inputs are recorded.
                    """

                    seen.append((mask.numpy(), x0.numpy(), classes.numpy(), replay.numpy()))
                    return tf.math.divide_no_nan(tf.reduce_sum(losses * mask), tf.reduce_sum(mask))

                model._classifier_kd_allocator = allocate
                # Count actual raw passes; auxiliary logits must add no stochastic execution.
                with patch.object(model.network, "compute_class", wraps=model.network.compute_class) as calls:
                    # V2 selects its classifier optimizer through the public phase fit adapter.
                    if version == 1:
                        result = model.train_step((images, labels, provenance))
                    # Alternating training updates only the classifier-owned optimizer group.
                    else:
                        model._set_clf_variables()
                        result = model.discriminator_train_step((images, labels, provenance))
                    self.assertEqual(calls.call_count, 1)
                target = tf.pad(tf.nn.softmax(tf.math.log([[.0001, .9999]]) / temperature),
                                [[0, 0], [0, 1]])
                expected_gradient = temperature * (tf.nn.softmax(tf.constant([[30., 0., -3.]]) / temperature) - target)
                actual = model.network.distil_classifier.weights[-1].numpy()
                np.testing.assert_allclose(actual, np.array([30., 0., -3.]) - .1 * expected_gradient[0],
                                           rtol=1e-5, atol=2e-6)
                self.assertEqual(len(seen), 1)
                np.testing.assert_array_equal(seen[0][0], [1., 0., 1.])
                np.testing.assert_array_equal(seen[0][2], labels)
                np.testing.assert_array_equal(seen[0][3], provenance)
                self.assertTrue(np.isfinite(result["clf_distil_loss"]))
                for before, after in zip(teacher_before, model.teacher_network.weights):
                    np.testing.assert_array_equal(before, after)

    def test_traced_soft_kd_after_head_growth_reloads_and_evaluates(self) -> None:
        """Grow a head, fit both actual graph paths, reload, and recompute label-free accuracy.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        images = tf.zeros((3, 4, 4, 1))
        labels = tf.constant([0, 2, 1])
        data = _dataset((images, labels), batch=3)
        for version in (1, 2):
            model = _wrapper(version=version, classes=None, eager=False)
            model._check_new_labels(y=tf.constant([0, 1]), verbose=False)
            model.compile(optimizer=tf.keras.optimizers.SGD(.1), loss="mse")
            _constant_head(model.network.distil_classifier, [120., 0.])
            # Use the public phase adapter to prepare variables and Keras graph functions.
            fit = model.fit if version == 1 else model.fit_discriminator
            history = fit(x=data, epochs=1, verbose=0)
            self.assertTrue(np.isfinite(history.history["clf_distil_loss"][0]))
            self.assertGreater(float(model.network.distil_classifier.weights[-1][1]), .09)
            with tempfile.TemporaryDirectory() as temporary:
                checkpoint = str(Path(temporary) / "grown.weights.h5")
                model.save_weights(checkpoint)
                restored = _wrapper(version=version, classes=None, eager=False)
                restored._check_new_labels(y=tf.constant([0, 1, 2]), verbose=False)
                restored.load_weights(checkpoint)
                nulls = tf.zeros(3, tf.int32)
                predictions = restored.network.predict_class((images, nulls, nulls))
                reference = model.network.predict_class((images, nulls, nulls))
                np.testing.assert_allclose(predictions, reference)
                expected = float(tf.reduce_mean(tf.cast(
                    tf.argmax(predictions, axis=-1, output_type=tf.int32) == labels, tf.float32)))
                # Alternating evaluation explicitly selects the classifier phase.
                options = {"test_part": "discriminator"} if version == 2 else {}
                result = restored.evaluate(x=data, verbose=0, return_dict=True, **options)
                self.assertAlmostEqual(result[restored.accuracy_tracker.name], expected)

    def test_auxiliary_kd_uses_stable_probability_mixture(self) -> None:
        """Temperature acts on the mean head distribution, including conflicting and saturated heads.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        model = _wrapper(auxiliary="distil", temperature=2., scope="current_and_replay")
        first = tf.Variable([[120., 0., -4.], [4., 0., 1.]])
        second = tf.Variable([[110., 0., -5.], [-1., 3., 0.]])
        labels = tf.constant([0, 1])
        teacher = tf.constant([[.0001, .9999], [.7, .3]])
        with tf.GradientTape() as tape:
            actual, _ = model.compute_clf_distil_ctr_loss(
                labels, [tf.nn.softmax(first), None, tf.nn.softmax(second)],
                teacher_labels=teacher, classes_logits_list=[first, None, second],
            )
        actual_gradients = tape.gradient(actual, [first, second])
        with tf.GradientTape() as tape:
            # Float64 probabilities remain representable in this independent mixture reference.
            mixture = (tf.nn.softmax(tf.cast(first, tf.float64)) +
                       tf.nn.softmax(tf.cast(second, tf.float64))) / 2.
            student = tf.math.log(mixture) / 2.
            target = tf.pad(tf.nn.softmax(tf.math.log(tf.cast(teacher, tf.float64)) / 2.),
                            [[0, 0], [0, 1]])
            expected = 4. * tf.reduce_mean(tf.reduce_sum(
                tf.math.xlogy(target, target) - target * tf.nn.log_softmax(student), axis=-1))
        expected_gradients = tape.gradient(expected, [first, second])
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        for measured, reference in zip(actual_gradients, expected_gradients):
            np.testing.assert_allclose(measured, reference, rtol=2e-5, atol=1e-7)

    def test_v2_nongrowing_refresh_retains_weight_order_and_checkpoint_layout(self) -> None:
        """Refreshing cached groups preserves trainable identity and HDF5 weight order.

        Returns:
            result (None): All public weight identities and values remain exact
                before refresh and after loading an equivalent wrapper.
        """

        model = _wrapper(version=2)
        model._register_optimizer_variables()
        before = [(value.name, id(value)) for value in model.weights]
        top_before = [(value.name, id(value)) for value in model.trainable_weights]
        model._register_optimizer_variables()
        self.assertEqual(before, [(value.name, id(value)) for value in model.weights])
        self.assertEqual(top_before, [(value.name, id(value)) for value in model.trainable_weights])
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = str(Path(temporary) / "nongrowing.weights.h5")
            model.save_weights(checkpoint)
            restored = _wrapper(version=2)
            restored._register_optimizer_variables()
            restored.load_weights(checkpoint)
            self.assertEqual(len(model.weights), len(restored.weights))
            for first, second in zip(model.weights, restored.weights):
                np.testing.assert_array_equal(first, second)

    def test_total_accuracy_matches_label_free_heads_across_scopes_and_modes(self) -> None:
        """Identical images use identical inference rules for both independent and auxiliary heads.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        images = tf.zeros((2, 4, 4, 1))
        labels = tf.constant([0, 2])
        for version in (1, 2):
            for mode in (None, "distil", "both"):
                model = _wrapper(version=version, auxiliary=mode)
                model.clf_acc_coef = 1.
                model.clf_distil_acc_coef = 1.
                model.ctr_acc_coef = 1.
                _constant_head(model.network.classifier, np.log([.1, .1, .8]).tolist())
                _constant_head(model.network.distil_classifier, np.log([.98, .01, .01]).tolist())
                # Include actual auxiliary heads whenever this configuration enables them.
                if mode is not None:
                    for stage in model.network.clf_layers_dicts:
                        # Non-regularized classifier stages have no auxiliary head.
                        if model.network.CTR in stage:
                            _constant_head(stage[model.network.CTR], np.log([.98, .01, .01]).tolist())
                ensemble = EnsembleAccuracy(model, network_name="raw", max_t=1,
                                            clf_acc_coef=1., clf_distil_acc_coef=1.,
                                            ctr_acc_coef=1. if mode is not None else 0.)
                scores = ensemble.ensemble_predict(images, training=False)
                np.testing.assert_array_equal(scores[0], scores[1])
                expected = float(tf.reduce_mean(tf.cast(tf.argmax(scores, axis=1,
                                              output_type=tf.int32) == labels, tf.float32)))
                for scope in ("old_classes", "replay_only", "current_and_replay"):
                    model.clf_distil_scope = scope
                    data = _dataset((images, labels, tf.constant([True, False])))
                    # V2 evaluation must explicitly select the classifier phase.
                    options = {"test_part": "discriminator"} if version == 2 else {}
                    result = model.evaluate(x=data, verbose=0, return_dict=True, **options)
                    self.assertEqual(result["total_accuracy"], expected)
                    self.assertEqual(expected, .5)

    def test_fit_validation_inherits_raw_or_ema_and_restores_graphs(self) -> None:
        """Real two-epoch fit validation and explicit alternating evaluations respect branch selection.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        data = _dataset((tf.zeros((2, 4, 4, 1)), tf.zeros(2, tf.int32)))
        for version in (1, 2):
            for configured in ("raw", "ema"):
                # Select the actual joint or phase wrapper with graph tracing enabled.
                wrapper_class = DiffusionClassifier if version == 1 else DiffusionClassifierV2
                model = wrapper_class(network=_network(2, distil=False), use_ema=True, ema_decay=.999,
                                      test_network_name=configured, test_steps=2,
                                      train_noisified_max_timesteps=None,
                                      test_noisified_max_timesteps=None, p_uncond=1.)
                model.compile(optimizer=tf.keras.optimizers.SGD(0.), loss="mse")
                _constant_head(model.network.classifier, [4., -4.])
                _constant_head(model.ema_network.classifier, [-4., 4.])
                # V2's public discriminator adapter sets both training and validation phase.
                fit = model.fit if version == 1 else model.fit_discriminator
                history = fit(x=data, validation_data=data, epochs=2, verbose=0)
                expected = 1. if configured == "raw" else 0.
                np.testing.assert_array_equal(history.history["val_classifier_accuracy"], [expected] * 2)
                cached_train = model.train_function
                for requested, accuracy in (("raw", 1.), ("ema", 0.), (None, expected)):
                    # A V2 evaluate call needs its phase after fit restores phase state.
                    options = {"test_part": "discriminator"} if version == 2 else {}
                    result = model.evaluate(x=data, network_name=requested, verbose=0, return_dict=True, **options)
                    self.assertEqual(result["classifier_accuracy"], accuracy)
                    self.assertEqual(model.test_network_name, configured)
                self.assertIsNone(model.test_function)
                self.assertIs(model.train_function, cached_train)

    def test_no_ema_validation_and_failed_override_restore_configured_branch(self) -> None:
        """Disabled EMA aliases raw and exceptions cannot retain a temporary evaluation selector.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        model = _wrapper(classes=2)
        _constant_head(model.network.classifier, [4., -4.])
        data = _dataset((tf.zeros((2, 4, 4, 1)), tf.zeros(2, tf.int32)))
        result = model.evaluate(x=data, network_name="ema", verbose=0, return_dict=True)
        self.assertEqual(result[model.accuracy_tracker.name], 1.)
        with patch.object(tf.keras.Model, "evaluate", side_effect=RuntimeError("fixture")):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                model.evaluate(x=data, network_name="ema")
        self.assertEqual(model.test_network_name, "raw")
        self.assertIsNone(model.test_function)

    def test_noise_kd_real_evaluation_is_independent_of_batch_partition(self) -> None:
        """Prepared evaluation batches aggregate only eligible rows through all three wrappers.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        classes = tf.constant([0, 1, 2, 2, 0])
        images = tf.zeros((5, 4, 4, 1))
        teacher_predictions = tf.reshape(tf.constant([1., 2., 9., 9., 3.]), (-1, 1, 1, 1)) \
            * tf.ones_like(images)
        eligible = tf.constant([1., 1., 0., 0., 1.])
        prepared = (images, images, tf.zeros(5, tf.int32), images,
                    classes + 1, tf.zeros(5, tf.int32), classes, teacher_predictions, eligible)
        for wrapper_class in (DiffusionModel, DiffusionClassifier, DiffusionClassifierV2):
            raw = _network(distil=False)
            # Base diffusion does not expose classifier-specific constructor options.
            options = {} if wrapper_class is DiffusionModel else {"clf_distil_loss_coef": 0.}
            model = wrapper_class(network=raw, teacher_network=_network(2, distil=False),
                                  use_ema=False, noise_distil_loss_coef=1., test_steps=2,
                                  map_preprocess=True, **options)
            model.compile(optimizer="sgd", loss="mse", run_eagerly=True)
            for variable in raw.weights:
                variable.assign(tf.zeros_like(variable))
            for batch in (1, 2, 3, 5):
                # Alternating wrappers expose the inherited generator evaluator explicitly.
                evaluate = model.evaluate_generator if wrapper_class is DiffusionClassifierV2 \
                    else model.evaluate
                result = evaluate(x=_dataset(prepared, batch), verbose=0, return_dict=True)
                self.assertAlmostEqual(result["noise_distil_loss"], 14. / 3., places=5)

    def test_progressive_batch_plateau_supports_both_directions(self) -> None:
        """Actual curriculum orchestration forwards max/min and uses the documented patience rule.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        model = _wrapper(classes=2)
        data = _dataset((tf.zeros((2, 4, 4, 1)), tf.zeros(2, tf.int32)))
        for direction, values in (("max", [.2, .4, .6, .6, .6]),
                                  ("min", [.8, .6, .4, .4, .4])):
            consumed = []

            def controlled_fit(bound: object, callbacks: list[tf.keras.callbacks.Callback],
                               initial_epoch: int, **kwargs: object) -> tf.keras.callbacks.History:
                """Supply controlled batch logs through the curriculum's actual callback list.

                Args:
                    bound (object): Wrapper on which callback state is installed.
                    callbacks (list[tf.keras.callbacks.Callback]): Actual curriculum callbacks to exercise.
                    initial_epoch (int): Epoch recorded in the synthetic history.
                    **kwargs (object): Remaining public fit arguments, ignored by this controlled log source.

                Returns:
                    history (tf.keras.callbacks.History): Controlled final accuracy and epoch after honoring callback-requested stopping.
                """

                bound.stop_training = False
                for callback in callbacks:
                    callback.set_model(bound)
                for batch, value in enumerate(values):
                    consumed.append(value)
                    for callback in callbacks:
                        callback.on_train_batch_end(batch, {model.accuracy_tracker.name: value})
                    # Stop only when the callback requests the end of this curriculum stage.
                    if bound.stop_training:
                        break
                history = tf.keras.callbacks.History()
                history.epoch = [initial_epoch]
                history.history = {model.accuracy_tracker.name: [consumed[-1]]}
                return history

            with patch.object(tf.keras.Model, "fit", controlled_fit):
                model.fit_progressively(stage_tasks=[("timesteps", (0, 4))], stage_epochs=1,
                                        final_epochs=0, pacing_type="plateau",
                                        earlystopping_type="batch_wise", monitor=model.accuracy_tracker.name,
                                        stopper_mode=direction, patience=2, min_delta=0.,
                                        x=data, verbose=0, stages_verbose=False)
            self.assertEqual(consumed, values)
            self.assertTrue(model.stop_training)
        auto = BatchLossPlateau(monitor="classifier_accuracy", patience=2, mode="auto")
        self.assertTrue(auto.maximize)
        with self.assertRaisesRegex(ValueError, "mode must"):
            BatchLossPlateau(mode="sideways")

    def test_unet_same_pass_logits_checkpoint_and_real_update(self) -> None:
        """Convolutional soft KD preserves logits, weight identities, reloads, and actual updates.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """

        model = _wrapper(classes=2, convolution=True)
        _constant_head(model.network.distil_classifier, [120., 0.])
        images = tf.zeros((2, 4, 4, 1))
        labels = tf.constant([0, 1])
        before_names = [weight.name for weight in model.network.weights]
        outputs = model.network((images, tf.zeros(2, tf.int32), labels),
                                full_return=True, return_logits=True, training=True)
        np.testing.assert_array_equal(outputs["distil_logits"], [[120., 0.]] * 2)

        @tf.function
        def traced(inputs: tuple[tf.Tensor, ...]) -> tuple:
            """Carry convolutional logits through the traced predict_class boundary.

            Args:
                inputs (tuple[tf.Tensor, ...]): Image, integer time, and class-condition tensors accepted by predict_class.

            Returns:
                outputs (tuple): Actual classifier outputs plus same-pass floating logits, preserving the network return order.
            """

            return model.network.predict_class(inputs, full_return=True,
                                               return_logits=True, training=True)

        traced_outputs = traced((images, tf.zeros(2, tf.int32), labels))
        np.testing.assert_array_equal(traced_outputs[-1]["distil_logits"], [[120., 0.]] * 2)
        self.assertEqual(before_names, [weight.name for weight in model.network.weights])
        result = model.train_step((images, labels))
        self.assertGreater(float(model.network.distil_classifier.weights[-1][1]), .09)
        self.assertTrue(np.isfinite(result["clf_distil_loss"]))
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = str(Path(temporary) / "wrapper.weights.h5")
            model.save_weights(checkpoint)
            restored = _wrapper(classes=2, convolution=True)
            restored.load_weights(checkpoint)
            np.testing.assert_allclose(model.network.predict_class((images, tf.zeros(2, tf.int32), labels)),
                                       restored.network.predict_class((images, tf.zeros(2, tf.int32), labels)))


# Direct invocation runs the same tests as unittest discovery.
if __name__ == "__main__":
    unittest.main()
