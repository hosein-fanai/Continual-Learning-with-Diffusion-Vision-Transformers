"""Numerical contracts for clean generator bounds and stable auxiliary CE."""

import unittest

import numpy as np
import tensorflow as tf

from common.tests.test_wrapper_verified_repairs import _constant_head
from diffusion import DiffusionClassifier, DiffusionModel, DiffusionTransformer
from common.tests.test_wrapper_verified_repairs import _network as classifier_network


def _network() -> DiffusionTransformer:
    """Create a tiny generator with one real auxiliary softmax head."""

    return DiffusionTransformer(
        num_classes=2, timesteps=4, image_size=4, channels=1, patch_size=2, 
        dim=4, depth=1, mha_num_heads=1, vit_block_mlp_ratio=1., 
        cls_token_type="new_weight", cls_token_regularizer_ids=[1], 
        cls_token_regularizer_kwargs={"start": 0, "end": 1}, seed=41
    )


class GeneratorAuditRepairTests(unittest.TestCase):
    """Keep native gradients and forward/inverse schedule identities observable."""

    def tearDown(self) -> None:
        """Release fixture models after each independent contract."""

        tf.keras.backend.clear_session()

    def test_clean_joint_epsilon_objective_rejects_without_changing_bounds(self) -> None:
        """The invalid transition cannot install inconsistent generator coordinates."""

        model = DiffusionModel(network=_network(), use_ema=False, test_steps=2, 
                               noise_loss_coef=1., image_loss_coef=1., preprocess_type=None)
        original = (model._active_min_timestep, model._active_max_timestep)
        with self.assertRaisesRegex(ValueError, "require modify_first_t=True"):
            model.set_timestep_bounds(0, 0)
        self.assertEqual((model._active_min_timestep, model._active_max_timestep), original)

    def test_valid_clean_and_scheduled_zero_conventions_reconstruct_perfect_targets(self) -> None:
        """Noiseless-first, x0 prediction and ordinary sampled zero keep distinct identities."""

        images = tf.ones((2, 4, 4, 1))
        for modified, direct_x0 in ((True, False), (False, True)):
            with self.subTest(modified=modified, direct_x0=direct_x0):
                model = DiffusionModel(
                    network=_network(), use_ema=False, test_steps=2, 
                    noise_loss_coef=1., image_loss_coef=1., modify_first_t=modified, 
                    swap_noise_image=direct_x0, preprocess_type=None
                )
                model.set_timestep_bounds(0, 0)
                x0, target, times, noisy, *_ = model.prep_inputs((images, tf.constant([0, 1])))
                restored, _ = model.denoise(noisy, times, target, reshape_coefs=True)
                np.testing.assert_array_equal(restored, x0)
        model = DiffusionModel(network=_network(), use_ema=False, test_steps=2, 
                               noise_loss_coef=1., image_loss_coef=1., preprocess_type=None)
        noisy, target, times = model.noisify(images, t=tf.zeros(2, tf.int32))
        restored, _ = model.denoise(noisy, times, target, reshape_coefs=True)
        np.testing.assert_allclose(restored, images, atol=2e-7)
        self.assertGreater(float(tf.reduce_sum(tf.abs(noisy - images))), 0.)

    def test_default_noise_only_and_separate_classifier_clean_remain_valid(self) -> None:
        """The coupled guard does not prohibit unrelated clean-input contracts."""

        model = DiffusionModel(network=_network(), use_ema=False, test_steps=2)
        model.set_timestep_bounds(0, 0)
        classifier = DiffusionClassifier(
            network=classifier_network(2, distil=False), use_ema=False, test_steps=2, 
            noise_loss_coef=1., image_loss_coef=1., clf_train_noisy_input_type="clean", 
            mask_by_nulls=False, preprocess_type=None
        )
        images = tf.ones((2, 4, 4, 1))
        clean, noise, _ = classifier.noisify(images, min_timesteps=0, max_timesteps=0)
        np.testing.assert_array_equal(clean, images)
        np.testing.assert_array_equal(noise, tf.zeros_like(images))
        self.assertEqual(classifier._active_max_timestep, classifier.timesteps)

    def test_native_generator_auxiliary_saturation_retains_same_pass_gradient(self) -> None:
        """Eager and graph generator heads give loss200 and bias gradient[-1,1]."""

        for graph in (False, True):
            with self.subTest(graph=graph):
                network = _network()
                head = network.layers_dicts[0][network.CTR]
                _constant_head(head, [-100., 100.])
                model = DiffusionModel(network=network, use_ema=False, test_steps=2)

                def evaluate(images: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                    """Differentiate the regularizer through the actual raw-network forward."""

                    with tf.GradientTape() as tape:
                        _, regularizers, _ = model.call_network(
                            images, tf.zeros(2, tf.int32), tf.ones(2, tf.int32), 
                            network_name="raw", training=True
                        )
                        loss, _ = model.compute_ctr_loss(tf.zeros(2, tf.int32), regularizers[0])
                    return loss, tape.gradient(loss, head.weights[-1])

                execute = tf.function(evaluate) if graph else evaluate
                loss, gradient = execute(tf.ones((2, 4, 4, 1)))
                self.assertAlmostEqual(float(loss), 200., places=5)
                np.testing.assert_allclose(gradient, [-1., 1.], atol=1e-7)
