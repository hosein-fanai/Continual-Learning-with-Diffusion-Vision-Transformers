"""Cross categorical ensemble inference modes against independent NumPy formulas.

The main matrix exhausts raw/EMA selection, compute mode, CFG combination,
weighting, timestep dropping, nonempty head subsets, prediction caps and stable
floating dtypes: 1,344 cases. Separate matrices cover traced variable batches,
class-count/label mapping/accuracy weights, and unavailable optional heads.
Fixtures expose deterministic probabilities depending on images, time, condition
and network identity; they do not estimate learned accuracy or noise quality.
Run with ``python -m unittest common.tests.test_inference_combination_matrix``.
"""

from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from common.runtime import derive_seed
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


HEAD_COEFFICIENTS = tuple(
    tuple(coefficient * bit for coefficient, bit in zip((0.7, 0.3, 0.5), bits))
    for bits in itertools.product((0, 1), repeat=3) if any(bits)
)
SIGNAL_POWERS = np.array([0.88, 0.66, 0.43, 0.21, 0.07], dtype=np.float64)
MASTER_SEED = 719


def _softmax(values: np.ndarray) -> np.ndarray:
    """Normalize float64 [..., C] logits along the last axis without overflow.

    Args:
        values (np.ndarray): Finite floating logits; input is not modified.

    Returns:
        np.ndarray: Float64 probabilities of identical shape summing to one.
    """

    centered = np.asarray(values, dtype=np.float64) - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(centered)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _fixture(class_count: int = 3, dynamic: bool = False, 
             regularizers: bool = True, distillation: bool = True) -> SimpleNamespace:
    """Create raw/EMA probability adapters with independent condition/time effects.

    Args:
        class_count (int): One or three output classes.
        dynamic (bool): Map original labels [40, 7, 99] to dense output columns.
        regularizers (bool): Expose two regularizer heads separated by None.
        distillation (bool): Expose an independent sixth probability output.

    Returns:
        SimpleNamespace: Wrapper-compatible fixture with deterministic noising,
        fixed seed, distinct raw/EMA adapters and an explicit label mapper.
    """

    def network(offset: float) -> SimpleNamespace:
        """Bind one network identity to a probability-producing adapter.

        Args:
            offset (float): Per-column logit offset distinguishing raw and EMA.

        Returns:
            SimpleNamespace: CFG classifier metadata and a full-return callable.
        """

        def predict(inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
                    max_encoder_num: int | None = None, full_return: bool = True, 
                    training: bool | None = None) -> tuple[object, ...]:
            """Compute multiple [B,C] heads from floating images and int32 IDs.

            Args:
                inputs (tuple): Floating [B,1,1,1] images and int32 [B] time/label IDs.
                max_encoder_num (int | None): Unused adapter compatibility option.
                full_return (bool): Must be true for this ensemble-only fixture.
                training (bool | None): Must be false: inference must not enable dropout.

            Returns:
                tuple: Primary probabilities, context, features, regularizers,
                latent values and optional independent distillation probabilities.
            """

            del max_encoder_num
            # Enforce the fixture's inference-only adapter contract before producing head scores.
            if not full_return or training is not False:
                raise AssertionError('Ensemble must request full inference outputs.')
            images, times, labels = inputs
            columns = tf.cast(tf.range(class_count), images.dtype)[None, :]
            feature = images[:, 0, 0, 0, None]
            time = tf.cast(times[:, None], images.dtype)
            condition = tf.cast(labels[:, None], images.dtype)
            logits = (feature * (0.6 - 0.4 * columns)
                      + time * (0.1 + 0.05 * columns)
                      + condition * (0.15 * columns - 0.1)
                      + offset * columns)
            primary = tf.nn.softmax(logits)
            heads = ([tf.nn.softmax(-0.6 * logits + 0.2 * columns), None, 
                      tf.nn.softmax(0.3 * logits - 0.15 * columns)]
                     if regularizers else [])
            result = (primary, None, [], heads, [])
            return result + tuple([tf.nn.softmax(-logits + 0.4 * columns)]) if distillation else result
        return SimpleNamespace(use_cfg=True, num_classes=class_count, 
                               num_labels=class_count + 1, 
                               dynamic_num_classes=dynamic, predict_class=predict)

    def rates(times: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Return float64 schedule amplitudes for int32 timestep IDs of any shape."""

        powers = tf.gather(tf.constant(SIGNAL_POWERS), times)
        return tf.sqrt(powers), tf.sqrt(1.0 - powers)

    def q_sample(images: tf.Tensor, times: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        """Keep dtype/shape and add 0.03*t; ignore noise to isolate aggregation math."""

        del noise
        return images + tf.cast(times[:, None, None, None], images.dtype) * tf.constant(0.03, images.dtype)

    def map_classes(labels: tf.Tensor) -> tf.Tensor:
        """Map known original int32 [B] or [B,1] labels to dense int32 [B]."""

        vocabulary = tf.constant([40, 7, 99][:class_count], tf.int32)
        return tf.argmax(tf.cast(tf.equal(tf.reshape(labels, [-1, 1]), vocabulary[None, :]), 
                                 tf.int32), axis=1, output_type=tf.int32)

    copies = {'raw': network(0.0), 'ema': network(0.35)}
    return SimpleNamespace(timesteps=5, seed=MASTER_SEED, get_network=copies.__getitem__, 
                           q_sample=q_sample, get_noise_and_signal_rates=rates, 
                           _map_classes=map_classes)


def _retained(drop_rate: float, dtype: str) -> np.ndarray:
    """Calculate seeded inverse-SNR Gumbel removal independently in NumPy.

    Args:
        drop_rate (float): Fraction 0, 0.4 or 1; at least one timestep remains.
        dtype (str): float32 or float64, matching the documented RNG arithmetic.

    Returns:
        np.ndarray: Sorted retained int32 timestep IDs. Only the shared stateless
        RNG primitive/seed derivation is reused; weighting and ranking are independent.
    """

    dropped = min(int(5 * drop_rate), 4)
    # No-drop inference must preserve every timestep without consulting randomness.
    if dropped == 0:
        return np.arange(5, dtype=np.int32)
    seed = derive_seed(MASTER_SEED, 'ensemble_accuracy', 'timestep_dropout')
    uniform = tf.random.stateless_uniform(
        tuple([5]), seed=(seed, 0), minval=np.finfo(dtype).tiny, 
        maxval=1.0, dtype=tf.as_dtype(dtype)
    ).numpy().astype(np.float64)
    snr = SIGNAL_POWERS / (1.0 - SIGNAL_POWERS)
    removal_scores = -np.log(snr) - np.log(-np.log(uniform))
    return np.sort(np.argsort(-removal_scores)[dropped:]).astype(np.int32)


def _expected(images: np.ndarray, class_count: int, network_name: str, 
              coefficients: tuple[float, float, float], separate: bool, 
              weighted: bool, drop_rate: float, dtype: str) -> np.ndarray:
    """Compute all-head inference by explicit NumPy image/time/condition loops.

    Args:
        images (np.ndarray): Floating [B,1,1,1] clean values.
        class_count (int): One or three output columns.
        network_name (str): raw or ema; changes the fixture's per-column bias.
        coefficients (tuple[float,float,float]): Primary, mean-regularizer and KD weights.
        separate (bool): Add the null vector and each real condition's diagonal,
            then softmax once after the weighted timestep mean.
        weighted (bool): Use normalized retained SNR rather than a uniform mean.
        drop_rate (float): Deterministic retained-timestep fraction control.
        dtype (str): Arithmetic used by the stateless selection stream.

    Returns:
        np.ndarray: Float64 [B,C] weighted scores, or normalized separate probabilities.
    """

    selected = _retained(drop_rate, dtype)
    snr = SIGNAL_POWERS[selected] / (1.0 - SIGNAL_POWERS[selected])
    weights = snr / snr.sum() if weighted else np.ones(len(selected)) / len(selected)
    columns = np.arange(class_count, dtype=np.float64)
    offset = 0.0 if network_name == 'raw' else 0.35
    answers = []
    for image in np.asarray(images, dtype=np.float64)[:, 0, 0, 0]:
        by_time = []
        for time in selected:
            predictions = []
            for condition in range(class_count + 1 if separate else 1):
                feature = image + 0.03 * time
                logits = (feature * (0.6 - 0.4 * columns)
                          + time * (0.1 + 0.05 * columns)
                          + condition * (0.15 * columns - 0.1) + offset * columns)
                primary = _softmax(logits)
                regularizer = (_softmax(-0.6 * logits + 0.2 * columns)
                               + _softmax(0.3 * logits - 0.15 * columns)) / 2.0
                distilled = _softmax(-logits + 0.4 * columns)
                predictions.append(coefficients[0] * primary + coefficients[1] * regularizer
                                   + coefficients[2] * distilled)
            score = predictions[0]
            # Real condition c contributes only its own class column c-1.
            if separate:
                score = score + np.diag(np.asarray(predictions[1:]))
            by_time.append(score)
        combined = np.sum(np.asarray(by_time) * weights[:, None], axis=0)
        answers.append(_softmax(combined) if separate else combined)
    return np.asarray(answers)


class InferenceCombinationMatrixTests(unittest.TestCase):
    """Check categorical inference interactions independently of learned model quality."""

    def tearDown(self) -> None:
        """Release traced graphs and Keras metric resources after each complete matrix."""

        tf.keras.backend.clear_session()

    def test_all_head_weight_condition_compute_and_dtype_combinations(self) -> None:
        """Match 1,344 Cartesian cases to independent selection/aggregation formulas."""

        exercised = 0
        for name, mode, separate, weighted, drop, coefficients, cap, dtype in itertools.product(
            ('raw', 'ema'), ('batched', 'chunked'), (False, True), (False, True), 
            (0.0, 0.4, 1.0), HEAD_COEFFICIENTS, (None, 3), ('float32', 'float64')
        ):
            with self.subTest(network=name, mode=mode, separate=separate, weighted=weighted, 
                              drop=drop, heads=coefficients, cap=cap, dtype=dtype):
                images = tf.constant([-0.9, 0.2, 1.3], tf.as_dtype(dtype))[:, None, None, None]
                metric = EnsembleAccuracy(
                    _fixture(), network_name=name, compute_type=mode, max_t=5, 
                    t_chunk_size=2, separate_probas=separate, weighted=weighted, 
                    t_range_drop_rate=drop, prediction_batch_size=cap, dtype=dtype, 
                    clf_acc_coef=coefficients[0], ctr_acc_coef=coefficients[1], 
                    clf_distil_acc_coef=coefficients[2]
                )
                np.testing.assert_array_equal(metric._select_timesteps(), _retained(drop, dtype))
                actual = metric.ensemble_predict(images, training=False).numpy()
                expected = _expected(images.numpy(), 3, name, coefficients, separate, weighted, drop, dtype)
                tolerance = 2e-6 if dtype == 'float32' else 2e-12
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
                exercised += 1
        self.assertEqual(exercised, 1344)

    def test_graph_dynamic_batches_class_counts_and_caps(self) -> None:
        """Check 32 traced configurations, each with both one-row and three-row input."""

        exercised = 0
        for mode, separate, cap, dtype, width in itertools.product(
            ('batched', 'chunked'), (False, True), (None, 3), 
            ('float32', 'float64'), (1, 3)
        ):
            with self.subTest(mode=mode, separate=separate, cap=cap, dtype=dtype, width=width):
                metric = EnsembleAccuracy(
                    _fixture(width), compute_type=mode, max_t=5, t_chunk_size=2, 
                    separate_probas=separate, weighted=True, t_range_drop_rate=0.4, 
                    prediction_batch_size=cap, dtype=dtype, 
                    clf_acc_coef=0.7, ctr_acc_coef=0.3, clf_distil_acc_coef=0.5
                )
                @tf.function(input_signature=[tf.TensorSpec((None, 1, 1, 1), tf.as_dtype(dtype))])
                def predict(images: tf.Tensor) -> tf.Tensor:
                    """Evaluate the captured ensemble under a symbolic batch length.

                    Args:
                        images (tf.Tensor): Float32/float64 [B,1,1,1] normalized
                            fixture pixels in the enclosing selected dtype.

                    Returns:
                        tf.Tensor: Same-dtype [B,width] scores for the captured
                            class count, cap, aggregation and noising settings.
                    """

                    return metric.ensemble_predict(images, training=False)
                for size in (1, 3):
                    images = tf.constant([-0.9, 0.2, 1.3][:size], tf.as_dtype(dtype))[:, None, None, None]
                    expected = _expected(images.numpy(), width, 'ema', (0.7, 0.3, 0.5), 
                                         separate, True, 0.4, dtype)
                    tolerance = 2e-6 if dtype == 'float32' else 2e-12
                    np.testing.assert_allclose(predict(images), expected, rtol=tolerance, atol=tolerance)
                self.assertEqual(predict.experimental_get_tracing_count(), 1)
                exercised += 1
        self.assertEqual(exercised, 32)

    def test_accuracy_labels_and_sample_weights_across_partial_batches(self) -> None:
        """Verify 64 label/branch/class/weight cases, including zero-weight and one-class data."""

        exercised = 0
        images = tf.constant([-0.9, 0.2, 1.3])[:, None, None, None]
        for name, dynamic, width, column, weights in itertools.product(
            ('raw', 'ema'), (False, True), (1, 3), (False, True), 
            (None, (1.0, 1.0, 1.0), (0.0, 0.5, 2.0), (0.0, 0.0, 0.0))
        ):
            with self.subTest(network=name, dynamic=dynamic, width=width, column=column, weights=weights):
                labels = np.array([0, width - 1, 0], dtype=np.int32)
                scores = _expected(images.numpy(), width, name, (1.0, 0.0, 0.0), 
                                   False, False, 0.0, 'float32')
                correct = (np.argmax(scores, axis=-1) == labels).astype(np.float64)
                row_weights = np.ones(3) if weights is None else np.asarray(weights)
                expected = np.sum(correct * row_weights) / row_weights.sum() if row_weights.sum() else 0.0
                original = np.array([40, 7, 99], np.int32)[labels] if dynamic else labels
                actual_labels = original[:, None] if column else original
                metric = EnsembleAccuracy(_fixture(width, dynamic), network_name=name, 
                                          max_t=5, prediction_batch_size=3)
                batches = []
                for start, stop in ((0, 2), (2, 3)):
                    batch = (images[start:stop], tf.constant(actual_labels[start:stop]))
                    batches.append(batch if weights is None else batch + tuple([row_weights[start:stop]]))
                self.assertAlmostEqual(float(metric.evaluate(batches, verbose=False)), expected, places=6)
                metric.reset_state()
                self.assertEqual(float(metric.result()), 0.0)
                exercised += 1
        self.assertEqual(exercised, 64)

    def test_optional_head_availability_is_consistent_across_modes(self) -> None:
        """Check all 56 availability/head-subset/mode cases against explicit required-head rules."""

        images = tf.zeros((2, 1, 1, 1))
        exercised = 0
        for regularizers, distilled, coefficients, mode in itertools.product(
            (False, True), (False, True), HEAD_COEFFICIENTS, ('batched', 'chunked')
        ):
            with self.subTest(regularizers=regularizers, distilled=distilled, heads=coefficients, mode=mode):
                metric = EnsembleAccuracy(
                    _fixture(regularizers=regularizers, distillation=distilled), 
                    compute_type=mode, max_t=5, t_chunk_size=2, 
                    clf_acc_coef=coefficients[0], ctr_acc_coef=coefficients[1], 
                    clf_distil_acc_coef=coefficients[2]
                )
                unavailable = ((coefficients[1] > 0 and not regularizers)
                               or (coefficients[2] > 0 and not distilled))
                # Reject only combinations whose positively weighted heads do not exist.
                if unavailable:
                    with self.assertRaisesRegex(ValueError, 'requires'):
                        metric.ensemble_predict(images, training=False)
                # Every available mixture still matches the independent weighted formula.
                else:
                    expected = _expected(images.numpy(), 3, 'ema', coefficients, 
                                         False, False, 0.0, 'float32')
                    np.testing.assert_allclose(metric.ensemble_predict(images, training=False), 
                                               expected, rtol=2e-6, atol=2e-6)
                exercised += 1
        self.assertEqual(exercised, 56)


# Allow focused command-line execution without running work during imports.
if __name__ == '__main__':
    unittest.main()
