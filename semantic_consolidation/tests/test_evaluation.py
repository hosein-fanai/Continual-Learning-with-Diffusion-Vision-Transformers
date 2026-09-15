"""Analytic probability, calibration, cost and real-network inference checks."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

import numpy as np
import tensorflow as tf

from common.mechanistic import calibration_metrics
from common.model import get_model
from semantic_consolidation.evaluation import (
    EnsembleEvaluationSettings, _predict, evaluate_checkpoint,
    fit_temperature, temperature_scale,
)


class _Network:
    """A counted classifier with analytic heads and the existing full-return API."""

    num_classes = 2
    num_labels = 3
    use_cfg = True
    compute_dtype = "float32"

    def __init__(self, conditional: bool = False, input_dependent: bool = False) -> None:
        """Initialize counted deterministic heads and one tracked scalar weight."""

        self.conditional = conditional
        self.input_dependent = input_dependent
        self.calls = 0
        self.examples = 0
        self.weights = [tf.Variable(0.7, trainable=True)]

    def predict_class(self, inputs: tuple, max_encoder_num: object = None,
                      full_return: bool = False, training: bool = False) -> object:
        """Count calls and return analytic distributions in the project head format."""

        # Evaluation must never enable training behavior in any head.
        if training is not False:
            raise AssertionError("The evaluator must always select inference mode.")
        x, times, labels = inputs
        count = int(tf.shape(x)[0])
        self.calls += 1
        self.examples += count
        primary = tf.tile([[0.8, 0.2]], [count, 1])
        # Conditional fixture rows make the candidate diagonal analytically known.
        if self.conditional:
            primary = tf.gather(tf.constant([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]), labels)
        # Input-dependent logits expose changes in stateless noising streams.
        if self.input_dependent:
            logits = tf.reduce_mean(x, axis=(1, 2, 3)) * self.weights[0] + tf.cast(times, tf.float32) * 0.01
            primary = tf.nn.softmax(tf.stack([logits, -logits], axis=1), axis=1)
        distillation = tf.tile([[0.2, 0.8]], [count, 1])
        regularizers = [tf.tile([[0.4, 0.6]], [count, 1]), None, tf.tile([[0.6, 0.4]], [count, 1])]
        return (primary, None, None, regularizers, None, distillation) if full_return else primary


class _Wrapper:
    """Minimal existing wrapper interface for counted probability fixtures."""

    timesteps = 8
    use_ema = False
    seen_classes = {4: 0, 7: 1}
    seed = 17

    def __init__(self, conditional: bool = False, input_dependent: bool = False) -> None:
        """Bind the analytic classifier with the selected fixture behavior."""

        self.network = _Network(conditional, input_dependent)

    def get_network(self, name: str) -> _Network:
        """Resolve the only available raw checkpoint and reject a branch mismatch."""

        # A missing EMA branch must not pass by silently returning raw weights.
        if name != "raw":
            raise AssertionError("Tests require the selected raw checkpoint.")
        return self.network

    def q_sample(self, x: tf.Tensor, times: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        """Apply a known affine noisy view without consuming stateful randomness."""

        return x + noise * 0.2

    def get_noise_and_signal_rates(self, times: tf.Tensor) -> tuple:
        """Return fixed signal/noise rates for the existing SNR weighting path."""

        return tf.ones_like(times, dtype=tf.float32) * 0.8, tf.ones_like(times, dtype=tf.float32) * 0.6


def _digest(network: object) -> str:
    """Hash all fixture weights for exact checkpoint-state invariance checks."""

    digest = hashlib.sha256()
    for variable in network.weights:
        value = variable.numpy()
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class ProbabilityTests(unittest.TestCase):
    """Check probability mixtures and scalar calibration against analytic results."""

    def test_head_weights_form_analytic_unit_mass_mixture(self) -> None:
        """Verify normalized branch weights and averaging of available regularizers."""

        wrapper = _Wrapper()
        settings = EnsembleEvaluationSettings(enabled=True, horizons=(2,), head_weights=(2., 1., 1.))
        probabilities, _ = _predict(wrapper, np.zeros((3, 2, 2, 1), "float32"), settings, 2, "validation")
        expected = np.tile([[0.575, 0.425]], (3, 1))
        np.testing.assert_allclose(probabilities, expected, atol=1e-7)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
        self.assertEqual(settings.normalized_head_weights, (0.5, 0.25, 0.25))

    def test_candidate_diagonal_uses_existing_explicit_softmax(self) -> None:
        """Verify null-plus-diagonal scores use the existing declared softmax."""

        wrapper = _Wrapper(conditional=True)
        settings = EnsembleEvaluationSettings(enabled=True, horizons=(1,), separate_probas=True)
        probabilities, cost = _predict(wrapper, np.zeros((2, 2, 2, 1), "float32"), settings, 1, "validation")
        expected = tf.nn.softmax([[1.1, 0.6]]).numpy()
        np.testing.assert_allclose(probabilities, np.repeat(expected, 2, axis=0), atol=1e-7)
        self.assertEqual(cost["candidate_condition_factor"], 3)
        self.assertEqual(cost["example_forwards"], 6)
        self.assertEqual(wrapper.network.examples, 6)

    def test_invalid_weights_and_budgets_fail_before_inference(self) -> None:
        """Reject undefined mixtures, nonfinite controls and invalid resource counts."""

        for values in (
            {"head_weights": (1., -0.1, 0.)}, {"head_weights": (0., 0., 0.)},
            {"head_weights": (1., np.nan, 0.)}, {"horizons": (0,)},
            {"horizons": (1, 1)}, {"noise_draws": 0}, {"seed": None},
            {"temperature_bounds": (0., 1.)}, {"temperature_bounds": (2., 4.)},
            {"batch_size": 2.5}, {"calibration_fraction": np.nan},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EnsembleEvaluationSettings(**values)

    def test_analytic_temperature_optimum_and_order_preservation(self) -> None:
        """Recover the exact temperature matching three correct targets out of four."""

        probabilities = np.tile([[0.9, 0.1]], (4, 1))
        labels = np.asarray([0, 0, 0, 1])
        fitted = fit_temperature(probabilities, labels)
        expected = np.log(9.) / np.log(3.)
        self.assertAlmostEqual(fitted["temperature"], expected, places=10)
        scaled = temperature_scale(probabilities, fitted["temperature"])
        np.testing.assert_allclose(scaled, np.tile([[0.75, 0.25]], (4, 1)), atol=1e-12)
        self.assertLess(fitted["nll_after"], fitted["nll_before"])
        np.testing.assert_array_equal(np.argmax(scaled, axis=1), np.argmax(probabilities, axis=1))
        np.testing.assert_array_equal(temperature_scale(probabilities, 1.), probabilities)
        with self.assertRaisesRegex(ValueError, "validation"):
            fit_temperature(probabilities, labels, split="test")
        with self.assertRaises(ValueError):
            fit_temperature(probabilities, labels + 0.5)

    def test_temperature_boundaries_uniform_and_zero_support(self) -> None:
        """Keep uniform/zero-support predictors and boundary optima well defined."""

        uniform = np.full((3, 2), 0.5)
        self.assertEqual(fit_temperature(uniform, [0, 1, 0])["temperature"], 1.)
        correct = fit_temperature(np.tile([[0.8, 0.2]], (3, 1)), [0, 0, 0])
        self.assertEqual(correct["temperature"], 0.05)
        self.assertTrue(correct["at_boundary"])
        scaled = temperature_scale(np.asarray([[1., 0.]]), 2.)
        self.assertTrue(np.isfinite(scaled).all())
        self.assertAlmostEqual(float(scaled.sum()), 1.)


class CheckpointEvaluationTests(unittest.TestCase):
    """Audit inference cost, calibration separation and checkpoint immutability."""

    @staticmethod
    def _inputs(count: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """Construct small deterministic images and alternating class targets."""

        return np.linspace(-1., 1., count * 4, dtype="float32").reshape(count, 2, 2, 1), np.arange(count) % 2

    def test_costs_match_actual_calls_with_candidate_class_factor(self) -> None:
        """Match reported classifier calls and processed rows to actual counters."""

        wrapper = _Wrapper(conditional=True)
        samples, labels = self._inputs(5)
        settings = EnsembleEvaluationSettings(
            enabled=True, horizons=(1, 3), batch_size=2, noise_draws=2,
            separate_probas=True, t_chunk_size=2,
        )
        result = evaluate_checkpoint(wrapper, samples, labels, settings)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        clean, first, third = result["variants"]
        self.assertEqual(clean["evaluation_cost"]["example_forwards"], 5)
        self.assertEqual(clean["evaluation_cost"]["network_forward_calls"], 3)
        self.assertEqual(first["evaluation_cost"]["example_forwards"], 30)
        self.assertEqual(third["evaluation_cost"]["example_forwards"], 90)
        self.assertEqual(third["evaluation_cost"]["network_forward_calls"], 12)
        self.assertEqual(sum(record["evaluation_cost"]["network_forward_calls"] for record in result["variants"]), wrapper.network.calls)
        self.assertEqual(sum(record["evaluation_cost"]["example_forwards"] for record in result["variants"]), wrapper.network.examples)
        for record in result["variants"]:
            self.assertGreater(record["evaluation_cost"]["latency_seconds"], 0.)

    def test_validation_partition_and_calibration_are_disjoint_and_counted(self) -> None:
        """Keep calibration/report indices disjoint and include both inference costs."""

        wrapper = _Wrapper(input_dependent=True)
        samples, labels = self._inputs()
        settings = EnsembleEvaluationSettings(enabled=True, horizons=(1, 2), batch_size=2, calibration_fraction=0.5)
        result = evaluate_checkpoint(wrapper, samples, labels, settings)
        self.assertEqual(len(result["evaluation_indices"]), 4)
        self.assertEqual(len(result["calibration_indices"]), 4)
        self.assertTrue(set(result["evaluation_indices"]).isdisjoint(result["calibration_indices"]))
        self.assertEqual(result["evaluation_class_counts"], {"0": 2, "1": 2})
        counted_calls, counted_examples = 0, 0
        for variant in result["variants"]:
            self.assertEqual(variant["temperature_fit"]["sample_count"], 4)
            self.assertEqual(variant["temperature_fit"]["class_counts"], {"0": 2, "1": 2})
            for name in ("evaluation_cost", "calibration_cost"):
                counted_calls += variant[name]["network_forward_calls"]
                counted_examples += variant[name]["example_forwards"]
        self.assertEqual(counted_calls, wrapper.network.calls)
        self.assertEqual(counted_examples, wrapper.network.examples)

    def test_test_labels_cannot_fit_temperature(self) -> None:
        """Changing test targets must not change independently fitted temperatures."""

        samples, labels = self._inputs()
        settings = EnsembleEvaluationSettings(enabled=True, horizons=(1,), calibration_fraction=0.5)
        with self.assertRaisesRegex(ValueError, "separate held-out validation"):
            evaluate_checkpoint(_Wrapper(), samples, labels, settings, split="test")
        calibration = samples.copy() + 0.01
        first = evaluate_checkpoint(_Wrapper(), samples, labels, settings, split="test",
                                    calibration_samples=calibration, calibration_labels=labels)
        second = evaluate_checkpoint(_Wrapper(), samples, 1 - labels, settings, split="test",
                                     calibration_samples=calibration, calibration_labels=labels)
        for a, b in zip(first["variants"], second["variants"]):
            self.assertEqual(a["temperature_fit"]["temperature"], b["temperature_fit"]["temperature"])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate_checkpoint(_Wrapper(), samples, labels, settings,
                                calibration_samples=samples, calibration_labels=labels)
        with self.assertRaisesRegex(ValueError, "validation"):
            evaluate_checkpoint(_Wrapper(), samples, labels, settings,
                                calibration_samples=calibration, calibration_labels=labels, calibration_split="test")

    def test_deterministic_compute_modes_and_global_rng_state(self) -> None:
        """Check repeated/chunked/batched equality without advancing the global RNG."""

        wrapper = _Wrapper(input_dependent=True)
        samples, _ = self._inputs()
        settings = EnsembleEvaluationSettings(enabled=True, horizons=(3,), noise_draws=2, batch_size=3, t_chunk_size=1)
        before = _digest(wrapper.network)
        tf.random.set_seed(121)
        expected_random = tf.random.uniform((4,)).numpy()
        tf.random.set_seed(121)
        first, _ = _predict(wrapper, samples, settings, 3, "validation")
        second, _ = _predict(wrapper, samples, settings, 3, "validation")
        batched, _ = _predict(wrapper, samples, replace(settings, compute_type="batched"), 3, "validation")
        np.testing.assert_array_equal(tf.random.uniform((4,)).numpy(), expected_random)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, batched, rtol=1e-6, atol=1e-7)
        self.assertEqual(before, _digest(wrapper.network))

    def test_calibration_and_class_support_fail_closed(self) -> None:
        """Reject singleton splitting, absent EMA, excessive horizons and future logits."""

        samples, labels = self._inputs(2)
        settings = EnsembleEvaluationSettings(enabled=True, horizons=(1,), calibration_fraction=0.5)
        with self.assertRaisesRegex(ValueError, "two validation rows"):
            evaluate_checkpoint(_Wrapper(), samples, labels, settings)
        with self.assertRaisesRegex(ValueError, "actual EMA"):
            evaluate_checkpoint(_Wrapper(), samples, labels, replace(settings, network_name="ema"))
        with self.assertRaisesRegex(ValueError, "schedule"):
            evaluate_checkpoint(_Wrapper(), samples, labels, replace(settings, horizons=(9,)))
        wrapper = _Wrapper()
        wrapper.seen_classes = {4: 0}
        with self.assertRaisesRegex(ValueError, "seen-class support"):
            evaluate_checkpoint(wrapper, samples, labels, settings)

    def test_real_classifier_checkpoint_is_unchanged(self) -> None:
        """Exercise a real tiny classifier and preserve every weight and optimizer step."""

        wrapper = get_model(
            model_name="dit_classifier", task="joint", image_shape=(4, 4, 1),
            class_num=2, seed=29, dtype_policy="float32", show_network_summary=False,
            model_kwargs={
                "timesteps": 4, "patch_size": 2, "dim": 4, "depth": 1,
                "mha_num_heads": 1, "vit_block_mlp_ratio": 1.,
                "clf_mha_num_heads": 1, "clf_vit_block_mlp_ratio": 1.,
                "classifier_mlp_ratio": 1, "compile_args": {"run_eagerly": True},
            },
            wrapper_kwargs={"use_ema": False, "p_uncond": 1., "test_steps": 2},
        )
        samples = np.linspace(-1., 1., 4 * 4 * 4, dtype="float32").reshape(4, 4, 4, 1)
        labels = np.asarray([0, 1, 0, 1])
        settings = EnsembleEvaluationSettings(enabled=True, horizons=(1, 3), batch_size=2, t_chunk_size=1)
        before = _digest(wrapper.network)
        optimizer_before = int(wrapper.optimizer.iterations.numpy())
        first = evaluate_checkpoint(wrapper, samples, labels, settings)
        second = evaluate_checkpoint(wrapper, samples, labels, settings)
        self.assertEqual(before, _digest(wrapper.network))
        self.assertEqual(optimizer_before, int(wrapper.optimizer.iterations.numpy()))
        for a, b in zip(first["variants"], second["variants"]):
            self.assertEqual(a["metrics"], b["metrics"])
        zero = tf.zeros((len(samples),), dtype=tf.int32)
        clean = wrapper.network.predict_class((samples, zero, zero), max_encoder_num=None, training=False).numpy()
        expected = calibration_metrics(clean, labels)
        for name, value in first["variants"][0]["metrics"].items():
            self.assertAlmostEqual(value, expected[name], places=6)


# Run the focused tests only when this file is invoked directly.
if __name__ == "__main__":
    unittest.main()
