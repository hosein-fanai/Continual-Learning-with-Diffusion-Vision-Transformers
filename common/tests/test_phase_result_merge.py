"""Regressions for public V2 phase-result merging and its training callers.

Small mappings and mocked phase execution exercise metric preservation without
training networks. The production merge and fit/evaluate dispatch methods run
directly; the progressive orchestration case also runs common.train.train_model.
"""

from __future__ import annotations

import unittest
from functools import partial
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock, patch

from common.train import train_model
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


def _model() -> MagicMock:
    """Provide V2 phase mocks while retaining the actual public merge behavior."""

    model = MagicMock(spec=DiffusionClassifierV2)
    model.merge_result_dicts.side_effect = partial(
        DiffusionClassifierV2.merge_result_dicts, model
    )
    model._test_part = None
    model.swap_noise_image = False
    return model


class PhaseResultMergeTests(unittest.TestCase):
    """Check that merged names preserve every value and leave source maps intact."""

    def test_default_names_preserve_history_inputs_and_value_identity(self) -> None:
        """Prefix shared metrics and preserve unique names with a shallow result."""

        generator = {"loss": [1.0], "noise_loss": [2.0]}
        discriminator = {"loss": [3.0], "classifier_accuracy": [0.8]}
        before = (dict(generator), dict(discriminator))

        result = _model().merge_result_dicts((generator, discriminator))

        self.assertEqual(result, {
            "generator_loss": [1.0], "noise_loss": [2.0],
            "discriminator_loss": [3.0], "classifier_accuracy": [0.8],
        })
        self.assertEqual((generator, discriminator), before)
        self.assertIs(result["generator_loss"], generator["loss"])
        self.assertIs(result["discriminator_loss"], discriminator["loss"])

    def test_absent_phases_preserve_name_alignment(self) -> None:
        """Discard None results without shifting prefixes or discarding empty maps."""

        model = _model()
        self.assertEqual(model.merge_result_dicts((None, {"loss": 2})), {"loss": 2})
        self.assertEqual(model.merge_result_dicts(({"loss": 1}, None)), {"loss": 1})
        self.assertEqual(model.merge_result_dicts((None, None)), {})
        self.assertEqual(model.merge_result_dicts(({}, {"loss": 2})), {"loss": 2})
        self.assertEqual(model.merge_result_dicts((), ()), {})
        self.assertEqual(
            model.merge_result_dicts(
                ({"loss": 1}, None, {"loss": 3}), ("first", "absent", "third")
            ),
            {"first_loss": 1, "third_loss": 3},
        )

    def test_partial_overlaps_across_three_phases(self) -> None:
        """Prefix each shared key only in the phases where it occurs."""

        mappings = ({"loss": 1}, {"loss": 2, "accuracy": 3}, {"accuracy": 4})
        before = tuple(dict(mapping) for mapping in mappings)
        result = _model().merge_result_dicts(mappings, ("a", "b", "c"))
        self.assertEqual(result, {
            "a_loss": 1, "b_loss": 2, "b_accuracy": 3, "c_accuracy": 4,
        })
        self.assertEqual(mappings, before)

    def test_same_mapping_in_two_phases_is_counted_twice(self) -> None:
        """Phase entries remain distinct when callers reuse one result object."""

        shared = {"loss": [1.0]}
        result = _model().merge_result_dicts((shared, shared))
        self.assertEqual(result, {"generator_loss": [1.0], "discriminator_loss": [1.0]})
        self.assertEqual(shared, {"loss": [1.0]})
        self.assertIs(result["generator_loss"], result["discriminator_loss"])

    def test_read_only_mappings_are_supported(self) -> None:
        """Merge Mapping inputs without requiring a mutable dictionary."""

        result = _model().merge_result_dicts((
            MappingProxyType({"loss": 1}), MappingProxyType({"loss": 2})
        ))
        self.assertEqual(result, {"generator_loss": 1, "discriminator_loss": 2})

    def test_misaligned_names_fail_before_skipping_or_mutating_inputs(self) -> None:
        """Reject missing or surplus names instead of silently truncating phases."""

        cases = (
            (({"loss": 1}, {"loss": 2}, {"third": 3}), ("generator", "discriminator")),
            (({"loss": 1}, {"loss": 2}), ("only",)),
            (({"loss": 1}, {"loss": 2}), ("a", "b", "surplus")),
            ((None, {"loss": 2}), ("only",)),
            ((None, None), ()),
        )
        for mappings, names in cases:
            before = tuple(None if item is None else dict(item) for item in mappings)
            with self.subTest(mappings=mappings, names=names), self.assertRaises(ValueError):
                _model().merge_result_dicts(mappings, names)
            self.assertEqual(mappings, before)

    def test_invalid_mapping_keys_and_prefixes_fail_clearly(self) -> None:
        """Reject nonmapping results and nonstring names at the public boundary."""

        cases = (
            ((1.0, {"loss": 2}), ("generator", "discriminator")),
            (([1.0], {"loss": 2}), ("generator", "discriminator")),
            (({1: 1}, {"loss": 2}), ("generator", "discriminator")),
            (({"loss": 1}, {"loss": 2}), (1, "discriminator")),
        )
        for mappings, names in cases:
            with self.subTest(mappings=mappings, names=names), self.assertRaises(TypeError):
                _model().merge_result_dicts(mappings, names)

    def test_ambiguous_output_names_raise_without_mutation(self) -> None:
        """Reject prefix collisions and duplicate phase names before any metric is lost."""

        cases = (
            (({"loss": 1, "generator_loss": 9}, {"loss": 2}), ("generator", "discriminator")),
            (({"loss": 1}, {"loss": 2, "generator_loss": 9}), ("generator", "discriminator")),
            (({"loss": 1}, {"loss": 2}), ("phase", "phase")),
        )
        for mappings, names in cases:
            before = tuple(dict(mapping) for mapping in mappings)
            with self.subTest(mappings=mappings, names=names), self.assertRaises(ValueError):
                _model().merge_result_dicts(mappings, names)
            self.assertEqual(mappings, before)


class PhaseMergeCallerTests(unittest.TestCase):
    """Exercise production callers against the public helper with isolated phase fits."""

    def test_combined_fit_uses_public_merge_and_preserves_histories(self) -> None:
        """Merge both Keras history mappings and retain their original loss entries."""

        model = _model()
        generator = {"loss": [1.0]}
        discriminator = {"loss": [2.0]}
        model.fit_generator.return_value = SimpleNamespace(history=generator)
        model.fit_discriminator.return_value = SimpleNamespace(history=discriminator)

        result = DiffusionClassifierV2.fit(
            model, gen_kwargs={"epochs": 2}, clf_kwargs={"epochs": 3}
        )

        self.assertEqual(result, {"generator_loss": [1.0], "discriminator_loss": [2.0]})
        model.fit_generator.assert_called_once_with(epochs=2)
        model.fit_discriminator.assert_called_once_with(epochs=3)
        model.merge_result_dicts.assert_called_once_with((generator, discriminator))
        self.assertEqual(generator, {"loss": [1.0]})
        self.assertEqual(discriminator, {"loss": [2.0]})

    def test_combined_evaluation_forces_dictionaries_and_uses_public_merge(self) -> None:
        """Combine both phase metrics even when the caller requests scalar results."""

        model = _model()
        generator = {"loss": 1.0}
        discriminator = {"loss": 2.0}
        model.evaluate_generator.return_value = generator
        model.evaluate_discriminator.return_value = discriminator
        dataset = object()

        result = DiffusionClassifierV2.evaluate(
            model, eval_both=True, x=dataset, return_dict=False, verbose=0
        )

        self.assertEqual(result, {"generator_loss": 1.0, "discriminator_loss": 2.0})
        model.evaluate_generator.assert_called_once_with(x=dataset, return_dict=True, verbose=0)
        model.evaluate_discriminator.assert_called_once_with(x=dataset, return_dict=True, verbose=0)
        model.merge_result_dicts.assert_called_once_with((generator, discriminator))
        self.assertEqual(generator, {"loss": 1.0})
        self.assertEqual(discriminator, {"loss": 2.0})

    def test_single_phase_evaluation_retains_unprefixed_metrics(self) -> None:
        """A discriminator-only evaluation handles the absent generator result."""

        model = _model()
        model._test_part = "discriminator"
        model.evaluate_discriminator.return_value = {"loss": 2.0}

        result = DiffusionClassifierV2.evaluate(model, verbose=0)

        self.assertEqual(result, {"loss": 2.0})
        model.evaluate_generator.assert_not_called()
        model.evaluate_discriminator.assert_called_once_with(verbose=0, return_dict=True)
        model.merge_result_dicts.assert_called_once_with((None, {"loss": 2.0}))

    def test_progressive_training_merges_through_public_helper(self) -> None:
        """The common trainer merges progressive and ordinary phase histories."""

        model = _model()
        generator = {"loss": [1.0]}
        discriminator = {"loss": [2.0]}
        model.fit_generator_progressively.return_value = SimpleNamespace(history=generator)
        model.fit_discriminator.return_value = SimpleNamespace(history=discriminator)
        dataset = object()

        with patch("common.train.validate_progressive_classifier_growth"), patch(
            "common.train.ImageGenerator", return_value=SimpleNamespace(results_path=None)
        ):
            result = train_model(
                model=model, trainset=dataset, fit_method="fit_progressively",
                fit_kwargs={"stage_tasks": ["timesteps"], "stage_epochs": 2},
                epochs=3, show_images=True, report_every_epoch=False,
                save_weights=False, save_config_=False, tensorboard=False,
                patience=0, verbose=0,
            )

        self.assertEqual(result, {"generator_loss": [1.0], "discriminator_loss": [2.0]})
        model.merge_result_dicts.assert_called_once_with(
            (generator, discriminator), ("generator", "discriminator")
        )
        self.assertEqual(model.fit_generator_progressively.call_args.kwargs["stage_epochs"], 2)
        self.assertNotIn("stage_epochs", model.fit_discriminator.call_args.kwargs)
        self.assertEqual(model.fit_discriminator.call_args.kwargs["epochs"], 3)
        self.assertEqual(generator, {"loss": [1.0]})
        self.assertEqual(discriminator, {"loss": [2.0]})
