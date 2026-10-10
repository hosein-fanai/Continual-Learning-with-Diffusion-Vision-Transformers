"""Bind serialized specialist teachers to the semantic student factory."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from semantic_consolidation.config import load_route_config
from semantic_consolidation.runner import run


class SemanticSpecialistDispatchTests(TestCase):
    """Check teacher loading and student RNG reset before any model construction."""

    def test_artifact_teachers_reach_factory_after_seed_reset(self) -> None:
        """Semantic attachment cannot silently discard a fixed specialist descriptor."""

        root = Path(__file__).resolve().parents[2]
        config = load_route_config(root / "semantic_consolidation/configs/smoke.yaml")
        descriptors = {"classifier": {"format": "keras", "path": "fixed.keras"}}
        config.common.continually_learn.specialist_teacher_descriptors = descriptors
        classifier, noise = object(), object()
        events = []

        def load(descriptors: dict, seed: int) -> dict:
            """Record a teacher graph load without constructing test-only networks."""

            events.append("load")
            return {"classifier": classifier, "noise": noise}

        def factory(*args: object, **kwargs: object) -> object:
            """Stop exactly at student construction after recording the passed networks."""

            events.append("factory")
            self.assertIs(kwargs["classifier_teacher_network"], classifier)
            self.assertIs(kwargs["noise_teacher_network"], noise)
            raise RuntimeError("factory boundary verified")

        with patch("common.runtime.configure_runtime", side_effect=lambda **kwargs: events.append("seed")), \
             patch("common.dataloader.get_datasets", return_value=(object(), object())), \
             patch("semantic_consolidation.provenance.source_provenance", return_value={}), \
             patch("common.specialist_teacher_artifacts.load_specialist_teacher_descriptors", side_effect=load) as loader, \
             patch("common.model.get_model", side_effect=factory):
            with self.assertRaisesRegex(RuntimeError, "factory boundary verified"):
                run(config)
        loader.assert_called_once_with(descriptors, seed=config.common.training.seed)
        self.assertEqual(events, ["seed", "load", "seed", "factory"])
