"""Route configuration regressions for shared YAML and immutable overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from semantic_consolidation.config import RouteConfig, RouteSettings, _merge, load_route_config, validate_route_config


class RouteYamlTests(unittest.TestCase):
    """Retain ordinary safe YAML behavior when loading semantic experiments."""

    def _load_text(self, text: str) -> RouteConfig:
        """Load one route fragment against the existing common base.

        Args:
            text (str): YAML common/route override fragment.

        Returns:
            config (RouteConfig): Validated configuration loaded from the fragment.

        Raises:
            yaml.YAMLError: If the fragment repeats a key or has invalid syntax.
            ValueError: If resolved settings violate the route protocol.
        """
        base = Path(__file__).resolve().parents[1] / "configs/common_v1.yaml"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.yaml"
            path.write_text(f"base_config: '{base.as_posix()}'\n{text}", encoding="utf-8")
            return load_route_config(path)

    def test_yaml_merge_override_matches_common_loader_semantics(self) -> None:
        """An explicit local value may override a inherited YAML merge key."""
        config = self._load_text(
            "route:\n  <<: {batch_size: 8, temperature: 0.2}\n  batch_size: 4\n"
        )
        self.assertEqual(config.route.batch_size, 4)
        self.assertEqual(config.route.temperature, 0.2)

    def test_explicit_duplicate_keys_are_still_rejected(self) -> None:
        """Ordinary merge support must not permit silently repeated scientific keys."""
        fragments = (
            "route:\n  temperature: 0.2\n  temperature: 0.4\n",
            "common:\n  training:\n    epochs: 1\n    epochs: 2\n",
            "route:\n  <<: {temperature: 0.2, temperature: 0.4}\n",
        )
        for fragment in fragments:
            with self.subTest(fragment=fragment), self.assertRaisesRegex(yaml.YAMLError, "duplicate key"):
                self._load_text(fragment)

    def test_nested_overrides_do_not_mutate_the_base_or_share_sequences(self) -> None:
        """Study and direct route loading share one independent-copy merge implementation."""
        base = {"route": {"noise_levels": [0, 2], "temperature": 0.1}, "tags": ["base"]}
        overrides = {"route": {"temperature": 0.2}, "tags": ["condition"]}
        before = deepcopy(base), deepcopy(overrides)
        merged = _merge(base, overrides)
        self.assertEqual(merged["route"], {"noise_levels": [0, 2], "temperature": 0.2})
        merged["route"]["noise_levels"].append(3)
        merged["tags"].append("later")
        self.assertEqual((base, overrides), before)

    def test_tmcl_augmentation_settings_round_trip_through_yaml(self) -> None:
        """The scientific policy and view count are explicit serialized controls."""
        config = self._load_text(
            "route:\n  image_augmentation: tmcl\n  augmentation_views: 6\n"
        )
        self.assertEqual(config.route.image_augmentation, "tmcl")
        self.assertEqual(config.route.augmentation_views, 6)

    def test_augmentation_defaults_preserve_the_historical_api(self) -> None:
        """Constructing old settings retains the unaugmented paired-view control."""
        settings = RouteSettings()
        self.assertEqual(settings.image_augmentation, "none")
        self.assertEqual(settings.augmentation_views, 4)

    def test_invalid_augmentation_controls_are_rejected(self) -> None:
        """A misspelled policy or unusable count fails before any phase training."""
        for value in ("random", "TMCL", "", True):
            with self.subTest(policy=value), self.assertRaisesRegex(ValueError, "image_augmentation"):
                RouteSettings(image_augmentation=value)
        for value in (0, 1, -1, 2.5, True, "4", None):
            with self.subTest(views=value), self.assertRaisesRegex(ValueError, "augmentation_views"):
                RouteSettings(image_augmentation="tmcl", augmentation_views=value)

    def test_tmcl_rejects_incompatible_image_geometry(self) -> None:
        """RGB color operations and the fixed crop cannot silently alter MNIST or padding."""
        for override in (
            {"dataset": {"name": "mnist"}},
            {"dataset": {"pad": 2}},
            {"model": {"kwargs": {"channels": 1}}},
            {"model": {"kwargs": {"image_size": 28}}},
        ):
            fragment = yaml.safe_dump({"common": override, "route": {"image_augmentation": "tmcl"}})
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "32x32 RGB CIFAR"):
                self._load_text(fragment)

    def test_shipped_cifar_recipes_enable_tmcl_and_mnist_retains_none(self) -> None:
        """Maintained benchmark entry points select the applicable published image policy."""
        project = Path(__file__).resolve().parents[2]
        for directory in (project / "semantic_consolidation/configs", project / "notebooks/thesis/configs"):
            for name in ("cifar10", "cifar100"):
                with self.subTest(directory=directory, dataset=name):
                    route = load_route_config(directory / f"{name}.yaml").route
                    self.assertEqual(route.image_augmentation, "tmcl")
                    self.assertEqual(route.augmentation_views, 4)
        route = load_route_config(project / "semantic_consolidation/configs/smoke.yaml").route
        self.assertEqual(route.image_augmentation, "none")

    def test_typed_cifar_geometry_uses_dataset_dimensions(self) -> None:
        """Shared model construction replaces typed MNIST geometry with CIFAR RGB."""
        config = self._load_text("route:\n  image_augmentation: tmcl\n")
        config.common.model.kwargs = {}
        config.common.model.dit_classifier.classifier_mlp_ratio = 1
        config.common.model.diffusion_classifier.use_ema = False
        config.common.model.diffusion_classifier.test_noisified_max_timesteps = 0
        for name in (None, "dit_classifier"):
            with self.subTest(model_name=name):
                config.common.model.name = name
                validate_route_config(config)


# Direct execution runs only these configuration regressions.
if __name__ == "__main__":
    unittest.main()
