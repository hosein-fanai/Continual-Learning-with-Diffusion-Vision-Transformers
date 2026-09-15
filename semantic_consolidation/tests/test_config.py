"""Route configuration regressions for shared YAML and immutable overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from semantic_consolidation.config import RouteConfig, _merge, load_route_config


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


# Direct execution runs only these configuration regressions.
if __name__ == "__main__":
    unittest.main()
