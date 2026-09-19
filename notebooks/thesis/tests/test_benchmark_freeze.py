"""Test-informed freezes retain native authentication and explicit interpretation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.experiment import materialize_run_plan, validate_frozen_confirmation
from notebooks.thesis import results_package, workflow
from notebooks.thesis.tests.test_recipe import SEEDS, TEMPLATES
from semantic_consolidation.config import load_route_config
from semantic_consolidation.study import validate_planned_config


PROVENANCE = {
    "test_informed": True,
    "independent_confirmation": False,
    "reason": "Synthetic fixture: prior official-test selection informed the recipe.",
}


class BenchmarkFreezeTests(unittest.TestCase):
    """Keep known selection exposure inseparable from the authenticated plan."""

    def test_all_runs_authenticate_benchmark_phase_settings_and_disclosure(self) -> None:
        """Materialize every declared stream without training or hiding test exposure."""
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            frozen = workflow.prepare_campaign(
                campaign, TEMPLATES, SEEDS, phase="benchmark",
                selection_provenance=PROVENANCE,
            )
            record, manifests = workflow._campaign(frozen)
            self.assertEqual(record["phase"], "benchmark")
            self.assertEqual(record["selection_provenance"], PROVENANCE)
            self.assertIn("notebooks/thesis/benchmark_selection.json", record["bound_files"])
            total = 0
            for dataset, manifest in manifests.items():
                self.assertEqual(manifest["phase"], "benchmark")
                with self.assertRaises(ValueError):
                    validate_frozen_confirmation(manifest, expected_hash=manifest["manifest_hash"])
                for entry in materialize_run_plan(manifest):
                    config = load_route_config(campaign / dataset / f"{entry['run_id']}.yaml")
                    validate_planned_config(config)
                    self.assertEqual(config.common.continually_learn.experiment_phase, "benchmark")
                    self.assertEqual(config.common.hpo["selection_provenance"], PROVENANCE)
                    self.assertEqual(config.common.optimizer.initial_learning_rate, .001)
                    weights = config.common.continually_learn.ensemble_accuracy_kwargs
                    self.assertEqual((weights["clf_acc_coef"], weights["clf_distil_acc_coef"]), (1., 0.))
                    total += 1
            self.assertEqual(total, 24)
            self.assertEqual(len(workflow.campaign_checklist(frozen)), 24)
            with patch.object(workflow, "_initialize", side_effect=lambda config, context: (config, context)):
                config, _ = workflow.load_run(frozen, "cifar10", "learned", repeat_index=None)
                self.assertEqual(config.common.continually_learn.experiment_phase, "benchmark")
            self.assertFalse(list(campaign.rglob("*.started.json")))
            self.assertFalse(list(campaign.rglob("*.completed.json")))
            context = results_package._context(record, manifests, {"runs": []}, "PROGRESS")
            self.assertIn("TEST-INFORMED BENCHMARK, not independent confirmation", context)
            self.assertIn("do not account for prior test-based selection", context)

            original = json.loads(frozen.read_text(encoding="utf-8"))
            for field, value, message in (
                ("phase", "confirmation", "phase differs"),
                ("selection_provenance", {}, "provenance differs"),
            ):
                with self.subTest(field=field):
                    altered = deepcopy(original)
                    altered[field] = value
                    frozen.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        workflow._campaign(frozen)
            frozen.write_text(json.dumps(original), encoding="utf-8")
            workflow._campaign(frozen)

    def test_benchmark_requires_explicit_disclosure_before_creating_artifacts(self) -> None:
        """Reject missing or contradictory disclosures before creating a campaign."""
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            for provenance in (None, {}, {**PROVENANCE, "independent_confirmation": True},
                               {**PROVENANCE, "reason": ""}):
                with self.subTest(provenance=provenance), self.assertRaisesRegex(ValueError, "provenance"):
                    workflow.prepare_campaign(campaign, TEMPLATES, SEEDS, phase="benchmark",
                                              selection_provenance=provenance)
                self.assertFalse(campaign.exists())

    def test_known_test_selection_cannot_be_frozen_as_confirmation(self) -> None:
        """Keep test-informed recipes outside the independent confirmation mode."""
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            with self.assertRaisesRegex(ValueError, "selected on the official test set"):
                workflow.prepare_campaign(campaign, TEMPLATES, SEEDS, selection_provenance=PROVENANCE)
            self.assertFalse(campaign.exists())

    def test_benchmark_does_not_allow_test_rows_for_live_validation(self) -> None:
        """Historical test exposure does not authorize live test-based selection."""
        configs = {name: load_route_config(path) for name, path in TEMPLATES.items()}
        configs["cifar10"].common.dataset.validation_source = "test"
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            with patch.object(workflow, "load_route_config", side_effect=lambda path: configs[Path(path).stem]), \
                    self.assertRaisesRegex(ValueError, "training-data validation split"):
                workflow.prepare_campaign(campaign, TEMPLATES, SEEDS, phase="benchmark",
                                          selection_provenance=PROVENANCE)
            self.assertFalse(campaign.exists())

    def test_caller_cannot_erase_recorded_test_selection(self) -> None:
        """Check template provenance before accepting supplied selection metadata."""
        configs = {name: load_route_config(path) for name, path in TEMPLATES.items()}
        configs["cifar10"].common.hpo["selection_provenance"] = deepcopy(PROVENANCE)
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            with patch.object(workflow, "load_route_config", side_effect=lambda path: configs[Path(path).stem]), \
                    self.assertRaisesRegex(ValueError, "selected on the official test set"):
                workflow.prepare_campaign(campaign, TEMPLATES, SEEDS,
                                          selection_provenance={"test_informed": False})
            self.assertFalse(campaign.exists())


# Support standalone execution as well as discovery.
if __name__ == "__main__":
    unittest.main()
