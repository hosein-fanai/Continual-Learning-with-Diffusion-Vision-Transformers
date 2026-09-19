"""Authenticate the 21-stream notebook scope without starting a training stream."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.experiment import materialize_run_plan
from notebooks.thesis import workflow
from semantic_consolidation.config import load_route_config, primary_accuracy_matrix_name
from semantic_consolidation.study import validate_planned_config


NOTEBOOKS = Path(__file__).resolve().parents[1]
TEMPLATES = {name: NOTEBOOKS / "configs" / f"{name}.yaml" for name in ("cifar10", "cifar100")}
SEEDS = [1103, 2207, 3301]
PROVENANCE = {
    "test_informed": True,
    "independent_confirmation": False,
    "reason": "Synthetic fixture: recorded test-informed selection is retained.",
}
SELECTED = {
    "cifar10": ["extra_joint", "learned"],
    "cifar100": ["baseline", "extra_joint", "learned", "random", "ce_only"],
}
EXPECTED_NOTEBOOKS = {
    ("cifar10", "extra_joint"): "03_CIFAR10_extra_joint.ipynb",
    ("cifar10", "learned"): "04_CIFAR10_learned.ipynb",
    ("cifar100", "baseline"): "05_CIFAR100_platform.ipynb",
    ("cifar100", "extra_joint"): "06_CIFAR100_extra_joint.ipynb",
    ("cifar100", "learned"): "07_CIFAR100_learned.ipynb",
    ("cifar100", "random"): "08_CIFAR100_random.ipynb",
    ("cifar100", "ce_only"): "09_CIFAR100_ce_only.ipynb",
}


class CampaignScopeTests(unittest.TestCase):
    """Exercise real native plans, explicit notebook mapping, and tamper guards."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare both supported scopes as temporary designs with no data loading."""
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-campaign-scope-")
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.baseline_bytes = (NOTEBOOKS / "02_CIFAR10_platform.ipynb").read_bytes()
        cls.frozen = workflow.prepare_campaign(
            cls.directory / "reduced", TEMPLATES, SEEDS, phase="benchmark",
            selection_provenance=PROVENANCE, scope="notebooks_03_09",
        )
        cls.legacy = workflow.prepare_campaign(
            cls.directory / "legacy", TEMPLATES, SEEDS, phase="benchmark",
            selection_provenance=PROVENANCE,
        )
        cls.record, cls.manifests = workflow._campaign(cls.frozen)
        cls.original = json.loads(cls.frozen.read_text(encoding="utf-8"))

    def _write_altered(self, record: dict) -> Path:
        """Write a separate record beside the fixture without changing its native files."""
        altered = self.frozen.parent / "altered_design.json"
        altered.write_text(json.dumps(record), encoding="utf-8")
        return altered

    def test_native_plan_has_exact_21_paired_streams_and_resolved_configs(self) -> None:
        """Keep all three independent paired seeds and full native recipes for seven methods."""
        self.assertEqual(self.record["campaign_scope"], "notebooks_03_09")
        self.assertEqual(self.record["campaign_version"], workflow.CAMPAIGN_VERSION)
        self.assertEqual(self.record["declared_conditions"], SELECTED)
        self.assertEqual(self.record["declared_stream_count"], 21)
        total = 0
        for dataset, manifest in self.manifests.items():
            plan = materialize_run_plan(manifest)
            total += len(plan)
            self.assertEqual(Counter(entry["condition"] for entry in plan),
                             {condition: 3 for condition in SELECTED[dataset]})
            self.assertEqual({(entry["condition"], entry["stream"]["stream_seed"]) for entry in plan},
                             {(condition, seed) for condition in SELECTED[dataset] for seed in SEEDS})
            self.assertEqual(manifest["spec"]["analysis_spec"]["condition_a"], "learned")
            self.assertEqual(manifest["spec"]["analysis_spec"]["condition_b"], "extra_joint")
            streams = {}
            for entry in plan:
                config = load_route_config(self.frozen.parent / dataset / f"{entry['run_id']}.yaml")
                validate_planned_config(config)
                self.assertEqual(config.common.continually_learn.experiment_phase, "benchmark")
                self.assertEqual(config.route.condition, workflow.CONDITIONS[dataset][entry["condition"]]["route"]["condition"])
                self.assertEqual(config.common.training.seed, entry["stream"]["stream_seed"])
                self.assertEqual(config.common.continually_learn.seed, entry["stream"]["stream_seed"])
                self.assertEqual(config.route.seed, entry["stream"]["stream_seed"])
                self.assertEqual(config.common.continually_learn.class_order, entry["stream"]["class_order"])
                self.assertEqual(config.common.continually_learn.task_groups, entry["stream"]["task_groups"])
                self.assertEqual(streams.setdefault(entry["block_id"], entry["stream"]), entry["stream"])
                policy = self.record["inference_policy"][dataset]
                self.assertEqual(policy["accuracy_matrix"], primary_accuracy_matrix_name(config))
                self.assertEqual(policy["ensemble_accuracy_kwargs"], config.common.continually_learn.ensemble_accuracy_kwargs)
                self.assertEqual(policy["evaluate_on"], "official_test_after_each_task")
            self.assertEqual(len(streams), 3)
        self.assertEqual(total, 21)
        self.assertFalse(list(self.directory.rglob("*.started.json")))
        self.assertFalse(list(self.directory.rglob("*.completed.json")))

    def test_checklist_and_bindings_select_only_notebooks_03_through_09(self) -> None:
        """Every notebook maps to its own native condition with three repeats and no 02 binding."""
        checklist = workflow.campaign_checklist(self.frozen, save=False)
        self.assertEqual(len(checklist), 21)
        self.assertEqual(Counter(checklist["notebook"]), {name: 3 for name in EXPECTED_NOTEBOOKS.values()})
        self.assertEqual({(row.dataset, row.condition): row.notebook for row in checklist.itertuples()}, EXPECTED_NOTEBOOKS)
        for _, rows in checklist.groupby("notebook"):
            self.assertEqual(set(rows["seed"]), set(SEEDS))
        self.assertEqual({path for path in self.record["bound_files"] if path.endswith(".ipynb")},
                         {f"notebooks/thesis/{name}" for name in EXPECTED_NOTEBOOKS.values()})
        self.assertEqual(self.record["completion_policy"]["required_streams"], 21)
        self.assertEqual(self.record["completion_policy"]["seeds_per_notebook"], 3)
        self.assertEqual(self.record["completion_policy"]["notebooks"], sorted(EXPECTED_NOTEBOOKS.values()))
        self.assertEqual((NOTEBOOKS / "02_CIFAR10_platform.ipynb").read_bytes(), self.baseline_bytes)

    def test_unplanned_baseline_fails_before_completion_or_runtime_initialization(self) -> None:
        """Report an excluded method as unplanned instead of incorrectly calling it complete."""
        for repeat in (None, 0):
            with self.subTest(repeat=repeat), patch.object(workflow, "_outputs") as outputs, \
                    patch.object(workflow, "_initialize") as initialize:
                with self.assertRaisesRegex(ValueError, "cifar10/baseline is not planned"):
                    workflow.load_run(self.frozen, "cifar10", "baseline", repeat_index=repeat)
                outputs.assert_not_called()
                initialize.assert_not_called()
        self.assertIn("baseline", workflow.CONDITIONS["cifar10"])

    def test_legacy_default_and_unscoped_24_records_remain_supported(self) -> None:
        """Preserve the historical default and accept old records without new policy fields."""
        record, manifests = workflow._campaign(self.legacy)
        self.assertEqual(record["campaign_version"], workflow.LEGACY_CAMPAIGN_VERSION)
        self.assertEqual(record["campaign_scope"], "notebooks_02_09")
        self.assertEqual(sum(len(materialize_run_plan(manifest)) for manifest in manifests.values()), 24)
        self.assertEqual(len(workflow.campaign_checklist(self.legacy, save=False)), 24)
        old = json.loads(self.legacy.read_text(encoding="utf-8"))
        for key in ("campaign_scope", "declared_conditions", "inference_policy", "completion_policy"):
            old.pop(key)
        old_path = self.legacy.parent / "historical_design.json"
        old_path.write_text(json.dumps(old), encoding="utf-8")
        _, old_manifests = workflow._campaign(old_path)
        self.assertEqual(old_manifests, manifests)

    def test_tampered_scope_count_conditions_seeds_and_policies_are_rejected(self) -> None:
        """Authenticate declarations against the fixed scope and native scientific settings."""
        for field, value, message in (
            ("declared_stream_count", 24, "scope declarations"),
            ("declared_conditions", {**SELECTED, "cifar10": ["baseline", "learned"]}, "scope declarations"),
            ("campaign_version", workflow.LEGACY_CAMPAIGN_VERSION, "scope declarations"),
            ("campaign_scope", "notebooks_02_09", "scope declarations"),
            ("seeds", [1103, 2207, 17], "scope declarations"),
            ("inference_policy", {}, "policy differs"),
            ("completion_policy", {"required_streams": 0}, "policy differs"),
        ):
            with self.subTest(field=field):
                altered = deepcopy(self.original)
                altered[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    workflow._campaign(self._write_altered(altered))
        altered = deepcopy(self.original)
        for key in ("campaign_scope", "declared_conditions", "inference_policy", "completion_policy"):
            altered.pop(key)
        altered["campaign_version"] = workflow.LEGACY_CAMPAIGN_VERSION
        with self.assertRaisesRegex(ValueError, "full 24-stream"):
            workflow._campaign(self._write_altered(altered))

    def test_valid_native_manifest_with_wrong_condition_membership_is_rejected(self) -> None:
        """A fully authenticated 24-scope C10 manifest cannot replace the declared 21-scope study."""
        legacy_record = json.loads(self.legacy.read_text(encoding="utf-8"))
        study = legacy_record["studies"]["cifar10"]
        foreign = self.frozen.parent / "foreign_cifar10_manifest.json"
        foreign.write_bytes((self.legacy.parent / study["manifest_path"]).read_bytes())
        altered = deepcopy(self.original)
        altered["studies"]["cifar10"] = {**study, "manifest_path": foreign.name}
        with self.assertRaisesRegex(ValueError, "Native study membership differs"):
            workflow._campaign(self._write_altered(altered))

    def test_invalid_scope_is_rejected_without_creating_a_campaign(self) -> None:
        """Reject an arbitrary treatment subset before native study publication."""
        destination = self.directory / "unknown_scope"
        with self.assertRaisesRegex(ValueError, "Campaign scope must be"):
            workflow.prepare_campaign(destination, TEMPLATES, SEEDS, phase="benchmark",
                                      selection_provenance=PROVENANCE, scope="arbitrary_subset")
        self.assertFalse(destination.exists())


# Support isolated execution as well as project test discovery.
if __name__ == "__main__":
    unittest.main()
