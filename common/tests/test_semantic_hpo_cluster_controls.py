"""Matched semantic controls retain global source identities and paired evidence."""

from copy import deepcopy
from pathlib import Path
from unittest import TestCase

from common import dit_hpo_runner as shared
from common import semantic_hpo_cluster as cluster
from common.semantic_hpo_cluster_controls import merge_controls
from common.tests import test_semantic_hpo_cluster as fixtures


class SemanticClusterControlTests(TestCase):
    """Use actual frozen semantic YAMLs with independent completion receipts."""

    def setUp(self) -> None:
        """Reuse the campaign fixture without inheriting its unrelated test methods."""

        self.fixture = fixtures.SemanticClusterTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.manifest = self.fixture.freeze()
        self.locals = {}
        for shard in self.fixture.plans:
            local = self.fixture.install(self.manifest, shard)
            self.locals[shard] = local
            for candidate in local["candidates"]:
                for seed in local["seeds"]:
                    self.fixture.receipt(local, shard, candidate, seed)
        self.learned = cluster.merge_confirmations(
            self.fixture.plans, self.manifest, self.fixture.campaign / "learned.json"
        )

    def write_controls(self) -> list[Path]:
        """Create two matched treatments whose paired learned improvement is known."""

        paths = []
        for shard, local in self.locals.items():
            root = Path(self.fixture.plans[shard]["control_root"])
            for candidate in local["candidates"]:
                for seed in local["seeds"]:
                    tail = Path(f"trial-{candidate['trial_number']:04d}") / f"seed-{seed}" / "completed.json"
                    learned = shared._read(root / "confirmations" / tail)
                    for condition in ("baseline", "unmodulated_feature_distillation"):
                        record = deepcopy(learned)
                        record["identity"]["control_condition"] = condition
                        result = record["result"]
                        result["control_condition"] = condition
                        result["objective"] -= 0.05
                        result["validation_continual_metrics"].update({
                            "retention_balanced_accuracy": result["objective"], 
                            "final_average_accuracy": 0.3, "final_old_task_accuracy": 0.2, "final_new_task_accuracy": 0.7
                        })
                        path = root / "controls" / condition / tail
                        shared._write(path, record)
                        paths.append(path)
        return paths

    def test_complete_controls_merge_all_eighteen_and_three_seed_pairs(self) -> None:
        """Paired effects and ordinary old/new metrics survive the global merge."""

        self.write_controls()
        report = merge_controls(self.fixture.plans, self.manifest, self.learned)
        self.assertEqual(report["completed_runs"], 18)
        self.assertEqual(report["required_runs"], 18)
        self.assertTrue(report["all_runs_complete"])
        self.assertTrue(report["all_pairs_complete"])
        for row in report["candidates"]:
            self.assertEqual(row["paired_seeds"], 3)
            self.assertAlmostEqual(row["mean_learned_minus_control"], 0.05)
            self.assertEqual([value["training_seed"] for value in row["per_seed"]], [101, 202, 303])
            self.assertEqual(row["mean_validation_metrics"]["final_old_task_accuracy"], 0.2)
            self.assertEqual(row["mean_validation_metrics"]["final_new_task_accuracy"], 0.7)

    def test_missing_controls_stay_incomplete_and_never_become_zero_scores(self) -> None:
        """One missing control removes one observation, not one candidate identity."""

        paths = self.write_controls()
        paths[0].unlink()
        report = merge_controls(self.fixture.plans, self.manifest, self.learned)
        self.assertEqual(report["completed_runs"], 17)
        self.assertFalse(report["all_runs_complete"])
        self.assertFalse(report["all_pairs_complete"])
        incomplete = [row for row in report["candidates"] if not row["comparable"]]
        self.assertEqual(len(incomplete), 1)
        self.assertEqual(incomplete[0]["completed_seeds"], 2)
        self.assertAlmostEqual(incomplete[0]["mean_learned_minus_control"], 0.05)

    def test_control_source_and_treatment_mutations_fail_closed(self) -> None:
        """Matching paths cannot authenticate foreign source trials or treatments."""

        path = self.write_controls()[0]
        original = shared._read(path)
        for key, value in (("source_trial_number", 999), ("control_condition", "learned")):
            changed = deepcopy(original)
            changed["result"][key] = value
            shared._write(path, changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                merge_controls(self.fixture.plans, self.manifest, self.learned)
        shared._write(path, original)

    def test_supplied_learned_score_cannot_replace_its_completion_receipt(self) -> None:
        """A changed summary cannot manufacture a paired treatment improvement."""

        self.write_controls()
        changed = deepcopy(self.learned)
        changed["candidates"][0]["scores_by_seed"]["101"] = 0.99
        with self.assertRaisesRegex(ValueError, "score differs"):
            merge_controls(self.fixture.plans, self.manifest, changed)
