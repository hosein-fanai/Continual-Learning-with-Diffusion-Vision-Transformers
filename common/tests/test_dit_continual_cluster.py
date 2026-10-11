"""Cross-host continual HPO clocks, global selection and paired receipt contracts."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common import dit_continual_cluster as cluster
from common import dit_hpo_runner as shared
from common.dit_continual_candidates import candidate_profile


class ContinualClusterTests(unittest.TestCase):
    """Exercise real immutable files and normalized candidates without GPU work."""

    def setUp(self) -> None:
        """Build independent host plans with identical scientific recipes."""

        temporary = tempfile.TemporaryDirectory(prefix="dit-cluster-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        identity = {
            "source_sha256": {"common/model.py": "source"}, 
            "versions": {"tensorflow": "2.20.0", "keras": "3.11.2"}, 
            "python": "3.12.3", "worker_policy": {"tf_memory_mib": 24576}
        }
        self.plans = {}
        with patch("common.dit_hpo_remote.inspect_remote", return_value=identity):
            for shard, seed in {"a": 42, "b": 43}.items():
                self.plans[shard] = shared.make_plan(
                    self.root, self.root / shard, epochs=20, concurrent_trials=24, 
                    experiment_hours=12, confirmation_reserve_hours=3, 
                    worker_gpu_memory_limit_mb=24576, search_profile="dit_continual_runner", 
                    continual_profile=candidate_profile(), seed=seed
                )
        self.campaign = self.root / "campaign"
        for shard, plan in self.plans.items():
            cluster.start_shared_campaign(plan, self.campaign, shard, started_at_unix=1000.0)

    def _exports(self) -> list[dict]:
        """Create two ranked studies with overlapping local trial identifiers."""

        records = []
        for shard, scores in {"a": [0.8, 0.6, 0.4], "b": [0.9, 0.7, 0.5]}.items():
            trials = [SimpleNamespace(
                number=index, value=score, state=SimpleNamespace(name="COMPLETE"), 
                params={"choice": f"{shard}-{index}"}
            ) for index, score in enumerate(scores)]
            plan = self.plans[shard]
            configs = Path(plan["study_root"]) / "configs"
            configs.mkdir(parents=True, exist_ok=True)
            for trial in trials:
                (configs / f"trial-{trial.number:04d}.yaml").write_text(
                    f"shard: {shard}\ntrial: {trial.number}\n", encoding="utf-8"
                )
            study = SimpleNamespace(get_trials=lambda deepcopy=False: trials)
            with patch.object(shared, "_load_study", return_value=study):
                records.append(cluster.export_search(plan, shard, self.campaign / f"export-{shard}.json"))
        return records

    def _manifest(self) -> dict:
        """Freeze four globally ranked candidates and three paired seeds."""

        return cluster.freeze_global_finalists(self._exports(), self.campaign / "global.json")

    def _receipt(self, local: dict, plan: dict, candidate: dict, score: float, seed: int) -> Path:
        """Write one authentic confirmation result using the shared receipt schema."""

        path = Path(plan["control_root"]) / "confirmations" / f"trial-{candidate['trial_number']:04d}" / f"seed-{seed}" / "completed.json"
        shared._write(path, {
            "identity": {
                "manifest_sha256": shared._digest(Path(plan["control_root"]) / "finalists.json"), 
                "config_sha256": candidate["config_sha256"], 
                "trial_number": candidate["trial_number"], "training_seed": seed
            }, 
            "result": {
                "objective": score, "objective_metric": "final_average_accuracy", 
                "objective_direction": "maximize", "objective_network": "raw", 
                "evaluation_source": "official_test", "task_groups": local["task_groups"], 
                "source_input_config_sha256": candidate["config_sha256"], 
                "source_trial_number": candidate["source_trial_number"], 
                "training_seed": seed, "dataset_seed": local["dataset_seed"]
            }
        })
        return path

    def test_shared_clock_never_restarts_and_normalized_seeds_match(self) -> None:
        """Different search seeds retain identical candidates and a single 12-hour clock."""

        self.assertEqual(cluster._recipe(self.plans["a"]), cluster._recipe(self.plans["b"]))
        clock = cluster.start_shared_campaign(self.plans["a"], self.campaign, "a")
        self.assertEqual(clock["budget"]["deadline_unix"], 44200.0)
        self.assertEqual(clock["budget"]["search_deadline_unix"], 33400.0)
        for plan in self.plans.values():
            self.assertEqual(shared._budget_state(plan), clock["budget"])
        with self.assertRaisesRegex(ValueError, "START time changed"):
            cluster.start_shared_campaign(self.plans["b"], self.campaign, "b", started_at_unix=2000.0)

    def test_clock_rejects_scientific_source_and_runtime_changes(self) -> None:
        """Host placement may vary while recipe, code and runtime must match."""

        for section, key, value in (
            ("hpo", "epochs", 10), 
            ("identity", "source_sha256", {"common/model.py": "changed"}), 
            ("identity", "versions", {"tensorflow": "other"})
        ):
            with self.subTest(key=key):
                changed = deepcopy(self.plans["b"])
                changed[section][key] = value
                with self.assertRaisesRegex(ValueError, "recipe or time policy changed"):
                    cluster.start_shared_campaign(changed, self.campaign, "b")

    def test_global_ranking_keeps_local_trial_numbers_distinct(self) -> None:
        """Trial zero on two hosts represents separate candidates with stable global IDs."""

        manifest = self._manifest()
        self.assertEqual(
            [(row["source_shard"], row["source_trial_number"]) for row in manifest["candidates"]], 
            [("b", 0), ("a", 0), ("b", 1), ("a", 1)]
        )
        self.assertEqual([row["global_candidate_id"] for row in manifest["candidates"]], [1, 2, 3, 4])
        for shard, plan in self.plans.items():
            local = cluster.install_global_finalists(plan, manifest, shard)
            self.assertEqual(len(local["candidates"]) * len(local["seeds"]), 8 if shard == "a" else 4)
            self.assertEqual(local["seeds"], [101, 202] if shard == "a" else [303])
            for candidate in local["candidates"]:
                self.assertEqual(shared._digest(candidate["input_config_path"]), candidate["config_sha256"])

    def test_global_selection_deduplicates_parameters_and_validates_integrity(self) -> None:
        """Duplicate recipes use one slot, whereas changed envelopes and YAML are rejected."""

        records = self._exports()
        duplicate = deepcopy(records)
        duplicate[1].pop("sha256")
        duplicate[1]["candidates"][0]["params"] = duplicate[0]["candidates"][0]["params"]
        duplicate[1] = cluster._seal(duplicate[1])
        selected = cluster.freeze_global_finalists(duplicate, self.campaign / "dedupe.json")
        params = [cluster._hash(row["params"]) for row in selected["candidates"]]
        self.assertEqual(len(set(params)), 4)
        self.assertEqual(selected["candidates"][0]["source_shard"], "b")
        changed = deepcopy(records)
        changed[0]["candidates"][0]["config_text"] = "tampered"
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            cluster.freeze_global_finalists(changed, self.campaign / "bad.json")
        changed[0].pop("sha256")
        changed[0] = cluster._seal(changed[0])
        with self.assertRaisesRegex(ValueError, "configuration identity"):
            cluster.freeze_global_finalists(changed, self.campaign / "bad.json")

    def test_duplicate_trial_within_shard_is_rejected(self) -> None:
        """A damaged export cannot represent one local trial as two candidates."""

        records = self._exports()
        records[0].pop("sha256")
        records[0]["candidates"][1]["source_trial_number"] = 0
        records[0] = cluster._seal(records[0])
        with self.assertRaisesRegex(ValueError, "configuration identity"):
            cluster.freeze_global_finalists(records, self.campaign / "bad.json")

    def test_global_selection_rejects_missing_hosts_changed_recipe_and_reused_seeds(self) -> None:
        """Global selection requires both sealed searches and independent paired repeats."""

        records = self._exports()
        with self.assertRaisesRegex(ValueError, "Export shards"):
            cluster.freeze_global_finalists(records[:1], self.campaign / "bad.json")
        changed = deepcopy(records)
        changed[1].pop("sha256")
        changed[1]["recipe"]["python"] = "other-runtime"
        changed[1] = cluster._seal(changed[1])
        with self.assertRaisesRegex(ValueError, "runtime identities"):
            cluster.freeze_global_finalists(changed, self.campaign / "bad.json")
        for seeds in ({"a": [101, 202], "b": [202, 404]}, {"a": [42, 202], "b": [303, 404]}):
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "Confirmation seeds"):
                cluster.freeze_global_finalists(records, self.campaign / "bad.json", seeds_by_shard=seeds)

    def test_merge_marks_missing_pairs_and_computes_three_seed_statistics(self) -> None:
        """Missing repeats remain incomplete and complete candidates use all paired seeds."""

        manifest = self._manifest()
        first_receipt = None
        for shard, plan in self.plans.items():
            local = cluster.install_global_finalists(plan, manifest, shard)
            for candidate in local["candidates"]:
                for seed in local["seeds"]:
                    # Deliberately leave the last candidate's final paired seed incomplete.
                    if candidate["trial_number"] == 4 and seed == 303:
                        continue
                    path = self._receipt(local, plan, candidate, 0.5 + seed / 1000.0, seed=seed)
                    first_receipt = path if first_receipt is None else first_receipt
        result = cluster.merge_global_confirmations(self.plans, manifest, self.campaign / "summary.json")
        self.assertFalse(result["all_pairs_complete"])
        self.assertEqual(result["candidates"][0]["completed_seeds"], 3)
        self.assertAlmostEqual(result["candidates"][0]["mean_final_average_accuracy"], 0.702)
        self.assertAlmostEqual(result["candidates"][0]["std_final_average_accuracy"], 0.101)
        self.assertFalse(result["candidates"][3]["comparable"])
        self.assertEqual(result["candidates"][3]["completed_seeds"], 2)
        record = shared._read(first_receipt)
        record["result"]["source_trial_number"] = 999
        shared._write(first_receipt, record)
        with self.assertRaisesRegex(ValueError, "global source or paired seed"):
            cluster.merge_global_confirmations(self.plans, manifest, self.campaign / "summary.json")

    def test_changed_installed_yaml_and_receipt_identity_are_rejected(self) -> None:
        """Resumption does not trust a replaced finalist file or a mismatched receipt."""

        manifest = self._manifest()
        plan = self.plans["a"]
        local = cluster.install_global_finalists(plan, manifest, "a")
        candidate = local["candidates"][0]
        path = self._receipt(local, plan, candidate, 0.8, seed=101)
        record = shared._read(path)
        record["identity"]["training_seed"] = 999
        shared._write(path, record)
        with self.assertRaisesRegex(ValueError, "different inputs"):
            cluster.merge_global_confirmations(self.plans, manifest, self.campaign / "summary.json")
        Path(candidate["input_config_path"]).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            cluster.install_global_finalists(plan, manifest, "a")
