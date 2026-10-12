"""Remote-only semantic shard clocks, portable selection and receipt contracts."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from common import dit_hpo_runner as shared
from common import semantic_hpo_cluster as cluster
from common import semantic_hpo_runner as semantic
from common.config import save_config
from common.semantic_hpo import build_semantic_config, normalize_semantic_profile
from common.tests.test_semantic_hpo import _Trial
from semantic_consolidation.config import load_route_config


class _RecipeTrial(_Trial):
    """Select distinct in-range learning rates without changing the study domain."""

    def suggest_float(self, name: str, low: float, high: float, step: float | None = None, log: bool = False) -> float:
        """Keep one stable distribution while producing distinct native trial settings."""

        value = super().suggest_float(name, low, high, step=step, log=log)
        # Vary one real parameter while retaining identical study distributions.
        if name == "learning_rate":
            value = low * (1 + self.number + getattr(self, "offset", 0))
            self.params[name] = value
        return value


class SemanticClusterTests(TestCase):
    """Exercise immutable envelopes and actual semantic YAMLs without model training."""

    def setUp(self) -> None:
        """Create two local fixtures representing separately admitted remote hosts."""

        temporary = TemporaryDirectory(prefix="semantic-cluster-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        native = load_route_config("semantic_consolidation/configs/cifar10.yaml")
        native.common.training.epochs = 2
        native.common.training.seed = 42
        native.common.continually_learn.seed = 42
        profile = normalize_semantic_profile({
            "student_config": native.common, "route_settings": native.route, 
            "selection_metric": "retention_balanced_accuracy"
        })
        self.plans = {}
        for shard, sampler_seed in (("a", 42), ("b", 43)):
            study = self.root / shard / "study"
            self.plans[shard] = {
                "version": 2, "study_root": str(study), "control_root": str(study / "notebook_runner"), 
                "study_name": "continual-dit_classifier-cifar10-semantic_consolidation_runner", 
                "time_budget": {"experiment_hours": 12.0, "confirmation_reserve_hours": 4.0}, 
                "hpo": {
                    "task": "continual", "model_name": "dit_classifier", "search_profile": semantic.PROFILE, 
                    "semantic_profile": deepcopy(profile), "seed": 42, "sampler_seed": sampler_seed, 
                    "objective_metrics": ["retention_balanced_accuracy"], "objective_directions": ["maximize"], 
                    "results_path": str(self.root / shard), "concurrent_trials": 8
                }, 
                "identity": {
                    "hostname": "host-" + shard, "source_sha256": {"common/semantic_hpo.py": "source"}, 
                    "versions": {"tensorflow": "2.20.0", "keras": "3.11.2"}, "python": "3.12.3"
                }
            }
        storage_patch = patch.object(cluster, "_storage_identity", side_effect=lambda plan: {
            "hostname": plan["identity"]["hostname"], "database_path": "/tmp/local-study.db", 
            "cache_id": plan["identity"]["hostname"], "study_root": plan["study_root"]
        })
        storage_patch.start()
        self.addCleanup(storage_patch.stop)
        self.campaign = self.root / "campaign"
        for shard, plan in self.plans.items():
            cluster.start_shared_campaign(plan, self.campaign, shard, started_at_unix=1000.0)

    def make_exports(self, waiting: bool = False) -> list[dict]:
        """Publish complete typed configurations and an authenticated closure snapshot."""

        records = []
        for shard_index, shard in enumerate(self.plans):
            values = [0.20, 0.15, 0.10] if shard == "a" else [0.25, 0.18, 0.12]
            plan = self.plans[shard]
            trials = []
            configs = Path(plan["study_root"]) / "configs"
            configs.mkdir(parents=True, exist_ok=True)
            for number, value in enumerate(values):
                trial = _RecipeTrial(number)
                trial.offset = 3 * shard_index
                config = build_semantic_config(
                    trial, "cifar10", 2, self.root / shard, plan["hpo"]["semantic_profile"], seed=42
                )
                save_config(config, configs / f"trial-{number:04d}.yaml")
                trials.append(SimpleNamespace(
                    number=number, value=value, params=trial.params, state=SimpleNamespace(name="COMPLETE")
                ))
            # Optional unstarted anchor evidence exercises early admission closure.
            if waiting:
                trials.append(SimpleNamespace(number=3, value=None, params={}, state=SimpleNamespace(name="WAITING")))
            profile = plan["hpo"]["semantic_profile"]
            marker = {
                "study_name": plan["study_name"], "fixed_recipe_sha256": profile["fixed_recipe_sha256"], 
                "objective_metric": "retention_balanced_accuracy", "task_seed": profile["task_seed"], 
                "dataset_seed": profile["dataset_seed"], "task_groups": profile["task_groups"], 
                "allocated_trials": len(trials), "trial_state_fingerprint": semantic._trial_snapshot(trials), 
                "unstarted_trial_numbers": [3] if waiting else [], "closed_at_unix": 2000.0
            }
            shared._write(Path(plan["control_root"]) / "search_closed.json", marker)
            study = SimpleNamespace(get_trials=lambda deepcopy=False: trials)
            with patch.object(shared, "_load_study", return_value=study):
                records.append(cluster.export_search(plan, shard, self.campaign / f"export-{shard}.json"))
        return records

    def freeze(self) -> dict:
        """Freeze three portable recipes from the two independent searches."""

        return cluster.freeze_global_finalists(self.make_exports(), self.campaign / "global.json")

    def install(self, manifest: dict, shard: str) -> dict:
        """Supply an already verified local admission closure for installation tests."""

        with patch.object(semantic, "validate_search_closed", return_value={"closed": True}):
            return cluster.install_global_finalists(self.plans[shard], manifest, shard)

    def receipt(self, local: dict, shard: str, candidate: dict, seed: int) -> Path:
        """Create one identity-bound semantic confirmation receipt."""

        plan = self.plans[shard]
        path = Path(plan["control_root"]) / "confirmations" / f"trial-{candidate['trial_number']:04d}" / f"seed-{seed}" / "completed.json"
        objective = 0.1 + seed / 1000.0
        shared._write(path, {
            "identity": {
                "manifest_sha256": shared._digest(Path(plan["control_root"]) / "finalists.json"), 
                "config_sha256": candidate["config_sha256"], "trial_number": candidate["trial_number"], 
                "training_seed": seed
            }, 
            "result": {
                "objective": objective, "objective_metric": "retention_balanced_accuracy", 
                "objective_direction": "maximize", "objective_network": "raw", 
                "search_profile": semantic.PROFILE, "fixed_recipe_sha256": local["fixed_recipe_sha256"], 
                "validation_continual_metrics": {"retention_balanced_accuracy": objective}, 
                "task_groups": local["task_groups"], "dataset_seed": local["dataset_seed"], 
                "task_seed": local["task_seed"], "training_seed": seed, 
                "source_trial_number": candidate["source_trial_number"], 
                "source_input_config_sha256": candidate["config_sha256"]
            }
        })
        return path

    def test_shared_clock_preserves_model_seed_and_allows_independent_sampler(self) -> None:
        """One absolute twelve-hour budget survives repeated START on either shard."""

        self.assertEqual(cluster._recipe(self.plans["a"]), cluster._recipe(self.plans["b"]))
        clock = cluster.start_shared_campaign(self.plans["a"], self.campaign, "a")
        self.assertEqual(clock["budget"]["deadline_unix"], 44200.0)
        self.assertEqual(clock["budget"]["search_deadline_unix"], 29800.0)
        with self.assertRaisesRegex(ValueError, "START time changed"):
            cluster.start_shared_campaign(self.plans["b"], self.campaign, "b", started_at_unix=2000.0)
        for section, key, value in (
            ("hpo", "seed", 43), ("identity", "source_sha256", {"common/semantic_hpo.py": "changed"})
        ):
            changed = deepcopy(self.plans["b"])
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "source identity or time policy"):
                cluster.start_shared_campaign(changed, self.campaign, "b")

    def test_early_admission_close_excludes_waiting_hints_before_deadline(self) -> None:
        """Measured capacity can close the search early without treating hints as runs."""

        with patch.object(cluster.time, "time", return_value=2000.0):
            exports = self.make_exports(waiting=True)
        self.assertTrue(all(record["unstarted_trial_numbers"] == [3] for record in exports))
        manifest = cluster.freeze_global_finalists(exports, self.campaign / "global.json")
        self.assertEqual(len(manifest["candidates"]), 3)
        damaged = deepcopy(exports)
        damaged[0].pop("sha256")
        damaged[0]["candidates"].pop()
        damaged[0] = cluster._seal(damaged[0])
        with self.assertRaisesRegex(ValueError, "every finite complete candidate"):
            cluster.freeze_global_finalists(damaged, self.campaign / "missing.json")

    def test_anchor_partition_preserves_exact_master_and_local_queue_identity(self) -> None:
        """Two shard queues jointly cover each canonical anchor once, not twice."""

        master = [{"base_recipe": name, "semantic_budget": "strong"} for name in ("one", "two", "three", "four")]
        allocation = {"a": master[::2], "b": master[1::2]}
        plans = deepcopy(self.plans)
        for shard, plan in plans.items():
            study = self.root / "partitioned" / shard
            plan["study_root"] = str(study)
            plan["control_root"] = str(study / "notebook_runner")
            plan["hpo"]["initial_trials"] = allocation[shard]
        campaign = self.root / "partitioned-clock"
        with patch("common.semantic_hpo_candidates.semantic_initial_trials", return_value=master):
            for shard, plan in plans.items():
                clock = cluster.start_shared_campaign(
                    plan, campaign, shard, started_at_unix=1000.0, anchors_by_shard=allocation
                )
            self.assertEqual(clock["recipe"]["anchors_by_shard"], allocation)
            self.assertEqual(cluster._recipe(plans["a"], allocation), cluster._recipe(plans["b"], allocation))
            self.assertNotEqual(cluster._recipe(plans["a"]), cluster._recipe(plans["b"]))
            duplicate = {"a": master, "b": master}
            with self.assertRaisesRegex(ValueError, "exactly once"):
                cluster.start_shared_campaign(plans["a"], self.root / "bad-clock", "a", anchors_by_shard=duplicate)
            wrong = deepcopy(plans["b"])
            wrong["hpo"]["initial_trials"] = allocation["a"]
            with self.assertRaisesRegex(ValueError, "declared shard anchor queue"):
                cluster.start_shared_campaign(wrong, campaign, "b", anchors_by_shard=allocation)
        self.assertFalse((self.root / "bad-clock" / "campaign_clock.json").exists())

    def test_global_ranking_preserves_source_trials_and_distributes_nine_repeats(self) -> None:
        """Overlapping local trial numbers receive unique global confirmation output IDs."""

        manifest = self.freeze()
        self.assertEqual(
            [(row["source_shard"], row["source_trial_number"]) for row in manifest["candidates"]], 
            [("b", 0), ("a", 0), ("b", 1)]
        )
        for shard, count in (("a", 6), ("b", 3)):
            local = self.install(manifest, shard)
            self.assertEqual(len(local["candidates"]) * len(local["seeds"]), count)
            self.assertEqual([row["trial_number"] for row in local["candidates"]], [1, 2, 3])
            for row in local["candidates"]:
                self.assertEqual(shared._digest(row["input_config_path"]), row["config_sha256"])

    def test_exports_reject_mutation_mismatched_sources_and_missing_shards(self) -> None:
        """Transport integrity and common scientific identity are independent checks."""

        exports = self.make_exports()
        with self.assertRaisesRegex(ValueError, "all distinct"):
            cluster.freeze_global_finalists(exports[:1], self.campaign / "missing.json")
        changed = deepcopy(exports)
        changed[0]["candidates"][0]["config_text"] = "changed"
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            cluster.freeze_global_finalists(changed, self.campaign / "changed.json")
        changed = deepcopy(exports)
        changed[1].pop("sha256")
        changed[1]["recipe"]["python"] = "other"
        changed[1] = cluster._seal(changed[1])
        with self.assertRaisesRegex(ValueError, "runtime identity"):
            cluster.freeze_global_finalists(changed, self.campaign / "runtime.json")
        with self.assertRaisesRegex(ValueError, "exactly once"):
            cluster.freeze_global_finalists(exports, self.campaign / "seeds.json", seeds_by_shard={"a": [101, 202], "b": [202]})

    def test_merge_uses_three_seeds_and_does_not_open_foreign_sqlite(self) -> None:
        """Transported receipts remain reportable with incomplete pairs left explicit."""

        manifest = self.freeze()
        first = None
        for shard in self.plans:
            local = self.install(manifest, shard)
            for candidate in local["candidates"]:
                for seed in local["seeds"]:
                    # One missing repeat must remain visibly incomplete in global statistics.
                    if candidate["trial_number"] == 3 and seed == 303:
                        continue
                    path = self.receipt(local, shard, candidate, seed=seed)
                    first = path if first is None else first
        with patch.object(shared, "_load_study", side_effect=AssertionError("Foreign SQLite must not open")):
            result = cluster.merge_confirmations(self.plans, manifest, self.campaign / "summary.json")
        self.assertFalse(result["all_pairs_complete"])
        self.assertAlmostEqual(result["candidates"][0]["mean_objective"], 0.302)
        self.assertAlmostEqual(result["candidates"][0]["std_objective"], 0.101)
        self.assertFalse(result["candidates"][2]["comparable"])
        receipt = shared._read(first)
        receipt["result"]["source_trial_number"] = 999
        shared._write(first, receipt)
        with self.assertRaisesRegex(ValueError, "global source or paired seed"):
            cluster.merge_confirmations(self.plans, manifest, self.campaign / "summary.json")


class SemanticFiveHostClusterTests(TestCase):
    """Five unequal hosts retain exact coverage and three paired seeds per finalist."""

    make_exports = SemanticClusterTests.make_exports
    install = SemanticClusterTests.install
    receipt = SemanticClusterTests.receipt

    def setUp(self) -> None:
        """Extend the two-host fixture with three independently identified caches."""

        SemanticClusterTests.setUp(self)
        self.weights = {"a": 6, "b": 3, "c": 4, "d": 2, "e": 1}
        for index, shard in enumerate(("c", "d", "e"), start=2):
            plan = deepcopy(self.plans["a"])
            study = self.root / shard / "study"
            plan["study_root"] = str(study)
            plan["control_root"] = str(study / "notebook_runner")
            plan["hpo"]["results_path"] = str(self.root / shard)
            plan["hpo"]["sampler_seed"] = 42 + index
            plan["identity"]["hostname"] = "host-" + shard
            self.plans[shard] = plan
            cluster.start_shared_campaign(plan, self.campaign, shard, started_at_unix=1000.0)

    def test_five_host_weighted_anchors_preserve_all_twenty_seven_once(self) -> None:
        """Unequal host capacity changes placement without omitting canonical treatment coverage."""

        master = [{"base_recipe": str(index), "semantic_budget": "strong"} for index in range(27)]
        with patch("common.semantic_hpo_candidates.semantic_initial_trials", return_value=master):
            allocation = cluster.partition_anchors({}, self.weights)
            self.assertEqual(set(allocation), set(self.weights))
            self.assertTrue(all(allocation.values()))
            self.assertEqual(
                sorted(cluster._hash(item) for values in allocation.values() for item in values), 
                sorted(cluster._hash(item) for item in master)
            )
            self.assertGreater(len(allocation["a"]), len(allocation["e"]))
            self.assertEqual(cluster._recipe(self.plans["a"], allocation)["anchors_by_shard"], allocation)
            duplicate = deepcopy(allocation)
            duplicate["e"].append(allocation["a"][0])
            with self.assertRaisesRegex(ValueError, "exactly once"):
                cluster._recipe(self.plans["a"], duplicate)

    def test_nine_weighted_bundles_and_eighteen_controls_merge_without_duplicate_work(self) -> None:
        """Empty local candidate seed lists neither launch work nor erase global three-seed evidence."""

        manifest = cluster.freeze_global_finalists(
            self.make_exports(), self.campaign / "global.json", shard_weights=self.weights
        )
        allocation = cluster._candidate_seed_allocation(manifest)
        self.assertEqual(sum(len(seeds) for rows in allocation.values() for seeds in rows.values()), 9)
        self.assertTrue(all(any(rows.values()) for rows in allocation.values()))
        self.assertTrue(any(not seeds for rows in allocation.values() for seeds in rows.values()))
        self.assertEqual(sum(len(seeds) for seeds in allocation["e"].values()), 1)
        learned_count = 0
        control_count = 0
        for shard in self.plans:
            local = self.install(manifest, shard)
            for candidate in local["candidates"]:
                for seed in candidate["seeds"]:
                    learned = self.receipt(local, shard, candidate, seed=seed)
                    for condition in ("baseline", "unmodulated_feature_distillation"):
                        record = shared._read(learned)
                        record["identity"]["control_condition"] = condition
                        record["result"]["control_condition"] = condition
                        path = Path(self.plans[shard]["control_root"]) / "controls" / condition / learned.relative_to(
                            Path(self.plans[shard]["control_root"]) / "confirmations"
                        )
                        shared._write(path, record)
            with patch.object(shared, "_launch", side_effect=AssertionError("Completed bundles cannot relaunch")):
                learned_count += len(semantic.run_confirmations(self.plans[shard]))
                with patch.object(semantic, "capacity_plan", return_value={"conservative_full_trial_seconds": 1.0}):
                    control_count += len(semantic.run_controls(self.plans[shard]))
            for row in semantic.confirmation_summary(self.plans[shard]):
                # An unassigned local candidate cannot be reported as an observed comparison.
                if row["required_seeds"] == 0:
                    self.assertFalse(row["comparable"])
        self.assertEqual((learned_count, control_count), (9, 18))
        result = cluster.merge_confirmations(self.plans, manifest, self.campaign / "summary.json")
        self.assertTrue(result["all_campaign_pairs_complete"])
        self.assertEqual(result["controls"]["required_runs"], 18)
        self.assertTrue(all(row["required_seeds"] == 3 for row in result["candidates"]))

    def test_five_host_exports_and_candidate_seed_coverage_fail_closed(self) -> None:
        """Missing hosts and duplicated or omitted paired seeds cannot authenticate selection."""

        exports = self.make_exports()
        with self.assertRaisesRegex(ValueError, "all distinct"):
            cluster.freeze_global_finalists(exports[:-1], self.campaign / "missing.json", shard_weights=self.weights)
        manifest = cluster.freeze_global_finalists(exports, self.campaign / "global.json", shard_weights=self.weights)
        for replacement in ([101, 101, 303], [101, 202]):
            damaged = deepcopy(manifest)
            for rows in damaged["seeds_by_candidate_by_shard"].values():
                rows["1"] = []
            damaged["seeds_by_candidate_by_shard"]["a"]["1"] = replacement
            with self.assertRaisesRegex(ValueError, "exactly once"):
                cluster._candidate_seed_allocation(damaged)
        local = self.install(manifest, "a")
        candidate = next(row for row in local["candidates"] if row["seeds"])
        candidate["seeds"] = []
        shared._write(Path(self.plans["a"]["control_root"]) / "finalists.json", local)
        with self.assertRaisesRegex(ValueError, "global allocation"):
            semantic.run_confirmations(self.plans["a"])
