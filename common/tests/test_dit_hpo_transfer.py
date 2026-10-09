"""Read-only Optuna transfer and scientific-identity regression checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import optuna

from common.config import Config, save_config
from common import dit_hpo_transfer as transfer


class DitHpoTransferTests(unittest.TestCase):
    """Use actual SQLite study storage without constructing or training models."""

    def setUp(self) -> None:
        """Build a source study with the maintained official-test protocol."""

        temporary = tempfile.TemporaryDirectory(prefix="dit-transfer-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / "source"
        self.dataset = getattr(self, "dataset", "CIFAR10")
        self.study_name = "generation-diffusion_transformer-" + self.dataset.lower()
        self.study_root = self.results / "generation" / "diffusion_transformer" / self.dataset.lower()
        self.control = self.study_root / "notebook_runner"
        self.control.mkdir(parents=True)
        self.manifest = self.root / "followup" / "transfer.json"
        common = {
            "task": "generation", "model_name": "diffusion_transformer", 
            "dataset_name": self.dataset.lower(), "epochs": 50, "fit_method": "fit", 
            "dtype_policy": "float32", "objective_metrics": ["generation_loss"], 
            "objective_directions": ["minimize"], "seed": 42, "n_startup_trials": 40, 
            "search_space_overrides": {"dim": [16, 32, 64, 128]}, "pruning": {"kind": "percentile"}
        }
        self.recipe = {"version": 2, "hpo": {
            **copy.deepcopy(common), "validation_source": "test", "validation_ratio": 0.0
        }, "source_sha256": {"common/hpo.py": "historical-source"}}
        self.spec = {
            **copy.deepcopy(common), "study_name": self.study_name, 
            "max_train_samples": None, "max_val_samples": None, "fit_kwargs": {}, 
            "effective_distillation": False, "use_ensemble_accuracy": False, 
            "model_overrides": {}, "wrapper_overrides": {}, 
            "data_selection": {"resolved": {
                "validation_source": "test", "validation_ratio": 0.0, 
                "effective_validation_ratio": 0.0, "drop_remainder": False
            }}
        }
        self.storage = optuna.storages.RDBStorage("sqlite:///" + (self.study_root / "study.db").as_posix())
        self.addCleanup(self.storage.engine.dispose)
        self.addCleanup(self.storage.remove_session)
        self.study = optuna.create_study(
            storage=self.storage, study_name=self.study_name, direction="minimize"
        )
        self._write_identity()

    def _write_identity(self) -> None:
        """Publish matching source recipe and study metadata."""

        fingerprint = transfer._fingerprint(self.spec)
        (self.control / "recipe.json").write_text(json.dumps(self.recipe), encoding="utf-8")
        (self.study_root / "study_spec.json").write_text(
            json.dumps({"spec": self.spec, "fingerprint": fingerprint}), encoding="utf-8"
        )
        self.study.set_user_attr("study_spec", self.spec)
        self.study.set_user_attr("study_spec_fingerprint", fingerprint)

    def _candidate(
        self, value: float | None, learning_rate: float = 0.001, dim: int = 128, 
        state: str = "COMPLETE", config_changes: dict | None = None
    ) -> int:
        """Add one stored trial plus its matching completed YAML configuration."""

        number = len(self.study.trials)
        params = {
            "learning_rate": learning_rate, "optimizer": "adam", "dim": dim, 
            "depth": 6, "mha_num_heads": 4, "mha_key_dim": None, "batch_size": 128, 
            "patch_size": 2, "dit_architecture_grid4": "plain", "time_freq_dim": 4, 
            "time_mlp_ratio": 2, "use_cfg": False, "loss_function": "mae"
        }
        config_path = self.study_root / "trials" / f"trial-{number:04d}" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        config = Config(
            dataset={"name": self.dataset, "validation_source": "test", "validation_ratio": 0.0, 
                     "drop_remainder": False}, 
            model={"name": "diffusion_transformer", "loss_function": "mae", 
                   "kwargs": {"compile_args": {"evaluation_loss": "mse"}}, 
                   "wrapper_kwargs": {"test_network_name": "ema", "use_ema": True}}, 
            hpo={"trial_number": number, "params": params, "objectives": [value], 
                 "objective_metrics": ["generation_loss"], "objective_directions": ["minimize"]}, 
            training={"task": "generation", "epochs": 50, "fit_method": "fit", 
                      "fit_kwargs": {}, "dtype_policy": "float32"}
        )
        for key, changed in (config_changes or {}).items():
            section, attribute = key.split(".")
            setattr(getattr(config, section), attribute, changed)
        save_config(config, config_path)
        self.study.add_trial(optuna.trial.create_trial(
            params=params, 
            distributions={
                **{key: optuna.distributions.CategoricalDistribution([value])
                   for key, value in params.items() if key not in {"learning_rate", "dim"}}, 
                "learning_rate": optuna.distributions.FloatDistribution(0.0003, 0.005, log=True), 
                "dim": optuna.distributions.CategoricalDistribution([32, 128])
            }, 
            value=value, user_attrs={"resolved_config_path": str(config_path)}, 
            state=getattr(optuna.trial.TrialState, state)
        ))
        return number

    def test_finite_rank_unique_hints_preserve_source(self) -> None:
        """Rank finite COMPLETE trials and strip capacity without importing scores."""

        self._candidate(0.3, learning_rate=0.002)
        self._candidate(0.2)
        self._candidate(0.1, dim=32)
        self._candidate(float("inf"), learning_rate=0.004)
        self._candidate(None, learning_rate=0.005, state="FAIL")
        before = (self.study_root / "study.db").read_bytes()
        manifest = transfer.freeze_transfer(self.results, self.manifest, top_k=3)
        self.assertEqual([trial["trial_number"] for trial in manifest["selected"]], [2, 0])
        self.assertEqual(len(manifest["initial_trials"]), 2)
        hint = manifest["initial_trials"][0]
        self.assertEqual(hint["learning_rate"], 0.001)
        self.assertEqual(hint["time_mlp_ratio"], 2)
        self.assertFalse(hint["use_cfg"])
        self.assertFalse(set(hint).intersection({
            "dim", "depth", "mha_num_heads", "mha_key_dim", "batch_size", "patch_size", 
            "dit_architecture_grid4", "value", "weights_path", "resume_checkpoint_dir"
        }))
        self.assertEqual((self.study_root / "study.db").read_bytes(), before)
        self.assertEqual(manifest["source"]["snapshot_trial_states"], {"COMPLETE": 4, "FAIL": 1})
        self.assertEqual(len(manifest["source"]["sqlite_snapshot_sha256"]), 64)

    def test_existing_manifest_ignores_growing_or_missing_source(self) -> None:
        """A resumed destination retains its original hints after upstream progresses."""

        self._candidate(0.4)
        manifest = transfer.freeze_transfer(self.results, self.manifest)
        before = self.manifest.read_bytes()
        self._candidate(0.1, learning_rate=0.002)
        with patch.object(transfer, "_snapshot_source", side_effect=AssertionError("must not reread")):
            resumed = transfer.freeze_transfer(self.results, self.manifest)
        self.assertEqual(resumed, manifest)
        self.assertEqual(self.manifest.read_bytes(), before)
        resumed["initial_trials"][0]["learning_rate"] = 0.1
        self.assertNotEqual(resumed, transfer.freeze_transfer(self.results, self.manifest))

    def test_changed_transfer_request_rejected(self) -> None:
        """Resuming cannot silently change source location or winner count."""

        self._candidate(0.4)
        transfer.freeze_transfer(self.results, self.manifest)
        with self.assertRaisesRegex(ValueError, "request changed"):
            transfer.freeze_transfer(self.results, self.manifest, top_k=2)
        with self.assertRaisesRegex(ValueError, "request changed"):
            transfer.freeze_transfer(self.root / "other", self.manifest)

    def test_manifest_tamper_rejected(self) -> None:
        """A changed hint invalidates the sealed transfer before destination launch."""

        self._candidate(0.4)
        manifest = transfer.freeze_transfer(self.results, self.manifest)
        manifest["initial_trials"][0]["learning_rate"] = 0.2
        with self.assertRaisesRegex(ValueError, "checksum"):
            transfer.validate_transfer_manifest(manifest)

    def test_architectural_hints_rejected_even_with_valid_digest(self) -> None:
        """A transfer manifest cannot restore the old architecture distribution."""

        self._candidate(0.4)
        manifest = transfer.freeze_transfer(self.results, self.manifest)
        manifest["initial_trials"][0]["dim"] = 128
        manifest["selected"][0]["hint_sha256"] = transfer._fingerprint(manifest["initial_trials"][0])
        manifest["transfer_sha256"] = transfer._fingerprint({
            key: value for key, value in manifest.items() if key != "transfer_sha256"
        })
        with self.assertRaisesRegex(ValueError, "architectural"):
            transfer.validate_transfer_manifest(manifest)

    def test_unstarted_source_is_actionable_and_does_not_create_database(self) -> None:
        """Setup fails before any destination worker when upstream has no study."""

        absent = self.root / "not-started"
        with self.assertRaisesRegex(FileNotFoundError, "Run the first notebook"):
            transfer.freeze_transfer(absent, self.manifest)
        self.assertFalse(absent.exists())
        self.assertFalse(self.manifest.exists())

    def test_no_finite_completed_results_rejected(self) -> None:
        """Failures and infinite completed objectives cannot become warm starts."""

        self._candidate(None, state="FAIL")
        self._candidate(float("inf"))
        with self.assertRaisesRegex(ValueError, "no finite COMPLETE"):
            transfer.freeze_transfer(self.results, self.manifest)
        self.assertFalse(self.manifest.exists())

    def test_source_directory_cannot_be_destination(self) -> None:
        """Even an uninitialized source cannot receive a transfer manifest."""

        with self.assertRaisesRegex(ValueError, "outside the source"):
            transfer.freeze_transfer(self.results, self.results / "transfer.json")

    def test_file_spec_checksum_rejected(self) -> None:
        """A corrupted sidecar cannot authenticate the source database."""

        self._candidate(0.1)
        (self.study_root / "study_spec.json").write_text(
            json.dumps({"spec": self.spec, "fingerprint": "wrong"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "checksum"):
            transfer.freeze_transfer(self.results, self.manifest)

    def test_database_and_file_spec_mismatch_rejected(self) -> None:
        """A matching file alone is insufficient when SQLite has a different identity."""

        self._candidate(0.1)
        self.study.set_user_attr("study_spec_fingerprint", "different")
        with self.assertRaisesRegex(ValueError, "Source Optuna study"):
            transfer.freeze_transfer(self.results, self.manifest)

    def test_split_validation_rejected(self) -> None:
        """Internal holdout results cannot be ranked as official-test feedback."""

        self._candidate(0.1)
        self.spec["data_selection"]["resolved"]["validation_source"] = "split"
        self._write_identity()
        with self.assertRaisesRegex(ValueError, "validation selection"):
            transfer.freeze_transfer(self.results, self.manifest)

    def test_shortened_source_protocol_rejected(self) -> None:
        """Smoke-test source results cannot seed the 50-epoch study as full results."""

        self._candidate(0.1)
        self.spec["max_train_samples"] = 128
        self._write_identity()
        with self.assertRaisesRegex(ValueError, "max_train_samples"):
            transfer.freeze_transfer(self.results, self.manifest)

    def test_mismatched_evaluation_loss_rejected(self) -> None:
        """MAE training is transferable only with fixed MSE evaluation units."""

        self._candidate(0.1, config_changes={"model.kwargs": {"compile_args": {"evaluation_loss": "mae"}}})
        with self.assertRaisesRegex(ValueError, "evaluated with MSE"):
            transfer.freeze_transfer(self.results, self.manifest)

    def test_legacy_mse_config_and_unexpanded_recipe_supported(self) -> None:
        """Older compatible MSE studies do not need an explicit evaluation override."""

        self.recipe["hpo"].pop("search_space_overrides")
        self.spec["search_space_overrides"] = {}
        self._write_identity()
        self._candidate(0.1, config_changes={"model.kwargs": {}, "model.loss_function": "mse"})
        self.assertEqual(len(transfer.freeze_transfer(self.results, self.manifest)["selected"]), 1)

    def test_resolved_config_objective_must_match_stored_result(self) -> None:
        """A completed result cannot be paired with a different resolved configuration."""

        self._candidate(0.1, config_changes={"training.epochs": 2})
        with self.assertRaisesRegex(ValueError, "epochs"):
            transfer.freeze_transfer(self.results, self.manifest)

    def test_wal_snapshot_includes_committed_results(self) -> None:
        """A SQLite backup includes committed WAL trials absent from a raw-file copy."""

        connection = sqlite3.connect(self.study_root / "study.db")
        self.addCleanup(connection.close)
        self.assertEqual(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
        self._candidate(0.2)
        self.assertTrue((self.study_root / "study.db-wal").exists())
        manifest = transfer.freeze_transfer(self.results, self.manifest)
        self.assertEqual(manifest["selected"][0]["value"], 0.2)


class Cifar100TransferTests(unittest.TestCase):
    """Keep source ranking, cached provenance and target datasets aligned."""

    def setUp(self) -> None:
        """Reuse real SQLite and resolved-config fixtures with 100-class data identity."""

        self.fixture = DitHpoTransferTests()
        self.fixture.dataset = "CIFAR100"
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def _freeze(self, **changes: object) -> dict:
        """Freeze the CIFAR100 fixture with optional public-API overrides."""

        options = {"top_k": 2, "dataset_name": "CiFaR100"}
        options.update(changes)
        return transfer.freeze_transfer(self.fixture.results, self.fixture.manifest, **options)

    def _resign(self, manifest: dict) -> dict:
        """Recompute the integrity hash to exercise independent semantic checks."""

        manifest["transfer_sha256"] = transfer._fingerprint({
            key: value for key, value in manifest.items() if key != "transfer_sha256"
        })
        return manifest

    def test_cifar100_ranking_hints_and_protocol_preserve_source(self) -> None:
        """Rank 100-class completions while copying no scores or old capacities into hints."""

        self.fixture._candidate(0.3, learning_rate=0.002)
        self.fixture._candidate(0.1)
        self.fixture._candidate(0.2, dim=32)
        before = (self.fixture.study_root / "study.db").read_bytes()
        manifest = self._freeze()
        self.assertEqual([item["trial_number"] for item in manifest["selected"]], [1, 0])
        self.assertEqual(manifest["request"]["dataset_name"], "cifar100")
        self.assertEqual(manifest["protocol"], {**transfer.PROTOCOL, "dataset_name": "cifar100"})
        self.assertEqual(manifest["source"]["study_name"], "generation-diffusion_transformer-cifar100")
        self.assertEqual(Path(manifest["source"]["study_root"]), self.fixture.study_root)
        self.assertEqual([hint["learning_rate"] for hint in manifest["initial_trials"]], [0.001, 0.002])
        self.assertTrue(all("dim" not in hint and "value" not in hint for hint in manifest["initial_trials"]))
        self.assertEqual((self.fixture.study_root / "study.db").read_bytes(), before)

    def test_cifar100_cached_manifest_retains_case_insensitive_identity(self) -> None:
        """Restarts reuse one immutable snapshot without rereading the growing source."""

        self.fixture._candidate(0.2)
        manifest = self._freeze()
        before = self.fixture.manifest.read_bytes()
        self.fixture._candidate(0.1, learning_rate=0.002)
        with patch.object(transfer, "_snapshot_source", side_effect=AssertionError("must not reread")):
            self.assertEqual(self._freeze(dataset_name="CIFAR100"), manifest)
        self.assertEqual(self.fixture.manifest.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "request changed"):
            self._freeze(dataset_name="CIFAR10")
        self.assertEqual(self.fixture.manifest.read_bytes(), before)

    def test_wrong_source_dataset_in_config_or_identity_is_rejected(self) -> None:
        """A CIFAR100 hierarchy cannot disguise CIFAR10 configurations or recipe identities."""

        self.fixture._candidate(0.2, config_changes={"dataset.name": "CIFAR10"})
        with self.assertRaisesRegex(ValueError, "Source trial dataset is not CIFAR100"):
            self._freeze()
        self.fixture.recipe["hpo"]["dataset_name"] = "cifar10"
        self.fixture._write_identity()
        with self.assertRaisesRegex(ValueError, "CIFAR100 source study"):
            self._freeze()
        self.assertFalse(self.fixture.manifest.exists())

    def test_resigned_dataset_inconsistencies_are_rejected(self) -> None:
        """Checksum recomputation cannot conceal disagreement among stored identities."""

        self.fixture._candidate(0.2)
        manifest = self._freeze()
        changes = [
            ("protocol", "dataset_name", "cifar10"), ("request", "dataset_name", "cifar10"), 
            ("source", "study_name", "generation-diffusion_transformer-cifar10"), 
            ("source", "study_root", str(self.fixture.study_root.parent / "cifar10"))
        ]
        for section, key, value in changes:
            changed = copy.deepcopy(manifest)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, "dataset"):
                transfer.validate_transfer_manifest(self._resign(changed))
        changed = copy.deepcopy(manifest)
        changed["source"]["recipe"]["hpo"]["dataset_name"] = "cifar10"
        with self.assertRaisesRegex(ValueError, "source dataset"):
            transfer.validate_transfer_manifest(self._resign(changed))
        changed = copy.deepcopy(manifest)
        changed["request"].pop("dataset_name")
        with self.assertRaisesRegex(ValueError, "request dataset"):
            transfer.validate_transfer_manifest(self._resign(changed))

    def test_malformed_manifest_envelopes_raise_value_error(self) -> None:
        """Malformed but re-signed envelopes fail through the public validation contract."""

        self.fixture._candidate(0.2)
        manifest = self._freeze()
        for key in ["protocol", "request", "source"]:
            changed = copy.deepcopy(manifest)
            changed[key] = None
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "manifest"):
                transfer.validate_transfer_manifest(self._resign(changed))

    def test_unsupported_dataset_rejected_before_destination_or_source_creation(self) -> None:
        """Only the two maintained image-study protocols may freeze source suggestions."""

        for dataset_name in ["MNIST", "cifar100-coarse", "imagenet", None]:
            with self.subTest(dataset_name=dataset_name), self.assertRaisesRegex(ValueError, "only CIFAR10 or CIFAR100"):
                self._freeze(dataset_name=dataset_name)
        self.assertFalse(self.fixture.manifest.parent.exists())

    def test_legacy_cifar10_manifest_keeps_request_shape_and_hash(self) -> None:
        """Explicit or implicit CIFAR10 calls reuse existing version-one snapshots unchanged."""

        legacy = DitHpoTransferTests()
        legacy.setUp()
        self.addCleanup(legacy.doCleanups)
        legacy._candidate(0.2)
        manifest = transfer.freeze_transfer(legacy.results, legacy.manifest)
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(set(manifest["request"]), {"source_results_path", "top_k"})
        before = legacy.manifest.read_bytes()
        with patch.object(transfer, "_snapshot_source", side_effect=AssertionError("must not reread")):
            self.assertEqual(transfer.freeze_transfer(
                legacy.results, legacy.manifest, dataset_name="cIfAr10"
            ), manifest)
        self.assertEqual(legacy.manifest.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "request changed"):
            transfer.freeze_transfer(legacy.results, legacy.manifest, dataset_name="CIFAR100")

    def test_cifar100_runner_seals_only_matching_dataset(self) -> None:
        """The public runner seals fresh hints and rejects a cross-dataset destination."""

        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock
        from common import dit_hpo_runner as runner
        from common.dit_hpo_backbones import FOLLOWUP_BRANCHES


        self.fixture._candidate(0.2)
        manifest = self._freeze()
        checkout = self.fixture.root / "checkout"
        checkout.mkdir()
        identity = {"source_sha256": {"common/hpo.py": "current-source"}, "versions": {}, 
                    "python": "test-python", "worker_policy": {"tf_memory_mib": 73728}}
        options = {
            "checkout_root": checkout, "results_path": self.fixture.root / "destination", 
            "dataset_name": "CIFAR100", "validation_source": "test", "validation_ratio": 0.0, 
            "search_space_overrides": {"dit_followup_branch": list(FOLLOWUP_BRANCHES)}, 
            "transfer_manifest": manifest
        }
        with patch.dict(sys.modules, {
            "common.dit_hpo_remote": SimpleNamespace(inspect_remote=Mock(return_value=identity))
        }):
            plan = runner.make_plan(**options)
            self.assertEqual(plan["hpo"]["dataset_name"], "CIFAR100")
            self.assertEqual(plan["transfer_manifest"], manifest)
            self.assertEqual(Path(plan["study_root"]).name, "cifar100")
            self.assertEqual(len(plan["hpo"]["initial_trials"]), 11)
            self.assertFalse((Path(plan["study_root"]) / "study.db").exists())
            self.assertFalse((Path(plan["control_root"]) / "budget.json").exists())
            self.assertEqual(runner.make_plan(**options)["hpo"], plan["hpo"])
            with self.assertRaisesRegex(ValueError, "target dataset differs"):
                runner.make_plan(**dict(options, dataset_name="CIFAR10"))
        recipe = json.loads((Path(plan["control_root"]) / "recipe.json").read_text())
        self.assertEqual(recipe["transfer_manifest"]["protocol"]["dataset_name"], "cifar100")

# Execute this focused suite only when explicitly invoked as a script.
if __name__ == "__main__":
    unittest.main()
