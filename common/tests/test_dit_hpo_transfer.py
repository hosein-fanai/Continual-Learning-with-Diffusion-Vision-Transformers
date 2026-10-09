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
        self.study_root = self.results / "generation" / "diffusion_transformer" / "cifar10"
        self.control = self.study_root / "notebook_runner"
        self.control.mkdir(parents=True)
        self.manifest = self.root / "followup" / "transfer.json"
        common = {
            "task": "generation", "model_name": "diffusion_transformer", 
            "dataset_name": "cifar10", "epochs": 50, "fit_method": "fit", 
            "dtype_policy": "float32", "objective_metrics": ["generation_loss"], 
            "objective_directions": ["minimize"], "seed": 42, "n_startup_trials": 40, 
            "search_space_overrides": {"dim": [16, 32, 64, 128]}, "pruning": {"kind": "percentile"}
        }
        self.recipe = {"version": 2, "hpo": {
            **copy.deepcopy(common), "validation_source": "test", "validation_ratio": 0.0
        }, "source_sha256": {"common/hpo.py": "historical-source"}}
        self.spec = {
            **copy.deepcopy(common), "study_name": transfer.STUDY_NAME, 
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
            storage=self.storage, study_name=transfer.STUDY_NAME, direction="minimize"
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
            dataset={"name": "CIFAR10", "validation_source": "test", "validation_ratio": 0.0, 
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


# Execute this focused suite only when explicitly invoked as a script.
if __name__ == "__main__":
    unittest.main()
