"""Focused confirmation contracts using public-API doubles, without training."""

from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from common.dit_hpo_confirmation import run_confirmation


class ConfirmationTests(TestCase):
    """Protect independent training and fixed-data finalist comparisons."""

    def setUp(self) -> None:
        """Create an isolated source and typed-config-shaped public-API doubles."""

        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "trial-0007.yaml"
        self.source.write_text("immutable source config\n", encoding="utf-8")
        self.digest = sha256(self.source.read_bytes()).hexdigest()
        self.output = self.root / "attempt"
        self.config = SimpleNamespace(
            model=SimpleNamespace(
                wrapper_name="diffusion_model", kwargs={"dim": 64, "seed": 42}, 
                wrapper_kwargs={"use_ema": True, "test_network_name": "ema", "seed": 42}, 
                weights_path="old/trained.weights.h5", name="diffusion_transformer"
            ), 
            dataset=SimpleNamespace(
                validation_source="split", trainset_len=None, split_metadata={}
            ), 
            continually_learn=SimpleNamespace(
                resume_from="old/checkpoint", checkpoint_dir="old/checkpoint"
            ), 
            hpo={
                "trial_number": 7, "params": {"capacity": "64x4"}, 
                "input_config_path": "old/input.yaml", "objectives": [0.001], 
                "checkpoint_dir": "old/checkpoint", "use_distillation": False
            }, 
            training=SimpleNamespace(
                task="generation", fit_method="fit", 
                dtype_policy="float32", deterministic_ops=False, 
                use_valset=True, fit_kwargs={}, epochs=50, patience=5, 
                results_path="old/runs", project_tag="old", 
                tensorboard_path="old/tensorboard", tensorboard_run_name="old", seed=42
            )
        )
        self.snapshots = {}
        self.saved_configs = []
        self.model = object()
        self.trainset = object()
        self.valset = object()
        self.history = {"noise_loss": [0.5]}
        self.modules = {}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module_name in (
            "common.config", "common.dataloader", "common.model", 
            "common.runtime", "common.train"
        ):
            self.modules[module_name] = ModuleType(module_name)
        self.modules["common.config"].load_config = Mock(return_value=self.config)
        self.modules["common.config"].save_config = Mock(side_effect=self.save_config)
        self.modules["common.dataloader"].get_datasets = Mock(side_effect=self.get_datasets)
        self.modules["common.model"].get_model = Mock(side_effect=self.get_model)
        self.modules["common.runtime"].configure_runtime = Mock()
        self.modules["common.train"].train_model = Mock(side_effect=self.train_model)
        self.modules["common.train"].report = Mock(
            return_value={"valset_ema_eval": {"noise_loss": 0.125}}
        )
        self.stack.enter_context(patch.dict("sys.modules", self.modules))

    def save_config(self, config: object, path: Path) -> None:
        """Record the public persistence call and create its output marker."""

        self.saved_configs.append((deepcopy(config), Path(path)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("saved config\n", encoding="utf-8")

    def get_datasets(self, config: object) -> tuple[object, object]:
        """Observe the data seed and emulate schedule-length resolution."""

        self.snapshots["data"] = deepcopy(config)
        config.dataset.trainset_len = 313
        return self.trainset, self.valset

    def get_model(self, config: object) -> object:
        """Observe the exact fresh-model configuration before training."""

        self.snapshots["model"] = deepcopy(config)
        return self.model

    def train_model(
        self, config: object, model: object, trainset: object, valset: object = None
    ) -> dict[str, list[float]]:
        """Check data/model forwarding and emulate resolved output paths."""

        self.assertIs(model, self.model)
        self.assertIs(trainset, self.trainset)
        self.assertIs(valset, self.valset)
        config.training.results_path = str(self.output / "runs" / "concrete")
        config.model.weights_path = str(Path(config.training.results_path) / "model.weights.h5")
        return self.history

    def test_fixed_split_fresh_weights_and_public_pipeline(self) -> None:
        """Keep split/shuffle identity while changing every model seed."""

        result = run_confirmation(self.source, self.output, 101, self.digest)
        data_config = self.snapshots["data"]
        model_config = self.snapshots["model"]
        self.assertEqual(data_config.training.seed, 42)
        self.assertEqual(model_config.training.seed, 101)
        self.assertEqual(model_config.model.kwargs["seed"], 101)
        self.assertEqual(model_config.model.wrapper_kwargs["seed"], 101)
        self.assertIsNone(model_config.model.weights_path)
        self.assertIsNone(model_config.continually_learn.resume_from)
        self.assertIsNone(model_config.continually_learn.checkpoint_dir)
        self.assertNotIn("objectives", model_config.hpo)
        self.assertNotIn("checkpoint_dir", model_config.hpo)
        self.assertEqual(model_config.dataset.trainset_len, 313)
        self.assertEqual(model_config.training.epochs, 50)
        self.assertEqual(model_config.training.patience, 5)
        self.assertEqual(model_config.model.kwargs["dim"], 64)
        self.assertEqual(result["objective"], 0.125)
        self.assertEqual(result["split_seed"], 42)
        self.assertEqual(result["training_seed"], 101)
        self.assertEqual(result["source_input_config_sha256"], self.digest)
        self.assertEqual(self.saved_configs[-1][0].hpo["objectives"], [0.125])
        self.modules["common.runtime"].configure_runtime.assert_called_once_with(
            dtype_policy="float32", deterministic_ops=False, seed=42
        )
        self.modules["common.train"].report.assert_called_once_with(
            self.config, self.history, self.model, self.trainset, valset=self.valset
        )

    def test_source_hash_mismatch_fails_before_loading(self) -> None:
        """Reject a changed finalist before data or training side effects."""

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            run_confirmation(self.source, self.output, 101, "wrong")
        self.modules["common.config"].load_config.assert_not_called()
        self.modules["common.dataloader"].get_datasets.assert_not_called()

    def test_existing_attempt_is_not_retrained(self) -> None:
        """Refuse to overwrite an attempt's frozen input on reexecution."""

        self.output.mkdir()
        (self.output / "confirmation-input.yaml").write_text("existing", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            run_confirmation(self.source, self.output, 101, self.digest)
        self.modules["common.model"].get_model.assert_not_called()

    def test_test_source_is_not_accepted_for_confirmation(self) -> None:
        """Prevent an incompatible source from changing validation population."""

        self.config.dataset.validation_source = "test"
        with self.assertRaisesRegex(ValueError, "seeded split/EMA"):
            run_confirmation(self.source, self.output, 101, self.digest)
        self.modules["common.dataloader"].get_datasets.assert_not_called()

    def test_missing_ema_score_has_no_fallback(self) -> None:
        """Never rank using raw-network or training metrics when EMA is absent."""

        self.modules["common.train"].report.return_value = {
            "valset_network_eval": {"noise_loss": 0.01}
        }
        with self.assertRaises(KeyError):
            run_confirmation(self.source, self.output, 101, self.digest)

    def test_nonfinite_score_is_not_a_completed_result(self) -> None:
        """Exclude divergent validation results from confirmation ranking."""

        self.modules["common.train"].report.return_value = {
            "valset_ema_eval": {"noise_loss": float("nan")}
        }
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            run_confirmation(self.source, self.output, 101, self.digest)

    def test_nonscalar_score_is_rejected(self) -> None:
        """Prevent accidental aggregation of a vector-valued validation score."""

        self.modules["common.train"].report.return_value = {
            "valset_ema_eval": {"noise_loss": [0.1, 0.2]}
        }
        with self.assertRaisesRegex(TypeError, "real scalar"):
            run_confirmation(self.source, self.output, 101, self.digest)


# Support direct focused execution without invoking the full repository suite.
if __name__ == "__main__":
    main()