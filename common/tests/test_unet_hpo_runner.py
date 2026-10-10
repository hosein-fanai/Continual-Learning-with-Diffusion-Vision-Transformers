"""UNet notebook routing and fresh-seed confirmation regression contracts."""

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from common import dit_hpo_runner as runner
from common.dit_hpo_confirmation import run_confirmation
from common.tests import test_dit_hpo_confirmation as confirmation_tests


class UNetHpoRunnerTests(TestCase):
    """Keep UNet studies isolated while preserving the shared worker protocol."""

    def setUp(self) -> None:
        """Create a mocked remote identity and a real temporary recipe store."""

        temporary = TemporaryDirectory(prefix="unet-hpo-runner-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        identity = {
            "source_sha256": {}, "versions": {}, "python": "test-python", 
            "worker_policy": {"tf_memory_mib": 12288}
        }
        self.remote = SimpleNamespace(inspect_remote=Mock(return_value=identity))

    def test_model_families_have_distinct_sealed_studies(self) -> None:
        """A UNet plan cannot load or overwrite the colocated default DiT study."""

        overrides = {"widths": ["32-64", "32-64-96"]}
        with patch.dict("sys.modules", {"common.dit_hpo_remote": self.remote}):
            dit = runner.make_plan(self.checkout, self.root / "results")
            unet = runner.make_plan(
                self.checkout, self.root / "results", 
                search_space_overrides=overrides, model_name="unet"
            )
            resumed = runner.make_plan(
                self.checkout, self.root / "results", 
                search_space_overrides=overrides, model_name="unet"
            )
            with self.assertRaisesRegex(ValueError, "recipe changed"):
                runner.make_plan(
                    self.checkout, self.root / "results", epochs=51, 
                    search_space_overrides=overrides, model_name="unet"
                )
        self.assertEqual(dit["hpo"]["model_name"], "diffusion_transformer")
        self.assertEqual(unet["hpo"], resumed["hpo"])
        self.assertEqual(unet["hpo"]["task"], "generation")
        self.assertEqual(unet["hpo"]["objective_metrics"], ["generation_loss"])
        self.assertEqual(unet["hpo"]["objective_directions"], ["minimize"])
        self.assertEqual(unet["study_name"], "generation-unet-cifar10")
        self.assertEqual(Path(unet["study_root"]), self.root / "results" / "generation" / "unet" / "cifar10")
        self.assertNotEqual(dit["control_root"], unet["control_root"])
        sealed = runner._read(Path(unet["control_root"]) / "recipe.json")["hpo"]
        scientific = {
            key: value for key, value in unet["hpo"].items()
            if key not in {"concurrent_trials", "worker_gpu_memory_limit_mb", "worker_gpu_ids"}
        }
        self.assertEqual(sealed, scientific)
        self.assertEqual(sealed["model_name"], "unet")
        self.assertEqual(sealed["search_space_overrides"], overrides)
        self.assertNotIn("concurrent_trials", sealed)

    def test_unet_rejects_dit_profiles_and_transfer_before_remote_inspection(self) -> None:
        """Incompatible scientific protocols fail before admission or persistence."""

        options = [
            {"search_profile": "dit_classifier_runner"}, 
            {"search_profile": "dit_continual_runner"}, 
            {"transfer_manifest": {"source": "dit"}}
        ]
        with patch.dict("sys.modules", {"common.dit_hpo_remote": self.remote}):
            for extra in options:
                with self.subTest(options=extra), self.assertRaisesRegex(ValueError, "UNet generation"):
                    runner.make_plan(self.checkout, self.root / "invalid", model_name="unet", **extra)
        self.remote.inspect_remote.assert_not_called()
        self.assertFalse((self.root / "invalid").exists())

    def test_unet_worker_forwards_pruning_and_model_to_public_hpo(self) -> None:
        """The admitted parallel worker keeps the selected model and search policy."""

        policy = {"type": "percentile", "percentile": 75.0}
        with patch.dict("sys.modules", {"common.dit_hpo_remote": self.remote}):
            plan = runner.make_plan(
                self.checkout, self.root / "results", concurrent_trials=2, 
                pruning=policy, model_name="unet"
            )
        factory = Mock()
        remote = SimpleNamespace(
            managed_worker=Mock(), 
            managed_parallel_coordinator=Mock(return_value=nullcontext(factory))
        )
        hpo = Mock()
        request = self.root / "request.json"
        receipt = self.root / "receipt.json"
        runner._write(request, {
            "plan": plan, "payload": {"kind": "search", "allocated_target": 8}, 
            "receipt_path": str(receipt), "deadline": None
        })
        with patch.dict("sys.modules", {
            "common.hpo": SimpleNamespace(run_hpo=hpo), "common.dit_hpo_remote": remote
        }), patch.object(runner, "_load_study", return_value=None):
            runner._worker(request)
        hpo.assert_called_once_with(**{**plan["hpo"], "n_trials": 8, "worker_context": factory})
        self.assertEqual(hpo.call_args.kwargs["model_name"], "unet")
        self.assertEqual(hpo.call_args.kwargs["pruning"], policy)
        self.assertTrue(receipt.exists())

    def test_unet_progress_counts_width_families_and_finite_completions(self) -> None:
        """Progress uses UNet architecture names and excludes failed or divergent trials."""

        with patch.dict("sys.modules", {"common.dit_hpo_remote": self.remote}):
            plan = runner.make_plan(self.checkout, self.root / "results", model_name="unet")
        trials = [
            SimpleNamespace(number=0, value=0.2, state=SimpleNamespace(name="COMPLETE"), params={"widths": "32-64"}), 
            SimpleNamespace(number=1, value=0.1, state=SimpleNamespace(name="COMPLETE"), params={"widths": "32-64-96"}), 
            SimpleNamespace(number=2, value=None, state=SimpleNamespace(name="FAIL"), params={"widths": "32-64"}), 
            SimpleNamespace(number=3, value=float("nan"), state=SimpleNamespace(name="COMPLETE"), params={"widths": "32-64"})
        ]
        study = SimpleNamespace(get_trials=lambda deepcopy=False: trials)
        with patch.object(runner, "_load_study", return_value=study):
            summary = runner.search_summary(plan)
        self.assertEqual(summary["allocated_trials"], 4)
        self.assertEqual(summary["completed_finite_trials"], 2)
        self.assertEqual(summary["architecture_counts"], {"32-64": 1, "32-64-96": 1})
        self.assertEqual(summary["best_trial"], 1)
        self.assertEqual(summary["best_validation_noise_loss"], 0.1)


class UNetHpoConfirmationTests(TestCase):
    """Replay UNet architecture choices without relaxing confirmation safeguards."""

    def setUp(self) -> None:
        """Reuse the established public training API fixture without model compute."""

        self.fixture = confirmation_tests.ConfirmationTests(methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.config.model.name = "unet"
        self.fixture.config.model.kwargs = {"widths": [32, 64, 96], "seed": 42}

    def test_unet_confirmation_preserves_architecture_and_uses_fresh_seed(self) -> None:
        """A UNet finalist keeps its validation split while discarding trained weights."""

        fixture = self.fixture
        result = run_confirmation(fixture.source, fixture.output, 101, fixture.digest)
        model = fixture.snapshots["model"]
        self.assertEqual(model.model.name, "unet")
        self.assertEqual(model.model.kwargs, {"widths": [32, 64, 96], "seed": 101})
        self.assertIsNone(model.model.weights_path)
        self.assertEqual(fixture.snapshots["data"].training.seed, 42)
        self.assertEqual(result["training_seed"], 101)
        self.assertEqual(result["objective"], 0.125)
        self.assertEqual(result["objective_network"], "ema")
        fixture.modules["common.train"].train_model.assert_called_once()

    def test_unet_confirmation_rejects_teacher_and_dit_profile_mix(self) -> None:
        """UNet acceptance does not extend to distillation or named DiT protocols."""

        fixture = self.fixture
        for policy in (
            {"use_distillation": True}, {"search_profile": "dit_classifier_runner"}, 
            {"search_profile": "dit_continual_runner"}
        ):
            with self.subTest(policy=policy):
                fixture.config.hpo = {"trial_number": 7, **policy}
                with self.assertRaisesRegex(ValueError, "ordinary teacher-free"):
                    run_confirmation(fixture.source, fixture.output, 101, fixture.digest)
        fixture.modules["common.dataloader"].get_datasets.assert_not_called()
        fixture.modules["common.train"].train_model.assert_not_called()


# Support focused standalone execution as well as unittest discovery.
if __name__ == "__main__":
    main()
