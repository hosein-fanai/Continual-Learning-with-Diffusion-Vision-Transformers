"""Process-boundary, lock, failure, and real TensorFlow worker regression checks."""

from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from common.hpo_process import (
    WorkerHandle, dataset_load_lock, finish_worker, start_worker, stop_workers, study_lock,
)
from common.hpo_worker import _json_value, run_worker


class HpoProcessTests(unittest.TestCase):
    """Exercise actual child processes without training or importing TensorFlow."""

    def setUp(self) -> None:
        """Give every child a private artifact directory and guaranteed cleanup."""

        temporary = tempfile.TemporaryDirectory(prefix="hpo-process-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "input.yaml"
        self.config.write_text("{}", encoding="utf-8")
        self.handles: list[WorkerHandle] = []
        self.addCleanup(stop_workers, self.handles)

    def _spawn(self, script: str) -> WorkerHandle:
        """Launch a small real Python child with the production handle shape."""

        index = len(self.handles)
        output = self.root / f"result-{index}.json"
        log = self.root / f"worker-{index}.log"
        stream = log.open("wb")
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", script, str(output)],
            stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
        )
        handle = WorkerHandle(process, output, log, stream)
        self.handles.append(handle)
        return handle

    def _payload(self, status: str = "complete") -> dict[str, object]:
        """Return a realistic, otherwise valid protocol envelope."""

        return {
            "status": status, "config_path": str(self.config),
            "results_path": str(self.root), "history": {"loss": [1.0]},
            "evaluations": {"valset_network_eval": {"classifier_accuracy": 0.5}},
            "error": None if status == "complete" else "expected training failure",
            "divergence": {"metric": "loss"} if status == "pruned" else None,
            "divergence_path": None,
        }

    def test_imports_do_not_initialize_tensorflow(self) -> None:
        """Coordinator transport and CLI imports remain standard-library only."""

        result = subprocess.run(
            [sys.executable, "-c", "import sys; import common.hpo_process; "
             "import common.hpo_worker; print('tensorflow' in sys.modules)"],
            text=True, capture_output=True, check=True, timeout=20,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_lock_contention_release_and_crash_recovery(self) -> None:
        """The OS lock rejects another process and releases after its death."""

        script = (
            "from pathlib import Path\n"
            "import sys,time\n"
            "from common.hpo_process import study_lock\n"
            f"with study_lock(Path({str(self.root)!r})):\n"
            " print('locked',flush=True)\n"
            " time.sleep(60)\n"
        )
        handle = self._spawn(script)
        deadline = time.monotonic() + 10
        while "locked" not in handle.log_path.read_text() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertIn("locked", handle.log_path.read_text())
        with self.assertRaisesRegex(RuntimeError, "coordinator"):
            with study_lock(self.root):
                self.fail("Concurrent coordinators must not acquire the same lock")
        handle.process.kill()
        handle.process.wait(timeout=10)
        with study_lock(self.root):
            self.assertTrue((self.root / ".coordinator.lock").exists())
        with study_lock(self.root):
            with study_lock(self.root / "independent"):
                self.assertTrue((self.root / ".coordinator.lock").exists())

    def test_launcher_validates_options_and_paths_before_spawn(self) -> None:
        """Invalid limits, input paths, and collisions never create a process."""

        cases = [
            {"threads": value} for value in (True, 0, -1, 1.5)
        ] + [
            {"gpu_memory_limit_mb": value} for value in (True, 0, -1, float("nan"), float("inf"), "10")
        ]
        with patch("common.hpo_process.subprocess.Popen") as popen:
            for case in cases:
                with self.subTest(case=case), self.assertRaises(ValueError):
                    settings = {"gpu_memory_limit_mb": None, **case}
                    start_worker(self.config, self.root / "r.json", self.root / "l.log", **settings)
            with self.assertRaises(FileNotFoundError):
                start_worker(self.root / "missing", self.root / "r", self.root / "l", gpu_memory_limit_mb=None)
            with self.assertRaises(ValueError):
                start_worker(self.config, self.config, self.root / "l", gpu_memory_limit_mb=None)
            popen.assert_not_called()

    def test_launcher_preserves_visibility_and_applies_limits(self) -> None:
        """Only a child environment gets thread, plotting, and GPU policy changes."""

        for cap in (None, 2048.0):
            with self.subTest(cap=cap), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
                child = Mock()
                child.poll.return_value = 0
                child.wait.return_value = 0
                with patch("common.hpo_process.subprocess.Popen", return_value=child) as popen:
                    handle = start_worker(
                        self.config, self.root / "r.json", self.root / "l.log",
                        gpu_memory_limit_mb=cap, threads=2,
                    )
                self.handles.append(handle)
                command = popen.call_args.args[0]
                options = popen.call_args.kwargs
                self.assertEqual(command[:4], [sys.executable, "-u", "-m", "common.hpo_worker"])
                self.assertIn("--watch-parent", command)
                self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], "7")
                self.assertEqual(options["env"]["TF_NUM_INTRAOP_THREADS"], "2")
                self.assertEqual(options["env"]["TF_NUM_INTEROP_THREADS"], "2")
                self.assertEqual(options["env"]["OMP_NUM_THREADS"], "2")
                self.assertEqual(options["env"]["MPLBACKEND"], "Agg")
                self.assertEqual(options["env"]["TF_FORCE_GPU_ALLOW_GROWTH"], "true" if cap is None else "false")
                self.assertEqual("--gpu-memory-limit-mb" in command, cap is not None)
                self.assertNotIn("shell", options)
                self.assertEqual(options["stdin"], subprocess.PIPE)
                stop_workers([handle])
                self.assertTrue(handle.log_file.closed)

    def test_launcher_failure_closes_log(self) -> None:
        """A missing executable cannot leave its opened log stream behind."""

        log = io.BytesIO()
        with patch.object(Path, "open", return_value=log), \
                patch("common.hpo_process.subprocess.Popen", side_effect=OSError("no executable")):
            with self.assertRaisesRegex(OSError, "no executable"):
                start_worker(self.config, self.root / "r", self.root / "l", gpu_memory_limit_mb=None)
        self.assertTrue(log.closed)

    def test_dataset_load_serializes_and_releases_after_error(self) -> None:
        """Loading locks block competitors and release after a failed extraction."""

        active = {"current": 0, "peak": 0}

        def load() -> object:
            """Measure loader overlap while the production cache lock is held."""

            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
            time.sleep(0.08)
            active["current"] -= 1
            return object()

        loader = Mock(side_effect=load)
        loaders = {"cifar10": loader, "cifar100": Mock(side_effect=ValueError("bad cache"))}

        def prepare(dataset_name: str) -> None:
            """Use the same narrow lock boundary as the production CIFAR loader."""

            with dataset_load_lock(dataset_name):
                loaders[dataset_name]()

        with patch("common.hpo_process.tempfile.gettempdir", return_value=str(self.root)):
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.assertEqual(list(executor.map(prepare, ["cifar10", "cifar10"])), [None, None])
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(active["peak"], 1)
            with self.assertRaisesRegex(ValueError, "bad cache"):
                prepare("cifar100")
            loaders["cifar100"].side_effect = None
            self.assertIsNone(prepare("cifar100"))
            with self.assertRaisesRegex(ValueError, "cifar10 and cifar100"):
                prepare("mnist")

    def test_cleanup_survives_interrupt_and_kills_stubborn_workers(self) -> None:
        """One interrupted termination cannot skip cleanup of later children."""

        handles = []
        for index in range(2):
            child = Mock()
            child.poll.return_value = None
            child.wait.side_effect = [subprocess.TimeoutExpired("worker", 0), 0]
            stream = io.BytesIO()
            handles.append(WorkerHandle(child, self.root / f"r{index}", self.root / f"l{index}", stream))
        handles[0].process.terminate.side_effect = KeyboardInterrupt
        stop_workers(handles)
        for handle in handles:
            handle.process.kill.assert_called_once()
            self.assertTrue(handle.log_file.closed)
            handle.process.stdin.close.assert_called_once()

    def test_all_valid_statuses_cross_real_process_boundary(self) -> None:
        """Complete, divergent, OOM, and ordinary errors preserve their evidence."""

        for status in ("complete", "pruned", "oom", "error"):
            with self.subTest(status=status):
                payload = self._payload(status)
                script = (
                    "import pathlib,sys\n"
                    f"pathlib.Path(sys.argv[1]).write_text({json.dumps(payload)!r})\n"
                    f"raise SystemExit({0 if status == 'complete' else 1})\n"
                )
                handle = self._spawn(script)
                handle.process.wait(timeout=10)
                self.assertEqual(finish_worker(handle), payload)
                self.assertTrue(handle.log_file.closed)
                self.assertTrue(handle.process.stdin.closed)

    def test_invalid_crashed_and_missing_payloads_are_rejected(self) -> None:
        """Malformed or partial reports cannot be mistaken for successful trials."""

        valid = self._payload()
        cases = [
            ("invalid-json", 0), ("[]", 0), (None, 3),
            (json.dumps(valid), 3),
            (json.dumps({**valid, "status": "unknown"}), 0),
            (json.dumps({**valid, "config_path": ""}), 0),
            (json.dumps({**valid, "history": None}), 0),
            (json.dumps({**valid, "evaluations": []}), 0),
            (json.dumps({**valid, "results_path": None}), 0),
            (json.dumps({**self._payload("error"), "error": None}), 1),
            (json.dumps({**self._payload("pruned"), "divergence": None}), 1),
        ]
        for contents, exit_code in cases:
            with self.subTest(contents=contents, exit_code=exit_code):
                script = "import pathlib,sys\n"
                # An absent output models interpreter crashes and forced kills.
                if contents is not None:
                    script += f"pathlib.Path(sys.argv[1]).write_text({contents!r})\n"
                script += f"raise SystemExit({exit_code})\n"
                handle = self._spawn(script)
                handle.process.wait(timeout=10)
                with self.assertRaisesRegex(RuntimeError, "worker-|exit"):
                    finish_worker(handle)
                self.assertTrue(handle.log_file.closed)

    def test_live_workers_cannot_be_finished_and_cleanup_is_idempotent(self) -> None:
        """Cancellation stops multiple children and never reads a running result."""

        handles = [self._spawn("import time; time.sleep(60)") for _ in range(2)]
        with self.assertRaisesRegex(RuntimeError, "still running"):
            finish_worker(handles[0])
        self.assertFalse(handles[0].log_file.closed)
        stop_workers(handles)
        stop_workers(handles)
        for handle in handles:
            self.assertIsNotNone(handle.process.poll())
            self.assertTrue(handle.log_file.closed)
            self.assertTrue(handle.process.stdin.closed)

    def test_parent_pipe_eof_exits_before_training(self) -> None:
        """The real worker exits when its coordinator's liveness descriptor closes."""

        handle = start_worker(
            self.config, self.root / "orphan.json", self.root / "orphan.log",
            gpu_memory_limit_mb=None,
        )
        self.handles.append(handle)
        handle.process.stdin.close()
        self.assertEqual(handle.process.wait(timeout=15), 1)
        self.assertFalse(handle.output_path.exists())

    def test_interrupted_popen_cannot_leave_a_training_child(self) -> None:
        """A created child times out safely if Popen never returns to its caller."""

        original_popen = subprocess.Popen
        output = self.root / "unacknowledged.json"
        log = self.root / "unacknowledged.log"

        def interrupted_launch(command: list[str], **options: object) -> subprocess.Popen:
            """Reproduce an interrupt after OS creation but before assignment."""

            child = original_popen(command, **options)
            self.handles.append(WorkerHandle(child, output, log, options["stdout"]))
            raise KeyboardInterrupt

        with patch("common.hpo_process.subprocess.Popen", side_effect=interrupted_launch):
            with self.assertRaises(KeyboardInterrupt):
                start_worker(self.config, output, log, gpu_memory_limit_mb=None)
        handle = self.handles[-1]
        self.assertFalse(handle.process.stdin.closed)
        self.assertEqual(handle.process.wait(timeout=15), 1)
        self.assertFalse(output.exists())
        self.assertIn("did not acknowledge worker startup", log.read_text())

    def test_real_tensorflow_trial_runs_in_isolated_worker(self) -> None:
        """Train/evaluate a tiny real profile through YAML, CLI, and JSON on CPU."""

        from common.config import load_config, save_config
        from common.hpo_profiles import build_joint_classifier_config
        from common.tests.test_joint_hpo_profile import _Trial

        config = build_joint_classifier_config(
            _Trial(), dataset_name="cifar10", epochs=1, seed=17,
            results_path=self.root / "runs", dtype_policy="float32",
            validation_source="test", max_train_samples=4, max_val_samples=2,
            search_space_overrides={
                "dim": [32], "depth": [3], "clf_depth": [1], "patch_size": [4],
                "mha_num_heads": [4], "clf_train_batch_fraction": [0.5],
                "clf_train_noisy_input_type": ["clean"],
                "clf_train_class_input_type": ["null_class_only"],
            },
        )
        config.dataset.batch_size = 4
        config.model.show_network_summary = False
        config.model.kwargs.update(dim=8, depth=1, mha_num_heads=1, clf_mha_num_heads=1, timesteps=4)
        config.model.wrapper_kwargs.update(test_steps=2)
        config.training.verbose = 0
        config.training.tensorboard = False
        config.reporting.save_history_plot = False
        config.reporting.save_final_images = False
        config.reporting.save_final_gifs = False
        save_config(config, self.config)
        original_popen = subprocess.Popen
        # Patch only data loading in the fresh process; all training remains real.
        fixture = (
            "import sys,runpy,numpy as np\n"
            "from unittest.mock import patch\n"
            "rng=np.random.default_rng(17)\n"
            "data=((rng.integers(0,256,(8,32,32,3),dtype=np.uint8),(np.arange(8)%2).reshape(-1,1)),"
            "(rng.integers(0,256,(4,32,32,3),dtype=np.uint8),(np.arange(4)%2).reshape(-1,1)))\n"
            "sys.argv=['common.hpo_worker',*sys.argv[1:]]\n"
            "with patch('tensorflow.keras.datasets.cifar10.load_data',return_value=data):\n"
            " runpy.run_module('common.hpo_worker',run_name='__main__')\n"
        )

        def synthetic_worker(command: list[str], **options: object) -> subprocess.Popen:
            """Replace only the executable shim so the real CLI sees synthetic rows."""

            return original_popen([command[0], "-u", "-c", fixture, *command[4:]], **options)

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3"}), \
                patch("common.hpo_process.subprocess.Popen", side_effect=synthetic_worker):
            handle = start_worker(
                self.config, self.root / "trained.json", self.root / "trained.log",
                gpu_memory_limit_mb=None,
            )
        self.handles.append(handle)
        handle.process.wait(timeout=180)
        result = finish_worker(handle)
        self.assertEqual(result["status"], "complete", handle.log_path.read_text())
        self.assertEqual(len(result["history"]["loss"]), 1)
        self.assertFalse(any(key.startswith("val_") for key in result["history"]))
        self.assertNotIn("model", result)
        self.assertIn("classifier_accuracy", result["evaluations"]["valset_network_eval"])
        self.assertIn("noise_loss", result["evaluations"]["valset_network_eval"])
        resolved = load_config(result["config_path"])
        self.assertEqual(resolved.dataset.trainset_len, 1)
        self.assertEqual(resolved.dataset.split_metadata["training_rows_per_epoch"], 4)
        self.assertEqual(resolved.model.wrapper_kwargs["clf_train_batch_fraction"], 0.5)


class _Diverged(FloatingPointError):
    """Represent the real guard's exception without importing TensorFlow."""

    def __init__(self) -> None:
        """Attach the evidence fields consumed by the worker."""

        super().__init__("loss is nonfinite")
        self.evidence = {"metric": "loss", "value": "nan"}
        self.evidence_path = None


class HpoWorkerTests(unittest.TestCase):
    """Check worker serialization, memory policy, and failure classification."""

    def _run_fake(self, root: Path, *, cap: float | None = None,
                  failure: Exception | None = None, metrics: object = None) -> tuple[dict, Mock, int]:
        """Inject the training boundary while running real worker envelope logic."""

        tensorflow = Mock()
        tensorflow.errors.ResourceExhaustedError = MemoryError
        tensorflow.config.list_physical_devices.return_value = ["gpu0"]
        config = SimpleNamespace(training=SimpleNamespace(results_path=str(root)))
        training = Mock()
        training.main.side_effect = failure
        training.main.return_value = {
            "model": object(), "history": {"loss": [1.0]},
            "evaluations": {"valset_network_eval": {"accuracy": 0.5 if metrics is None else metrics}},
            "results_path": str(root),
        }
        modules = {
            "tensorflow": tensorflow,
            "common.train": training,
            "common.config": SimpleNamespace(load_config=Mock(return_value=config), save_config=Mock()),
            "common.callbacks.hpo_guard": SimpleNamespace(TrainingDiverged=_Diverged),
        }
        output = root / "result.json"
        with patch.dict(sys.modules, modules), patch.dict(os.environ), contextlib.redirect_stderr(io.StringIO()):
            code = run_worker(root / "input.yaml", output, gpu_memory_limit_mb=cap, threads=2)
        return json.loads(output.read_text()), tensorflow, code

    def test_growth_and_cap_configure_devices_before_training(self) -> None:
        """The saved config and JSON payload omit live model objects."""

        for cap in (None, 1536.0):
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload, tensorflow, code = self._run_fake(root, cap=cap)
                self.assertEqual(code, 0)
                self.assertEqual(payload["status"], "complete")
                self.assertEqual(payload["config_path"], str(root / "config.yaml"))
                self.assertNotIn("model", payload)
                tensorflow.config.threading.set_intra_op_parallelism_threads.assert_called_once_with(2)
                tensorflow.config.threading.set_inter_op_parallelism_threads.assert_called_once_with(2)
                # Growth and explicit limits are mutually exclusive policies.
                if cap is None:
                    tensorflow.config.experimental.set_memory_growth.assert_called_once_with("gpu0", True)
                    tensorflow.config.set_logical_device_configuration.assert_not_called()
                # A capped worker must never configure memory growth as well.
                else:
                    tensorflow.config.experimental.set_memory_growth.assert_not_called()
                    tensorflow.config.LogicalDeviceConfiguration.assert_called_once_with(memory_limit=cap)
                    tensorflow.config.set_logical_device_configuration.assert_called_once()

    def test_worker_failures_and_unsupported_metrics_are_published(self) -> None:
        """Exceptions always produce an actionable atomic envelope when possible."""

        cases = [(_Diverged(), "pruned"), (MemoryError("oom"), "oom"), (ValueError("bad config"), "error")]
        for failure, status in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                payload, _, code = self._run_fake(Path(directory), failure=failure)
                self.assertEqual(code, 1)
                self.assertEqual(payload["status"], status)
                self.assertTrue(payload["error"])
                self.assertEqual(payload["history"], {})
        with tempfile.TemporaryDirectory() as directory:
            payload, _, code = self._run_fake(Path(directory), metrics=object())
            self.assertEqual(code, 1)
            self.assertEqual(payload["status"], "error")
            self.assertIn("Unsupported worker metric", payload["error"])
            self.assertEqual(payload["evaluations"], {})
            self.assertFalse((Path(directory) / "result.json.tmp").exists())

    def test_nonfinite_metrics_reach_coordinator_unchanged(self) -> None:
        """Final scoring retains responsibility for pruning nonfinite objectives."""

        with tempfile.TemporaryDirectory() as directory:
            payload, _, code = self._run_fake(Path(directory), metrics=float("inf"))
            self.assertEqual(code, 0)
            self.assertEqual(payload["evaluations"]["valset_network_eval"]["accuracy"], float("inf"))

    def test_invalid_direct_options_produce_error_without_import(self) -> None:
        """Direct CLI calls validate limits even when not launched by the helper."""

        for arguments in ({"threads": False}, {"gpu_memory_limit_mb": float("nan")}):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "result.json"
                with contextlib.redirect_stderr(io.StringIO()):
                    code = run_worker(Path(directory) / "input.yaml", output, **arguments)
                self.assertEqual(code, 1)
                self.assertEqual(json.loads(output.read_text())["status"], "error")


# Permit running this focused transport suite without the repository-wide suite.
if __name__ == "__main__":
    unittest.main()
