"""Resource-admission and startup-order checks for isolated HPO workers."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from common.hpo_process import finish_worker, start_worker, stop_workers, normalize_worker_gpu_ids
from common.hpo_worker import run_worker


class HpoWorkerResourceTests(unittest.TestCase):
    """Keep resource leases alive until every owned child is confirmed stopped."""

    def setUp(self) -> None:
        """Prepare one valid input without importing a training framework."""

        temporary = tempfile.TemporaryDirectory(prefix="hpo-resource-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "input.yaml"
        self.config.write_text("{}", encoding="utf-8")
        self.events = []
        self.child = Mock(pid=24680)
        self.child.poll.return_value = None
        self.child.wait.side_effect = self._wait
        self.child.stdin.write.side_effect = lambda value: self.events.append("authorize")
        self.child.stdin.flush.side_effect = lambda: self.events.append("flush")
        self.register = Mock(side_effect=lambda pid: self.events.append(("register", pid)))
        self.handles = []
        self.addCleanup(stop_workers, self.handles)

    def _wait(self, **kwargs: object) -> int:
        """Record that the process was reaped before resource cleanup."""

        self.events.append("wait")
        self.child.poll.return_value = 0
        return 0

    @contextmanager
    def _resource(self) -> object:
        """Expose resource registration and record its held lifetime."""

        self.events.append("enter")
        try:
            yield {"environment": {"CUDA_VISIBLE_DEVICES": "0", "ADMITTED": "yes"}, "register": self.register}
        finally:
            self.events.append("release")

    def _start(self) -> object:
        """Launch the mocked process through the complete production transport."""

        with patch("common.hpo_process.subprocess.Popen", return_value=self.child) as popen:
            handle = start_worker(
                self.config, self.root / "result.json", self.root / "worker.log", 
                gpu_memory_limit_mb=2048, worker_context=self._resource
            )
        self.handles.append(handle)
        return handle, popen

    def test_registration_precedes_authorization_and_cleanup_follows_exit(self) -> None:
        """An admitted child receives its environment before its startup byte."""

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
            handle, popen = self._start()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(popen.call_args.kwargs["env"]["ADMITTED"], "yes")
        self.assertEqual(self.events[:4], ["enter", ("register", 24680), "authorize", "flush"])
        self.assertNotIn("release", self.events)
        stop_workers([handle])
        self.assertLess(self.events.index("wait"), self.events.index("release"))
        stop_workers([handle])
        self.assertEqual(self.events.count("release"), 1)
        self.assertIsNone(handle.resource_stack)

    def test_explicit_gpu_is_visible_before_child_authorization(self) -> None:
        """Transport sends only the selected GPU while leaving the coordinator hidden."""

        for gpu_id, expected in ((1, "1"), ("GPU-second", "GPU-second")):
            with self.subTest(gpu_id=gpu_id), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}), \
                    patch("common.hpo_process.subprocess.Popen", return_value=self.child) as popen:
                handle = start_worker(
                    self.config, self.root / "result.json", self.root / "worker.log", 
                    gpu_memory_limit_mb=2048, gpu_id=gpu_id
                )
                self.handles.append(handle)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
                self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], expected)
                self.assertEqual(popen.call_args.kwargs["env"]["HPO_WORKER_GPU_ID"], expected)
                stop_workers([handle])

    def test_gpu_context_receives_the_assigned_device_before_authorization(self) -> None:
        """GPU-aware admission holds the selected device through process reaping."""

        selections = []

        @contextmanager
        def resource(gpu_id: str) -> object:
            """Return an admission record for exactly the requested device."""

            selections.append(gpu_id)
            self.events.append("enter")
            try:
                yield {"environment": {"CUDA_VISIBLE_DEVICES": gpu_id}, "register": self.register}
            finally:
                self.events.append("release")

        with patch("common.hpo_process.subprocess.Popen", return_value=self.child) as popen:
            handle = start_worker(
                self.config, self.root / "result.json", self.root / "worker.log", 
                gpu_memory_limit_mb=2048, gpu_id="GPU-second", gpu_worker_context=resource
            )
        self.handles.append(handle)
        self.assertEqual(selections, ["GPU-second"])
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-second")
        self.assertEqual(self.events[:4], ["enter", ("register", 24680), "authorize", "flush"])
        stop_workers([handle])
        self.assertLess(self.events.index("wait"), self.events.index("release"))

    def test_conflicting_resource_gpu_fails_before_spawning_and_releases(self) -> None:
        """A resource hook cannot move a worker onto another GPU after slot selection."""

        with patch("common.hpo_process.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "does not match"):
                start_worker(
                    self.config, self.root / "result.json", self.root / "worker.log", 
                    gpu_memory_limit_mb=2048, worker_context=self._resource, gpu_id=1
                )
        popen.assert_not_called()
        self.register.assert_not_called()
        self.assertEqual(self.events, ["enter", "release"])

    def test_gpu_selectors_and_admission_modes_are_validated_before_entry(self) -> None:
        """Malformed selectors and ambiguous admission cannot create children or leases."""

        factory = Mock()
        invalid = [
            {"gpu_id": "0,1"}, {"gpu_id": -1}, {"gpu_id": True}, 
            {"gpu_worker_context": factory}, 
            {"gpu_id": 0, "worker_context": factory, "gpu_worker_context": factory}
        ]
        for options in invalid:
            with self.subTest(options=options), patch("common.hpo_process.subprocess.Popen") as popen:
                with self.assertRaises(ValueError):
                    start_worker(
                        self.config, self.root / "result.json", self.root / "worker.log", 
                        gpu_memory_limit_mb=2048, **options
                    )
                popen.assert_not_called()
        factory.assert_not_called()
        self.assertIsNone(normalize_worker_gpu_ids(None))
        self.assertEqual(normalize_worker_gpu_ids([0, "01", "GPU-second"]), ("0", "1", "GPU-second"))
        for selection in ([], "0", [0, "00"], [" 0"], ["GPU-x,1"], [False], ["-1"]):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                normalize_worker_gpu_ids(selection)

    def test_registration_failure_never_authorizes_and_reaps_before_release(self) -> None:
        """A failed ownership record cannot authorize framework initialization."""

        self.register.side_effect = RuntimeError("registration failed")
        with patch("common.hpo_process.subprocess.Popen", return_value=self.child):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                start_worker(
                    self.config, self.root / "result.json", self.root / "worker.log", 
                    gpu_memory_limit_mb=2048, worker_context=self._resource
                )
        self.child.stdin.write.assert_not_called()
        self.assertLess(self.events.index("wait"), self.events.index("release"))

    def test_spawn_failure_releases_admission(self) -> None:
        """A failed process creation leaves no external resource reservation."""

        with patch("common.hpo_process.subprocess.Popen", side_effect=OSError("spawn failed")):
            with self.assertRaisesRegex(OSError, "spawn failed"):
                start_worker(
                    self.config, self.root / "result.json", self.root / "worker.log", 
                    gpu_memory_limit_mb=2048, worker_context=self._resource
                )
        self.assertEqual(self.events, ["enter", "release"])
        self.register.assert_not_called()

    def test_invalid_arguments_do_not_enter_resource_context(self) -> None:
        """Admission happens only after ordinary transport validation succeeds."""

        resource = Mock()
        with self.assertRaises(ValueError):
            start_worker(
                self.config, self.root / "result.json", self.root / "worker.log", 
                gpu_memory_limit_mb=-1, worker_context=resource
            )
        resource.assert_not_called()

    def test_unreaped_worker_retains_admission_until_later_cleanup(self) -> None:
        """Failed termination cannot free a reservation while its child survives."""

        handle, _ = self._start()
        self.child.wait.side_effect = subprocess.TimeoutExpired("worker", 0)
        stop_workers([handle])
        self.assertNotIn("release", self.events)
        self.assertIsNotNone(handle.resource_stack)
        self.child.wait.side_effect = self._wait
        stop_workers([handle])
        self.assertEqual(self.events.count("release"), 1)

    def test_finish_releases_completed_or_malformed_worker_resources(self) -> None:
        """Result validation cannot retain admission after a confirmed child exit."""

        for malformed in [False, True]:
            with self.subTest(malformed=malformed):
                self.events.clear()
                self.child.poll.return_value = None
                handle, _ = self._start()
                with self.assertRaisesRegex(RuntimeError, "still running"):
                    finish_worker(handle)
                self.assertNotIn("release", self.events)
                handle.output_path.write_text(
                    "invalid-json" if malformed else json.dumps({
                        "status": "error", "error": "training failed", "config_path": str(self.config)
                    }), 
                    encoding="utf-8"
                )
                self.child.poll.return_value = 1
                # Malformed payloads fail while still releasing completed resources.
                if malformed:
                    with self.assertRaisesRegex(RuntimeError, "invalid"):
                        finish_worker(handle)
                # A valid error envelope remains available for HPO finalization.
                else:
                    self.assertEqual(finish_worker(handle)["status"], "error")
                self.assertEqual(self.events.count("release"), 1)

    def test_worker_admission_failure_precedes_tensorflow_import(self) -> None:
        """An invalid lease produces an error envelope without loading TensorFlow."""

        prepare = Mock(side_effect=RuntimeError("invalid lease"))
        module = SimpleNamespace(prepare_hpo_worker=prepare)
        original_import = __import__

        def guarded_import(module_name: str, *args: object, **kwargs: object) -> object:
            """Reject any framework import after failed worker authentication."""

            # The worker must return its admission failure before loading TensorFlow.
            if module_name == "tensorflow":
                self.fail("TensorFlow imported before valid admission")
            return original_import(module_name, *args, **kwargs)

        output = self.root / "result.json"
        with patch.dict(os.environ, {"DIT_HPO_PARALLEL_IDENTITY": "{}"}),                 patch.dict(sys.modules, {"common.dit_hpo_remote": module}),                 patch("builtins.__import__", side_effect=guarded_import), redirect_stderr(io.StringIO()):
            code = run_worker(self.config, output, gpu_memory_limit_mb=2048)
        self.assertEqual(code, 1)
        self.assertIn("invalid lease", json.loads(output.read_text())["error"])
        prepare.assert_called_once_with(2048)


    def _run_admitted(
        self, 
        physical_devices: list | None = None, 
        logical_devices: list | None = None, 
        tensorflow_version: str = "2.20.0", 
        admitted: bool = True, 
        explicit_gpu: bool = False
    ) -> tuple:
        """Run the worker envelope with an admitted, fully mocked GPU runtime."""

        tensorflow = Mock()
        tensorflow.__version__ = tensorflow_version
        tensorflow.errors.ResourceExhaustedError = MemoryError
        tensorflow.config.list_physical_devices.return_value = ["gpu0"] if physical_devices is None else physical_devices
        tensorflow.config.set_logical_device_configuration.side_effect = lambda *args: self.events.append("cap")

        def logical_inventory(kind: str) -> list:
            """Record when GPU initialization follows its memory configuration."""

            self.events.append("logical")
            return ["logical0"] if logical_devices is None else logical_devices

        tensorflow.config.list_logical_devices.side_effect = logical_inventory
        config = SimpleNamespace(training=SimpleNamespace(results_path=str(self.root)))
        train = Mock(return_value={
            "results_path": str(self.root), "history": {"loss": [1.0]}, "evaluations": {}
        })
        prepare = Mock()
        modules = {
            "tensorflow": tensorflow, 
            "common.dit_hpo_remote": SimpleNamespace(prepare_hpo_worker=prepare), 
            "common.config": SimpleNamespace(load_config=Mock(return_value=config), save_config=Mock()), 
            "common.train": SimpleNamespace(main=train), 
            "common.callbacks.hpo_guard": SimpleNamespace(TrainingDiverged=FloatingPointError)
        }
        output = self.root / "admitted.json"
        environment = dict(os.environ)
        environment.pop("DIT_HPO_PARALLEL_IDENTITY", None)
        environment.pop("HPO_WORKER_GPU_ID", None)
        # Admission and explicit routing independently require a working GPU.
        if admitted:
            environment["DIT_HPO_PARALLEL_IDENTITY"] = "{}"
        # Direct API workers carry routing even without notebook admission metadata.
        if explicit_gpu:
            environment.update({"HPO_WORKER_GPU_ID": "GPU-selected", "CUDA_VISIBLE_DEVICES": "GPU-selected"})
        with patch.dict(os.environ, environment, clear=True), \
                patch.dict(sys.modules, modules), redirect_stderr(io.StringIO()):
            code = run_worker(self.config, output, gpu_memory_limit_mb=2048)
        # Only admitted notebook workers invoke the notebook's resource verifier.
        if admitted:
            prepare.assert_called_once_with(2048)
        # Generic API routing retains its own lightweight device checks.
        else:
            prepare.assert_not_called()
        return code, json.loads(output.read_text()), tensorflow, train

    def test_admitted_worker_initializes_one_gpu_after_memory_cap(self) -> None:
        """Valid admitted trials configure a cap before logical GPU initialization."""

        code, payload, tensorflow, train = self._run_admitted()
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(self.events, ["cap", "logical"])
        tensorflow.config.LogicalDeviceConfiguration.assert_called_once_with(memory_limit=2048)
        train.assert_called_once()

    def test_admitted_worker_rejects_missing_or_multiple_physical_gpus(self) -> None:
        """An admitted GPU lease can never silently train on CPU or another device."""

        for devices in [[], ["gpu0", "gpu1"]]:
            with self.subTest(devices=devices):
                code, payload, tensorflow, train = self._run_admitted(physical_devices=devices)
                self.assertEqual(code, 1)
                self.assertIn("exactly one physical GPU", payload["error"])
                tensorflow.config.set_logical_device_configuration.assert_not_called()
                tensorflow.config.list_logical_devices.assert_not_called()
                train.assert_not_called()

    def test_admitted_worker_rejects_gpu_initialization_failure(self) -> None:
        """A physical inventory without a usable logical GPU is not training admission."""

        code, payload, tensorflow, train = self._run_admitted(logical_devices=[])
        self.assertEqual(code, 1)
        self.assertIn("did not initialize successfully", payload["error"])
        self.assertEqual(self.events, ["cap", "logical"])
        train.assert_not_called()

    def test_explicit_routing_without_admission_requires_one_working_gpu(self) -> None:
        """Generic GPU selection cannot silently fall back to CPU or multiple devices."""

        for physical, logical, expected in (([], [], 1), (["gpu0", "gpu1"], [], 1), (["gpu0"], [], 1), (["gpu0"], ["logical0"], 0)):
            with self.subTest(physical=physical, logical=logical):
                self.events.clear()
                code, payload, tensorflow, train = self._run_admitted(
                    physical_devices=physical, logical_devices=logical, admitted=False, explicit_gpu=True
                )
                self.assertEqual(code, expected)
                # Only one successfully initialized GPU may reach the training call.
                if expected == 0:
                    train.assert_called_once()
                    self.assertEqual(self.events, ["cap", "logical"])
                # Missing, extra, or failed GPUs leave training unstarted.
                else:
                    self.assertEqual(payload["status"], "error")
                    train.assert_not_called()

    def test_admitted_worker_rejects_imported_runtime_mismatch(self) -> None:
        """A different imported TensorFlow cannot bypass the preflight package check."""

        code, payload, tensorflow, train = self._run_admitted(tensorflow_version="2.19.0")
        self.assertEqual(code, 1)
        self.assertIn("imported TensorFlow runtime", payload["error"])
        tensorflow.config.list_physical_devices.assert_not_called()
        train.assert_not_called()

# This focused suite performs no TensorFlow imports or training.
if __name__ == "__main__":
    unittest.main()
