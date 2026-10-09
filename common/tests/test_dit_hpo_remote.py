"""Check GPU admission orchestration without claiming slots or importing TF."""

from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
from typing import Any, Iterator
import unittest
from unittest.mock import Mock, patch

from common import dit_hpo_remote as remote


class AdmissionBlocked(RuntimeError):
    """Represent the deployed allocator's fail-closed admission error."""


class FakeProcess:
    """Track subprocess lifetime without creating a real worker."""

    def __init__(self, events: list[str]) -> None:
        """Retain the lifecycle event log and initial running state."""

        self.pid = 42
        self.events = events
        self.returncode = None

    def poll(self) -> int | None:
        """Return the fake worker's current exit status."""

        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        """Record that the coordinator reaped the worker."""

        self.events.append("wait")
        self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        """Record a termination request without pretending it was reaped."""

        self.events.append("terminate")

    def kill(self) -> None:
        """Record forced termination of the fake worker."""

        self.events.append("kill")


@contextmanager
def fake_launch(runtime_kind: str = "runpod") -> Iterator[tuple[FakeProcess, SimpleNamespace, SimpleNamespace, list[str]]]:
    """Replace admission, OS pipes, and subprocess creation with event mocks."""

    events = []
    process = FakeProcess(events)
    lease = SimpleNamespace(
        record={"slot": 0, "owner": {"pid": 1, "start_ticks": "2"}}, 
        register_worker=Mock(side_effect=lambda pid: events.append("register")), 
        close=Mock(side_effect=lambda: events.append("close"))
    )
    allocator = SimpleNamespace(AdmissionBlocked=AdmissionBlocked, acquire=Mock(return_value=lease))
    policy = {"max_jobs": 2, "tf_memory_mib": 12288, "reserved_memory_mib": 13312}
    # Hosted fixtures have one worker and a budget appropriate to measured 15-GiB capacity.
    if runtime_kind != "runpod":
        policy = {"max_jobs": 1, "tf_memory_mib": 11264, "reserved_memory_mib": 12288}
    identity = {
        "checkout_root": "/workspace/test", "pool_host": "a100_2", 
        "runtime_kind": runtime_kind, "worker_policy": policy
    }

    def start_process(command: list[str], **kwargs: Any) -> FakeProcess:
        """Retain the launch configuration and record process creation."""

        events.append("spawn")
        process.launch_options = kwargs
        return process

    patches = [
        (remote, "_verify_snapshot", Mock(return_value=identity)), 
        (remote, "_in_existing_allocation", Mock(return_value=False)), 
        (remote, "_allocator", Mock(return_value=allocator)), 
        (remote, "sys", SimpleNamespace(modules={})), 
        (remote.subprocess, "Popen", Mock(side_effect=start_process)), 
        (remote.os, "pipe", Mock(return_value=(101, 102))), 
        (remote.os, "close", Mock()), 
        (remote.os, "write", Mock(side_effect=lambda descriptor, value: events.append("gate"))), 
        (remote.time, "sleep", Mock())
    ]
    with ExitStack() as stack:
        for target, attribute, replacement in patches:
            stack.enter_context(patch.object(target, attribute, replacement))
        yield process, lease, allocator, events


@contextmanager
def hosted_checkout(runtime_kind: str = "colab", total_mib: int = 15360) -> Iterator[Path]:
    """Build a source-only hosted checkout and mock live provider/GPU evidence."""

    markers = {"COLAB_RELEASE_TAG": "test"} if runtime_kind == "colab" else {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}
    versions = {"tensorflow": "2.20.0", "keras": "3.11.2", "optuna": "5.0.0"}
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        (root / "common").mkdir()
        (root / "common/hpo.py").write_text("# source identity\n")
        (root / "common/gpu_resource_slots.py").write_text("# admission identity\n")
        stack.enter_context(patch.dict(remote.os.environ, markers, clear=True))
        stack.enter_context(patch.object(remote, "sys", SimpleNamespace(modules={})))
        stack.enter_context(patch.object(remote, "_ADMITTED_HOSTED_RUNTIME", None))
        stack.enter_context(patch.object(remote.platform, "system", return_value="Linux"))
        stack.enter_context(patch.object(remote.socket, "gethostname", return_value="hosted-test"))
        stack.enter_context(patch.object(remote.importlib.metadata, "version", side_effect=versions.__getitem__))
        stack.enter_context(patch.object(remote, "_query", side_effect=lambda arguments:
            [["GPU-hosted", "Hosted accelerator", str(total_mib - 512), str(total_mib)]]
            if "-i" in arguments else []))
        yield root

@contextmanager
def fake_parallel() -> Iterator[dict[str, Any]]:
    """Model independent lease owners and exact child registrations without GPUs."""

    events = []
    owner = {"pid": 1, "start_ticks": "owner", "parent_pid": 0, "namespace_pids": [1]}
    coordinator = {"pid": 20, "start_ticks": "coordinator", "parent_pid": 1, "namespace_pids": [20]}
    identities = {
        1: owner, 20: coordinator, 
        30: {"pid": 30, "start_ticks": "worker0", "parent_pid": 20, "namespace_pids": [30]}, 
        31: {"pid": 31, "start_ticks": "worker1", "parent_pid": 20, "namespace_pids": [31]}
    }
    records = {
        0: {"slot": 0, "owner": owner, "workers": [coordinator]}, 
        1: {"slot": 1, "owner": coordinator, "workers": []}
    }
    lease = SimpleNamespace(record=records[1], close=Mock(side_effect=lambda: events.append("close-second")))

    def acquire(**kwargs: Any) -> SimpleNamespace:
        """Reject overlapping ownership before granting the coordinator its own slot."""

        # Distinct slots must never claim the coordinator identity simultaneously.
        if any(item["pid"] == 20 for item in records[0]["workers"]):
            raise AdmissionBlocked("This process already owns or overlaps an active slot identity.")
        events.append("acquire-second")
        return lease

    allocator = SimpleNamespace(AdmissionBlocked=AdmissionBlocked, acquire=Mock(side_effect=acquire), _identity=identities.__getitem__)
    identity = {
        "checkout_root": "/workspace/test", "runtime_kind": "runpod", "pool_host": "a100_2", 
        "gpu_uuid": "GPU-test", "concurrent_trials": 2, 
        "worker_policy": {"max_jobs": 2, "tf_memory_mib": 12288, "reserved_memory_mib": 13312}
    }

    def edit(slot: int, expected_owner: dict[str, Any], worker: dict[str, Any], expected_identity: dict[str, Any], module: Any, add: bool) -> None:
        """Track exact registration mutation with the same slot-owner boundary."""

        # Test fixtures must preserve both the verified reservation and slot owner.
        if expected_owner != records[slot]["owner"] or expected_identity != identity:
            raise RuntimeError("Unexpected lease identity.")
        records[slot]["workers"] = [item for item in records[slot]["workers"] if item != worker]
        # Only registration writes add the verified process to its selected slot.
        if add:
            records[slot]["workers"].append(worker)
        events.append(("add-" if add else "remove-") + str(worker["pid"]) + "-slot" + str(slot))

    patches = [
        (remote, "sys", SimpleNamespace(modules={})), 
        (remote, "_arm_parent_exit", Mock()), 
        (remote, "_validate_worker_lease", Mock(return_value=records[0])), 
        (remote, "_verify_snapshot", Mock(return_value=identity)), 
        (remote, "_allocator", Mock(return_value=allocator)), 
        (remote, "_process_identity", Mock(side_effect=identities.__getitem__)), 
        (remote, "_edit_lease_worker", Mock(side_effect=edit)), 
        (remote, "_stop_owned_worker", Mock(side_effect=lambda worker: events.append("reap-" + str(worker["pid"])))), 
        (remote.os, "getpid", Mock(return_value=20)), 
        (remote.signal, "signal", Mock()), 
        (remote.signal, "getsignal", Mock(return_value=signal.SIG_DFL)), 
        (remote.time, "sleep", Mock())
    ]
    with ExitStack() as stack:
        stack.enter_context(patch.dict(remote.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True))
        for target, attribute, replacement in patches:
            stack.enter_context(patch.object(target, attribute, replacement))
        yield {
            "identity": identity, "records": records, "allocator": allocator, 
            "lease": lease, "events": events, "identities": identities
        }
@contextmanager
def fake_hosted_parallel() -> Iterator[dict[str, Any]]:
    """Extend the two-owner fixture with four atomically acquired hosted leases."""

    with fake_parallel() as fixture, patch.object(remote, "_ADMITTED_HOSTED_RUNTIME", None):
        identity = fixture["identity"]
        identity["runtime_kind"] = "colab"
        identity["pool_host"] = "colab"
        identity["concurrent_trials"] = 5
        identity["worker_policy"]["max_jobs"] = 5
        remote.os.environ["CONTINUAL_RUNTIME"] = "colab"
        coordinator = fixture["identities"][20]
        leases = []
        for slot in range(1, 5):
            fixture["records"][slot] = {
                "slot": slot, "owner": coordinator, "workers": [], 
                "version": 2, "allocation_group": "hosted-group"
            }
            fixture["identities"][30 + slot] = {
                "pid": 30 + slot, "start_ticks": "worker" + str(slot), 
                "parent_pid": 20, "namespace_pids": [30 + slot]
            }
            leases.append(SimpleNamespace(
                record=fixture["records"][slot], 
                close=Mock(side_effect=lambda value=slot: fixture["events"].append("close-slot" + str(value)))
            ))

        def acquire_many(**kwargs: Any) -> list[SimpleNamespace]:
            """Require detached CPU ownership before granting the complete group."""

            # The parent registration must be removed before the CPU owns a group.
            if any(item["pid"] == 20 for item in fixture["records"][0]["workers"]):
                raise AdmissionBlocked("This process already owns or overlaps an active slot identity.")
            fixture["events"].append("acquire-group")
            return leases

        fixture["allocator"].acquire_many = Mock(side_effect=acquire_many)
        fixture["leases"] = leases
        yield fixture


@contextmanager
def fake_multi_gpu() -> Iterator[dict[str, Any]]:
    """Model five notebook-owned reservations split over two physical devices."""

    with fake_hosted_parallel() as fixture:
        identity = fixture["identity"]
        identity["gpu_ids"] = [0, 1]
        identity["worker_policy"]["max_jobs"] = 3
        identity["gpus"] = [{
            "gpu_id": gpu, "gpu_uuid": "GPU-test" if gpu == 0 else "GPU-second", 
            "gpu_name": "Test GPU", "total_mib": 81920, "free_mib": 80000, "processes": [], 
            "concurrent_trials": count, "worker_policy": {**identity["worker_policy"], "max_jobs": count}
        } for gpu, count in [(0, 3), (1, 2)]]
        records = {}
        allocations = []
        for gpu, count in [(0, 3), (1, 2)]:
            for slot in range(count):
                records[(gpu, slot)] = {
                    "gpu": str(gpu), "gpu_uuid": identity["gpus"][gpu]["gpu_uuid"], "memory_mb": 13312, 
                    "slot": slot, "owner": fixture["identities"][1], "workers": []
                }
                allocations.append({"gpu_id": gpu, "slot": slot, "owner": fixture["identities"][1]})
        records[(0, 0)]["workers"] = [fixture["identities"][20]]
        remote.os.environ["DIT_HPO_GROUP_LEASES"] = json.dumps(allocations)

        def edit(slot: int, owner: dict[str, Any], worker: dict[str, Any], expected: dict[str, Any], allocator: Any, add: bool) -> None:
            """Track registrations by device and slot with exact notebook ownership."""

            record = records[(expected["gpu_id"], slot)]
            # The selected-device view must retain the correct physical UUID.
            if record["owner"] != owner or record["gpu_uuid"] != expected["gpu_uuid"]:
                raise RuntimeError("Unexpected multi-GPU lease identity.")
            record["workers"] = [item for item in record["workers"] if item != worker]
            # Registration adds only the caller's exact owned process.
            if add:
                record["workers"].append(worker)
            fixture["events"].append(("add-" if add else "remove-") + str(worker["pid"]))

        @contextmanager
        def locked(slot: int, owner: dict[str, Any], expected: dict[str, Any]) -> Iterator[tuple[dict[str, Any], Path]]:
            """Verify the requested group entry addresses its own physical GPU."""

            record = records[(expected["gpu_id"], slot)]
            # Reject a UUID or owner mismatch even if a slot number exists elsewhere.
            if record["gpu_uuid"] != expected["gpu_uuid"] or record["owner"] != owner:
                raise RuntimeError("Unexpected multi-GPU reservation.")
            yield record, Path("/unused")

        with patch.object(remote, "_edit_lease_worker", side_effect=edit), \
        patch.object(remote, "_locked_lease_record", side_effect=locked):
            fixture["multi_records"] = records
            yield fixture

class RemoteAdmissionTests(unittest.TestCase):
    """Exercise ownership, memory reservation, and worker cleanup boundaries."""

    def test_multi_gpu_inventory_assigns_round_robin_counts_and_common_cap(self) -> None:
        """Uneven quotas and heterogeneous capacity derive one enforceable worker cap."""

        with hosted_checkout(total_mib=81920) as root:
            def inventory(arguments: list[str]) -> list[list[str]]:
                """Return independently measured capacities for two selected GPUs."""

                # The process query reports only the second GPU's live workload.
                if "-i" not in arguments:
                    return [["GPU-four", "901", "python", "1024"]]
                gpu = arguments[arguments.index("-i") + 1]
                return [["GPU-two" if gpu == "2" else "GPU-four", "Test GPU", "40000", "81920" if gpu == "2" else "16384"]]

            with patch.object(remote, "_query", side_effect=inventory):
                identity = remote.inspect_remote(root, concurrent_trials=5, gpu_ids=[2, 4])
            self.assertEqual(identity["gpu_ids"], [2, 4])
            self.assertEqual([item["concurrent_trials"] for item in identity["gpus"]], [3, 2])
            self.assertEqual([item["worker_policy"]["tf_memory_mib"] for item in identity["gpus"]], [5632, 5632])
            self.assertEqual(identity["gpus"][0]["processes"], [])
            self.assertEqual(identity["gpus"][1]["processes"][0]["pid"], 901)
            self.assertEqual(remote._device_identity(identity, "GPU-four")["gpu_id"], 4)
            self.assertEqual(remote._device_identity(identity, "4")["gpu_uuid"], "GPU-four")
            with self.assertRaisesRegex(RuntimeError, "absent"):
                remote._device_identity(identity, 0)

    def test_explicit_worker_memory_roundtrips_and_remains_available_to_confirmations(self) -> None:
        """A verified 24-GiB request survives reinspection and a serial confirmation."""

        with hosted_checkout(total_mib=81920) as root:
            identity = remote.inspect_remote(root, concurrent_trials=3, worker_gpu_memory_limit_mb=24576)
            self.assertEqual(identity["worker_gpu_memory_limit_mb"], 24576)
            self.assertEqual(identity["worker_policy"], {
                "max_jobs": 3, "tf_memory_mib": 24576, "reserved_memory_mib": 25600
            })
            self.assertEqual(remote._verify_snapshot(root, identity), identity)
            confirmation = remote.serial_worker_identity(root, identity)
            self.assertEqual(confirmation["concurrent_trials"], 1)
            self.assertEqual(confirmation["worker_gpu_memory_limit_mb"], 24576)
            self.assertEqual(confirmation["worker_policy"], {
                "max_jobs": 1, "tf_memory_mib": 24576, "reserved_memory_mib": 25600
            })
            automatic = remote.inspect_remote(root, concurrent_trials=3)
            self.assertNotIn("worker_gpu_memory_limit_mb", automatic)
            self.assertEqual(automatic["worker_policy"]["tf_memory_mib"], remote.TF_MEMORY_MIB)

    def test_explicit_worker_memory_must_fit_every_device_quota_without_clipping(self) -> None:
        """An incompatible requested reservation fails instead of silently shrinking."""

        with hosted_checkout(total_mib=81920) as root:
            with self.assertRaisesRegex(ValueError, "budget and concurrency"):
                remote.inspect_remote(root, concurrent_trials=4, worker_gpu_memory_limit_mb=24576)
            def inventory(arguments: list[str]) -> list[list[str]]:
                """Provide two independently measured capacities for the explicit cap."""

                # The process query has no live GPU workers in this fixture.
                if "-i" not in arguments:
                    return []
                gpu = arguments[arguments.index("-i") + 1]
                return [["GPU-" + gpu, "Test GPU", "40000", "81920" if gpu == "0" else "16384"]]

            with patch.object(remote, "_query", side_effect=inventory):
                with self.assertRaisesRegex(ValueError, "budget and concurrency"):
                    remote.inspect_remote(root, concurrent_trials=2, gpu_ids=[0, 1], worker_gpu_memory_limit_mb=24576)
        with patch.object(remote, "_remote_root") as inspect:
            for budget in [True, 0, -1, 24576.5]:
                with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "positive integer"):
                    remote.inspect_remote("/unused", worker_gpu_memory_limit_mb=budget)
            inspect.assert_not_called()

    def test_multi_gpu_selection_rejects_duplicate_boolean_and_unused_devices(self) -> None:
        """Only exact distinct physical indices with positive assigned quotas are valid."""

        with patch.object(remote, "_remote_root") as inspect:
            for selection in [[], [0, 0], [False], [-1], [0, 1, 2]]:
                with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "gpu_ids"):
                    remote.inspect_remote("/unused", concurrent_trials=2, gpu_ids=selection)
            inspect.assert_not_called()

    def test_container_evidence_enables_generic_remote_runtime(self) -> None:
        """A Linux container can use generic GPU admission without provider branding."""

        with patch.dict(remote.os.environ, {}, clear=True), \
        patch.object(remote, "sys", SimpleNamespace(modules={})), \
        patch.object(remote, "_ADMITTED_HOSTED_RUNTIME", None), \
        patch.object(remote.Path, "is_file", side_effect=lambda: True):
            self.assertEqual(remote._runtime_kind("generic-container"), "container")

    def test_multi_gpu_contexts_bind_uuid_and_do_not_exceed_device_quota(self) -> None:
        """Five admitted workers bind three slots on GPU zero and two on GPU one."""

        with fake_multi_gpu() as fixture:
            with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]) as factory:
                with ExitStack() as stack:
                    contexts = [stack.enter_context(factory(gpu)) for gpu in ["GPU-test", "GPU-second", "GPU-test", "GPU-second", "GPU-test"]]
                    for index, context in enumerate(contexts):
                        context["register"](30 + index)
                        self.assertEqual(context["environment"]["CUDA_VISIBLE_DEVICES"], "GPU-test" if index % 2 == 0 else "GPU-second")
                    with self.assertRaisesRegex(RuntimeError, "already in use"):
                        with factory("GPU-second"):
                            self.fail("A third worker exceeded GPU one's quota.")
                    self.assertEqual(sum(bool(record["workers"]) for record in fixture["multi_records"].values()), 5)
            fixture["allocator"].acquire.assert_not_called()
            fixture["allocator"].acquire_many.assert_not_called()
            self.assertEqual(fixture["multi_records"][(0, 0)]["workers"], [fixture["identities"][20]])
            self.assertTrue(all(not record["workers"] for key, record in fixture["multi_records"].items() if key != (0, 0)))

    def test_multi_gpu_factory_accepts_core_normalized_physical_indices(self) -> None:
        """Core-normalized decimal selectors preserve device and CUDA identity."""

        with fake_multi_gpu() as fixture:
            with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]) as factory:
                with factory("1") as context:
                    context["register"](30)
                    self.assertEqual(context["environment"]["CUDA_VISIBLE_DEVICES"], "1")
                    self.assertEqual(context["environment"]["DIT_HPO_LEASE_GPU"], "1")
                    self.assertEqual(fixture["multi_records"][(1, 0)]["workers"], [fixture["identities"][30]])
    def test_multi_gpu_context_rejects_one_process_on_two_devices(self) -> None:
        """One process identity cannot consume reservations on both physical GPUs."""

        with fake_multi_gpu() as fixture:
            with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]) as factory:
                with factory("GPU-test") as first, factory("GPU-second") as second:
                    first["register"](30)
                    with self.assertRaisesRegex(RuntimeError, "another admitted context"):
                        second["register"](30)

    def test_multi_gpu_context_requires_complete_group_before_workers(self) -> None:
        """A truncated cross-device reservation group cannot start training children."""

        with fake_multi_gpu() as fixture:
            group = json.loads(remote.os.environ["DIT_HPO_GROUP_LEASES"])
            remote.os.environ["DIT_HPO_GROUP_LEASES"] = json.dumps(group[:-1])
            with self.assertRaisesRegex(RuntimeError, "distinct slots"):
                with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]):
                    self.fail("An incomplete GPU reservation group was admitted.")
            self.assertEqual(fixture["multi_records"][(0, 0)]["workers"], [fixture["identities"][20]])

    def test_serial_identity_rechecks_full_plan_then_selects_first_gpu_only(self) -> None:
        """Confirmation reserves one selected device only after full-plan validation."""

        expected = {"gpu_ids": [2, 4], "concurrent_trials": 5}
        derived = {"gpu_ids": [2], "concurrent_trials": 1}
        events = []
        with patch.object(remote, "_verify_snapshot", side_effect=lambda *args: events.append("verify") or expected), \
        patch.object(remote, "inspect_remote", side_effect=lambda *args, **kwargs: events.append("derive") or derived) as inspect:
            self.assertIs(remote.serial_worker_identity("/workspace/test", expected), derived)
            inspect.assert_called_once_with("/workspace/test", concurrent_trials=1, gpu_ids=[2])
            self.assertEqual(events, ["verify", "derive"])
        with patch.object(remote, "_verify_snapshot", side_effect=RuntimeError("source changed")), \
        patch.object(remote, "inspect_remote") as inspect:
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                remote.serial_worker_identity("/workspace/test", expected)
            inspect.assert_not_called()
    def test_explicit_single_worker_uses_existing_lease_without_extra_acquisition(self) -> None:
        """The shared routed-worker API supports one worker on one admitted GPU."""

        with fake_parallel() as fixture:
            fixture["identity"]["concurrent_trials"] = 1
            with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]) as factory:
                with factory("GPU-test") as context:
                    context["register"](30)
                    self.assertEqual(context["environment"]["CUDA_VISIBLE_DEVICES"], "GPU-test")
            fixture["allocator"].acquire.assert_not_called()

    def test_multi_gpu_acquisition_rolls_back_before_capacity_retry(self) -> None:
        """A full second GPU cannot leave first-GPU reservations held while queued."""

        with fake_launch("colab") as (process, lease, allocator, events):
            identity = remote._verify_snapshot.return_value
            identity.update({"concurrent_trials": 2, "gpu_ids": [0, 1], "gpu_uuid": "GPU-first"})
            identity["gpus"] = [{
                "gpu_id": gpu, "gpu_uuid": "GPU-first" if gpu == 0 else "GPU-second", 
                "gpu_name": "Test GPU", "free_mib": 80000, "total_mib": 81920, "processes": [], 
                "concurrent_trials": 1, "worker_policy": identity["worker_policy"]
            } for gpu in [0, 1]]
            lease.record["gpu"] = "0"
            other = SimpleNamespace(
                record={"gpu": "1", "slot": 0, "owner": lease.record["owner"]}, 
                close=Mock(side_effect=lambda: events.append("close-other"))
            )
            allocator.acquire_many = Mock(side_effect=[
                [lease], AdmissionBlocked("Insufficient free GPU memory"), [lease], [other]
            ])
            remote.time.sleep.side_effect = lambda seconds: events.append("retry")
            with patch.object(remote, "_parallel_launch_guard", side_effect=lambda *args: ExitStack()):
                with remote.launch_worker(["python", "worker.py"], "/workspace/test", identity) as worker:
                    worker.wait()
            self.assertEqual(events[:2], ["close", "retry"])
            self.assertEqual(allocator.acquire_many.call_count, 4)
            self.assertEqual(events[-2:], ["close-other", "close"])
            self.assertEqual(process.launch_options["env"]["CUDA_VISIBLE_DEVICES"], "GPU-first")
            self.assertEqual(len(json.loads(process.launch_options["env"]["DIT_HPO_GROUP_LEASES"])), 2)

    def test_selected_gpu_worker_rejects_visibility_for_another_device(self) -> None:
        """A valid GPU-one lease cannot initialize a worker exposing GPU zero."""

        with fake_multi_gpu() as fixture:
            environment = {
                "CONTINUAL_RUNTIME": "colab", "CUDA_VISIBLE_DEVICES": "GPU-test", 
                "DIT_HPO_PARALLEL_IDENTITY": json.dumps(fixture["identity"]), 
                "DIT_HPO_PARALLEL_COORDINATOR": json.dumps(fixture["identities"][20]), 
                "DIT_HPO_LEASE_GPU": "1", "DIT_HPO_LEASE_SLOT": "0", 
                "DIT_HPO_LEASE_OWNER": json.dumps(fixture["identities"][1])
            }
            with patch.dict(remote.os.environ, environment, clear=True), patch.object(remote.os, "getpid", return_value=30):
                with self.assertRaisesRegex(RuntimeError, "visibility"):
                    remote.prepare_hpo_worker(12288)
    def test_registered_before_gate_and_reaped_before_release(self) -> None:
        """Do not initialize workers before registration or release live slots."""

        with fake_launch() as (process, lease, allocator, events):
            with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}) as worker:
                self.assertIs(worker, process)
                worker.wait()
            allocator.acquire.assert_called_once_with(memory_mb=13312, gpu=0, max_jobs=2)
            lease.register_worker.assert_called_once_with(42)
            self.assertEqual(events, ["spawn", "register", "gate", "wait", "close"])
            self.assertEqual(process.launch_options["pass_fds"], tuple([101]))
            self.assertEqual(process.launch_options["env"]["CUDA_VISIBLE_DEVICES"], "0")

    def test_interruption_reaps_only_owned_worker_before_release(self) -> None:
        """A notebook interrupt terminates and reaps its child before release."""

        with fake_launch() as (process, lease, allocator, events):
            with self.assertRaises(KeyboardInterrupt):
                with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}):
                    raise KeyboardInterrupt()
            self.assertEqual(events[-3:], ["terminate", "wait", "close"])

    def test_capacity_shortage_queues_and_rechecks(self) -> None:
        """Occupied slots wait without introducing a global deadline."""

        with fake_launch() as (process, lease, allocator, events):
            allocator.acquire.side_effect = [
                AdmissionBlocked("Both remote workload slots are occupied; no third job may start."), 
                lease
            ]
            with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}) as worker:
                worker.wait()
            self.assertEqual(allocator.acquire.call_count, 2)
            remote.time.sleep.assert_called_once_with(15)
            self.assertEqual(remote._verify_snapshot.call_count, 3)

    def test_unknown_ownership_fails_without_launch_or_queue(self) -> None:
        """Unknown GPU/kernel workers must not be bypassed as a capacity issue."""

        with fake_launch() as (process, lease, allocator, events):
            allocator.acquire.side_effect = AdmissionBlocked("Unmapped GPU/kernel workers block admission")
            with self.assertRaisesRegex(AdmissionBlocked, "Unmapped"):
                with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}):
                    self.fail("A worker was launched despite unknown ownership.")
            self.assertEqual(events, [])
            remote.time.sleep.assert_not_called()
            lease.close.assert_not_called()

    def test_parent_death_signal_armed(self) -> None:
        """A worker requests Linux termination when the coordinator exits."""

        library = SimpleNamespace(prctl=Mock(return_value=0))
        with patch.object(remote.ctypes, "CDLL", return_value=library):
            remote._arm_parent_exit()
        library.prctl.assert_called_once_with(1, signal.SIGTERM, 0, 0, 0)

    def test_parent_death_signal_failure_is_fatal(self) -> None:
        """Do not continue GPU work if orphan prevention is unavailable."""

        library = SimpleNamespace(prctl=Mock(return_value=-1))
        with patch.object(remote.ctypes, "CDLL", return_value=library):
            with self.assertRaises(OSError):
                remote._arm_parent_exit()


    def test_hosted_provider_and_measured_memory_without_campaign_files(self) -> None:
        """Colab and Kaggle need source, exact packages, and measured device capacity."""

        for runtime_kind in ["colab", "kaggle"]:
            with self.subTest(runtime_kind=runtime_kind), hosted_checkout(runtime_kind) as root:
                identity = remote.inspect_remote(root)
                self.assertEqual(identity["runtime_kind"], runtime_kind)
                self.assertEqual(identity["gpu_uuid"], "GPU-hosted")
                self.assertEqual(identity["worker_policy"], {
                    "max_jobs": 1, "tf_memory_mib": 11264, "reserved_memory_mib": 12288
                })
                self.assertEqual(set(identity["source_sha256"]), {"common/hpo.py", "common/gpu_resource_slots.py"})
                self.assertFalse((root / remote.CAMPAIGN).exists())

    def test_pruning_callback_sources_are_sealed_and_detect_changed_behavior(self) -> None:
        """Guard and reporting changes invalidate the remote scientific source identity."""

        with hosted_checkout() as root:
            directory = root / "common/callbacks"
            directory.mkdir()
            paths = [directory / "hpo_guard.py", directory / "hpo_pruning.py"]
            for path in paths:
                path.write_text("# initial callback behavior\n", encoding="utf-8")
            original = remote.inspect_remote(root)
            self.assertEqual(set(original["source_sha256"]), {
                "common/hpo.py", "common/gpu_resource_slots.py", 
                "common/callbacks/hpo_guard.py", "common/callbacks/hpo_pruning.py"
            })
            for path in paths:
                with self.subTest(path=path.name):
                    path.write_text("# changed callback behavior\n", encoding="utf-8")
                    changed = remote.inspect_remote(root)
                    key = path.relative_to(root).as_posix()
                    self.assertNotEqual(original["source_sha256"][key], changed["source_sha256"][key])
                    with self.assertRaisesRegex(RuntimeError, "changed"):
                        remote._verify_snapshot(root, original)
                    path.write_text("# initial callback behavior\n", encoding="utf-8")

    def test_hosted_large_device_retains_maximum_worker_cap(self) -> None:
        """A larger GPU does not silently expand the existing per-worker TF budget."""

        with hosted_checkout(total_mib=81920) as root:
            self.assertEqual(remote.inspect_remote(root)["worker_policy"]["tf_memory_mib"], 12288)

    def test_hosted_memory_policy_fails_without_headroom(self) -> None:
        """A device too small for a positive worker budget fails before execution."""

        with hosted_checkout(total_mib=4096) as root:
            with self.assertRaisesRegex(RuntimeError, "sufficient memory"):
                remote.inspect_remote(root)

    def test_runtime_override_alone_does_not_authorize_a_laptop(self) -> None:
        """A configuration override is not independent evidence of an online host."""

        with patch.dict(remote.os.environ, {"CONTINUAL_RUNTIME": "colab"}, clear=True), \
        patch.object(remote, "sys", SimpleNamespace(modules={})), \
        patch.object(remote, "_ADMITTED_HOSTED_RUNTIME", None), \
        patch.object(remote.Path, "is_file", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "laptop execution"):
                remote._runtime_kind("unrecognized-machine")

    def test_colab_module_marker_is_recognized_in_parent(self) -> None:
        """The parent can recognize an active Colab module without a release tag."""

        with patch.dict(remote.os.environ, {}, clear=True), \
        patch.object(remote, "sys", SimpleNamespace(modules={"google.colab": object()})):
            self.assertEqual(remote._runtime_kind("hosted-test"), "colab")

    def test_known_pool_host_keeps_pool_policy_with_hosted_markers(self) -> None:
        """Inherited provider markers cannot bypass a supplied host's allocator."""

        with patch.dict(remote.os.environ, {"COLAB_RELEASE_TAG": "test"}, clear=True):
            self.assertEqual(remote._runtime_kind(next(iter(remote.VERIFIED_HOSTS))), "runpod")

    def test_windows_is_rejected_before_checkout_access(self) -> None:
        """A hosted marker cannot authorize a Windows notebook runtime."""

        with patch.object(remote.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "laptop execution"):
                remote._remote_root(Path("C:/unread-checkout"))

    def test_changed_worker_policy_invalidates_runtime_snapshot(self) -> None:
        """A different admission budget requires a new verified plan."""

        with hosted_checkout() as root:
            identity = remote.inspect_remote(root)
            expected = {**identity, "worker_policy": {**identity["worker_policy"], "max_jobs": 2}}
            with self.assertRaisesRegex(RuntimeError, "worker_policy"):
                remote._verify_snapshot(root, expected)

    def test_hosted_launch_uses_one_slot_and_passes_verified_provider(self) -> None:
        """Hosted workers retain the gate protocol while using their measured budget."""

        with fake_launch("colab") as (process, lease, allocator, events):
            with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}) as worker:
                worker.wait()
            allocator.acquire.assert_called_once_with(memory_mb=12288, gpu=0, max_jobs=1)
            self.assertEqual(process.launch_options["env"]["CONTINUAL_RUNTIME"], "colab")
            self.assertEqual(events, ["spawn", "register", "gate", "wait", "close"])

    def test_hosted_worker_rejects_second_slot_before_lock_access(self) -> None:
        """The worker cannot accept a supplied-pool slot outside hosted policy."""

        environment = {
            "DIT_HPO_GATE_FD": "101", "DIT_HPO_LEASE_SLOT": "1", 
            "DIT_HPO_LEASE_OWNER": '{"pid": 1, "start_ticks": "2"}'
        }
        with patch.dict(remote.os.environ, environment, clear=True), \
        patch.object(remote.os, "read", return_value=b"1"), \
        patch.object(remote.os, "close"):
            with self.assertRaisesRegex(RuntimeError, "slot is invalid"):
                remote._validate_worker_lease({"worker_policy": {"max_jobs": 1}})

    def test_inherited_provider_is_established_only_after_lease_validation(self) -> None:
        """Module-only Colab children cannot inherit admission before their lease passes."""

        events = []
        identity = {"runtime_kind": "colab"}
        with patch.dict(remote.os.environ, {"CONTINUAL_RUNTIME": "colab"}, clear=True), \
        patch.object(remote, "sys", SimpleNamespace(modules={})), \
        patch.object(remote, "_ADMITTED_HOSTED_RUNTIME", None), \
        patch.object(remote, "_arm_parent_exit", side_effect=lambda: events.append("parent-signal")), \
        patch.object(remote, "_validate_worker_lease", side_effect=RuntimeError("lease denied")), \
        patch.object(remote, "_verify_snapshot") as verify:
            with self.assertRaisesRegex(RuntimeError, "lease denied"):
                with remote.managed_worker("/unused", identity):
                    self.fail("An unauthenticated worker was admitted.")
            self.assertIsNone(remote._ADMITTED_HOSTED_RUNTIME)
            verify.assert_not_called()
            self.assertEqual(events, ["parent-signal"])

    def test_authenticated_module_only_colab_worker_initializes_with_verified_cap(self) -> None:
        """A registered child inherits Colab identity before initializing its capped GPU."""

        events = []
        identity = {
            "runtime_kind": "colab", 
            "worker_policy": {"max_jobs": 1, "tf_memory_mib": 11264, "reserved_memory_mib": 12288}
        }
        configuration = SimpleNamespace(
            list_physical_devices=Mock(side_effect=lambda kind: events.append("discover") or ["gpu0"]), 
            LogicalDeviceConfiguration=Mock(side_effect=lambda memory_limit: {"memory_limit": memory_limit}), 
            set_logical_device_configuration=Mock(side_effect=lambda device, options: events.append("cap")), 
            list_logical_devices=Mock(side_effect=lambda kind: events.append("initialize") or ["logical0"])
        )
        tensorflow = SimpleNamespace(__version__="2.20.0", config=configuration)
        original_import = __import__

        def import_module(module_name: str, *args: Any, **kwargs: Any) -> Any:
            """Intercept only TensorFlow while preserving ordinary Python imports."""

            # Supplying a fake module at the import boundary avoids loading a real framework.
            if module_name == "tensorflow":
                events.append("import-tf")
                return tensorflow
            return original_import(module_name, *args, **kwargs)

        def verify_identity(checkout_root: str, expected_identity: dict[str, Any]) -> dict[str, Any]:
            """Observe inherited provider identity after the worker lease has passed."""

            events.append("snapshot")
            self.assertEqual(remote._runtime_kind("hosted-child"), "colab")
            self.assertEqual(expected_identity, identity)
            return identity

        with patch.dict(remote.os.environ, {"CONTINUAL_RUNTIME": "colab"}, clear=True), \
        patch.object(remote, "sys", SimpleNamespace(modules={})), \
        patch.object(remote, "_ADMITTED_HOSTED_RUNTIME", None), \
        patch.object(remote, "_arm_parent_exit", side_effect=lambda: events.append("parent-signal")), \
        patch.object(remote, "_validate_worker_lease", side_effect=lambda value: events.append("lease")), \
        patch.object(remote, "_verify_snapshot", side_effect=verify_identity), \
        patch("builtins.__import__", side_effect=import_module):
            with remote.managed_worker("/hosted/project", identity):
                events.append("body")
            self.assertEqual(remote.os.environ["KERAS_HOME"], "/hosted/.keras")
        configuration.LogicalDeviceConfiguration.assert_called_once_with(memory_limit=11264)
        configuration.set_logical_device_configuration.assert_called_once_with("gpu0", [{"memory_limit": 11264}])
        self.assertEqual(events, ["parent-signal", "lease", "snapshot", "import-tf", "discover", "cap", "initialize", "body"])

    def test_parallel_hosted_policy_covers_both_reserved_workers(self) -> None:
        """Two hosted workers divide measured capacity while preserving headroom."""

        with hosted_checkout(total_mib=15360) as root:
            identity = remote.inspect_remote(root, concurrent_trials=2)
            self.assertEqual(identity["concurrent_trials"], 2)
            self.assertEqual(identity["worker_policy"], {
                "max_jobs": 2, "tf_memory_mib": 5120, "reserved_memory_mib": 6144
            })
            self.assertLessEqual(2 * identity["worker_policy"]["reserved_memory_mib"] + 2048, identity["free_mib"])

    def test_parallel_count_requires_exact_supported_integer(self) -> None:
        """Booleans and nonpositive values cannot describe exact workload counts."""

        for value in [True, 1.0, 0, -1]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                remote.inspect_remote("/unused", concurrent_trials=value)

    def test_parallel_workers_use_distinct_owners_and_slots(self) -> None:
        """Each core HPO child is registered to one independently reserved slot."""

        with fake_parallel() as fixture:
            with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]) as factory:
                self.assertEqual(remote.os.environ["CUDA_VISIBLE_DEVICES"], "-1")
                with factory() as first, factory() as second:
                    first["register"](30)
                    second["register"](31)
                    self.assertEqual([item["pid"] for item in fixture["records"][0]["workers"]], [30])
                    self.assertEqual([item["pid"] for item in fixture["records"][1]["workers"]], [31])
                    self.assertEqual(first["environment"]["DIT_HPO_LEASE_SLOT"], "0")
                    self.assertEqual(second["environment"]["DIT_HPO_LEASE_SLOT"], "1")
                    self.assertNotEqual(first["environment"]["DIT_HPO_LEASE_OWNER"], second["environment"]["DIT_HPO_LEASE_OWNER"])
                    self.assertEqual(json.loads(first["environment"]["DIT_HPO_PARALLEL_IDENTITY"]), fixture["identity"])
                    self.assertEqual(first["environment"]["CUDA_VISIBLE_DEVICES"], "0")
                    with self.assertRaisesRegex(RuntimeError, "already in use"):
                        with factory():
                            self.fail("A third worker entered the two-slot pool.")
                self.assertEqual(fixture["records"][0]["workers"], [])
            fixture["allocator"].acquire.assert_called_once_with(memory_mb=13312, gpu=0, max_jobs=2)
            self.assertEqual(fixture["events"][:2], ["remove-20-slot0", "acquire-second"])
            self.assertEqual(fixture["events"][-2:], ["close-second", "add-20-slot0"])
            self.assertEqual(remote.os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_parallel_interruption_reaps_before_releasing_and_restores_parent(self) -> None:
        """An interrupted batch drains both owned children before dropping leases."""

        with fake_parallel() as fixture:
            with self.assertRaises(KeyboardInterrupt):
                with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]) as factory:
                    with factory() as first, factory() as second:
                        first["register"](30)
                        second["register"](31)
                        raise KeyboardInterrupt()
            events = fixture["events"]
            self.assertLess(events.index("reap-30"), events.index("close-second"))
            self.assertLess(events.index("reap-31"), events.index("close-second"))
            self.assertEqual(events[-2:], ["close-second", "add-20-slot0"])
            self.assertEqual([item["pid"] for item in fixture["records"][0]["workers"]], [20])

    def test_parallel_unknown_ownership_restores_first_registration(self) -> None:
        """Failure acquiring a second slot restores the outer lease without bypassing admission."""

        with fake_parallel() as fixture:
            fixture["allocator"].acquire.side_effect = AdmissionBlocked("Unmapped GPU/kernel workers block admission")
            with self.assertRaisesRegex(AdmissionBlocked, "Unmapped"):
                with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]):
                    self.fail("Parallel work started with unknown ownership.")
            self.assertEqual(fixture["events"], ["remove-20-slot0", "add-20-slot0"])
            fixture["lease"].close.assert_not_called()
            remote.time.sleep.assert_not_called()

    def test_parallel_capacity_wait_does_not_import_or_initialize_tensorflow(self) -> None:
        """The second reservation is queued before any framework-dependent work."""

        with fake_parallel() as fixture:
            fixture["allocator"].acquire.side_effect = [
                AdmissionBlocked("Both remote workload slots are occupied; no third job may start."), 
                fixture["lease"]
            ]
            with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]):
                self.assertNotIn("tensorflow", remote.sys.modules)
            remote.time.sleep.assert_called_once_with(15)
            self.assertEqual(fixture["allocator"].acquire.call_count, 2)

    def test_parallel_resource_context_rejects_second_registration(self) -> None:
        """One process context cannot share a slot with a second training child."""

        with fake_parallel() as fixture:
            with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]) as factory:
                with factory() as context:
                    context["register"](30)
                    with self.assertRaisesRegex(RuntimeError, "already registered"):
                        context["register"](31)

    def test_parallel_core_worker_verifies_cap_and_exact_registration(self) -> None:
        """The reused HPO worker authenticates admission without importing TensorFlow."""

        with fake_parallel() as fixture:
            worker = fixture["identities"][30]
            environment = {
                "DIT_HPO_PARALLEL_IDENTITY": json.dumps(fixture["identity"]), 
                "DIT_HPO_PARALLEL_COORDINATOR": json.dumps(fixture["identities"][20]), 
                "DIT_HPO_LEASE_SLOT": "0", "DIT_HPO_LEASE_OWNER": json.dumps(fixture["identities"][1])
            }

            @contextmanager
            def locked(slot: int, owner: dict[str, Any], expected: dict[str, Any]) -> Iterator[tuple[dict[str, Any], Path]]:
                """Supply a verified live allocation record at the shared lock boundary."""

                self.assertEqual(slot, 0)
                self.assertEqual(owner["pid"], 1)
                self.assertEqual(expected, fixture["identity"])
                yield {"workers": [worker]}, Path("/unused")

            with patch.dict(remote.os.environ, environment, clear=True), \
            patch.object(remote.os, "getpid", return_value=30), \
            patch.object(remote, "_locked_lease_record", side_effect=locked):
                remote.prepare_hpo_worker(12288)
                self.assertNotIn("tensorflow", remote.sys.modules)
                self.assertNotIn("DIT_HPO_PARALLEL_IDENTITY", remote.os.environ)
            with patch.dict(remote.os.environ, environment, clear=True):
                with self.assertRaisesRegex(RuntimeError, "memory cap"):
                    remote.prepare_hpo_worker(8192)

    def test_parallel_core_worker_rejects_reused_parent_pid(self) -> None:
        """A matching parent number cannot authorize a different process lifetime."""

        with fake_parallel() as fixture:
            environment = {
                "DIT_HPO_PARALLEL_IDENTITY": json.dumps(fixture["identity"]), 
                "DIT_HPO_PARALLEL_COORDINATOR": json.dumps({**fixture["identities"][20], "start_ticks": "old"}), 
                "DIT_HPO_LEASE_SLOT": "0", "DIT_HPO_LEASE_OWNER": json.dumps(fixture["identities"][1])
            }
            with patch.dict(remote.os.environ, environment, clear=True), \
            patch.object(remote.os, "getpid", return_value=30):
                with self.assertRaisesRegex(RuntimeError, "coordinator identity changed"):
                    remote.prepare_hpo_worker(12288)
    def test_parallel_launch_guard_spans_admission_through_final_release(self) -> None:
        """The coordinator turn is acquired before the first GPU lease and released last."""

        with fake_launch() as (process, lease, allocator, events):
            @contextmanager
            def guard(checkout_root: Path, expected_identity: dict[str, Any]) -> Iterator[None]:
                """Track the lifetime of the independent pair-acquisition guard."""

                events.append("guard-enter")
                try:
                    yield
                finally:
                    events.append("guard-exit")

            allocator.acquire.side_effect = lambda **kwargs: events.append("acquire-first") or lease
            with patch.object(remote, "_parallel_launch_guard", side_effect=guard):
                with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {"concurrent_trials": 2}) as worker:
                    worker.wait()
            self.assertEqual(events, ["guard-enter", "acquire-first", "spawn", "register", "gate", "wait", "close", "guard-exit"])

    def test_interrupted_parallel_guard_wait_takes_no_gpu_lease(self) -> None:
        """A second parallel notebook waits without holding either shared GPU slot."""

        with fake_launch() as (process, lease, allocator, events), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "dit-hpo-parallel-gpu-0.lock").open("a") as held:
                remote.fcntl.flock(held.fileno(), remote.fcntl.LOCK_EX)
                with patch.object(remote, "LOCK_ROOT", root), \
                patch.object(remote.time, "sleep", side_effect=KeyboardInterrupt()):
                    with self.assertRaises(KeyboardInterrupt):
                        with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {"concurrent_trials": 2}):
                            self.fail("A second parallel notebook took a GPU lease while waiting.")
            allocator.acquire.assert_not_called()
            self.assertEqual(events, [])

    def test_exact_registration_edit_preserves_other_workers(self) -> None:
        """Updating one allocation identity keeps unrelated registered processes intact."""

        original_edit = remote._edit_lease_worker
        with tempfile.TemporaryDirectory() as directory, fake_parallel() as fixture:
            root = Path(directory)
            owner = fixture["identities"][1]
            worker = fixture["identities"][20]
            other = {"pid": 99, "start_ticks": "unrelated"}
            record = {
                "slot": 0, "owner": owner, "workers": [worker, other], 
                "gpu_uuid": "GPU-test", "memory_mb": 13312
            }
            path = root / "joint-notebook-gpu-0-slot0.json"
            path.write_text(json.dumps(record))
            writer = lambda destination, value: destination.write_text(json.dumps(value))
            fixture["allocator"]._write_record = Mock(side_effect=writer)
            with (root / "joint-notebook-gpu-0.lock").open("a") as held, patch.object(remote, "LOCK_ROOT", root):
                remote.fcntl.flock(held.fileno(), remote.fcntl.LOCK_EX)
                with remote._locked_lease_record(0, owner, fixture["identity"]) as (saved, saved_path):
                    self.assertEqual(saved, record)
                    self.assertEqual(saved_path, path)
                original_edit(0, owner, worker, fixture["identity"], fixture["allocator"], add=False)
                self.assertEqual(json.loads(path.read_text())["workers"], [other])
                original_edit(0, owner, worker, fixture["identity"], fixture["allocator"], add=True)
                self.assertEqual(json.loads(path.read_text())["workers"], [other, worker])
                with self.assertRaisesRegex(RuntimeError, "reservation changed"):
                    with remote._locked_lease_record(0, owner, {**fixture["identity"], "gpu_uuid": "GPU-other"}):
                        self.fail("A different physical GPU was accepted.")
            with patch.object(remote, "LOCK_ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "no longer held"):
                    with remote._locked_lease_record(0, owner, fixture["identity"]):
                        self.fail("An unlocked stale record was accepted.")
    def test_five_hosted_workers_retain_per_worker_h100_budget(self) -> None:
        """Five hosted reservations fit measured H100 capacity with fixed headroom."""

        with hosted_checkout(total_mib=81559) as root:
            identity = remote.inspect_remote(root, concurrent_trials=5)
            self.assertEqual(identity["worker_policy"], {
                "max_jobs": 5, "tf_memory_mib": 12288, "reserved_memory_mib": 13312
            })
            self.assertLessEqual(5 * 13312 + 2048, identity["free_mib"])

    def test_supplied_pool_still_rejects_five_workers_before_inventory(self) -> None:
        """Hosted concurrency cannot change the campaign's two-worker policy."""

        with patch.object(remote, "_remote_root", return_value=(Path("/workspace/test"), "supplied")), \
        patch.object(remote, "_runtime_kind", return_value="runpod"), \
        patch.object(remote, "_query") as query:
            with self.assertRaisesRegex(ValueError, "only one or two"):
                remote.inspect_remote("/workspace/test", concurrent_trials=5)
            query.assert_not_called()

    def test_five_workers_reserve_unique_slots_and_reject_sixth_context(self) -> None:
        """Five contexts authenticate distinct slots after atomic group admission."""

        with fake_hosted_parallel() as fixture:
            with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]) as factory:
                self.assertEqual(remote.os.environ["CUDA_VISIBLE_DEVICES"], "-1")
                with ExitStack() as stack:
                    contexts = [stack.enter_context(factory()) for _ in range(5)]
                    for slot, context in enumerate(contexts):
                        context["register"](30 + slot)
                        self.assertEqual(context["environment"]["DIT_HPO_LEASE_SLOT"], str(slot))
                        self.assertEqual(context["environment"]["CUDA_VISIBLE_DEVICES"], "0")
                        self.assertEqual([item["pid"] for item in fixture["records"][slot]["workers"]], [30 + slot])
                    with self.assertRaisesRegex(RuntimeError, "already in use"):
                        with factory():
                            self.fail("A sixth worker exceeded the admitted group.")
                self.assertTrue(all(not record["workers"] for record in fixture["records"].values()))
            fixture["allocator"].acquire.assert_not_called()
            fixture["allocator"].acquire_many.assert_called_once_with(memory_mb=13312, gpu=0, max_jobs=5, count=4)
            events = fixture["events"]
            self.assertEqual(events[:2], ["remove-20-slot0", "acquire-group"])
            self.assertEqual(events[-5:], ["close-slot4", "close-slot3", "close-slot2", "close-slot1", "add-20-slot0"])
            for pid in range(30, 35):
                self.assertLess(events.index("reap-" + str(pid)), events.index("close-slot4"))
            self.assertEqual(remote.os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_five_worker_interruption_drains_group_before_releasing_leases(self) -> None:
        """Every registered child is reaped before interrupted group reservations close."""

        with fake_hosted_parallel() as fixture:
            with self.assertRaises(KeyboardInterrupt):
                with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]) as factory:
                    with ExitStack() as stack:
                        for pid in range(30, 35):
                            stack.enter_context(factory())["register"](pid)
                        raise KeyboardInterrupt()
            events = fixture["events"]
            for pid in range(30, 35):
                self.assertLess(events.index("reap-" + str(pid)), events.index("close-slot4"))
            self.assertEqual([item["pid"] for item in fixture["records"][0]["workers"]], [20])
            self.assertTrue(all(lease.close.call_count == 1 for lease in fixture["leases"]))

    def test_five_worker_group_failure_restores_parent_without_partial_leases(self) -> None:
        """An ownership failure during atomic admission leaves no group to release."""

        with fake_hosted_parallel() as fixture:
            fixture["allocator"].acquire_many.side_effect = AdmissionBlocked("Unmapped GPU/kernel workers block admission")
            with self.assertRaisesRegex(AdmissionBlocked, "Unmapped"):
                with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]):
                    self.fail("Unknown ownership admitted a hosted group.")
            self.assertEqual(fixture["events"], ["remove-20-slot0", "add-20-slot0"])
            self.assertTrue(all(lease.close.call_count == 0 for lease in fixture["leases"]))
            remote.time.sleep.assert_not_called()

    def test_five_worker_group_waits_before_framework_imports(self) -> None:
        """Capacity waits retry the atomic group with the CPU coordinator hidden."""

        with fake_hosted_parallel() as fixture:
            fixture["allocator"].acquire_many.side_effect = [
                AdmissionBlocked("Insufficient free GPU memory"), fixture["leases"]
            ]
            with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]):
                self.assertNotIn("tensorflow", remote.sys.modules)
                self.assertEqual(remote.os.environ["CUDA_VISIBLE_DEVICES"], "-1")
            remote.time.sleep.assert_called_once_with(15)
            self.assertEqual(fixture["allocator"].acquire_many.call_count, 2)

    def test_five_worker_contexts_reject_duplicate_process_registration(self) -> None:
        """A child cannot count against two reservations in the same hosted group."""

        with fake_hosted_parallel() as fixture:
            with remote.managed_parallel_coordinator("/workspace/test", fixture["identity"]) as factory:
                with factory() as first, factory() as second:
                    first["register"](30)
                    with self.assertRaisesRegex(RuntimeError, "another admitted context"):
                        second["register"](30)
                    self.assertEqual(fixture["records"][1]["workers"], [])

    def test_fifth_worker_authenticates_exact_cap_and_coordinator(self) -> None:
        """A slot-four child retains the same identity gate and per-worker cap."""

        with fake_hosted_parallel() as fixture:
            worker = fixture["identities"][34]
            environment = {
                "CONTINUAL_RUNTIME": "colab", 
                "DIT_HPO_PARALLEL_IDENTITY": json.dumps(fixture["identity"]), 
                "DIT_HPO_PARALLEL_COORDINATOR": json.dumps(fixture["identities"][20]), 
                "DIT_HPO_LEASE_SLOT": "4", "DIT_HPO_LEASE_OWNER": json.dumps(fixture["identities"][20])
            }

            @contextmanager
            def locked(slot: int, owner: dict[str, Any], expected: dict[str, Any]) -> Iterator[tuple[dict[str, Any], Path]]:
                """Check the worker addresses its actual fifth reservation."""

                self.assertEqual(slot, 4)
                self.assertEqual(owner, fixture["identities"][20])
                self.assertEqual(expected, fixture["identity"])
                yield {"workers": [worker]}, Path("/unused")

            with patch.dict(remote.os.environ, environment, clear=True), \
            patch.object(remote.os, "getpid", return_value=34), \
            patch.object(remote, "_locked_lease_record", side_effect=locked):
                remote.prepare_hpo_worker(12288)
                self.assertNotIn("tensorflow", remote.sys.modules)
            with patch.dict(remote.os.environ, environment, clear=True):
                with self.assertRaisesRegex(RuntimeError, "memory cap"):
                    remote.prepare_hpo_worker(8192)

    def test_fifth_slot_uses_its_own_lifetime_lock(self) -> None:
        """Higher-slot authentication cannot accept a held slot-one lock."""

        with tempfile.TemporaryDirectory() as directory, fake_hosted_parallel() as fixture:
            root = Path(directory)
            owner = fixture["identities"][20]
            record = {
                "slot": 4, "owner": owner, "workers": [], 
                "gpu_uuid": "GPU-test", "memory_mb": 13312, "version": 2
            }
            path = root / "joint-notebook-gpu-0-slot4.json"
            path.write_text(json.dumps(record))
            with patch.object(remote, "LOCK_ROOT", root):
                with (root / "joint-notebook-gpu-0-slot1.lock").open("a") as wrong:
                    remote.fcntl.flock(wrong.fileno(), remote.fcntl.LOCK_EX)
                    with self.assertRaisesRegex(RuntimeError, "no longer held"):
                        with remote._locked_lease_record(4, owner, fixture["identity"]):
                            self.fail("The wrong slot lock authenticated a higher slot.")
                with (root / "joint-notebook-gpu-0-slot4.lock").open("a") as held:
                    remote.fcntl.flock(held.fileno(), remote.fcntl.LOCK_EX)
                    with remote._locked_lease_record(4, owner, fixture["identity"]) as (saved, saved_path):
                        self.assertEqual(saved, record)
                        self.assertEqual(saved_path, path)

    def test_five_worker_group_guard_waits_before_first_reservation(self) -> None:
        """Concurrent hosted groups cannot each hold one slot while waiting."""

        with fake_launch() as (process, lease, allocator, events), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "dit-hpo-parallel-gpu-0.lock").open("a") as held:
                remote.fcntl.flock(held.fileno(), remote.fcntl.LOCK_EX)
                with patch.object(remote, "LOCK_ROOT", root), \
                patch.object(remote.time, "sleep", side_effect=KeyboardInterrupt()):
                    with self.assertRaises(KeyboardInterrupt):
                        with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {"concurrent_trials": 5}):
                            self.fail("A second hosted group reserved a slot before its turn.")
            allocator.acquire.assert_not_called()
            self.assertEqual(events, [])

    def test_nested_coordinator_detects_fifth_slot_registration(self) -> None:
        """Nested launch detection includes every published hosted slot record."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = {"pid": 34, "start_ticks": "worker4", "parent_pid": 20}
            record = {"owner": {"pid": 20, "start_ticks": "coordinator"}, "workers": [worker]}
            (root / "joint-notebook-gpu-0-slot4.json").write_text(json.dumps(record))
            with patch.object(remote, "LOCK_ROOT", root), patch.object(remote, "_process_identity", return_value=worker):
                self.assertTrue(remote._in_existing_allocation(34))

    def test_coordinator_can_own_a_sibling_confirmation_on_another_gpu(self) -> None:
        """A CPU coordinator may launch the next repeat beside a live sibling."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = {"pid": 20, "start_ticks": "coordinator", "parent_pid": 1}
            path = root / "joint-notebook-gpu-1-slot0.json"
            for device in [1, "1"]:
                with self.subTest(device=device):
                    record = {"owner": owner, "workers": [], "gpu": device}
                    path.write_text(json.dumps(record))
                    with patch.object(remote, "LOCK_ROOT", root), patch.object(remote, "_process_identity", return_value=owner):
                        self.assertFalse(remote._in_existing_allocation(20, gpu_ids=[0, 2]))

    def test_launch_scopes_existing_owner_check_to_selected_gpu(self) -> None:
        """The real launch boundary allows a verified owner on another device."""

        existing_allocation = remote._in_existing_allocation
        owner = {"pid": remote.os.getpid(), "start_ticks": "coordinator", "parent_pid": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"owner": owner, "workers": [], "gpu": "1"}
            (root / "joint-notebook-gpu-1-slot0.json").write_text(json.dumps(record))
            with fake_launch() as (process, lease, allocator, events):
                with patch.object(remote, "LOCK_ROOT", root), \
                patch.object(remote, "_process_identity", return_value=owner), \
                patch.object(remote, "_in_existing_allocation", wraps=existing_allocation) as guard:
                    with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}) as worker:
                        self.assertIs(worker, process)
                        worker.wait()
                    guard.assert_called_once_with(owner["pid"], gpu_ids=[0])
                allocator.acquire.assert_called_once_with(memory_mb=13312, gpu=0, max_jobs=2)
                self.assertEqual(events, ["spawn", "register", "gate", "wait", "close"])

    def test_launch_rejects_existing_owner_on_selected_gpu(self) -> None:
        """Device scoping cannot admit overlapping leases at the launch boundary."""

        existing_allocation = remote._in_existing_allocation
        owner = {"pid": remote.os.getpid(), "start_ticks": "coordinator", "parent_pid": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"owner": owner, "workers": [], "gpu": "0"}
            (root / "joint-notebook-gpu-0-slot0.json").write_text(json.dumps(record))
            with fake_launch() as (process, lease, allocator, events):
                with patch.object(remote, "LOCK_ROOT", root), \
                patch.object(remote, "_process_identity", return_value=owner), \
                patch.object(remote, "_in_existing_allocation", wraps=existing_allocation) as guard:
                    with self.assertRaisesRegex(RuntimeError, "already belongs"):
                        with remote.launch_worker(["python", "worker.py"], Path("/workspace/test"), {}):
                            self.fail("An existing same-device owner admitted a second lease.")
                    guard.assert_called_once_with(owner["pid"], gpu_ids=[0])
                allocator.acquire.assert_not_called()
                self.assertEqual(events, [])

    def test_coordinator_cannot_admit_a_second_lease_on_selected_gpu(self) -> None:
        """Same-device ownership retains the nested admission restriction."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = {"pid": 20, "start_ticks": "coordinator", "parent_pid": 1}
            record = {"owner": owner, "workers": [], "gpu": "1"}
            (root / "joint-notebook-gpu-1-slot0.json").write_text(json.dumps(record))
            with patch.object(remote, "LOCK_ROOT", root), patch.object(remote, "_process_identity", return_value=owner):
                self.assertTrue(remote._in_existing_allocation(20, gpu_ids=[0, 1]))
                self.assertTrue(remote._in_existing_allocation(20))

    def test_registered_worker_cannot_launch_on_another_gpu(self) -> None:
        """Foreign-device worker membership cannot become coordinator authority."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = {"pid": 34, "start_ticks": "worker4", "parent_pid": 20}
            record = {"owner": {"pid": 20, "start_ticks": "coordinator"}, "workers": [worker], "gpu": "1"}
            (root / "joint-notebook-gpu-1-slot4.json").write_text(json.dumps(record))
            with patch.object(remote, "LOCK_ROOT", root), patch.object(remote, "_process_identity", return_value=worker):
                self.assertTrue(remote._in_existing_allocation(34, gpu_ids=[0]))

    def test_unverified_owner_device_still_blocks_admission(self) -> None:
        """Missing or malformed device data cannot authorize a sibling launch."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = {"pid": 20, "start_ticks": "coordinator", "parent_pid": 1}
            path = root / "joint-notebook-gpu-1-slot0.json"
            invalid_devices = [{}, {"gpu": None}, {"gpu": True}, {"gpu": -1}, {"gpu": 1.0}, {"gpu": "-1"}, {"gpu": "GPU-other"}]
            for extra in invalid_devices:
                with self.subTest(extra=extra):
                    path.write_text(json.dumps({"owner": owner, "workers": [], **extra}))
                    with patch.object(remote, "LOCK_ROOT", root), patch.object(remote, "_process_identity", return_value=owner):
                        self.assertTrue(remote._in_existing_allocation(20, gpu_ids=[0]))

    def test_reused_pid_is_not_treated_as_lease_membership(self) -> None:
        """Owner and worker matching both retain the process start identity."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = {"pid": 20, "start_ticks": "old-process", "parent_pid": 1}
            current = {**stale, "start_ticks": "new-process"}
            record = {"owner": stale, "workers": [stale], "gpu": "0"}
            (root / "joint-notebook-gpu-0-slot0.json").write_text(json.dumps(record))
            with patch.object(remote, "LOCK_ROOT", root), patch.object(remote, "_process_identity", return_value=current):
                self.assertFalse(remote._in_existing_allocation(20, gpu_ids=[0]))
                self.assertFalse(remote._in_existing_allocation(20))

    def test_removed_sibling_record_does_not_interrupt_admission(self) -> None:
        """Concurrent sibling cleanup may remove a record after discovery."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = {"pid": 20, "start_ticks": "coordinator", "parent_pid": 1}
            path = root / "joint-notebook-gpu-1-slot0.json"
            path.write_text(json.dumps({"owner": owner, "workers": [], "gpu": "1"}))
            with patch.object(remote, "LOCK_ROOT", root), \
            patch.object(remote, "_process_identity", return_value=owner), \
            patch.object(Path, "read_text", side_effect=FileNotFoundError()):
                self.assertFalse(remote._in_existing_allocation(20, gpu_ids=[0]))


# Allow the focused checks to run directly as a standalone test module.
if __name__ == "__main__":
    unittest.main()
