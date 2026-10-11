"""Check launch identities, persistent deadlines and node-local SQLite bootstrap."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import dit_continual_cluster_runtime as runtime
from common import dit_hpo_runner as shared


class ClusterRuntimeTests(unittest.TestCase):
    """Exercise orchestration boundaries without GPU or learning-framework work."""

    def setUp(self) -> None:
        """Create one isolated shared-control tree for every test."""

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = runtime.campaign_settings()
        self.request = runtime.prepare_cluster(self.root, self.root / "checkout", settings=self.settings)

    def ready(self, shard: str, timestamp: float = 100.0) -> dict:
        """Publish distinct eight-device readiness for a synthetic container."""

        gpus = [
            {
                "gpu_uuid": f"GPU-{shard}-{index}", "gpu_name": "NVIDIA A100-SXM4-80GB", 
                "total_mib": 81920, "free_mib": 81152, "processes": [], "concurrent_trials": 3, 
                "worker_policy": {"reserved_memory_mib": 25600}
            }
            for index in range(8)
        ]
        status = {
            "shard_id": shard, "phase": "ready", "heartbeat_unix": timestamp, 
            "request_sha256": self.request["sha256"], 
            "identity": {"hostname": self.settings["hosts"][shard], "gpus": gpus}, 
            "recipe": {"same": "scientific recipe"}
        }
        shared._write(self.root / "services" / shard / "status.json", status)
        return status

    def test_settings_are_independent_and_target_sixteen_gpus(self) -> None:
        """Caller edits must not change another notebook's defaults."""

        first = runtime.campaign_settings()
        first["confirmation_seeds"]["a"].append(999)
        second = runtime.campaign_settings()
        self.assertEqual(second["confirmation_seeds"]["a"], [101, 202])
        self.assertEqual(len(second["gpu_ids"]) * len(second["hosts"]), 16)
        self.assertEqual(second["search_hours"] + second["confirmation_hours"], 12)
        self.assertEqual(second["top_k"] * len(second["confirmation_seeds"]["a"]), 8)
        self.assertEqual(sum(len(seeds) for seeds in second["confirmation_seeds"].values()), 3)

    def test_request_immutable_and_preparation_does_not_start(self) -> None:
        """Repeated setup retains the exact recipe and does not spend the campaign clock."""

        self.assertEqual(self.request, runtime.prepare_cluster(self.root, self.root / "checkout"))
        self.assertFalse((self.root / "START.json").exists())
        changed = deepcopy(self.settings)
        changed["epochs"] += 1
        with self.assertRaisesRegex(ValueError, "immutable"):
            runtime.prepare_cluster(self.root, self.root / "checkout", settings=changed)

    def test_launch_is_idempotent_after_services_finish(self) -> None:
        """A second notebook launch cannot reset the first start time."""

        self.ready("a")
        self.ready("b")
        with patch.object(runtime.time, "time", return_value=110.0):
            first = runtime.launch_cluster(self.root)
        self.ready("a", timestamp=0)
        with patch.object(runtime.time, "time", return_value=10000.0):
            second = runtime.launch_cluster(self.root)
        self.assertEqual(first, second)
        self.assertEqual(first["started_at_unix"], 110.0)

    def test_launch_rejects_stale_service(self) -> None:
        """A dead peer's old inventory must never authorize a new campaign."""

        self.ready("a", timestamp=1)
        self.ready("b")
        with patch.object(runtime.time, "time", return_value=110.0), self.assertRaisesRegex(RuntimeError, "fresh"):
            runtime.launch_cluster(self.root)
        self.assertFalse((self.root / "START.json").exists())

    def test_launch_rejects_duplicate_physical_gpus(self) -> None:
        """Two host labels cannot masquerade as sixteen distinct devices."""

        first = self.ready("a")
        second = self.ready("b")
        second["identity"]["gpus"] = first["identity"]["gpus"]
        shared._write(self.root / "services/b/status.json", second)
        with patch.object(runtime.time, "time", return_value=110.0), self.assertRaisesRegex(RuntimeError, "distinct"):
            runtime.launch_cluster(self.root)

    def test_launch_rejects_source_or_recipe_mismatch(self) -> None:
        """Comparable objectives require matching sealed scientific identities."""

        self.ready("a")
        second = self.ready("b")
        second["recipe"] = {"same": "different code"}
        shared._write(self.root / "services/b/status.json", second)
        with patch.object(runtime.time, "time", return_value=110.0), self.assertRaisesRegex(RuntimeError, "matching"):
            runtime.launch_cluster(self.root)

    def test_busy_device_blocks_launch(self) -> None:
        """Existing GPU processes remain protected by readiness admission."""

        self.ready("a")
        second = self.ready("b")
        second["identity"]["gpus"][0]["processes"] = [{"pid": 123}]
        shared._write(self.root / "services/b/status.json", second)
        with patch.object(runtime.time, "time", return_value=110.0), self.assertRaisesRegex(RuntimeError, "idle"):
            runtime.launch_cluster(self.root)

    def test_wait_stops_on_peer_failure(self) -> None:
        """An impossible peer artifact must fail immediately instead of burning the budget."""

        shared._write(self.root / "services/b/status.json", {"phase": "failed"})
        with self.assertRaisesRegex(RuntimeError, "Peer service failed"):
            runtime._wait_files(self.root, [self.root / "missing.json"], deadline=200)

    def test_wait_obeys_absolute_deadline(self) -> None:
        """No cross-host barrier can wait past the campaign cleanup deadline."""

        with patch.object(runtime.time, "time", return_value=196), self.assertRaises(TimeoutError):
            runtime._wait_files(self.root, [self.root / "missing.json"], deadline=200)

    def test_recovery_keeps_original_start(self) -> None:
        """Startup repairs consume the original budget instead of buying another twelve hours."""

        self.ready("a")
        self.ready("b")
        with patch.object(runtime.time, "time", return_value=110.0):
            result = runtime.launch_cluster(self.root, started_at_unix=50.0)
            self.assertEqual(result["started_at_unix"], 50.0)
            with self.assertRaisesRegex(ValueError, "differs"):
                runtime.launch_cluster(self.root, started_at_unix=60.0)

    def test_recovery_rejects_future_start(self) -> None:
        """A supplied recovery timestamp cannot postpone the deadline."""

        with patch.object(runtime.time, "time", return_value=110.0), self.assertRaisesRegex(ValueError, "no later"):
            runtime.launch_cluster(self.root, started_at_unix=111.0)

    def test_sqlite_bootstrap_routes_to_authenticated_storage(self) -> None:
        """Each shard uses the native initializer on its node-local cache."""

        plan = {"study_root": str(self.root / "study"), "control_root": str(self.root / "control")}
        with patch.object(runtime, "prepare_storage", return_value={"test": "local"}) as prepare:
            self.assertEqual(runtime._enable_storage(plan, "a"), {"test": "local"})
        prepare.assert_called_once_with(plan, local_root=Path("/tmp/dit_continual_cluster_sqlite/a"))


# Run only explicit separate-process unittest invocations.
if __name__ == "__main__":
    unittest.main()
