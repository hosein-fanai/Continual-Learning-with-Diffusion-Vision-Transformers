"""Check portable GPU leases with fake inventory and isolated real file locks."""

from contextlib import contextmanager, ExitStack
import fcntl
import json
from pathlib import Path
import tempfile
from typing import Any, Iterator
import unittest
from unittest.mock import patch

from common import gpu_resource_slots as slots


@contextmanager
def isolated_allocator() -> Iterator[dict[str, Any]]:
    """Replace host discovery while preserving lock and record operations."""

    owner = {"pid": 101, "start_ticks": "1001", "namespace_pids": [101], "parent_pid": 1}
    second = {"pid": 202, "start_ticks": "2002", "namespace_pids": [202], "parent_pid": 1}
    worker = {"pid": 303, "start_ticks": "3003", "namespace_pids": [303], "parent_pid": 101}
    identities = {101: owner, 202: second, 303: worker}
    inventory = {
        "uuid": "GPU-hosted-test", "model": "Tesla T4", "free_mb": 15000, 
        "total_mb": 15360, "processes": []
    }
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        stack.enter_context(patch.object(slots, "LOCK_ROOT", Path(directory)))
        stack.enter_context(patch.object(slots, "_gpu_inventory", return_value=inventory))
        stack.enter_context(patch.object(slots, "_kernel_pids", return_value=set()))
        stack.enter_context(patch.object(slots, "_identity", side_effect=lambda pid: dict(identities[pid])))
        pid = stack.enter_context(patch.object(slots.os, "getpid", return_value=101))
        yield {"pid": pid, "inventory": inventory, "identities": identities, "paths": slots._paths(0)}


class PortableSlotTests(unittest.TestCase):
    """Verify the hosted one-worker policy and shared lease protocol."""

    def test_t4_budget_holds_protocol_lock_and_releases_record(self) -> None:
        """Admit a hosted GPU by measured memory and retain its lifetime lock."""

        with isolated_allocator() as fixture:
            with slots.acquire(memory_mb=11264, gpu=0, max_jobs=1) as lease:
                self.assertEqual(lease.record["version"], 1)
                self.assertEqual(lease.record["slot"], 0)
                self.assertEqual(lease.record["gpu_uuid"], "GPU-hosted-test")
                self.assertEqual(lease.record["required_mb_at_admission"], 13312)
                self.assertEqual(lease.record["memory_mb"], 11264)
                self.assertEqual(fixture["paths"]["locks"][0].name, "joint-notebook-gpu-0.lock")
                self.assertEqual(fixture["paths"]["records"][0].name, "joint-notebook-gpu-0-slot0.json")
                with fixture["paths"]["locks"][0].open("a") as probe:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertFalse(fixture["paths"]["records"][0].exists())
            self.assertTrue(lease.handle.closed)

    def test_one_worker_policy_blocks_free_second_slot(self) -> None:
        """A hosted one-slot policy must also see allocations in shared locks."""

        with isolated_allocator() as fixture:
            with slots.acquire(memory_mb=4096, gpu=0, max_jobs=1) as first:
                fixture["pid"].return_value = 202
                with self.assertRaisesRegex(slots.AdmissionBlocked, "Both remote workload slots are occupied"):
                    slots.acquire(memory_mb=4096, gpu=0, max_jobs=1)
                self.assertEqual(json.loads(fixture["paths"]["records"][0].read_text()), first.record)
                self.assertFalse(fixture["paths"]["records"][1].exists())

    def test_reservations_cover_memory_not_yet_used(self) -> None:
        """Existing idle allocations retain their full outstanding memory budget."""

        with isolated_allocator() as fixture:
            with slots.acquire(memory_mb=6000, gpu=0, max_jobs=2):
                fixture["pid"].return_value = 202
                with self.assertRaisesRegex(slots.AdmissionBlocked, "need 17048 MiB"):
                    slots.acquire(memory_mb=9000, gpu=0, max_jobs=2)
                with slots.acquire(memory_mb=5000, gpu=0, max_jobs=2) as second:
                    self.assertEqual(second.record["slot"], 1)
                    self.assertEqual(second.record["required_mb_at_admission"], 13048)

    def test_unknown_workers_fail_without_leaving_an_allocation(self) -> None:
        """Do not assume an unregistered GPU process or kernel is harmless."""

        with isolated_allocator() as fixture:
            fixture["inventory"]["processes"] = [{"pid": 404, "memory_mb": 1}]
            with self.assertRaisesRegex(slots.AdmissionBlocked, "Unmapped GPU/kernel workers"):
                slots.acquire(memory_mb=4096, gpu=0, max_jobs=1)
            self.assertFalse(any(path.exists() for path in fixture["paths"]["records"]))
            fixture["inventory"]["processes"] = []
            with patch.object(slots, "_kernel_pids", return_value={404}):
                with self.assertRaisesRegex(slots.AdmissionBlocked, "Unmapped GPU/kernel workers"):
                    slots.acquire(memory_mb=4096, gpu=0, max_jobs=1)
            with slots.acquire(memory_mb=4096, gpu=0, max_jobs=1):
                pass

    def test_registered_worker_must_descend_from_owner(self) -> None:
        """Publish only verified descendants and retain exact process identity."""

        with isolated_allocator() as fixture:
            with slots.acquire(memory_mb=4096, gpu=0, max_jobs=1) as lease:
                lease.register_worker(303)
                recorded = json.loads(fixture["paths"]["records"][0].read_text())
                self.assertEqual(recorded["workers"], [fixture["identities"][303]])
                fixture["identities"][202]["parent_pid"] = 0
                with self.assertRaisesRegex(slots.AdmissionBlocked, "Only descendants"):
                    lease.register_worker(202)
                self.assertEqual(json.loads(fixture["paths"]["records"][0].read_text()), recorded)

    def test_pid_reuse_blocks_second_admission(self) -> None:
        """A process number alone cannot validate a previously recorded owner."""

        with isolated_allocator() as fixture:
            with slots.acquire(memory_mb=4096, gpu=0, max_jobs=2):
                fixture["identities"][101]["start_ticks"] = "changed"
                fixture["pid"].return_value = 202
                with self.assertRaisesRegex(slots.AdmissionBlocked, "PID was reused"):
                    slots.acquire(memory_mb=4096, gpu=0, max_jobs=2)

    def test_existing_reservation_overrun_blocks_admission(self) -> None:
        """Measured usage beyond an existing budget stops new work."""

        with isolated_allocator() as fixture:
            with slots.acquire(memory_mb=4096, gpu=0, max_jobs=2):
                fixture["pid"].return_value = 202
                fixture["inventory"]["processes"] = [{"pid": 101, "memory_mb": 4097}]
                with self.assertRaisesRegex(slots.AdmissionBlocked, "exceeds its declared memory budget"):
                    slots.acquire(memory_mb=4096, gpu=0, max_jobs=2)

    def test_changed_owner_record_is_preserved_on_close(self) -> None:
        """Lease cleanup must never remove a record assigned to another owner."""

        with isolated_allocator() as fixture:
            lease = slots.acquire(memory_mb=4096, gpu=0, max_jobs=1)
            path = fixture["paths"]["records"][0]
            changed = dict(lease.record)
            changed["owner"] = fixture["identities"][202]
            path.write_text(json.dumps(changed))
            try:
                with self.assertRaisesRegex(slots.AdmissionBlocked, "ownership changed"):
                    lease.close()
                self.assertEqual(json.loads(path.read_text()), changed)
                self.assertFalse(lease.handle.closed)
            finally:
                path.write_text(json.dumps(lease.record))
                lease.close()


class HostedGroupTests(unittest.TestCase):
    """Verify atomic groups, shared CPU ownership, and higher-slot discovery."""

    def test_group_dimensions_require_exact_integer_counts(self) -> None:
        """Do not interpret Boolean or fractional values as workload counts."""

        with isolated_allocator():
            for count, max_jobs in [(True, 5), (1, True), (1.5, 5), (1, 2.5)]:
                with self.subTest(count=count, max_jobs=max_jobs):
                    with self.assertRaisesRegex(ValueError, 'positive integer count <= max_jobs'):
                        slots.acquire_many(memory_mb=4096, max_jobs=max_jobs, count=count)

    def test_five_slots_include_one_parent_and_four_group_workers(self) -> None:
        """Reserve a full hosted coordinator group with the notebook lease held."""

        with isolated_allocator() as fixture, ExitStack() as stack:
            fixture['inventory']['free_mb'] = 80000
            fixture['inventory']['total_mb'] = 81920
            first = stack.enter_context(slots.acquire(memory_mb=13312, max_jobs=5))
            fixture['pid'].return_value = 202
            leases = slots.acquire_many(memory_mb=13312, max_jobs=5, count=4)
            for lease in leases:
                stack.enter_context(lease)
            self.assertEqual(first.record['version'], 1)
            self.assertEqual([lease.record['slot'] for lease in leases], [1, 2, 3, 4])
            self.assertEqual({lease.record['version'] for lease in leases}, {2})
            self.assertEqual(len({lease.record['allocation_group'] for lease in leases}), 1)
            self.assertTrue(all(lease.record['required_mb_at_admission'] == 68608 for lease in leases))
            self.assertEqual(len(slots._paths(0, max_jobs=1)['records']), 5)

    def test_smaller_caller_counts_active_slots_above_its_limit(self) -> None:
        """Discover high slot records even after both original slots are released."""

        with isolated_allocator() as fixture, ExitStack() as stack:
            fixture['inventory']['free_mb'] = 80000
            leases = slots.acquire_many(memory_mb=4096, max_jobs=5, count=5)
            for lease in leases:
                stack.enter_context(lease)
            leases[0].close()
            leases[1].close()
            fixture['pid'].return_value = 202
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'workload slots are occupied'):
                slots.acquire(memory_mb=4096, max_jobs=2)

    def test_group_capacity_shortage_creates_no_partial_records(self) -> None:
        """Keep the existing reservation intact when the entire group cannot fit."""

        with isolated_allocator() as fixture:
            fixture['inventory']['free_mb'] = 80000
            with slots.acquire(memory_mb=4096, max_jobs=5):
                fixture['pid'].return_value = 202
                with self.assertRaisesRegex(slots.AdmissionBlocked, 'workload slots are occupied'):
                    slots.acquire_many(memory_mb=4096, max_jobs=5, count=5)
                self.assertEqual([path.name for path in slots.LOCK_ROOT.glob('*.json')], ['joint-notebook-gpu-0-slot0.json'])

    def test_group_memory_shortage_creates_no_partial_records(self) -> None:
        """Budget every new slot before writing the first group allocation."""

        with isolated_allocator() as fixture:
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'need 18432 MiB'):
                slots.acquire_many(memory_mb=4096, max_jobs=4, count=4)
            self.assertEqual(list(slots.LOCK_ROOT.glob('*.json')), [])
            with slots.acquire(memory_mb=4096, max_jobs=1):
                pass

    def test_failed_group_publication_releases_records_and_locks(self) -> None:
        """An interrupted second record write rolls back the whole new group."""

        with isolated_allocator() as fixture:
            fixture['inventory']['free_mb'] = 80000
            original_write = slots._write_record

            def failing_write(path: Path, record: dict[str, Any]) -> None:
                """Simulate a filesystem failure after publishing the first slot."""

                # Only the second publication fails; the first exercises rollback.
                if record['slot'] == 1:
                    raise OSError('simulated record write failure')
                original_write(path, record)

            with patch.object(slots, '_write_record', side_effect=failing_write):
                with self.assertRaisesRegex(slots.AdmissionBlocked, 'simulated record write failure'):
                    slots.acquire_many(memory_mb=4096, max_jobs=5, count=4)
            self.assertEqual(list(slots.LOCK_ROOT.glob('*.json')), [])
            with ExitStack() as stack:
                for lease in slots.acquire_many(memory_mb=4096, max_jobs=5, count=5):
                    stack.enter_context(lease)

    def test_same_owner_cannot_extend_an_existing_group(self) -> None:
        """Repeated admission remains forbidden for ordinary and grouped callers."""

        with isolated_allocator() as fixture, ExitStack() as stack:
            fixture['inventory']['free_mb'] = 80000
            for lease in slots.acquire_many(memory_mb=4096, max_jobs=5, count=2):
                stack.enter_context(lease)
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'already owns or overlaps'):
                slots.acquire(memory_mb=4096, max_jobs=5)
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'already owns or overlaps'):
                slots.acquire_many(memory_mb=4096, max_jobs=5, count=2)

    def test_gpu_active_owner_cannot_request_a_group(self) -> None:
        """Shared ownership is allowed only for a coordinator without GPU use."""

        with isolated_allocator() as fixture:
            fixture['inventory']['free_mb'] = 80000
            fixture['inventory']['processes'] = [{'pid': 101, 'memory_mb': 1}]
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'grouped lease owner is using the GPU'):
                slots.acquire_many(memory_mb=4096, max_jobs=5, count=4)
            self.assertEqual(list(slots.LOCK_ROOT.glob('*.json')), [])

    def test_existing_group_owner_gpu_use_blocks_admission(self) -> None:
        """Never charge one coordinator's memory repeatedly across group leases."""

        with isolated_allocator() as fixture, ExitStack() as stack:
            fixture['inventory']['free_mb'] = 80000
            for lease in slots.acquire_many(memory_mb=4096, max_jobs=5, count=2):
                stack.enter_context(lease)
            fixture['pid'].return_value = 202
            fixture['inventory']['processes'] = [{'pid': 101, 'memory_mb': 1}]
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'grouped lease owner is using the GPU'):
                slots.acquire(memory_mb=4096, max_jobs=5)

    def test_group_workers_belong_to_exactly_one_slot(self) -> None:
        """Reject duplicate worker registrations and registering a grouped owner."""

        with isolated_allocator() as fixture, ExitStack() as stack:
            fixture['inventory']['free_mb'] = 80000
            leases = slots.acquire_many(memory_mb=4096, max_jobs=5, count=2)
            for lease in leases:
                stack.enter_context(lease)
            leases[0].register_worker(303)
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'already belongs to another'):
                leases[1].register_worker(303)
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'owner cannot be registered'):
                leases[0].register_worker(101)

    def test_group_workers_are_charged_to_their_own_budgets(self) -> None:
        """Only each slot's observed worker memory reduces its reservation."""

        with isolated_allocator() as fixture, ExitStack() as stack:
            fixture['inventory']['free_mb'] = 80000
            fixture['identities'][404] = {'pid': 404, 'start_ticks': '4004', 'namespace_pids': [404], 'parent_pid': 101}
            leases = slots.acquire_many(memory_mb=4096, max_jobs=5, count=2)
            for lease in leases:
                stack.enter_context(lease)
            leases[0].register_worker(303)
            leases[1].register_worker(404)
            fixture['pid'].return_value = 202
            fixture['inventory']['processes'] = [{'pid': 303, 'memory_mb': 1000}, {'pid': 404, 'memory_mb': 2000}]
            with slots.acquire(memory_mb=4096, max_jobs=5) as extra:
                self.assertEqual(extra.record['required_mb_at_admission'], 11336)
            fixture['inventory']['processes'][0]['memory_mb'] = 4097
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'exceeds its declared memory budget'):
                slots.acquire(memory_mb=4096, max_jobs=5)

    def test_group_worker_pid_reuse_blocks_admission(self) -> None:
        """Grouped ownership cannot hide a recycled worker process number."""

        with isolated_allocator() as fixture, ExitStack() as stack:
            fixture['inventory']['free_mb'] = 80000
            leases = slots.acquire_many(memory_mb=4096, max_jobs=5, count=2)
            for lease in leases:
                stack.enter_context(lease)
            leases[0].register_worker(303)
            fixture['identities'][303]['start_ticks'] = 'reused'
            fixture['pid'].return_value = 202
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'PID was reused'):
                slots.acquire(memory_mb=4096, max_jobs=5)

    def test_overlapping_worker_records_fail_closed(self) -> None:
        """Externally corrupted group records cannot double count one GPU worker."""

        with isolated_allocator() as fixture, ExitStack() as stack:
            fixture['inventory']['free_mb'] = 80000
            leases = slots.acquire_many(memory_mb=4096, max_jobs=5, count=2)
            for lease in leases:
                stack.enter_context(lease)
            leases[0].register_worker(303)
            changed = dict(leases[1].record)
            changed['workers'] = [fixture['identities'][303]]
            slots._write_record(leases[1].paths['records'][1], changed)
            fixture['pid'].return_value = 202
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'identities overlap'):
                slots.acquire(memory_mb=4096, max_jobs=5)


@contextmanager
def isolated_multi_gpu_allocator() -> Iterator[dict[str, Any]]:
    """Give each device an independent inventory while sharing lease ownership."""

    with isolated_allocator() as fixture:
        inventories = {
            '0': fixture['inventory'], 
            '1': {'uuid': 'GPU-second-test', 'model': 'A100', 'free_mb': 20000, 'total_mb': 81920, 'processes': []}
        }
        with patch.object(slots, '_gpu_inventory', side_effect=lambda gpu: inventories[str(gpu)]):
            fixture['inventories'] = inventories
            yield fixture


class MultiGpuSlotTests(unittest.TestCase):
    """Keep per-device budgets independent and global worker ownership exclusive."""

    def test_one_coordinator_reserves_and_releases_independent_gpu_groups(self) -> None:
        """A CPU owner can hold several slots on each of two distinct GPUs."""

        with isolated_multi_gpu_allocator() as fixture, ExitStack() as first_stack:
            first = slots.acquire_many(memory_mb=4096, gpu=0, max_jobs=2, count=2)
            for lease in first:
                first_stack.enter_context(lease)
            with ExitStack() as second_stack:
                second = slots.acquire_many(memory_mb=4096, gpu=1, max_jobs=2, count=2)
                for lease in second:
                    second_stack.enter_context(lease)
                self.assertEqual([lease.record['slot'] for lease in first], [0, 1])
                self.assertEqual([lease.record['slot'] for lease in second], [0, 1])
                self.assertEqual({lease.record['required_mb_at_admission'] for lease in first + second}, {10240})
                self.assertEqual({lease.record['gpu_uuid'] for lease in first + second}, {'GPU-hosted-test', 'GPU-second-test'})
            self.assertTrue(all(path.exists() for path in first[0].paths['records']))
            self.assertFalse(any(path.exists() for path in second[0].paths['records']))
            fixture['pid'].return_value = 202
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'workload slots are occupied'):
                slots.acquire(memory_mb=4096, gpu=0, max_jobs=2)
            with slots.acquire(memory_mb=4096, gpu=1, max_jobs=1):
                pass

    def test_other_gpu_owned_kernels_do_not_consume_selected_gpu_capacity(self) -> None:
        """Recognize another device's notebook and worker without charging its budget."""

        with isolated_multi_gpu_allocator() as fixture:
            with slots.acquire(memory_mb=4096, gpu=0, max_jobs=1) as first:
                first.register_worker(303)
                fixture['pid'].return_value = 202
                fixture['inventories']['0']['processes'] = [{'pid': 303, 'memory_mb': 4000}]
                with patch.object(slots, '_kernel_pids', return_value={101, 202, 303}):
                    with slots.acquire(memory_mb=16000, gpu=1, max_jobs=1) as second:
                        self.assertEqual(second.record['required_mb_at_admission'], 18048)

    def test_worker_on_wrong_gpu_remains_unknown_despite_foreign_registration(self) -> None:
        """Ownership on GPU zero cannot authorize the same process on GPU one."""

        with isolated_multi_gpu_allocator() as fixture:
            with slots.acquire(memory_mb=4096, gpu=0, max_jobs=1) as first:
                first.register_worker(303)
                fixture['pid'].return_value = 202
                fixture['inventories']['1']['processes'] = [{'pid': 303, 'memory_mb': 1}]
                with patch.object(slots, '_kernel_pids', return_value={101, 303}):
                    with self.assertRaisesRegex(slots.AdmissionBlocked, r'GPU PIDs=\[303\]'):
                        slots.acquire(memory_mb=4096, gpu=1, max_jobs=1)
                self.assertFalse(any(path.exists() for path in slots._paths(1)['records']))

    def test_unowned_kernel_still_blocks_with_verified_foreign_leases(self) -> None:
        """A known notebook on another device cannot hide an unrelated kernel."""

        with isolated_multi_gpu_allocator() as fixture:
            with slots.acquire(memory_mb=4096, gpu=0, max_jobs=1):
                fixture['pid'].return_value = 202
                with patch.object(slots, '_kernel_pids', return_value={101, 404}):
                    with self.assertRaisesRegex(slots.AdmissionBlocked, r'kernel PIDs=\[404\]'):
                        slots.acquire(memory_mb=4096, gpu=1, max_jobs=1)

    def test_foreign_stale_record_does_not_establish_kernel_ownership(self) -> None:
        """An unlocked record is historical data even if its process remains alive."""

        with isolated_multi_gpu_allocator() as fixture:
            with slots.acquire(memory_mb=4096, gpu=0, max_jobs=1) as first:
                record = dict(first.record)
            first.paths['records'][0].write_text(json.dumps(record))
            fixture['pid'].return_value = 202
            with patch.object(slots, '_kernel_pids', return_value={101}):
                with self.assertRaisesRegex(slots.AdmissionBlocked, r'kernel PIDs=\[101\]'):
                    slots.acquire(memory_mb=4096, gpu=1, max_jobs=1)
            self.assertEqual(json.loads(first.paths['records'][0].read_text()), record)

    def test_foreign_pid_reuse_and_gpu_uuid_change_fail_closed(self) -> None:
        """Foreign aliases require verified process lifetime and physical GPU identity."""

        with isolated_multi_gpu_allocator() as fixture:
            with slots.acquire(memory_mb=4096, gpu=0, max_jobs=1):
                fixture['pid'].return_value = 202
                with patch.object(slots, '_kernel_pids', return_value={101}):
                    fixture['identities'][101]['start_ticks'] = 'reused'
                    with self.assertRaisesRegex(slots.AdmissionBlocked, 'PID was reused'):
                        slots.acquire(memory_mb=4096, gpu=1, max_jobs=1)
                    fixture['identities'][101]['start_ticks'] = '1001'
                    fixture['inventories']['0']['uuid'] = 'GPU-replaced'
                    with self.assertRaisesRegex(slots.AdmissionBlocked, 'active slot allocation record is invalid'):
                        slots.acquire(memory_mb=4096, gpu=1, max_jobs=1)

    def test_one_worker_cannot_register_on_two_gpu_groups(self) -> None:
        """A shared CPU coordinator assigns each descendant to exactly one lease."""

        with isolated_multi_gpu_allocator(), ExitStack() as stack:
            first = slots.acquire_many(memory_mb=4096, gpu=0, max_jobs=2, count=2)
            second = slots.acquire_many(memory_mb=4096, gpu=1, max_jobs=2, count=2)
            for lease in first + second:
                stack.enter_context(lease)
            first[0].register_worker(303)
            with self.assertRaisesRegex(slots.AdmissionBlocked, 'another active GPU slot identity'):
                second[0].register_worker(303)
            self.assertEqual(second[0].record['workers'], [])
            self.assertEqual(len(first[0].record['workers']), 1)


# Run this focused suite in a separate remote Python process.
if __name__ == "__main__":
    unittest.main()
