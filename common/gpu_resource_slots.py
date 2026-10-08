"""Reserve online GPU workloads with shared process and memory accounting.

Each lease covers one workload and a total memory budget across its registered
processes. Acquire before importing TensorFlow; register an owned notebook kernel
before it starts GPU work. Unmapped NVIDIA/container PID identities block further
admission instead of assuming that distinct process lists refer to the same job.

The caller must verify the online runtime before using this allocator. GPU models
are not an admission policy: measured memory and verifiable ownership determine
capacity. Lock paths and record version match the deployed campaign allocator.
"""

import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
from types import TracebackType
from typing import Any, TextIO
from uuid import uuid4


LOCK_ROOT = Path('/tmp')
HEADROOM_MB = 2048


class AdmissionBlocked(RuntimeError):
    """The current process, lock, or memory inventory cannot admit another job."""


def _identity(pid: int) -> dict[str, Any]:
    """Read a Linux process identity and only its explicit namespace aliases."""

    directory = Path('/proc') / str(pid)
    fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    aliases = {int(pid)}
    for line in (directory / 'status').read_text().splitlines():
        # Only explicit namespace aliases identify the same process.
        if line.startswith('NSpid:'):
            aliases.update(int(value) for value in line.split()[1:])
    return {'pid': int(pid), 'start_ticks': fields[19], 'namespace_pids': sorted(aliases), 'parent_pid': int(fields[1])}


def _verified_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Reject PID reuse before using a record to identify a running process."""

    current = _identity(record['pid'])
    # Reused PIDs cannot establish continuity of slot ownership.
    if current['start_ticks'] != record['start_ticks']:
        raise AdmissionBlocked('A slot process PID was reused; its ownership cannot be verified.')
    return current


def _paths(gpu: int | str, max_jobs: int = 2) -> dict[str, Any]:
    """Discover every published slot while retaining the legacy slot-zero lock."""

    stem = 'joint-notebook-gpu-' + str(gpu).replace('/', '_')
    count = max(2, max_jobs)
    for path in LOCK_ROOT.glob(stem + '-slot*'):
        suffix = path.name[len(stem + '-slot'):]
        number, extension = os.path.splitext(suffix)
        # Previously allocated higher slots remain visible to smaller callers.
        if number.isdigit() and extension in ['.lock', '.json']:
            count = max(count, int(number) + 1)
    return {
        'mutex': LOCK_ROOT / (stem + '-admission.lock'), 
        'locks': [LOCK_ROOT / (stem + '.lock')] + [LOCK_ROOT / (stem + '-slot' + str(slot) + '.lock') for slot in range(1, count)], 
        'records': [LOCK_ROOT / (stem + '-slot' + str(slot) + '.json') for slot in range(count)]
    }


def _write_record(path: Path, record: dict[str, Any]) -> None:
    """Publish a complete allocation record while the admission mutex is held."""

    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def _gpu_inventory(gpu: int | str) -> dict[str, Any]:
    """Query one GPU and its compute processes without initializing a runtime."""

    output = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,name,memory.free,memory.total', 
        '--format=csv,noheader,nounits'
    ], text=True, timeout=15)
    [row] = list(csv.reader(output.splitlines(), skipinitialspace=True))
    uuid, model, free_mb, total_mb = row
    applications = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', 
        '--format=csv,noheader,nounits'
    ], text=True, timeout=15)
    processes = []
    for row in csv.reader(applications.splitlines(), skipinitialspace=True):
        # Account only for compute processes on the selected GPU.
        if len(row) == 3 and row[0] == uuid:
            try:
                memory_mb = int(row[2])
            except ValueError as error:
                raise AdmissionBlocked('A GPU process has unknown memory usage; admission is blocked.') from error
            processes.append({'pid': int(row[1]), 'memory_mb': memory_mb})
    return {'uuid': uuid, 'model': model, 'free_mb': int(free_mb), 'total_mb': int(total_mb), 'processes': processes}


def _kernel_pids() -> set[int]:
    """Read visible notebook kernels conservatively, including idle kernels."""

    result = set()
    for directory in Path('/proc').iterdir():
        # Non-numeric proc entries do not identify running processes.
        if not directory.name.isdigit():
            continue
        try:
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        # Visible notebook kernels need ownership even before GPU use.
        if 'ipykernel_launcher' in command or 'ipykernel/__main__' in command:
            result.add(int(directory.name))
    return result


class Lease:
    """Hold one slot until explicitly closed or the process exits."""

    def __init__(self, handle: TextIO, paths: dict[str, Any], record: dict[str, Any]) -> None:
        """Retain the lock handle and its exact allocation identity."""

        self.handle = handle
        self.paths = paths
        self.record = record

    def register_worker(self, pid: int) -> None:
        """Register a verified descendant before that worker starts GPU work."""

        try:
            worker = _identity(pid)
            ancestor = worker
            visited = set()
            while ancestor['pid'] != self.record['owner']['pid']:
                # Reject cycles or ancestry ending before the verified owner.
                if ancestor['pid'] in visited or ancestor['parent_pid'] <= 0:
                    raise AdmissionBlocked('Only descendants of the lease owner may use its workload slot.')
                visited.add(ancestor['pid'])
                ancestor = _identity(ancestor['parent_pid'])
            _verified_identity(self.record['owner'])
        except OSError as error:
            raise AdmissionBlocked('The proposed worker process or ancestry is not verifiable.') from error
        with (LOCK_ROOT / 'joint-notebook-worker-registration.lock').open('a') as registration_mutex, self.paths['mutex'].open('a') as mutex:
            fcntl.flock(registration_mutex.fileno(), fcntl.LOCK_EX)
            fcntl.flock(mutex.fileno(), fcntl.LOCK_EX)
            # A closed lease cannot retain or modify an allocation.
            if self.handle.closed:
                raise AdmissionBlocked('Cannot register a worker on a closed lease.')
            _check_registration(_paths(self.record['gpu']), self.record, worker)
            for other in _foreign_allocations(self.record['gpu']):
                # A trial process is assigned to one device and one workload lease.
                if set(worker['namespace_pids']) & other['aliases']:
                    raise AdmissionBlocked('The worker already belongs to another active GPU slot identity.')
            self.record['workers'] = [item for item in self.record['workers'] if item['pid'] != worker['pid']]
            self.record['workers'].append(worker)
            _write_record(self.paths['records'][self.record['slot']], self.record)

    def close(self) -> None:
        """Remove only this lease record and release its lifetime slot lock."""

        # A closed lease cannot retain or modify an allocation.
        if self.handle.closed:
            return
        with self.paths['mutex'].open('a') as mutex:
            fcntl.flock(mutex.fileno(), fcntl.LOCK_EX)
            path = self.paths['records'][self.record['slot']]
            # Only an existing record needs its ownership checked before removal.
            if path.exists():
                saved = json.loads(path.read_text())
                # Never remove a record that now belongs to another process.
                if saved['owner'] != self.record['owner'] or saved.get('allocation_group') != self.record.get('allocation_group'):
                    raise AdmissionBlocked('Slot record ownership changed; refusing to remove it.')
                path.unlink()
            self.handle.close()

    def __enter__(self) -> "Lease":
        """Return the acquired lease without taking another reservation."""

        return self

    def __exit__(self, exception_type: type[BaseException] | None, exception: BaseException | None, traceback: TracebackType | None) -> None:
        """Release the allocation when the caller leaves the context."""

        self.close()


def _allocation(record: dict[str, Any], slot: int, gpu_uuid: str) -> dict[str, Any]:
    """Validate one live reservation and separate owner aliases from workers."""

    group = record.get('allocation_group')
    valid_version = record.get('version') == 1 and group is None
    valid_group = record.get('version') == 2 and isinstance(group, str) and bool(group)
    # Grouped records use a distinct version so older allocators fail closed.
    if not (valid_version or valid_group) or record.get('gpu_uuid') != gpu_uuid or record.get('slot') != slot or not isinstance(record.get('memory_mb'), int) or record['memory_mb'] <= 0:
        raise AdmissionBlocked('An active slot allocation record is invalid; admission is blocked.')
    owner = _verified_identity(record['owner'])
    owner_aliases = set(owner['namespace_pids'])
    worker_aliases = set()
    for worker in record.get('workers', []):
        try:
            identity = _verified_identity(worker)
        except FileNotFoundError:
            continue
        aliases = set(identity['namespace_pids'])
        # A grouped CPU coordinator cannot also become a GPU worker.
        if group is not None and aliases & owner_aliases:
            raise AdmissionBlocked('A grouped lease owner cannot be registered as a GPU worker.')
        worker_aliases.update(aliases)
    return {
        'owner': owner, 'group': group, 'owner_aliases': owner_aliases, 
        'aliases': owner_aliases | worker_aliases, 
        'gpu_aliases': worker_aliases if group is not None else owner_aliases | worker_aliases, 
        'memory_mb': record['memory_mb']
    }


def _check_overlap(allocation: dict[str, Any], previous: dict[str, Any]) -> None:
    """Permit only the intentional shared CPU owner within one lease group."""

    overlap = allocation['aliases'] & previous['aliases']
    same_owner = allocation['owner']['pid'] == previous['owner']['pid'] and allocation['owner']['start_ticks'] == previous['owner']['start_ticks']
    same_group = allocation['group'] is not None and allocation['group'] == previous['group'] and same_owner
    permitted = allocation['owner_aliases'] & previous['owner_aliases'] if same_group else set()
    # Worker identities may never be charged to two workload reservations.
    if overlap - permitted:
        raise AdmissionBlocked('Active slot process identities overlap; admission is blocked.')


def _check_registration(paths: dict[str, Any], record: dict[str, Any], worker: dict[str, Any]) -> None:
    """Reject workers already assigned to another live slot under the mutex."""

    aliases = set(worker['namespace_pids'])
    # Group owners remain CPU-only even if a caller registers the owner itself.
    if record.get('allocation_group') is not None and aliases & set(record['owner']['namespace_pids']):
        raise AdmissionBlocked('A grouped lease owner cannot be registered as a GPU worker.')
    for slot, path in enumerate(paths['locks']):
        # The current lease may retain or refresh its own worker identity.
        if slot == record['slot']:
            continue
        with path.open('a') as probe:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                record_path = paths['records'][slot]
                # Held legacy locks cannot establish exclusive worker ownership.
                if not record_path.exists():
                    raise AdmissionBlocked('An active legacy slot has no validated owner/memory record; wait for it to finish.')
                other = _allocation(json.loads(record_path.read_text()), slot, record['gpu_uuid'])
                # A registered worker must belong to exactly one workload slot.
                if aliases & other['aliases']:
                    raise AdmissionBlocked('The worker already belongs to another active slot identity.')


def _foreign_allocations(gpu: int | str) -> list[dict[str, Any]]:
    """Read verified live leases on other devices without charging their memory."""

    allocations = []
    inventories = {}
    for record_path in LOCK_ROOT.glob('joint-notebook-gpu-*-slot*.json'):
        device, slot_text = record_path.stem.removeprefix('joint-notebook-gpu-').rsplit('-slot', 1)
        # Current-device ownership is already checked under its admission mutex.
        if device == str(gpu) or not slot_text.isdigit():
            continue
        slot = int(slot_text)
        paths = _paths(device, max_jobs=slot + 1)
        try:
            with paths['locks'][slot].open('a') as probe:
                try:
                    fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    try:
                        record = json.loads(record_path.read_text())
                    except FileNotFoundError:
                        # Cleanup removes the record immediately before unlocking it.
                        continue
                    # A record must describe the physical device named by its lock.
                    if str(record['gpu']) != device:
                        raise AdmissionBlocked('A foreign GPU slot record has an invalid identity.')
                    # Verify each physical GPU UUID once per foreign-lease inspection.
                    if device not in inventories:
                        inventories[device] = _gpu_inventory(device)
                    allocations.append(_allocation(record, slot, inventories[device]['uuid']))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise AdmissionBlocked('Foreign GPU resource ownership could not be verified: ' + str(error)) from error
    return allocations


def acquire(memory_mb: int = 12288, gpu: int | str = 0, max_jobs: int = 2) -> Lease:
    """Reserve one workload while prohibiting repeated admission by its owner.

    The integer MiB budget covers this workload's owner and registered workers.
    The caller enforces that budget in the GPU runtime. Admission retains 2048
    MiB headroom and considers all active slots, including higher numbered slots.
    """

    return acquire_many(memory_mb=memory_mb, gpu=gpu, max_jobs=max_jobs, count=1)[0]


def acquire_many(memory_mb: int = 12288, gpu: int | str = 0, max_jobs: int = 2, count: int = 1) -> list[Lease]:
    """Reserve a complete group of workload slots or leave no new allocations.

    Each slot receives the integer MiB budget. Positive integer count and max_jobs
    describe the requested group size and total allowed workloads. A multi-slot
    group shares a CPU-only owner; each GPU process must be registered with one
    distinct lease. Separate calls by an already admitted owner remain forbidden.
    """

    # Group dimensions must describe a realizable collection of workload slots.
    if type(count) is not int or type(max_jobs) is not int or count < 1 or max_jobs < count:
        raise ValueError('A workload group requires positive integer count <= max_jobs.')
    paths = _paths(gpu, max_jobs=max_jobs)
    handles = []
    selected = []
    published = []
    with paths['mutex'].open('a') as mutex:
        fcntl.flock(mutex.fileno(), fcntl.LOCK_EX)
        paths = _paths(gpu, max_jobs=max_jobs)
        try:
            inventory = _gpu_inventory(gpu)
            owner = _identity(os.getpid())
            owned_aliases = set(owner['namespace_pids'])
            allocations = []
            free_slots = []
            for slot, path in enumerate(paths['locks']):
                handle = path.open('a')
                handles.append(handle)
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    record_path = paths['records'][slot]
                    # A held lock without a record has unverifiable legacy ownership.
                    if not record_path.exists():
                        raise AdmissionBlocked('An active legacy slot has no validated owner/memory record; wait for it to finish.')
                    allocation = _allocation(json.loads(record_path.read_text()), slot, inventory['uuid'])
                    # Only one atomic call may admit a process and its lease group.
                    if allocation['aliases'] & owned_aliases:
                        raise AdmissionBlocked('This process already owns or overlaps an active slot identity.')
                    for previous in allocations:
                        _check_overlap(allocation, previous)
                    allocations.append(allocation)
                else:
                    free_slots.append((slot, handle))
            # Count all live reservations even when callers request fewer slots.
            if len(allocations) + count > max_jobs or len(free_slots) < count:
                raise AdmissionBlocked('Both remote workload slots are occupied or the requested workload group exceeds available slots; wait for capacity.')
            all_aliases = owned_aliases | {pid for allocation in allocations for pid in allocation['aliases']}
            unknown_gpu = [item['pid'] for item in inventory['processes'] if item['pid'] not in all_aliases]
            unknown_kernels = _kernel_pids() - all_aliases
            # A verified notebook on another device does not consume this GPU's slots.
            if unknown_kernels:
                foreign_aliases = {pid for allocation in _foreign_allocations(gpu) for pid in allocation['aliases']}
                unknown_kernels -= foreign_aliases
            # Unowned GPU processes or kernels make further admission unsafe.
            if unknown_gpu or unknown_kernels:
                raise AdmissionBlocked('Unmapped GPU/kernel workers block admission; GPU PIDs=' + str(unknown_gpu) + ', kernel PIDs=' + str(sorted(unknown_kernels)))
            cpu_owners = owned_aliases if count > 1 else set()
            for allocation in allocations:
                # Existing grouped owners stay CPU-only even after partial cleanup.
                if allocation['group'] is not None:
                    cpu_owners.update(allocation['owner_aliases'])
            # Shared owner memory cannot be assigned unambiguously to one slot.
            if any(item['pid'] in cpu_owners for item in inventory['processes']):
                raise AdmissionBlocked('A grouped lease owner is using the GPU; admission is blocked.')
            remaining_reserved = 0
            for allocation in allocations:
                observed = sum(item['memory_mb'] for item in inventory['processes'] if item['pid'] in allocation['gpu_aliases'])
                # A worker over its reservation invalidates the memory budget.
                if observed > allocation['memory_mb']:
                    raise AdmissionBlocked('An existing workload exceeds its declared memory budget.')
                remaining_reserved += allocation['memory_mb'] - observed
            required_mb = count * memory_mb + HEADROOM_MB + remaining_reserved
            # Leave headroom and cover every requested and existing reservation.
            if inventory['free_mb'] < required_mb:
                raise AdmissionBlocked('Insufficient free GPU memory: need ' + str(required_mb) + ' MiB including reservations/headroom, have ' + str(inventory['free_mb']) + ' MiB.')
            group = uuid4().hex if count > 1 else None
            leases = []
            for slot, chosen_handle in free_slots[:count]:
                record = {
                    'version': 2 if group is not None else 1, 'slot': slot, 'gpu': str(gpu), 'gpu_uuid': inventory['uuid'], 
                    'model': inventory['model'], 'memory_mb': memory_mb, 'headroom_mb': HEADROOM_MB, 
                    'owner': owner, 'workers': [], 'free_mb_at_admission': inventory['free_mb'], 
                    'required_mb_at_admission': required_mb
                }
                # Version-two records explicitly identify their shared CPU owner group.
                if group is not None:
                    record['allocation_group'] = group
                published.append((paths['records'][slot], record))
                _write_record(paths['records'][slot], record)
                leases.append(Lease(chosen_handle, paths, record))
            selected = [lease.handle for lease in leases]
            return leases
        except (OSError, KeyError, json.JSONDecodeError) as error:
            raise AdmissionBlocked('Resource ownership or device inventory could not be verified: ' + str(error)) from error
        finally:
            try:
                # Failed publication must not leave a partial group allocation behind.
                if not selected:
                    for path, record in published:
                        # Remove only records written by this unsuccessful admission.
                        if path.exists() and json.loads(path.read_text()) == record:
                            path.unlink()
            finally:
                for handle in handles:
                    # Release probes even when cleanup encounters a filesystem failure.
                    if handle not in selected and not handle.closed:
                        handle.close()
