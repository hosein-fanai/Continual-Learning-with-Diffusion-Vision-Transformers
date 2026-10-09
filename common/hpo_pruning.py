"""Standard-library epoch reports for coordinator-owned HPO pruning decisions.

Training children never touch Optuna storage. Unique attempt and request tokens
bind each atomic report/decision exchange to one trial and zero-based epoch.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import errno
import json
import math
from pathlib import Path
import time
from uuid import uuid4


class TrialPerformancePruned(RuntimeError):
    """A coordinator decision to stop a finite, poorly performing trial."""

    def __init__(self, evidence: Mapping[str, object], evidence_path: Path | None = None) -> None:
        """Retain JSON-safe partial metrics and their optional persisted location."""

        self.evidence = deepcopy(dict(evidence))
        self.evidence_path = evidence_path
        super().__init__(
            f"Performance pruning at epoch {self.evidence['epoch']}: "
            f"{self.evidence['metric']}={self.evidence['value']}"
        )


def create_exchange(directory: Path, monitor: str, trial_number: int) -> dict[str, object]:
    """Create an identity for an already owned temporary exchange directory."""

    return validate_exchange({
        "protocol_version": 1, 
        "directory": str(Path(directory).resolve()), 
        "token": uuid4().hex, 
        "monitor": monitor, 
        "trial_number": trial_number
    })


def _valid_token(value: object) -> bool:
    """Recognize a generated UUID token without accepting path components."""

    return isinstance(value, str) and len(value) == 32 and all(
        character in "0123456789abcdef" for character in value
    )


def validate_exchange(exchange: Mapping[str, object]) -> dict[str, object]:
    """Validate process identity fields without importing a framework or study."""

    # Exact fields prevent stale protocol versions from being interpreted silently.
    if not isinstance(exchange, Mapping) or set(exchange) != {
        "protocol_version", "directory", "token", "monitor", "trial_number"
    } or type(exchange["protocol_version"]) is not int or exchange["protocol_version"] != 1:
        raise ValueError("Malformed HPO pruning exchange identity.")
    # A private absolute directory and UUID identify a single launch attempt.
    if not isinstance(exchange["directory"], str) or not Path(exchange["directory"]).is_absolute() \
            or not _valid_token(exchange["token"]):
        raise ValueError("Malformed HPO pruning exchange directory or token.")
    # Trial and metric identities must agree with the coordinator's policy.
    if not isinstance(exchange["monitor"], str) or not exchange["monitor"].strip() \
            or isinstance(exchange["trial_number"], bool) \
            or not isinstance(exchange["trial_number"], int) or exchange["trial_number"] < 0:
        raise ValueError("Malformed HPO pruning monitor or trial number.")
    return dict(exchange)


def write_atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish finite JSON atomically and remove an incomplete adjacent file."""

    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(dict(payload), allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_atomic_json(path: Path, retry_missing: bool = True) -> object:
    """Read a publication with a bounded grace period for storage visibility.

    Retry incomplete JSON and transient filesystem visibility at most twenty
    times over one second. Missing terminal results get the same grace period;
    polling channels may return immediately before their first publication.
    Schema and identity validation remains with the caller and is never retried.
    Persistent malformed JSON or I/O errors propagate after the final attempt.
    """

    transient_errors = {errno.EAGAIN, errno.EINTR, errno.EIO, errno.ETIMEDOUT}
    transient_errors.add(getattr(errno, "ESTALE", errno.EIO))
    for attempt in range(21):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            # Polling channels may legitimately have no first publication yet.
            if isinstance(error, FileNotFoundError):
                # Let the scheduler advance immediately to its other workers.
                if not retry_missing:
                    raise
            # Permission and other permanent I/O failures are not visibility races.
            elif isinstance(error, OSError) and error.errno not in transient_errors:
                raise
            # Persistent corruption must remain visible instead of hanging a worker.
            if attempt == 20:
                raise
            time.sleep(0.05)


def validate_report(exchange: Mapping[str, object], report: object) -> dict[str, object]:
    """Reject cross-attempt, cross-trial, malformed, and nonfinite epoch reports."""

    exchange = validate_exchange(exchange)
    # The report has an exact schema so readers cannot silently miss an identity.
    if not isinstance(report, dict) or set(report) != {
        "protocol_version", "token", "trial_number", "monitor", "step", "request_token", "value"
    }:
        raise ValueError("Malformed HPO pruning epoch report.")
    for field in ("protocol_version", "token", "trial_number", "monitor"):
        # Reused directories or another trial's report must fail closed.
        if report[field] != exchange[field] or type(report[field]) is not type(exchange[field]):
            raise ValueError(f"HPO pruning report has a different {field}.")
    # Epoch identity cannot be a boolean, fractional step, or path-like token.
    if isinstance(report["step"], bool) or not isinstance(report["step"], int) \
            or report["step"] < 0 or not _valid_token(report["request_token"]):
        raise ValueError("Malformed HPO pruning epoch or request token.")
    # Nonfinite objectives belong to the divergence guard, never Optuna ranking.
    if isinstance(report["value"], bool) or not isinstance(report["value"], (int, float)) \
            or not math.isfinite(report["value"]):
        raise ValueError("HPO pruning reports require a finite scalar value.")
    return dict(report)


def read_report(exchange: Mapping[str, object]) -> dict[str, object] | None:
    """Read a complete epoch report, or return None before its atomic publication."""

    exchange = validate_exchange(exchange)
    try:
        value = read_atomic_json(Path(exchange["directory"]) / "report.json", retry_missing=False)
    except FileNotFoundError:
        return None
    return validate_report(exchange, value)


def decision_path(exchange: Mapping[str, object], report: Mapping[str, object]) -> Path:
    """Locate only the response belonging to this validated epoch request."""

    validated = validate_report(exchange, dict(report))
    return Path(exchange["directory"]) / f"decision-{validated['request_token']}.json"


def report_epoch(
    exchange: Mapping[str, object], 
    step: int, 
    value: float
) -> tuple[bool, dict[str, object]]:
    """Publish one epoch and wait for its exact coordinator decision.

    The worker's independent parent-pipe watcher terminates this process if its
    coordinator exits while a decision is pending. The coordinator owns removal
    of the exchange directory only after reaping the worker.

    Args:
        exchange (Mapping[str, object]): Validated attempt/trial/monitor identity.
        step (int): Zero-based completed epoch, strictly increasing per worker.
        value (float): Finite monitored scalar after validation and EMA evaluation.

    Returns:
        tuple[bool, dict[str, object]]: Prune decision and the exact report evidence.

    Raises:
        ValueError: A report or reply is malformed or has a different identity.
        OSError: The coordinator-owned exchange is inaccessible.
    """

    exchange = validate_exchange(exchange)
    report = validate_report(exchange, {
        "protocol_version": exchange["protocol_version"], 
        "token": exchange["token"], 
        "trial_number": exchange["trial_number"], 
        "monitor": exchange["monitor"], 
        "step": step, 
        "request_token": uuid4().hex, 
        "value": value
    })
    reply_path = decision_path(exchange, report)
    write_atomic_json(Path(exchange["directory"]) / "report.json", report)
    while True:
        try:
            reply = read_atomic_json(reply_path, retry_missing=False)
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        # A response must repeat every identity and the exact finite measurement.
        if not isinstance(reply, dict) or set(reply) != {"report", "prune"} \
                or not isinstance(reply["prune"], bool) \
                or validate_report(exchange, reply["report"]) != report:
            raise ValueError("HPO pruning decision does not match its epoch report.")
        return reply["prune"], report
