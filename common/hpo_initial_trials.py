"""Immutable, retry-safe partial suggestions for a new Optuna study."""

from __future__ import annotations

import hashlib
import json
import math

from collections.abc import Mapping, Sequence
from typing import Any


def normalize_initial_trials(
    initial_trials: Sequence[Mapping[str, object]] | None
) -> list[dict[str, object]] | None:
    """Copy JSON-scalar parameter hints without carrying scores or training state.

    Args:
        initial_trials: Ordered partial parameter dictionaries, or None to retain
            the existing unseeded study contract.

    Returns:
        A detached list of parameter dictionaries, or None.

    Raises:
        TypeError: The outer sequence, mappings, names or values are unsupported.
        ValueError: A floating-point parameter is nonfinite.
    """

    # Omission preserves legacy study identity and behavior.
    if initial_trials is None:
        return None
    # Strings and mappings cannot stand in for an ordered list of trials.
    if not isinstance(initial_trials, Sequence) or isinstance(initial_trials, (str, bytes)):
        raise TypeError("initial_trials must be a sequence of parameter mappings.")
    result = []
    for point in initial_trials:
        # Partial trial suggestions contain named scalar parameters only.
        if not isinstance(point, Mapping):
            raise TypeError("Each initial trial must be a parameter mapping.")
        copied = {}
        for parameter, value in point.items():
            # Names follow the same nonempty string protocol as Optuna suggestions.
            if not isinstance(parameter, str) or not parameter:
                raise TypeError("Initial-trial parameter names must be nonempty strings.")
            # Nested metadata or foreign numeric scalar types are not parameters.
            if value is not None and type(value) not in (str, bool, int, float):
                raise TypeError("Initial-trial values must be JSON scalars.")
            # Nonfinite numbers cannot be sealed reproducibly in a JSON recipe.
            if type(value) is float and not math.isfinite(value):
                raise ValueError("Initial-trial values must be finite.")
            copied[parameter] = value
        result.append(copied)
    return result


def initial_trials_digest(initial_trials: Sequence[Mapping[str, object]]) -> str:
    """Hash the exact ordered list of partial suggestions.

    Args:
        initial_trials: JSON-scalar parameter dictionaries in launch order.

    Returns:
        A SHA-256 digest with mapping keys sorted and trial order preserved.

    Raises:
        TypeError: The values cannot be normalized or serialized.
        ValueError: A floating-point value is nonfinite.
    """

    normalized = normalize_initial_trials(initial_trials)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enqueue_initial_trials(
    study: Any, 
    initial_trials: Sequence[Mapping[str, object]], 
    max_new_trials: int | None = None
) -> int:
    """Queue each partial hint once, preserving the caller's total attempt budget.

    Call inside the study coordinator lock after identity validation and sampler
    state persistence. Atomic enqueue markers survive a crash before this helper
    returns. Recovered retries may retain the same marker without duplicating
    the original warm-start request. No objective value is imported.

    Args:
        study: Optuna Study owned by the current coordinator.
        initial_trials: Immutable ordered partial parameter hints.
        max_new_trials: Remaining allocation allowance, or None for all hints.

    Returns:
        Number of newly queued trials.

    Raises:
        ValueError: Existing markers disagree with the sealed hint list.
        TypeError: Hints contain unsupported structures or values.
    """

    points = normalize_initial_trials(initial_trials)
    digest = initial_trials_digest(points)
    seen = set()
    for trial in study.get_trials(deepcopy=False):
        marker = trial.user_attrs.get("initial_trial")
        # Ordinary trials have no warm-start provenance.
        if marker is None:
            continue
        # Markers identify the exact ordered source of a queued suggestion.
        if not isinstance(marker, dict) or marker.get("sha256") != digest:
            raise ValueError("Initial-trial provenance differs from this study.")
        index = marker.get("index")
        # Reject altered marker identities rather than silently skipping hints.
        if type(index) is not int or not 0 <= index < len(points):
            raise ValueError("Initial-trial provenance contains an invalid index.")
        seen.add(index)
    queued = 0
    for index, point in enumerate(points):
        # A previous enqueue or its recovery retry already represents this hint.
        if index in seen:
            continue
        # Total-budget callers cannot allocate beyond their remaining allowance.
        if max_new_trials is not None and queued >= max_new_trials:
            break
        study.enqueue_trial(
            point, user_attrs={"initial_trial": {"sha256": digest, "index": index}}
        )
        queued += 1
    return queued
