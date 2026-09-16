"""Count shared live tensor payloads without importing an experimental route."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import tensorflow as tf


def tensor_inventory(groups: dict[str, Iterable[Any]]) -> dict:
    """Inventory tensor shapes/dtypes and deduplicate shared object references.

    The manifest identifies tensors by deterministic inventory order, without
    exposing process-specific object IDs. Ownership can overlap between raw
    student, teacher, optimizer, and frozen scorer; group totals therefore must
    not be added to obtain the unique total. Object identity does not detect
    distinct Tensor objects that alias allocator storage, so these declared
    payload bytes are explicitly not a physical-memory measurement.
    """
    unique, group_ids, references = {}, {}, 0
    for group, variables in groups.items():
        group_ids[group] = set()
        for value in variables:
            references += 1
            identity = id(value)
            group_ids[group].add(identity)
            # Register each live tensor once despite overlapping model or optimizer ownership.
            if identity not in unique:
                shape = tf.TensorShape(value.shape)
                elements = shape.num_elements()
                # Tensor resource inventory requires fully defined shapes.
                if elements is None:
                    raise ValueError("Tensor resource inventory requires fully defined shapes.")
                dtype = tf.as_dtype(value.dtype)
                unique[identity] = {
                    # Retaining this reference prevents id reuse when a caller
                    # provides a generator rather than a stable variable list.
                    "object": value,
                    "entry": {"index": len(unique), "name": getattr(value, "name", None),
                              "shape": shape.as_list(), "dtype": dtype.name,
                              "bytes": int(elements) * dtype.size, "groups": []},
                }
            entry = unique[identity]["entry"]
            # Record a role once even if its variable list repeats the same tensor reference.
            if group not in entry["groups"]:
                entry["groups"].append(group)
    entries = [value["entry"] for value in unique.values()]
    shared = [entry for entry in entries if len(entry["groups"]) > 1]
    return {
        "identity_rule": "same live Python tensor object counted once",
        "unique_tensor_count": len(entries),
        "unique_tensor_bytes": sum(entry["bytes"] for entry in entries),
        "tensor_reference_count": references,
        "duplicate_tensor_references": references - len(entries),
        "shared_tensor_count": len(shared),
        "shared_tensor_bytes": sum(entry["bytes"] for entry in shared),
        "groups": {
            group: {"tensor_count": len(identities),
                    "tensor_bytes": sum(unique[identity]["entry"]["bytes"] for identity in identities)}
            for group, identities in group_ids.items()
        },
        "tensors": entries,
    }
