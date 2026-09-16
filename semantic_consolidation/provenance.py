"""Record executable source identity and the numerical runtime for thesis runs."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import tensorflow as tf

from common.study_artifacts import source_files


def runtime_provenance() -> dict:
    """Record the installed numerical environment, including the active Keras API.

    Returns:
        runtime (dict): JSON-compatible dict of installed package versions, active
            Keras/backend, build details and physical device metadata.

    Raises:
        OSError: If installed distribution metadata cannot be read.
        RuntimeError: If TensorFlow runtime/device metadata cannot be queried.
    """
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in metadata.distributions() if distribution.metadata.get("Name")}
    keras_version = getattr(tf.keras, "__version__", None)
    version_function = getattr(tf.keras, "version", None)
    # Some supported Keras APIs expose the active version as a callable.
    if keras_version is None and callable(version_function):
        keras_version = version_function()
    return {
        "installed_packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "active_keras_version": keras_version,
        "keras_backend": tf.keras.backend.backend(),
        "tf_use_legacy_keras": os.environ.get("TF_USE_LEGACY_KERAS"),
        "tensorflow_build": dict(tf.sysconfig.get_build_info()),
        "tensor_float_32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "physical_devices": [
            {"name": device.name, "device_type": device.device_type,
             "details": tf.config.experimental.get_device_details(device)}
            for device in tf.config.list_physical_devices()
        ],
    }


def source_provenance() -> dict:
    """Hash local implementation files; no documents or dataset pixels are read.

    Returns:
        provenance (dict): Dict of per-source SHA-256, combined source identity, software
            versions and numerical runtime metadata.

    Raises:
        OSError: If a production source or installed package metadata file cannot be read.
    """

    files = source_files()
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {
        "source_sha256": combined, "files": files,
        "python": sys.version, "tensorflow": tf.__version__, "numpy": np.__version__,
        "platform": platform.platform(),
        "visible_devices": [device.name for device in tf.config.get_visible_devices()],
        "runtime": runtime_provenance(),
        "interpretation": "Source identity and runtime metadata; not an independently authenticated archive.",
    }


def save_provenance(value: dict, directory: str | Path) -> None:
    """Save the source digests and numerical runtime metadata beside run artifacts.

    Args:
        value (dict): Source/runtime provenance dict returned by source_provenance,
            containing only JSON-compatible values.
        directory (str | Path): Output directory for this operation, resolved using ordinary
            pathlib path semantics.

    Returns:
        saved (None): None; writes source_provenance.json in the existing run directory.

    Raises:
        OSError: If the provenance destination cannot be written.
        TypeError: If value contains a non-JSON-serializable object.
    """
    with (Path(directory) / "source_provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
