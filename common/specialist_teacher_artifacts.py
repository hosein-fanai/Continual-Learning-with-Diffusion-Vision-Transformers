"""Bind serializable specialist inputs to the existing model-loading APIs.

Descriptor normalization reads configuration and artifact bytes without importing
TensorFlow. The worker verifies those same identities before constructing teachers.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from common.config import Config, load_config, normalize_training_task


def _file_identity(path: object, expected: object = None) -> tuple[str, str]:
    """Resolve one model artifact and reject changes to an already sealed digest."""

    selected = Path(path).expanduser().resolve()
    # Descriptor identities cover one complete artifact, not an inferred directory.
    if not selected.is_file():
        raise FileNotFoundError(f"Specialist teacher artifact is not a file: {selected}")
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    # A resumed study must consume the coordinator's exact artifact bytes.
    if expected is not None and expected != actual:
        raise ValueError(f"Specialist teacher artifact identity changed: {selected}")
    return str(selected), actual


def specialist_teacher_config(descriptor: Mapping[str, object]) -> Config:
    """Read a native descriptor's detached inline Config or its original YAML.

    Identity verification belongs to descriptor normalization. This accessor
    supports inspecting an already normalized recipe without model imports.
    """

    # Keras archives require their model loader rather than native Config parsing.
    if descriptor.get("format") != "config":
        raise ValueError("A native specialist Config requires format='config'.")
    # A single source keeps configuration identity unambiguous during transport.
    if ("config" in descriptor) == ("path" in descriptor):
        raise ValueError("Supply exactly one specialist Config mapping or YAML path.")
    # Detach inline recipes so factory preparation cannot mutate the sealed input.
    if "config" in descriptor:
        # Only mappings can represent native Config constructor arguments.
        if not isinstance(descriptor["config"], Mapping):
            raise TypeError("An inline specialist config must be a mapping.")
        return Config(**deepcopy(dict(descriptor["config"])))
    return load_config(descriptor["path"])


def _validate_native_specialist(config: Config, role: str) -> None:
    """Restrict artifact roles to native owners with the required output contract."""

    task = normalize_training_task(config.training.task)
    joint = task == "joint" and config.model.name == "dit_classifier" \
        and config.model.wrapper_name in (None, "diffusion_classifier")
    generation = task == "generation" and config.model.name in ("unet", "diffusion_transformer") \
        and config.model.wrapper_name in (None, "diffusion_model")
    # Every role must expose its required class-probability or epsilon output.
    if not joint and not (role == "noise" and generation):
        raise ValueError(
            "Classifier specialists require a joint/dit_classifier V1 Config; "
            "noise specialists also accept generation/unet or generation/diffusion_transformer Configs."
        )
    # Recursive loaders would escape this finite authenticated artifact declaration.
    if config.continually_learn.specialist_teacher_descriptors:
        raise ValueError("A specialist Config cannot recursively load specialist teachers.")
    # The native wrapper owns its classifier head and independent training recipe.
    if config.model.classifier_name is not None:
        raise ValueError("A native specialist must own its attached head, not a separate classifier.")
    wrapper = config.model.wrapper_kwargs or (
        config.model.diffusion_classifier.kwargs() if joint else config.model.diffusion_model.kwargs()
    )
    # A serialized specialist cannot hide another runtime teacher dependency.
    if any(key.endswith("teacher_network") and value is not None for key, value in wrapper.items()):
        raise ValueError("A specialist Config cannot contain live or recursive teachers.")


def normalize_specialist_teacher_descriptors(
    descriptors: Mapping[str, object] | None
) -> dict[str, dict[str, object]]:
    """Seal compiled classifiers and native Config inputs without loading models.

    ``classifier`` accepts ``format='keras'`` with a compiled ``path``, or a
    native joint DiT V1 Config. ``noise`` accepts native UNet/DiT generation or
    joint DiT V1 Configs. ``format='config'`` takes exactly one YAML ``path`` or
    inline ``config`` mapping. SHA256 seals YAML bytes or canonical resolved
    inline Config JSON respectively. Native optional model.weights_path is
    resolved relative to YAML or, for inline recipes, the current directory;
    descriptor weights_path can supply it instead. Single-file weight artifacts
    are sealed separately. Re-normalization verifies both identities.

    Teachers are initialization artifacts, not an implicit original-label remapping.
    Use a fresh target Dense head above a pretrained image backbone. Native UNet
    ``num_classes=None`` enables incremental vocabulary growth; a fixed full-width
    native head retains dense schedule-ID conditions. Previously trained target
    heads must already agree with that vocabulary; no output weights are relabeled.
    """

    # Omitting specialists retains the existing previous-snapshot-only API.
    if descriptors is None:
        return {}
    # The role mapping must survive ordinary YAML and subprocess transport.
    if not isinstance(descriptors, Mapping):
        raise TypeError("specialist_teacher_descriptors must be a mapping.")
    # The previous role is generated by the continual learner, not an artifact slot.
    if set(descriptors) - {"classifier", "noise"}:
        raise ValueError("Specialist teacher roles must be classifier or noise.")
    normalized = {}
    for role, descriptor in descriptors.items():
        # Each independent role owns a complete serializable artifact declaration.
        if not isinstance(descriptor, Mapping):
            raise TypeError(f"The {role} specialist descriptor must be a mapping.")
        selected_format = descriptor.get("format")
        # Only classifier roles support ordinary compiled Keras archives.
        if selected_format not in ("keras", "config") or (role == "noise" and selected_format != "config"):
            raise ValueError("Classifier specialists require keras/config; noise specialists require config.")
        allowed = {"format", "path", "sha256"}
        # Native Configs may pair their architecture with a separate weights file.
        if selected_format == "config":
            allowed.update({"config", "weights_path", "weights_sha256"})
        # Reject ignored metadata that might otherwise appear to affect a teacher.
        if set(descriptor) - allowed:
            raise ValueError(f"Unknown {role} specialist descriptor fields: {sorted(set(descriptor) - allowed)}")
        # Native recipes are verified before constructing any framework objects.
        if selected_format == "config":
            # Authenticate sealed bytes before parsing a possibly replaced YAML file.
            if "path" in descriptor and "config" not in descriptor:
                path, digest = _file_identity(descriptor["path"], descriptor.get("sha256"))
            config = specialist_teacher_config(descriptor)
            _validate_native_specialist(config, role)
            # Inline recipes bind resolved configuration rather than external file bytes.
            if "config" in descriptor:
                # JSON round trips normalize integer mapping keys and tuple containers.
                resolved = json.loads(json.dumps(asdict(config), allow_nan=False))
                canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False)
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                # A resumed recipe must retain the coordinator's sealed settings.
                if descriptor.get("sha256") is not None and descriptor["sha256"] != digest:
                    raise ValueError("Inline specialist Config identity changed.")
                item = {"format": "config", "config": resolved, "sha256": digest}
                source_directory = Path.cwd()
            # YAML descriptors retain their authenticated path and relative weight base.
            else:
                item = {"format": "config", "path": path, "sha256": digest}
                source_directory = Path(path).parent
            weights = descriptor.get("weights_path", config.model.weights_path)
            # A supplied weight file joins the immutable teacher identity.
            if weights is not None:
                weights = Path(weights).expanduser()
                # Inline weights resolve once; sealed descriptors retain absolute paths.
                if not weights.is_absolute():
                    weights = source_directory / weights
                weights_path, weights_digest = _file_identity(weights, descriptor.get("weights_sha256"))
            # An architecture-only teacher is initialized through the native factory.
            else:
                # A digest cannot authenticate an absent artifact.
                if descriptor.get("weights_sha256") is not None:
                    raise ValueError("weights_sha256 requires a weights_path.")
                weights_path = weights_digest = None
            item.update(weights_path=weights_path, weights_sha256=weights_digest)
        # A Keras archive carries an ordinary classifier graph and its compilation.
        else:
            path, digest = _file_identity(descriptor["path"], descriptor.get("sha256"))
            # The Keras loader expects an archive containing the compiled classifier.
            if Path(path).suffix != ".keras":
                raise ValueError("Classifier specialists require a compiled .keras model artifact.")
            item = {"format": "keras", "path": path, "sha256": digest}
        normalized[role] = item
    return normalized


def load_specialist_teacher_descriptors(
    descriptors: Mapping[str, object] | None, 
    seed: int | None = None
) -> dict[str, object]:
    """Verify sealed files and load independent specialists through existing APIs.

    Keras restores the classifier's compile settings and optimizer. Native
    Config loading preserves its own diffusion schedule, preprocessing, optimizer
    and optional initial weights; the student wrapper checks compatibility before
    distillation. An optional study seed gives each role an independent construction
    stream, so enabling another teacher does not change its initial random weights.
    This function does not train either teacher.
    """

    normalized = normalize_specialist_teacher_descriptors(descriptors)
    # Previous-only and no-KD trials allocate no specialist models.
    if not normalized:
        return {}

    import tensorflow as tf

    from common.model import get_model
    from common.runtime import configure_runtime, derive_seed


    teachers = {}
    for role, descriptor in normalized.items():
        # Each role has a reproducible draw independent of other attached roles.
        if seed is not None:
            configure_runtime(
                dtype_policy=tf.keras.mixed_precision.global_policy().name, 
                seed=derive_seed(seed, "specialist_teacher", role)
            )
        # Keras restores the ordinary image classifier's independent optimizer.
        if descriptor["format"] == "keras":
            teacher = tf.keras.models.load_model(descriptor["path"], compile=True, safe_mode=True)
            # Existing image-teacher training requires the caller's compile recipe.
            if not teacher.compiled or getattr(teacher, "optimizer", None) is None:
                raise ValueError("The classifier specialist must be saved with its own compile settings and optimizer.")
        # Native Config loading preserves the denoiser's own diffusion semantics.
        else:
            config = specialist_teacher_config(descriptor)
            config.model.weights_path = descriptor["weights_path"]
            config.model.show_network_summary = False
            teacher = get_model(config)
        teachers[role] = teacher
    return teachers
