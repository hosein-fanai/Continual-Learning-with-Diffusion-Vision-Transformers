"""Temporary affine control and class-balanced sampling for route one.

Only learned vectors persist across increments. Image arrays belong to the
current increment's existing current/replay pool and are released afterwards.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from common.runtime import derive_seed


class ModulationBank:
    """Keep independent class variables so updating new classes cannot move old ones."""

    def __init__(self, settings: object, dimension: int, seed: int) -> None:
        """Set the fixed feature width and initialization stream for independent class gates.

        Args:
            settings (object): Validated settings instance for this component; its fields select
                the behavior described above.
            dimension (int): Positive integer width D of the existing classifier hidden
                projection.
            seed (int): Explicit integer random seed; local or derived streams preserve
                reproducibility without reseeding caller-owned generators.

        Returns:
            initialized (None): None; starts an empty bank with fixed projection width and seed.

        Raises:
            None: Width and settings consistency are checked by the route controller before use.
        """
        self.settings = settings
        self.dimension = dimension
        self.seed = seed
        self.vectors: dict[int, tuple[tf.Variable, tf.Variable]] = {}

    def add(self, class_ids: list[int]) -> None:
        """Initialize only missing classes with a reproducible, task-independent seed.

        Args:
            class_ids (list[int]): Integer class IDs whose missing independent gain and bias
                vectors should be initialized.

        Returns:
            added (None): None; only missing classes receive independent trainable float32
                gain/bias vectors [D].

        Raises:
            ValueError: If width, initialization scale or seed cannot define the requested
                variables.
        """

        for class_id in class_ids:
            # Adding an already-seen class must preserve its learned gate variables.
            if class_id in self.vectors:
                continue
            rng = np.random.default_rng(derive_seed(self.seed, "modulator", class_id))
            values = rng.normal(
                0., self.settings.modulation_init_std, (2, self.dimension)
            ).astype("float32")
            self.vectors[class_id] = tuple(
                tf.Variable(value, name=f"modulation_{class_id}_{kind}")
                for kind, value in zip(("gain", "bias"), values)
            )

    def apply(self, features: tf.Tensor, class_id: int) -> tf.Tensor:
        """Apply one selected class's control to EVERY row, including negatives.

        Args:
            features (tf.Tensor): Finite numeric matrix of shape [N, D]; objective and
                classifier paths compute in float32.
            class_id (int): Nonnegative integer class ID selecting one gate or one class cohort.

        Returns:
            modulated (tf.Tensor): Float32 matrix [N, D] after applying one class gate to every
                row; normalization is performed by the objective.

        Raises:
            KeyError: If class_id has no initialized gate.
            ValueError: If feature and gate shapes are incompatible.
        """

        gain, bias = self.vectors[int(class_id)]
        return affine_modulation(features, gain, bias, self.settings)

    def frozen(self) -> dict[int, tuple[tf.Tensor, tf.Tensor]]:
        """Copy values into detached constants, without sharing mutable variables.

        Returns:
            snapshot (dict[int, tuple[tf.Tensor, tf.Tensor]]): Dict from integer class ID to
                detached float32 gain/bias tensors [D], with no mutable variable alias.

        Raises:
            AttributeError: If stored variables do not expose eager numpy values.
        """

        return {
            class_id: tuple(tf.constant(value.numpy()) for value in values)
            for class_id, values in self.vectors.items()
        }

    @property
    def nbytes(self) -> int:
        """Persistent raw gain/bias storage, excluding allocator/Python overhead.

        Returns:
            nbytes (int): Integer raw gain/bias payload bytes across all currently retained
                classes.

        Raises:
            AttributeError: If stored gate variables cannot expose eager values.
        """

        return sum(value.numpy().nbytes for pair in self.vectors.values() for value in pair)


def affine_modulation(
    features: tf.Tensor, gain: tf.Tensor, bias: tf.Tensor, settings: object
) -> tf.Tensor:
    """Bound channel gain around one and bias around zero before normalization.

    Args:
        features (tf.Tensor): Float-compatible feature matrix [N, D], cast to tf.float32
            before the affine transformation.
        gain (tf.Tensor): Float32 channel vector [D] of unconstrained gain parameters,
            bounded through tanh.
        bias (tf.Tensor): Float32 channel vector [D] of unconstrained bias parameters,
            bounded through tanh.
        settings (object): Validated settings instance for this component; its fields select
            the behavior described above.

    Returns:
        modulated (tf.Tensor): Float32 feature matrix [N, D]; each channel gain is
            1+gain_limit*tanh(gain), with a bounded additive bias.

    Raises:
        ValueError: If features/gates have incompatible shapes.
        tf.errors.InvalidArgumentError: If TensorFlow cannot broadcast the channel vectors.
    """

    features = tf.cast(features, tf.float32)
    return features * (1. + settings.gain_limit * tf.tanh(gain)) + (
        settings.bias_limit * tf.tanh(bias)
    )


class ClassBalancedPool:
    """Select distinct row indices: half focus-class positives, balanced negatives.

    Negative classes receive round-robin slots in a seeded random class order;
    exhausted classes are removed. Within each class, rows are shuffled. Batches
    shrink when a class has fewer than batch_size/2 positive rows. They require
    at least two positives and one negative. Identical pixels at different pool
    indices remain distinct observations; no image deduplication is performed.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Validate aligned observations and index the represented classes for pair sampling.

        Args:
            images (np.ndarray): Finite numeric array [N, ...] with at least one feature axis;
                copied/cast to float32.
            labels (np.ndarray): One-dimensional nonnegative integer vector [N], representable
                in int32; at least two classes are required.

        Returns:
            initialized (None): None; stores float32 images, int32 labels and per-class integer
                index arrays.

        Raises:
            ValueError: If labels, image rank, finiteness, row alignment, minimum rows or
                represented classes are invalid.
        """
        self.images = np.asarray(images, dtype="float32")
        labels = np.asarray(labels)
        # Phase labels must be nonnegative sparse integer IDs representable as int32.
        if labels.ndim != 1 or labels.dtype.kind not in "iu" or np.any(labels < 0) \
        or np.any(labels > np.iinfo(np.int32).max):
            raise ValueError("Phase labels must be nonnegative sparse integer IDs representable as int32.")
        self.labels = labels.astype("int32", copy=False)
        # Phase images must have a sample axis and feature dimensions.
        if self.images.ndim < 2:
            raise ValueError("Phase images must have a sample axis and feature dimensions.")
        # A modulation pool needs at least three aligned rows.
        if len(self.images) != len(self.labels) or len(self.images) < 3:
            raise ValueError("A modulation pool needs at least three aligned rows.")
        # The phase pool contains nonfinite pixels.
        if not np.isfinite(self.images).all():
            raise ValueError("The phase pool contains nonfinite pixels.")
        self.indices = {
            int(label): np.flatnonzero(self.labels == label)
            for label in np.unique(self.labels)
        }
        # Modulation requires at least two represented classes.
        if len(self.indices) < 2:
            raise ValueError("Modulation requires at least two represented classes.")

    def draw(
        self, focus: int, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pixels, sparse labels and positive mask for one shared modulator.

        Args:
            focus (int): Integer selected-class ID; its observations define positives and all
                other classes define negatives.
            batch_size (int): Positive integer maximum number of rows per batch; any additional
                phase-specific minimum is described above.
            rng (np.random.Generator): Caller-owned numpy.random.Generator used to sample and
                shuffle row indices.

        Returns:
            batch (tuple[np.ndarray, np.ndarray, np.ndarray]): (images, labels, positive):
                float32 sample array, int32 sparse vector and bool selected-class mask; indices
                are distinct within the batch.

        Raises:
            ValueError: If batch_size is not an integer at least four, focus is absent/invalid
                or its class lacks two positive rows.
        """

        # A phase batch_size must be an integer >= 4.
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 4:
            raise ValueError("A phase batch_size must be an integer >= 4.")
        # The focus class must be an integer represented in the phase pool.
        if isinstance(focus, bool) or not isinstance(focus, (int, np.integer)) or focus not in self.indices:
            raise ValueError("The focus class must be an integer represented in the phase pool.")
        positive = self.indices[focus]
        count = min(batch_size // 2, len(positive))
        # A focus class needs two distinct positive occurrences to define attraction.
        if count < 2:
            raise ValueError(f"Class {focus} needs two positive rows; increase exposure.")
        selected = list(rng.choice(positive, size=count, replace=False))
        negative_classes = rng.permutation([c for c in self.indices if c != focus])
        available = {int(c): list(rng.permutation(self.indices[c])) for c in negative_classes}
        negatives = []
        while len(negatives) < count and available:
            for class_id in list(available):
                negatives.append(available[class_id].pop())
                # Remove exhausted negative classes instead of sampling repeated indices.
                if not available[class_id]:
                    del available[class_id]
                # Stop once negative slots match the selected positive count.
                if len(negatives) == count:
                    break
        selected = np.asarray(selected + negatives, dtype="int64")
        rng.shuffle(selected)
        labels = self.labels[selected]
        return self.images[selected], labels, labels == focus
