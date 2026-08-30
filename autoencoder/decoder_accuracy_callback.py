"""Epoch callback for measuring the class fidelity of VAE generations."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import callbacks

from collections.abc import Callable
from numbers import Integral

from common.runtime import derive_seed


class DecoderAccuracyCallback(callbacks.Callback):
    """Add conditional generator classification accuracy to epoch logs.

    Keras sets ``model`` when training begins.  That model must implement
    ``generate(samples_per_class=..., onehot_y_output=False)`` and return
    generated vectors plus integer class IDs, as
    :class:`VariationalAutoencoder` does in conditional mode.

    Attributes:
        samples_per_class (int): Generated examples requested for each class;
            initialized from the constructor value.
        classifier (tf.keras.Model | Callable): Maps generated vectors to class
            scores; initialized from the constructor value.
    """

    def __init__(
        self: DecoderAccuracyCallback, 
        classifier: tf.keras.Model | Callable[..., tf.Tensor], 
        samples_per_class: int = 500,
        seed: int | None = None,
    ) -> None:
        """Initialize generation count and evaluation classifier.

        Args:
            classifier (tf.keras.Model | Callable): Callable accepting
                ``(x_gen, training=False)`` and returning scores shaped
                ``[samples, classes]``.
            samples_per_class (int): Non-boolean positive number of generations
                requested for each class previously seen by the attached VAE.
            seed (int | None): Optional experiment seed. A stable epoch-specific
                child seed is passed to the VAE so callback sampling is
                reproducible without repeating the same latent batch every
                epoch.

        Returns:
            None.

        Raises:
            TypeError: If the classifier is not callable or the sample count is
                not a non-boolean integer.
            ValueError: If the sample count is not positive.
        """

        super().__init__()

        # Require a callable classifier for generated-sample evaluation.
        if not callable(classifier):
            raise TypeError("classifier must be callable.")
        # Reject booleans and non-integral generation counts.
        if isinstance(samples_per_class, bool) \
        or not isinstance(samples_per_class, Integral):
            raise TypeError("samples_per_class must be a non-boolean integer.")
        # Require at least one generated sample per class.
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive.")

        self.samples_per_class = int(samples_per_class)
        self.classifier = classifier
        # Validate once and retain the master seed for deterministic epoch
        # streams. ``derive_seed`` also normalizes NumPy integral values.
        derive_seed(seed, "decoder_accuracy", 0)
        self.seed = None if seed is None else int(seed)

    def on_epoch_end(
        self: DecoderAccuracyCallback, 
        epoch: int, 
        logs: dict[str, object] | None = None
    ) -> None:
        """Generate examples, classify them, and log exact-match accuracy.

        Args:
            epoch (int): Zero-based completed epoch index used to derive an
                independent callback sampling stream.
            logs (dict[str, object] | None): Epoch log mapping. Any supplied
                dictionary receives ``"decoder_accuracy"`` as a NumPy scalar;
                ``None`` is replaced locally.

        Returns:
            None.

        Raises:
            ValueError: If the attached model has no seen conditional classes
                or generated/classifier shapes are incompatible.
        """

        # Create a log mapping when Keras supplies no mapping.
        if logs is None:
            logs = {}

        x_gen, y_true = self.model.generate(
            samples_per_class=self.samples_per_class, 
            onehot_y_output=False,
            seed=derive_seed(self.seed, "decoder_accuracy", int(epoch)),
        )

        # Avoid reporting an undefined accuracy for an empty generation.
        if len(y_true) == 0:
            raise ValueError(
                "Decoder accuracy requires at least one generated sample."
            )

        # Forward inference state to Keras classifiers.
        if isinstance(self.classifier, tf.keras.layers.Layer):
            y_pred = self.classifier(x_gen, training=False)
        # Support generic callables that accept only the generated samples.
        else:
            y_pred = self.classifier(x_gen)
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)
        tf.debugging.assert_equal(
            tf.shape(y_pred), 
            tf.shape(y_true), 
            message="Generated labels and classifier predictions must align."
        )

        classifier_policy = getattr(self.classifier, "dtype_policy", None)
        model_policy = getattr(self.model, "dtype_policy", None)
        stable_dtype = getattr(
            classifier_policy,
            "variable_dtype",
            getattr(
                model_policy,
                "variable_dtype",
                tf.keras.mixed_precision.global_policy().variable_dtype,
            ),
        )
        corrects = tf.cast(y_pred == y_true, dtype=stable_dtype)
        acc = tf.reduce_mean(corrects)

        logs["decoder_accuracy"] = acc.numpy()


def run_self_tests() -> dict[str, str]:
    """Run generation, classification, logging, and boundary callback tests.

    Tests validate constructor defaults, forwarding of generation arguments,
    exact and partial accuracy, classifier inference mode, nonempty/empty/None
    log mappings, empty generated batches, incompatible shapes, and callbacks
    that have not yet been attached to a model.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"DecoderAccuracyCallback": "passed"}`` after all
        assertions succeed.
    """

    from types import SimpleNamespace


    calls = {"generate": [], "training": []}


    def generate(
        samples_per_class: int, 
        onehot_y_output: bool,
        seed: int | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return deterministic class-coded samples for callback testing.

        Args:
            samples_per_class (int): Requested examples per class.
            onehot_y_output (bool): Requested label encoding flag.
            seed (int | None): Epoch-derived callback seed.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Class-coded features and integer IDs.
        """

        calls["generate"].append(
            (samples_per_class, onehot_y_output, seed)
        )
        labels = tf.repeat(
            tf.constant([0, 1], tf.int64), 
            samples_per_class
        )

        return tf.cast(labels[:, None], tf.float32), labels


    def perfect_classifier(
        inputs: tf.Tensor, 
        training: bool = False
    ) -> tf.Tensor:
        """Classify a class ID stored in the first input column.

        Args:
            inputs (tf.Tensor): Class-coded features shaped ``[batch, 1]``.
            training (bool): Inference flag supplied by the callback.

        Returns:
            tf.Tensor: Two-class one-hot scores.
        """

        calls["training"].append(training)

        return tf.one_hot(
            tf.cast(inputs[:, 0], tf.int32), depth=2
        )


    default_callback = DecoderAccuracyCallback(perfect_classifier)
    assert default_callback.samples_per_class == 500
    assert default_callback.classifier is perfect_classifier

    callback = DecoderAccuracyCallback(
        classifier=perfect_classifier, 
        samples_per_class=2, 
        seed=17,
    )
    callback.set_model(SimpleNamespace(generate=generate))
    logs = {"loss": 0.5}
    assert callback.on_epoch_end(3, logs) is None
    assert logs["loss"] == 0.5
    assert float(logs["decoder_accuracy"]) == 1.0
    assert calls["generate"] == [(
        2,
        False,
        derive_seed(17, "decoder_accuracy", 3),
    )]
    assert calls["training"] == [False]


    def half_correct_classifier(
        inputs: tf.Tensor, 
        training: bool = False
    ) -> tf.Tensor:
        """Return predictions that are correct for only class zero.

        Args:
            inputs (tf.Tensor): Class-coded features shaped ``[batch, 1]``.
            training (bool): Inference flag supplied by the callback.

        Returns:
            tf.Tensor: Scores always selecting class zero.
        """

        del training

        return tf.one_hot(
            tf.zeros(tf.shape(inputs)[0], tf.int32), 
            depth=2
        )


    partial_callback = DecoderAccuracyCallback(half_correct_classifier, 3)
    partial_callback.set_model(SimpleNamespace(generate=generate))
    partial_logs = {"existing": 1}
    partial_callback.on_epoch_end(0, partial_logs)
    assert float(partial_logs["decoder_accuracy"]) == 0.5


    def plain_classifier(inputs: tf.Tensor) -> tf.Tensor:
        """Classify samples without accepting a Keras training argument.

        Args:
            inputs (tf.Tensor): Class-coded features shaped ``[batch, 1]``.

        Returns:
            tf.Tensor: Matching two-class one-hot scores.
        """

        return tf.one_hot(
            tf.cast(inputs[:, 0], tf.int32), depth=2
        )


    plain_callback = DecoderAccuracyCallback(plain_classifier, 1)
    plain_callback.set_model(SimpleNamespace(generate=generate))
    plain_logs = {}
    plain_callback.on_epoch_end(0, plain_logs)
    assert float(plain_logs["decoder_accuracy"]) == 1.0

    empty_logs = {}
    callback.on_epoch_end(4, empty_logs)
    assert float(empty_logs["decoder_accuracy"]) == 1.0
    assert callback.on_epoch_end(5, None) is None


    def generate_empty(
        samples_per_class: int, 
        onehot_y_output: bool,
        seed: int | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return a correctly shaped empty generated batch.

        Args:
            samples_per_class (int): Requested positive count; unused.
            onehot_y_output (bool): Requested label encoding flag.
            seed (int | None): Optional callback seed; unused.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Empty features and labels.
        """

        assert samples_per_class == 1 and onehot_y_output is False
        assert seed is None


        return (tf.zeros((0, 1), tf.float32), 
            tf.zeros((0,), tf.int64))


    try:
        DecoderAccuracyCallback(perfect_classifier, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("A zero generation count must fail at construction.")

    empty_callback = DecoderAccuracyCallback(perfect_classifier, 1)
    empty_callback.set_model(SimpleNamespace(generate=generate_empty))
    try:
        empty_callback.on_epoch_end(0, {"sentinel": True})
    except ValueError:
        pass
    else:
        raise AssertionError("An empty generated batch must fail clearly.")

    for invalid_count in (True, 1.5):
        try:
            DecoderAccuracyCallback(perfect_classifier, invalid_count)
        except TypeError:
            pass
        else:
            raise AssertionError("Callback sample counts must be integers.")


    def generate_bad_labels(
        samples_per_class: int, 
        onehot_y_output: bool,
        seed: int | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return intentionally incompatible prediction and label lengths.

        Args:
            samples_per_class (int): Requested count; unused.
            onehot_y_output (bool): Requested label encoding flag; unused.
            seed (int | None): Optional callback seed; unused.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Two samples and three labels.
        """

        del samples_per_class, onehot_y_output, seed

        return (tf.zeros((2, 1), tf.float32), 
            tf.zeros((3,), tf.int64))


    invalid_callback = DecoderAccuracyCallback(perfect_classifier, 1)
    invalid_callback.set_model(SimpleNamespace(generate=generate_bad_labels))
    incompatible_logs = {"sentinel": True}
    try:
        invalid_callback.on_epoch_end(0, incompatible_logs)
    except tf.errors.InvalidArgumentError:
        assert incompatible_logs == {"sentinel": True}, (
            "A shape failure must occur before the callback mutates logs."
        )
    else:
        raise AssertionError("Incompatible label and prediction shapes must fail.")

    unattached = DecoderAccuracyCallback(perfect_classifier, 1)
    try:
        unattached.on_epoch_end(0, {"sentinel": True})
    except AttributeError:
        pass
    else:
        raise AssertionError("An unattached callback must not generate data.")

    return {"DecoderAccuracyCallback": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
