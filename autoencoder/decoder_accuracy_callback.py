"""Epoch callback for measuring the class fidelity of VAE generations."""

import tensorflow as tf
from tensorflow.keras import callbacks


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
        self, 
        classifier, 
        samples_per_class=500, 
    ):
        """Initialize generation count and evaluation classifier.

        Args:
            classifier (tf.keras.Model | Callable): Callable accepting
                ``(x_gen, training=False)`` and returning scores shaped
                ``[samples, classes]``.
            samples_per_class (int): Nonnegative generations requested for each
                class previously seen by the attached conditional VAE.

        Returns:
            None.
        """

        super().__init__()
        self.samples_per_class = samples_per_class
        self.classifier = classifier

    def on_epoch_end(self, epoch, logs=None):
        """Generate examples, classify them, and log exact-match accuracy.

        Args:
            epoch (int): Zero-based completed epoch index; unused otherwise.
            logs (dict[str, object] | None): Epoch log mapping.  A nonempty
                supplied dictionary receives ``"decoder_accuracy"`` as a NumPy
                scalar; ``None`` (or an empty dictionary) is replaced locally.

        Returns:
            None.

        Raises:
            ValueError: If the attached model has no seen conditional classes
                or generated/classifier shapes are incompatible.
        """

        logs = logs or {}

        x_gen, y_true = self.model.generate(
            samples_per_class=self.samples_per_class, 
            onehot_y_output=False
        )

        y_pred = self.classifier(x_gen, training=False)
        y_pred = tf.argmax(y_pred, axis=1)

        corrects = tf.cast(y_pred == y_true, dtype=tf.float16)
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
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return deterministic class-coded samples for callback testing.

        Args:
            samples_per_class (int): Requested examples per class.
            onehot_y_output (bool): Requested label encoding flag.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Class-coded features and integer IDs.
        """

        calls["generate"].append(
            (samples_per_class, onehot_y_output)
        )
        labels = tf.repeat(
            tf.constant([0, 1], tf.int64), 
            samples_per_class
        )

        return tf.cast(labels[:, None], tf.float32), labels


    def perfect_classifier(
        inputs: tf.Tensor, 
        training: bool = False, 
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
    )
    callback.set_model(SimpleNamespace(generate=generate))
    logs = {"loss": 0.5}
    assert callback.on_epoch_end(3, logs) is None
    assert logs["loss"] == 0.5
    assert float(logs["decoder_accuracy"]) == 1.0
    assert calls["generate"] == [(2, False)]
    assert calls["training"] == [False]


    def half_correct_classifier(
        inputs: tf.Tensor, 
        training: bool = False, 
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

    empty_logs = {}
    callback.on_epoch_end(4, empty_logs)
    assert empty_logs == {}, (
        "The current `logs or {}` implementation replaces an empty mapping "
        "instead of mutating it."
    )
    assert callback.on_epoch_end(5, None) is None


    def generate_empty(
        samples_per_class: int, 
        onehot_y_output: bool, 
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return a correctly shaped empty generated batch.

        Args:
            samples_per_class (int): Requested count; expected to be zero.
            onehot_y_output (bool): Requested label encoding flag.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Empty features and labels.
        """

        assert samples_per_class == 0 and onehot_y_output is False


        return (tf.zeros((0, 1), tf.float32), 
            tf.zeros((0,), tf.int64))


    zero_callback = DecoderAccuracyCallback(perfect_classifier, 0)
    zero_callback.set_model(SimpleNamespace(generate=generate_empty))
    zero_logs = {"sentinel": True}
    zero_callback.on_epoch_end(0, zero_logs)
    assert bool(tf.math.is_nan(zero_logs["decoder_accuracy"]))


    def generate_bad_labels(
        samples_per_class: int,
        onehot_y_output: bool,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return intentionally incompatible prediction and label lengths.

        Args:
            samples_per_class (int): Requested count; unused.
            onehot_y_output (bool): Requested label encoding flag; unused.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Two samples and three labels.
        """

        del samples_per_class, onehot_y_output

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
        assert float(incompatible_logs["decoder_accuracy"]) == 0.0, (
            "Depending on TensorFlow's device execution path, incompatible "
            "one-dimensional lengths either raise InvalidArgumentError or "
            "collapse tensor equality to scalar false."
        )

    unattached = DecoderAccuracyCallback(perfect_classifier, 1)
    try:
        unattached.on_epoch_end(0, {"sentinel": True})
    except AttributeError:
        pass
    else:
        raise AssertionError("An unattached callback must not generate data.")

    return {"DecoderAccuracyCallback": "passed"}


if __name__ == "__main__":
    print(run_self_tests())
