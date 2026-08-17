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
