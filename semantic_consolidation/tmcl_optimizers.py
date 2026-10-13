"""Native TensorFlow optimizers with the reference TMCL update equations.

The official TMCL implementation uses PyTorch AdamW. Keras 3.11.2 AdamW instead
inherits Adam's epsilon-hat formulation. Changing only its epsilon value does
not make the two updates equivalent: the reference adds epsilon after bias
correction of the second moment. This subclass changes that update while
retaining native Keras slots, decay/exclusions, serialization and callbacks.

Primary implementations consulted:
https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/adamw.py
https://github.com/keras-team/keras/blob/v3.11.2/keras/src/optimizers/adam.py
https://github.com/keras-team/keras/blob/v3.11.2/keras/src/optimizers/adamw.py
https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/sgd.py
https://github.com/keras-team/keras/blob/v3.11.2/keras/src/optimizers/sgd.py

The supported TMCL route uses float32 parameters and one optimizer for each
group with the same update frequency. Keras owns a group iteration counter;
it does not reproduce PyTorch's separate parameter clocks when parameters are
intermittently omitted. IndexedSlices are interpreted as their summed dense
gradient, including repeated indices, rather than claiming sparse PyTorch
AdamW support. No PyTorch dependency or global optimizer replacement is added.
"""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="SemanticConsolidation")
class TMCLAdamW(tf.keras.optimizers.AdamW):
    """Apply AdamW with epsilon outside the bias-corrected second-moment root.

    For moment estimates ``m`` and ``v`` after update ``t``, the adaptive update is
    ``lr * m / (1 - beta_1**t) / (sqrt(v / (1 - beta_2**t)) + epsilon)``.
    AMSGrad substitutes the running maximum of uncorrected second moments before
    this correction. Native Keras applies decoupled weight decay separately.

    This preserves the reference equation, not bitwise equality across kernels or
    frameworks. In particular, native Keras decay uses subtraction, and sparse
    gradients are explicitly densified for this small native DiT adaptation.
    """

    def __init__(self, learning_rate: float | tf.keras.optimizers.schedules.LearningRateSchedule = 0.001, 
                 weight_decay: float = 0.004, beta_1: float = 0.9, beta_2: float = 0.999, 
                 epsilon: float = 1e-8, amsgrad: bool = False, name: str = "tmcl_adamw", 
                 **kwargs: object) -> None:
        """Bind the reference epsilon with ordinary native AdamW configuration.

        TMCL callers explicitly supply their experiment's learning rate and
        weight decay. Additional native options remain framework-owned.
        """

        super().__init__(
            learning_rate=learning_rate, weight_decay=weight_decay, beta_1=beta_1, 
            beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad, name=name, **kwargs
        )

    def build(self, var_list: list[tf.Variable]) -> None:
        """Create native optimizer slots only for the supported float32 route."""

        # Mixed parameter precision changes the supported reference calculation.
        if any(tf.as_dtype(variable.dtype) != tf.float32 for variable in var_list):
            raise ValueError("TMCLAdamW supports float32 parameters only.")
        super().build(var_list)

    def update_step(self, gradient: tf.Tensor | tf.IndexedSlices, variable: tf.Variable, 
                    learning_rate: tf.Tensor) -> None:
        """Update native moments and apply the reference bias-corrected denominator.

        Bias corrections use float64 scalar arithmetic before conversion to
        the parameter dtype, matching the reference's Python-scalar powers.
        Parameter and moment arithmetic remains float32. Sparse indices are
        summed before squaring, so duplicated embedding indices have their
        mathematically correct dense-gradient second moment.
        """

        gradient = tf.cast(tf.convert_to_tensor(gradient), variable.dtype)
        rate = tf.cast(learning_rate, variable.dtype)
        index = self._get_variable_index(variable)
        momentum = self._momentums[index]
        velocity = self._velocities[index]
        self.assign_add(momentum, (gradient - momentum) * tf.cast(1. - self.beta_1, variable.dtype))
        self.assign(
            velocity, tf.cast(self.beta_2, variable.dtype) * velocity
            + tf.cast(1. - self.beta_2, variable.dtype) * tf.square(gradient)
        )
        # AMSGrad preserves the native maximum slot and checkpoint representation.
        if self.amsgrad:
            maximum = self._velocity_hats[index]
            self.assign(maximum, tf.maximum(maximum, velocity))
            velocity = maximum
        step = tf.cast(self.iterations + 1, tf.float64)
        beta_1 = tf.convert_to_tensor(self.beta_1, dtype=tf.float64)
        beta_2 = tf.convert_to_tensor(self.beta_2, dtype=tf.float64)
        correction_1 = tf.cast(1. - tf.pow(beta_1, step), variable.dtype)
        correction_2_root = tf.cast(tf.sqrt(1. - tf.pow(beta_2, step)), variable.dtype)
        denominator = tf.sqrt(velocity) / correction_2_root + tf.cast(self.epsilon, variable.dtype)
        self.assign_sub(variable, (rate / correction_1) * momentum / denominator)


@tf.keras.utils.register_keras_serializable(package="SemanticConsolidation")
class TMCLSGD(tf.keras.optimizers.SGD):
    """Use an unscaled momentum buffer for the reference cosine-LR linear probe.

    With zero dampening, ``buffer = momentum * buffer + gradient`` followed by
    ``parameter -= current_lr * buffer``. Native Keras instead stores a buffer
    containing historical learning rates; these differ when the rate changes.
    The optional Nesterov direction is ``gradient + momentum * buffer``.

    The reference readout has zero weight decay. Nonzero decay is unsupported
    here because native Keras decouples it while reference SGD couples it into
    the gradient. Native slots and configuration serialization are retained.
    """

    def __init__(self, learning_rate: float | tf.keras.optimizers.schedules.LearningRateSchedule = 0.01, 
                 momentum: float = 0.9, nesterov: bool = False, weight_decay: float | None = None, 
                 name: str = "tmcl_sgd", **kwargs: object) -> None:
        """Bind the supported zero-decay, zero-dampening reference probe update."""

        # Nonzero native SGD decay is a different optimization equation.
        if weight_decay is not None and weight_decay != 0.:
            raise ValueError("TMCLSGD supports the reference probe's zero weight decay only.")
        super().__init__(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov, 
            weight_decay=weight_decay, name=name, **kwargs
        )

    def build(self, var_list: list[tf.Variable]) -> None:
        """Create native momentum slots only for the supported float32 readout."""

        # The readout reference and its state tests use float32 parameter buffers.
        if any(tf.as_dtype(variable.dtype) != tf.float32 for variable in var_list):
            raise ValueError("TMCLSGD supports float32 parameters only.")
        super().build(var_list)

    def update_step(self, gradient: tf.Tensor | tf.IndexedSlices, variable: tf.Variable, 
                    learning_rate: tf.Tensor) -> None:
        """Apply the current rate to a native slot containing unscaled gradients."""

        gradient = tf.cast(tf.convert_to_tensor(gradient), variable.dtype)
        rate = tf.cast(learning_rate, variable.dtype)
        direction = gradient
        # A zero-momentum ablation has no native momentum slot.
        if self.momentum != 0.:
            buffer = self.momentums[self._get_variable_index(variable)]
            coefficient = tf.cast(self.momentum, variable.dtype)
            self.assign(buffer, coefficient * buffer + gradient)
            direction = buffer
            # Reference Nesterov uses the freshly updated, unscaled buffer.
            if self.nesterov:
                direction = gradient + coefficient * buffer
        self.assign_sub(variable, rate * direction)
