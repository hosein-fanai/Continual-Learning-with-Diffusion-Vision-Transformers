import tensorflow as tf
from tensorflow.keras import layers, models

from common.argument_saver import ArgumentSaverLayer


class AdaLNZero(ArgumentSaverLayer):

    def __init__(
        self, 
        dim: int, 
        gate_dim: int | None = None, 
        mlp_ratio: float | None = None, 
        return_gate: bool = True, 
        no_adaptation: bool = False, 
        epsilon: float = 1e-6, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._save_init_args(locals())

        self.gate_dim = self.gate_dim if self.gate_dim is not None else self.dim

        self.norm = layers.LayerNormalization(
            center=False, 
            scale=False, 
            epsilon=self.epsilon, 
            name="layer_norm"
        )

        self.mlp_output_dim = self.dim * 2 + (
            self.gate_dim if self.return_gate else 0
        )
        if self.mlp_ratio is None:
            mlp_first_layer = layers.Activation(
                "swish", 
                name=f"{self.name}/mlp/first_layer"
            )
        else:
            mlp_first_layer = layers.Dense(
                int(self.dim * self.mlp_ratio), 
                activation="swish", 
                # kernel_initializer="zeros", 
                name=f"{self.name}/mlp/first_layer"
            )
        self.mlp = models.Sequential([
            mlp_first_layer, 
            layers.Dense(
                self.mlp_output_dim, 
                kernel_initializer="zeros", 
                name=f"{self.name}/mlp/final_layer"
            )
        ], name="mlp") if not self.no_adaptation else None

    def call(self, inputs, training=None):
        x, cond = inputs

        h = self.norm(x, training=training)

        if self.no_adaptation:
            if self.return_gate:
                return h, 1.
            return h

        params = self.mlp(cond, training=training)
        params = tf.expand_dims(params, 1)

        if self.return_gate:
            shift, scale, gate = tf.split(
                params,
                [self.dim, self.dim, self.gate_dim],
                axis=-1
            )
            return h * (1 + scale) + shift, gate

        shift, scale = tf.split(params, 2, axis=-1)
        return h * (1 + scale) + shift
