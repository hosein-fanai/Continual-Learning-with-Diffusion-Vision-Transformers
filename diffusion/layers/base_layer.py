from tensorflow.keras import layers, models

from common.argument_saver import ArgumentSaverLayer

from diffusion.layers.adaptive_layer_normalization_zero import AdaLNZero


class BaseLayer(ArgumentSaverLayer):
    """
    
    """

    def __init__(
        self, 
        use_layer_norm: bool = False, 
        ln_dim: int | None = None, 
        ln_mlp_ratio: int | None = None, 
        ln_no_adaptation: bool = False, 
        mlp_ratio: float | None = None, 
        mlp_activation_func: str = "swish", 
        mlp_output_dim: int | None = None, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._check_assertions(locals())
        self._save_init_args(locals())

    def _check_assertions(self, local_vars):
        if local_vars["use_layer_norm"] and \
        not local_vars["ln_no_adaptation"]:
            assert local_vars["ln_dim"] is not None, \
                "ln_dim cannot be None when use_layer_norm is true."

    def _create_layer_norm(
        self, 
        dim: int | None = None, 
        gate_dim: int | None = None, 
        mlp_ratio: float | None = None, 
        return_gate: bool = True, 
        no_adaptation: bool | None = None, 
        use_layer_norm: bool | None = None, 
        name: str | None = None, 
    ):
        """
        """

        dim = self.ln_dim if dim is None else dim
        gate_dim = self.ln_dim if gate_dim is None else gate_dim
        mlp_ratio = None if mlp_ratio is None and self.ln_mlp_ratio is None \
                    else mlp_ratio or self.ln_mlp_ratio
        no_adaptation = self.ln_no_adaptation if no_adaptation is None else no_adaptation
        use_layer_norm = self.use_layer_norm if use_layer_norm is None else use_layer_norm
        name = f"{self.name}/layer_norm" if name is None else name

        layer_norm = AdaLNZero(
            dim=dim, 
            gate_dim=gate_dim, 
            mlp_ratio=mlp_ratio, 
            return_gate=return_gate, 
            no_adaptation=no_adaptation, 
            name=name
        ) if use_layer_norm else None

        return layer_norm

    def _create_mlp(
        self, 
        prev_output_dim: int, 
        mlp_ratio: float | None = None, 
        mlp_activation_func: str | None = None, 
        mlp_output_dim: int | None = None, 
    ):
        """
        """

        mlp_ratio = self.mlp_ratio if mlp_ratio is None else mlp_ratio
        mlp_activation_func = self.mlp_activation_func if mlp_activation_func is None \
                            else mlp_activation_func
        mlp_output_dim = self.mlp_output_dim if mlp_output_dim is None else mlp_output_dim

        self.prev_output_dim = int(prev_output_dim)
        if mlp_output_dim is not None:
            self.output_dim = mlp_output_dim
            mlp = models.Sequential(name=f"{self.name}/mlp")
            if mlp_ratio is not None:
                mlp.add(layers.Dense(
                    int(prev_output_dim * mlp_ratio), 
                    activation=mlp_activation_func, 
                    name=f"{mlp.name}/first_layer"
                ))

            mlp.add(layers.Dense(
                mlp_output_dim, 
                name=f"{mlp.name}/final_layer"
            ))
        else:
            self.output_dim = prev_output_dim
            mlp = None

        return mlp
