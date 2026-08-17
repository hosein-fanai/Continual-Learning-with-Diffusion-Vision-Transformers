"""Selection, merging, normalization, and projection of saved features."""

import tensorflow as tf

from typing import get_args

from diffusion.layers.embedding import MergeType
from diffusion.layers.base_layer import BaseLayer


class FeatureHandler(BaseLayer):
    """Select and combine intermediate feature tensors.

    Transformer builders use this layer for skip connections and feature
    aggregation. ``ids`` indexes the ordered feature list using ordinary
    Python list rules: ``0`` is the first feature, ``-1`` the most recent,
    ordering is preserved, and repeated IDs repeat a feature. Additional
    tensors from ``second_list`` are appended after the selected features.

    Args:
        ids: Default list of integer feature indices. ``None`` defers the
            choice to :meth:`call`; in that case every call must pass ``ids``.
            An empty list plus an empty ``second_list`` returns ``None``.
        connect_axis: Tensor axis used for ``connect_type="concat"``. All
            dimensions other than this axis must match.
        connect_type: ``"concat"`` concatenates selected tensors;
            ``"add"`` sums them and therefore requires broadcast-compatible
            shapes.
        **kwargs: :class:`BaseLayer` options. Useful keys include
            ``use_layer_norm``, ``ln_dim``, ``ln_mlp_ratio``,
            ``ln_no_adaptation``, ``mlp_ratio``, ``mlp_activation_func``, and
            ``mlp_output_dim``, followed by standard Keras layer options.
            ``ln_dim`` is required by the current constructor because it is
            passed to the MLP factory even when no MLP is created. When adaptive
            normalization is enabled, it must also match the merged last-axis
            width and :meth:`call` needs ``cond``.

    Inputs:
        A sequence of same-rank tensors, normally each shaped
        ``[batch, tokens, channels]``. Optional secondary tensors follow the
        same merge-compatibility rules.

    Outputs:
        The merged floating ``tf.Tensor``, optionally normalized/projected, or
        ``None`` when no tensors were selected.
    """

    def __init__(
        self, 
        ids: list[int] | None = None, 
        connect_axis: int = -1, 
        connect_type: MergeType = "concat", 
        **kwargs
    ):
        """Initialize feature-selection and post-merge processing options.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())
        self._check_fh_assertions(locals())

        self.layer_norm = self._create_layer_norm(
            return_gate=False
        )
        self.mlp = self._create_mlp(
            self.ln_dim
        )

    def _check_fh_assertions(self, local_vars):
        """Validate merge mode and projection dimensions.

        Args:
            local_vars: Constructor-local mapping containing ``connect_type``.

        Returns:
            ``None``. Invalid modes or a projected MLP without ``ln_dim`` raise
            ``AssertionError``.
        """

        assert local_vars["connect_type"] in get_args(MergeType), \
            f"connect_type can be one of {get_args(MergeType)}."

        if self.mlp_output_dim is not None:
            assert self.ln_dim is not None, \
                "ln_dim cannot be None when mlp_output_dim is not None."

    def call(
        self, 
        features_list, 
        second_list=None, 
        ids=None, 
        cond=None, 
        training=None
    ):
        """Select, merge, and optionally transform feature tensors.

        Args:
            features_list: Indexable sequence of ``tf.Tensor`` objects.
            second_list: Optional iterable of tensors appended to the selected
                primary features. ``None`` means no secondary features.
            ids: Optional call-specific ``list[int]`` overriding ``self.ids``.
                Negative and duplicate indices follow Python list semantics.
            cond: Optional condition tensor shaped ``[batch, condition_dim]``.
                It is required only by adaptive layer normalization.
            training: Optional Keras training flag forwarded to nested layers.

        Returns:
            ``tf.Tensor | None``. Concatenation preserves tensor dtype and
            changes only ``connect_axis``; addition follows TensorFlow
            broadcasting. A configured MLP changes the last dimension to
            ``mlp_output_dim``.
        """

        second_list = [] if second_list is None else second_list
        ids = self.ids if ids is None else ids

        if len(ids) == 0 and len(second_list) == 0:
            return None

        selected_features = [
            features_list[id_] for id_ in ids
        ] + list(second_list)

        if self.connect_type == "concat":
            x = tf.concat(
                selected_features, 
                axis=self.connect_axis
            )
        else:
            x = sum(selected_features)

        x = self.layer_norm(
            (x, cond), 
            training=training
        ) if self.layer_norm is not None else x
        x = self.mlp(
            x, 
            training=training
        ) if self.mlp is not None else x

        return x
