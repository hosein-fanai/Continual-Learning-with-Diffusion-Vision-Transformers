"""Selection, merging, normalization, and projection of saved features."""

import tensorflow as tf

from typing import Any, Sequence, get_args

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
        grid_size: Optional square token-grid side produced by the merge. This
            is metadata for downstream spatial-shape inference and does not
            alter feature selection or merging.
        **kwargs: :class:`BaseLayer` options. Useful keys include
            ``use_layer_norm``, ``ln_dim``, ``ln_mlp_ratio``,
            ``ln_no_adaptation``, ``mlp_ratio``, ``mlp_activation_func``, and
            ``mlp_output_dim``, followed by standard Keras layer options.
            ``ln_dim`` is required only when normalization or an output MLP is
            enabled. In those cases it must match the merged last-axis width;
            adaptive normalization also requires ``cond`` at call time.

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
        grid_size: int | None = None, 
        **kwargs: Any
    ) -> None:
        """Initialize feature-selection and post-merge processing options.

        Args:
            ids (list[int] | None): Default feature indices, or ``None`` to
                require call-specific indices.
            connect_axis (int): Concatenation axis.
            connect_type (MergeType): Either ``"concat"`` or ``"add"``.
            grid_size (int | None): Optional output token-grid side retained
                for downstream architecture inference.
            **kwargs (Any): Typed :class:`BaseLayer` and Keras options.

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

    def _check_fh_assertions(self, local_vars: dict[str, Any]) -> None:
        """Validate merge mode and projection dimensions.

        Args:
            local_vars (dict[str, Any]): Constructor-local mapping containing
                ``connect_type``.

        Returns:
            ``None``. Invalid modes or a projected MLP without ``ln_dim`` raise
            ``ValueError``.
        """

        # Restrict feature merging to concatenation or addition.
        if local_vars["connect_type"] not in get_args(MergeType):
            raise ValueError(
                f"connect_type must be one of {get_args(MergeType)}."
            )

        # A configured projection requires a known merged input width.
        if self.mlp_output_dim is not None:
            # Reject projection setup when that merged width is missing.
            if self.ln_dim is None:
                raise ValueError(
                    "ln_dim cannot be None when mlp_output_dim is not None."
                )

    def call(
        self, 
        features_list: Sequence[tf.Tensor], 
        second_list: Sequence[tf.Tensor] | None = None, 
        ids: list[int] | None = None, 
        cond: tf.Tensor | None = None, 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor | None:
        """Select, merge, and optionally transform feature tensors.

        Args:
            features_list (Sequence[tf.Tensor]): Indexable feature sequence.
            second_list (Sequence[tf.Tensor] | None): Optional tensors appended to the selected
                primary features. ``None`` means no secondary features.
            ids (list[int] | None): Optional call-specific indices overriding ``self.ids``.
                Negative and duplicate indices follow Python list semantics.
            cond (tf.Tensor | None): Optional condition tensor shaped ``[batch, condition_dim]``.
                It is required only by adaptive layer normalization.
            training (bool | tf.Tensor | None): Optional Keras training flag.

        Returns:
            ``tf.Tensor | None``. Concatenation preserves tensor dtype and
            changes only ``connect_axis``; addition follows TensorFlow
            broadcasting. A configured MLP changes the last dimension to
            ``mlp_output_dim``.
        """

        second_list = [] if second_list is None else second_list
        ids = self.ids if ids is None else ids

        # Require call-time IDs when selection was deferred at construction.
        if ids is None:
            raise ValueError("ids must be supplied either at construction or call time.")

        # Return no feature when both primary and secondary selections are empty.
        if len(ids) == 0 and len(second_list) == 0:
            return None

        selected_features = [
            features_list[id_] for id_ in ids
        ] + list(second_list)

        # Concatenate selected features along the configured axis.
        if self.connect_type == "concat":
            x = tf.concat(
                selected_features, 
                axis=self.connect_axis
            )
        # Otherwise combine the selected features by elementwise addition.
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


def run_self_tests() -> dict[str, str]:
    """Run compact tests for every :class:`FeatureHandler` branch.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry success mapping after selection, merge, normalization,
        projection, override, empty-input, config, and invalid-input tests.
    """

    import numpy as np


    features = [
        tf.ones((2, 2, 2), dtype=tf.float32),
        tf.fill((2, 2, 2), 2.0),
        tf.fill((2, 2, 2), 3.0),
    ]

    for invalid_mode in ("multiply", "", None):
        try:
            FeatureHandler(ids=[0], connect_type=invalid_mode, ln_dim=2)
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown feature merge modes must fail.")

    unconfigured = FeatureHandler(ids=[0])
    np.testing.assert_array_equal(unconfigured(features).numpy(), features[0].numpy())

    concat = FeatureHandler(ids=[-1, 0, 0], connect_type="concat", ln_dim=2)
    concatenated = concat(features)
    assert concatenated.shape == (2, 2, 6)
    np.testing.assert_array_equal(
        concatenated[0, 0].numpy(), [3.0, 3.0, 1.0, 1.0, 1.0, 1.0],
    )
    overridden = concat(features, ids=[1], second_list=[features[0]])
    assert overridden.shape == (2, 2, 4)

    add = FeatureHandler(ids=[0, 1], connect_type="add", ln_dim=2)
    np.testing.assert_array_equal(add(features).numpy(), tf.fill((2, 2, 2), 3.0))
    np.testing.assert_array_equal(
        add(features, ids=[], second_list=[features[2]]).numpy(), features[2].numpy(),
    )
    assert add(features, ids=[], second_list=[]) is None

    deferred = FeatureHandler(ids=None, connect_type="add", ln_dim=2)
    assert deferred(features, ids=[2]).shape == (2, 2, 2)
    try:
        deferred(features)
    except ValueError:
        pass
    else:
        raise AssertionError("A deferred selector requires call-time ids.")

    normalized = FeatureHandler(
        ids=[0], connect_type="add", ln_dim=2,
        use_layer_norm=True, ln_no_adaptation=True,
    )(features, cond=None, training=True)
    np.testing.assert_allclose(normalized.numpy(), np.zeros((2, 2, 2)), atol=1e-6)

    adaptive = FeatureHandler(
        ids=[0], connect_type="add", ln_dim=2,
        use_layer_norm=True, mlp_output_dim=3, mlp_ratio=2,
    )
    projected = adaptive(features, cond=tf.ones((2, 4)), training=False)
    assert projected.shape == (2, 2, 3)
    assert adaptive.prev_output_dim == 2 and adaptive.output_dim == 3

    axis_one = FeatureHandler(ids=[0, 1], connect_axis=1, ln_dim=2)
    assert axis_one(features).shape == (2, 4, 2)
    try:
        concat([tf.ones((1, 2, 2)), tf.ones((1, 3, 3))], ids=[0, 1])
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("Incompatible concatenation shapes must fail.")
    try:
        add(features, ids=[99])
    except IndexError:
        pass
    else:
        raise AssertionError("Out-of-range feature IDs must fail.")

    restored = FeatureHandler.from_config(add.get_config())
    assert restored.ids == [0, 1] and restored.connect_type == "add"

    dtype_layer = FeatureHandler(ids=[0], ln_dim=2, dtype="float64")
    dtype_output = dtype_layer([tf.ones((1, 2, 2), dtype=tf.float64)])
    assert dtype_layer.compute_dtype == "float64"
    assert dtype_output.dtype == tf.float64

    return {"FeatureHandler": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
