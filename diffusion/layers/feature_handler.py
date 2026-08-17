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


def run_self_tests() -> dict[str, str]:
    """Run compact tests for every :class:`FeatureHandler` branch.

    Args:
        None.

    Returns:
        A one-entry success mapping after selection, merge, normalization,
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
        except AssertionError:
            pass
        else:
            raise AssertionError("Unknown feature merge modes must fail.")

    try:
        FeatureHandler(ids=[0])
    except TypeError:
        pass
    else:
        raise AssertionError("The documented ln_dim construction limit changed.")

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
    except TypeError:
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


if __name__ == "__main__":
    print(run_self_tests())
