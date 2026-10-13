"""Scoped internal DiT modulations for the opt-in TMCL representation route.

The native encoder and its variables remain authoritative. Eager calls temporarily
inject residual affine gates into attention projections and preactivation MLP
linears, then restore every native method and activation. Ordinary model calls
outside this adapter retain their original computation and checkpoint graph.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from threading import Lock
from typing import Callable, Iterator

import numpy as np
import tensorflow as tf

from common.runtime import derive_seed
from diffusion.layers.block.vision_transformer_block import VisionTransformerBlock
from diffusion.models.transformer.di_t_classifier import DiTClassifier


class LayerModulationBank(tf.Module):
    """Store independent class gain/bias vectors for a fixed ordered layer map."""

    def __init__(self, sites: list[dict[str, object]], init_std: float = 0.02, 
                 seed: int = 0) -> None:
        """Declare structural gate slices and deterministic per-class initialization."""

        super().__init__(name="tmcl_modulation_bank")
        self.sites = deepcopy(sites)
        self.init_std = init_std
        self.seed = seed
        self.dimension = sum(int(np.prod(site["shape"])) for site in sites)
        object.__setattr__(self, "vectors", {})

    @property
    def classes(self) -> list[int]:
        """Return the sorted currently retained class IDs."""

        return sorted(self.vectors)

    def add(self, class_ids: list[int]) -> None:
        """Add absent classes without replacing or changing any retained variable."""

        # Exact integer identifiers are part of persistent bank identity.
        if any(type(value) is not int or value < 0 for value in class_ids):
            raise ValueError("Modulation class IDs must be nonnegative Python integers.")
        for class_id in class_ids:
            # Existing class state survives repeated declarations unchanged.
            if class_id in self.vectors:
                continue
            rng = np.random.default_rng(derive_seed(self.seed, "tmcl_layer_gate", class_id))
            values = rng.normal(0., self.init_std, (2, self.dimension)).astype("float32")
            pair = []
            for kind, value in zip(("gain", "bias"), values):
                attribute = f"class_{class_id}_{kind}"
                variable = tf.Variable(value, name=attribute)
                setattr(self, attribute, variable)
                pair.append(variable)
            self.vectors[class_id] = tuple(pair)

    def variables_for(self, class_ids: list[int] | None = None) -> list[tf.Variable]:
        """Select complete independent class variables for a phase optimizer."""

        selected = self.classes if class_ids is None else class_ids
        return [value for class_id in selected for value in self.vectors[class_id]]

    def apply(self, outputs: tf.Tensor, site_index: int, class_ids: tf.Tensor) -> tf.Tensor:
        """Apply per-row class gates across tokens and all projection feature axes."""

        ids = tf.convert_to_tensor(class_ids)
        # Fractional or silently rounded identifiers could select the wrong memory.
        if not ids.dtype.is_integer:
            raise ValueError("Modulation selections must have integer dtype.")
        tf.debugging.assert_rank(ids, 1)
        tf.debugging.assert_equal(tf.shape(ids)[0], tf.shape(outputs)[0])
        # A modulated call must reference an initialized persistent bank.
        if not self.classes:
            raise ValueError("No modulation classes have been initialized.")
        matches = tf.equal(tf.cast(ids[:, None], tf.int64), tf.constant(self.classes, tf.int64)[None, :])
        tf.debugging.assert_equal(tf.reduce_sum(tf.cast(matches, tf.int32), axis=1), 1)
        rows = tf.argmax(tf.cast(matches, tf.int32), axis=1, output_type=tf.int32)
        site = self.sites[site_index]
        start = sum(int(np.prod(item["shape"])) for item in self.sites[:site_index])
        width = int(np.prod(site["shape"]))
        shape = list(site["shape"])
        tf.debugging.assert_equal(tf.shape(outputs)[-len(shape):], shape)
        broadcast = tf.concat([tf.shape(outputs)[:1], tf.ones(tf.rank(outputs) - len(shape) - 1, tf.int32), shape], axis=0)
        selected = []
        for kind in range(2):
            values = tf.stack([self.vectors[class_id][kind][start:start + width] for class_id in self.classes])
            selected.append(tf.cast(tf.reshape(tf.gather(values, rows), broadcast), outputs.dtype))
        return outputs * (1. + selected[0]) + selected[1]

    def snapshot(self) -> dict[str, object]:
        """Copy the structural identity and numeric bank values for exact recovery."""

        return {"sites": deepcopy(self.sites), "seed": self.seed, 
                "init_std": self.init_std, "classes": self.classes, 
                "gain": [self.vectors[key][0].numpy().copy() for key in self.classes], 
                "bias": [self.vectors[key][1].numpy().copy() for key in self.classes]}

    def restore(self, state: dict[str, object]) -> None:
        """Restore matching class/layout state after validating the entire payload."""

        # Existing banks may not lose classes or reinterpret an old feature coordinate.
        if (state.get("sites") != self.sites or state.get("seed") != self.seed
                or state.get("init_std") != self.init_std or state.get("classes") != sorted(set(state.get("classes", [])))
                or not set(self.classes).issubset(state.get("classes", []))):
            raise ValueError("Modulation checkpoint layout or class identity differs.")
        classes = state["classes"]
        # Validate all shapes before adding or assigning any persistent state.
        if any(len(state.get(kind, [])) != len(classes) for kind in ("gain", "bias")):
            raise ValueError("Modulation checkpoint class arrays are misaligned.")
        for kind in ("gain", "bias"):
            for value in state[kind]:
                array = np.asarray(value)
                # Exact dtype and width distinguish native state from corrupted data.
                if array.dtype != np.dtype("float32") or array.shape != tuple([self.dimension]) or not np.isfinite(array).all():
                    raise ValueError("Modulation checkpoint arrays must be finite matching float32 vectors.")
        self.add(classes)
        for index, class_id in enumerate(classes):
            for position, kind in enumerate(("gain", "bias")):
                self.vectors[class_id][position].assign(state[kind][index])


class TMCLDiTAdapter:
    """Expose clean native classifier features with encoder-wide class modulations.

Images are already in wrapper model coordinates. The feature endpoint is the
native classifier token pool, before its hidden projection and class output.
No class label is supplied to the diffusion condition: zero is the native CFG
null ID. Only the separate modulation bank consumes selected class IDs.
"""

    def __init__(self, wrapper: object, modulation_init_std: float = 0.02, 
                 seed: int = 0) -> None:
        """Inspect an already built native DiT and preserve its weight ownership."""

        self.wrapper = wrapper
        self.network = wrapper.network
        # This route's internal projection contract has only been implemented for DiT.
        if type(self.network) is not DiTClassifier:
            raise TypeError("TMCLDiTAdapter requires the native DiTClassifier.")
        # Null conditioning and fixed float32 are explicit supported-route identities.
        if not self.network.use_cfg or self.network.dtype_policy.name != "float32":
            raise ValueError("TMCL DiT requires CFG null conditioning and float32.")
        # Stochastic latent reconstruction is not the clean feature protocol.
        if any(getattr(self.network, prefix + "reshaper_kwargs", {}).get("add_kl", False)
               for prefix in ("", "clf_")):
            raise ValueError("TMCL clean features do not support variational reshapers.")
        # Scoped hooks require the real native projections to exist before inspection.
        if not self.network.built:
            raise ValueError("Build the native DiT before creating its TMCL adapter.")
        # Multiple adapters around one native model must share temporary-call ownership.
        if not hasattr(self.network, "_tmcl_adapter_lock"):
            object.__setattr__(self.network, "_tmcl_adapter_lock", Lock())
        self._lock = self.network._tmcl_adapter_lock
        self._layers: list[tf.keras.layers.Layer] = []
        sites = []
        blocks = []
        all_blocks = []
        self._block_specs = []
        self._encoder_stop = self.network.max_encoder_num
        for branch, stages in (("encoder", self.network.layers_dicts), ("classifier", self.network.clf_layers_dicts)):
            for stage_index, stage in enumerate(stages):
                block = stage.get(self.network.VTB)
                # Every registered transformer must have a recognized structural owner.
                if block is not None:
                    all_blocks.append(block)
                active = branch == "classifier" or self._encoder_stop < 0 or stage_index < self._encoder_stop
                # Only blocks actually traversed by clean classification receive gates.
                if block is not None and active:
                    blocks.append(block)
                    has_cls = self.network.has_cls_token if branch == "encoder" else self.network.clf_has_cls_token
                    has_distil = self.network.has_distil_token if branch == "encoder" else self.network.clf_has_distil_token
                    self._block_specs.append({
                        "key": f"block_{len(blocks) - 1}", "branch": branch, "stage": stage_index + 1, 
                        "width": int(block.mlp_output_dim), "has_cls": bool(has_cls), 
                        "has_distil": bool(has_distil), 
                        "pooling": "native_cls" if has_cls and not self.network.force_global_avg_pooling else "native_gap_without_distil"
                    })
        discovered = [layer for layer in self.network._flatten_layers(include_self=False, recursive=True)
                      if isinstance(layer, VisionTransformerBlock)]
        # Shared or unrecognized nested blocks cannot receive an unambiguous depth policy.
        if (len({id(block) for block in all_blocks}) != len(all_blocks)
                or {id(block) for block in all_blocks} != {id(block) for block in discovered}):
            raise ValueError("TMCL requires distinct native transformer blocks in explicit depth order.")
        self._blocks = blocks
        for block_index, block in enumerate(blocks):
            for attention_name in ("mha", "mha2"):
                attention = getattr(block, attention_name, None)
                # Decoder blocks have a second attention branch; plain blocks do not.
                if attention is None:
                    continue
                for role in ("query", "key", "value", "output"):
                    layer = getattr(attention, f"_{role}_dense", None)
                    # Missing native MHA projections indicate an unbuilt or incompatible core.
                    if layer is None or not layer.built:
                        raise ValueError("Build every native attention block before creating TMCL gates.")
                    shape = list(layer.kernel.shape[-1:]) if role == "output" else list(layer.kernel.shape[-2:])
                    self._layers.append(layer)
                    sites.append({"key": f"block_{block_index}/{attention_name}/{role}", "shape": shape})
            for layer_index, layer in enumerate(block.mlp.layers):
                # Modulate both MLP linear outputs before their optional activation.
                if isinstance(layer, tf.keras.layers.Dense):
                    self._layers.append(layer)
                    sites.append({"key": f"block_{block_index}/mlp_{layer_index}", "shape": [layer.units]})
        # An empty injection map would silently revert to an unrelated feature-only method.
        if not sites:
            raise ValueError("TMCL requires at least one native attention/MLP block.")
        self.bank = LayerModulationBank(sites, init_std=modulation_init_std, seed=seed)
        excluded = []
        for attribute in ("classifier", "distil_classifier", "labels_embed_reg", "clf_labels_embed_reg"):
            layer = getattr(self.network, attribute, None)
            # Optional classifier and auxiliary heads are outside representation training.
            if layer is not None:
                excluded.extend(layer.variables)
        # The unpatchifier contributes only when the classifier explicitly consumes noises.
        if not self.network.aggregate_from_noises and self.network.unpatchifier is not None:
            excluded.extend(self.network.unpatchifier.variables)
        for stages in (self.network.layers_dicts, self.network.clf_layers_dicts):
            for stage in stages:
                # Class-token regularizer heads are not feature transformations.
                if self.network.CTR in stage:
                    excluded.extend(stage[self.network.CTR].variables)
        excluded_ids = {id(variable) for variable in excluded}
        self._backbone_variables = [variable for variable in self.network.trainable_variables
                                    if id(variable) not in excluded_ids]

    @property
    def backbone_variables(self) -> list[tf.Variable]:
        """Return eligible native feedforward variables, without gates or class heads."""

        return list(self._backbone_variables)

    def _hook(self, original: Callable, activation: Callable | None, 
              site_index: int, class_ids: tf.Tensor) -> Callable:
        """Create one projection call preserving native linear computation."""

        def call(inputs: tf.Tensor, **kwargs: object) -> tf.Tensor:
            """Modulate the native linear output and then restore its activation order."""

            outputs = self.bank.apply(original(inputs, **kwargs), site_index, class_ids)
            return activation(outputs) if activation is not None else outputs

        return call

    @contextmanager
    def modulated(self, class_ids: tf.Tensor | None) -> Iterator[None]:
        """Own an eager forward exclusively and restore hooks even after exceptions."""

        # Python method hooks cannot safely survive tracing or overlap on one network.
        if not tf.executing_eagerly():
            raise RuntimeError("TMCL scoped modulation requires eager execution.")
        # A shared network cannot admit overlapping or nested temporary hook ownership.
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent or nested TMCL adapter calls are unsupported.")
        restored = []
        try:
            # Unmodulated calls use the exact native methods without temporary replacements.
            if class_ids is not None:
                for index, layer in enumerate(self._layers):
                    own_call = "call" in vars(layer)
                    original = layer.call
                    activation = getattr(layer, "activation", None)
                    restored.append((layer, own_call, original, activation))
                    # Native Dense embeds its activation; gates belong before that activation.
                    if activation is not None:
                        object.__setattr__(layer, "activation", tf.keras.activations.linear)
                    object.__setattr__(layer, "call", self._hook(original, activation, index, class_ids))
            yield
        finally:
            for layer, own_call, original, activation in reversed(restored):
                # Restore an existing instance override or remove only our temporary override.
                if own_call:
                    object.__setattr__(layer, "call", original)
                # Ordinary class-defined call methods must remain class-defined after cleanup.
                else:
                    object.__delattr__(layer, "call")
                # Preserve the exact original activation object, including custom functions.
                if activation is not None:
                    object.__setattr__(layer, "activation", activation)
            self._lock.release()

    def features(self, images: tf.Tensor, class_ids: tf.Tensor | None = None, 
                 training: bool = False) -> tf.Tensor:
        """Return native pooled clean features with optional per-row class modulation."""

        with self.modulated(class_ids):
            times = tf.zeros(tf.shape(images)[:1], dtype=tf.int32)
            outputs = self.network.predict_class(
                (images, times, tf.zeros_like(times)), 
                max_encoder_num=None, full_return=True, training=training
            )
            features = self.network.classifier_feature_extractor(outputs[2][-1], training=training)
            return tf.cast(features, tf.float32)

    def readout_spec(self, last_layers: int = 4) -> dict[str, object]:
        """Declare a frozen last-block readout on an unambiguous sequential feature route.

        The native DiT has no added shared ConViT output LayerNorm. Its native
        class token is pooled where available; forced/no-class-token pooling
        averages tokens while excluding only distillation tokens. Those explicit
        DiT choices are recorded rather than labelled paper CLS reproduction.
        """

        # Readouts must not silently shrink the requested depth or use another route.
        if type(last_layers) is not int or last_layers < 1 or last_layers > len(self._blocks):
            raise ValueError("Requested readout depth exceeds the active transformer path.")
        # An architecture mutation requires a newly inspected adapter and checkpoint identity.
        if self.network.max_encoder_num != self._encoder_stop:
            raise ValueError("The native encoder stop changed after adapter construction.")
        mappings = self.network.feature_aggregation_ids_dict
        expected = self.network.depth if self._encoder_stop < 0 else self._encoder_stop
        # Multiple entry/later merges need a distinct, explicitly audited ancestry policy.
        if (self.network.aggregate_from_noises or set(mappings) != {1}
                or list(mappings[1]) != [expected]):
            raise ValueError("TMCL readout requires one final-depth classifier entry without noise or later merges.")
        for branch, stages in (("encoder", self.network.layers_dicts), ("classifier", self.network.clf_layers_dicts)):
            for index, stage in enumerate(stages):
                # Inactive encoder stages do not contribute to this feature route.
                if branch == "encoder" and self._encoder_stop >= 0 and index >= self._encoder_stop:
                    continue
                # Cross connections and token reshapers need a separately audited ancestry/prefix map.
                if any(key in stage for key in (self.network.CAC, self.network.CAA, self.network.R)):
                    raise ValueError("TMCL readout requires sequential blocks without skip/cross branches or token reshapers.")
                # The native terminal connector selects the immediate predecessor and is sequential.
                if self.network.FC in stage and list(stage[self.network.FC].ids) not in ([-1], [index]):
                    raise ValueError("TMCL readout cannot use a connector that skips or merges feature ancestors.")
        selected = deepcopy(self._block_specs[-last_layers:])
        return {"endpoint": "native_dit_block_outputs", "last_layers": last_layers, 
                "layer_order": "encoder_then_classifier", "layers": selected, 
                "width": sum(item["width"] for item in selected), 
                "extra_normalization": None, "modulated": False, "training": False}

    def _capture_hook(self, original: Callable, index: int, 
                      captured: list[tuple[int, tf.Tensor]]) -> Callable:
        """Capture a native block output without changing that block's computation."""

        def call(inputs: object, **kwargs: object) -> tf.Tensor:
            """Record each actual native eager execution in chronological order."""

            output = original(inputs, **kwargs)
            captured.append((index, output))
            return output

        return call

    def readout_features(self, images: tf.Tensor, last_layers: int = 4) -> tf.Tensor:
        """Concatenate stopped pooled block outputs for separate linear/kNN evaluation."""

        spec = self.readout_spec(last_layers)
        captured = []
        restored = []
        with self.modulated(None):
            try:
                for index, block in enumerate(self._blocks):
                    original = block.call
                    restored.append((block, "call" in vars(block), original))
                    object.__setattr__(block, "call", self._capture_hook(original, index, captured))
                times = tf.zeros(tf.shape(images)[:1], dtype=tf.int32)
                self.network.predict_class(
                    (images, times, tf.zeros_like(times)), 
                    max_encoder_num=None, full_return=True, training=False
                )
            finally:
                for block, owned, original in reversed(restored):
                    # Restore pre-existing instance overrides exactly after any failed call.
                    if owned:
                        object.__setattr__(block, "call", original)
                    # Remove only this context's override when call belongs to the class.
                    else:
                        object.__delattr__(block, "call")
        # A repeated, skipped or reordered call invalidates the declared last-layer topology.
        if [index for index, _ in captured] != list(range(len(self._blocks))):
            raise ValueError("Native execution differs from the declared transformer readout path.")
        pooled = []
        for (_, tokens), layer in zip(captured[-last_layers:], spec["layers"]):
            tf.debugging.assert_rank(tokens, 3)
            tf.debugging.assert_equal(tf.shape(tokens)[-1], layer["width"])
            # Native CLS pooling selects the first token before any distillation token.
            if layer["pooling"] == "native_cls":
                feature = tokens[:, 0]
            # Native GAP retains any class token and removes only the distillation position.
            else:
                # Distillation tokens are not part of the primary classifier feature pool.
                if layer["has_distil"]:
                    position = int(layer["has_cls"])
                    tokens = tf.concat([tokens[:, :position], tokens[:, position + 1:]], axis=1)
                feature = tf.reduce_mean(tokens, axis=1)
            pooled.append(tf.cast(feature, tf.float32))
        return tf.stop_gradient(tf.concat(pooled, axis=-1))

    def snapshot(self) -> dict[str, object]:
        """Copy complete native network and gate values without optimizer/RNG claims."""

        return {"network_config": self.network.get_config(), 
                "network": [value.copy() for value in self.network.get_weights()], 
                "bank": self.bank.snapshot()}

    def restore(self, state: dict[str, object]) -> None:
        """Restore an exact adapter snapshot after structural native-network checks."""

        current = self.network.get_weights()
        # This numeric helper cannot authorize architecture changes or class expansion.
        if (state.get("network_config") != self.network.get_config()
                or len(state.get("network", [])) != len(current)
                or any(np.asarray(left).shape != right.shape or np.asarray(left).dtype != right.dtype
                       or not np.isfinite(left).all() for left, right in zip(state["network"], current))):
            raise ValueError("TMCL network snapshot structure differs from the current native model.")
        self.bank.restore(state["bank"])
        self.network.set_weights(state["network"])
