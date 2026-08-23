# Transformer blocks

This directory provides the condition-aware attention blocks used by the raw
networks in `diffusion.models.transformer`. They operate on token sequences;
the diffusion wrappers in `diffusion.models.wrapper` supply training and
sampling behavior around the completed raw network.

## `VisionTransformerBlock`

The block applies two residual branches:

1. adaptive normalization → multi-head attention → condition gate → DropPath;
2. adaptive normalization → MLP → condition gate → DropPath.

```python
from diffusion.layers.block.vision_transformer_block import VisionTransformerBlock

block = VisionTransformerBlock(
    dim=128, 
    num_heads=4, 
    key_dim=32, 
    mlp_ratio=4, 
    mlp_output_dim=128, 
    drop_prob=0.1, 
    ln_mlp_ratio=2, 
    name="encoder/block_1", 
    dtype="float32", 
)

y = block((x, condition), training=True)
# x: [B, T, 128], condition: [B, C], y: [B, T, 128]
```

`key_dim` and `value_dim` are per-head widths. `key_dim=None` resolves to
`dim // num_heads`; ensure that value is positive. `query_dim` controls the
attention residual width, while `mlp_output_dim` controls the final width.
Dense residual projections are created when widths differ.

The public call can also replace attention inputs:

```python
y = block(
    (x, condition), 
    queries=query_tokens, 
    values=context_tokens, 
    mask=attention_mask, 
    training=False, 
)
```

`queries` is `[B,Tq,Dq]`, `values` is `[B,Tv,Dv]`, and a Keras attention mask is
broadcastable to `[B,Tq,Tv]` with nonzero/true entries permitting attention.
Because the result is added to `x`, `Tq` must normally equal `T`.

Useful forwarded `**kwargs` are `ln_mlp_ratio`, `ln_no_adaptation`, and
`mlp_output_dim`, followed by Keras `name`, `dtype`, and `trainable`.
`use_layer_norm`, `ln_dim`, `mlp_ratio`, and `mlp_activation_func` are already
supplied by this class and must not be repeated.

`VisionTransformerBlock.from_config(block.get_config())` is supported; the
constructor discards its forced normalization keys before initialization.

## `DiTDecoderBlock`

The decoder extends the block to three residual branches:

1. self-attention over decoder tokens, optionally using `causal_mask`;
2. cross-attention from decoder tokens to `values`;
3. the feed-forward MLP.

```python
from diffusion.layers.block.di_t_decoder_block import DiTDecoderBlock

decoder = DiTDecoderBlock(dim=128, num_heads=4, mlp_ratio=4)
y = decoder(
    (decoder_tokens, condition), 
    values=encoder_tokens, 
    causal_mask=lower_triangular_mask, 
    training=True, 
)
```

The causal mask affects only the first attention branch. The public decoder API
does not expose a cross-attention mask. `queries=None` uses the normalized
decoder tokens for cross-attention queries; `values=None` also uses the decoder
tokens, making the second branch another self-attention pass. Do not pass
`gate_query_flag` in `**kwargs`: the decoder fixes it to `False`.

`DiTDecoderBlock.from_config(block.get_config())` is also supported. The
constructor discards the serialized `gate_query_flag`, and its parent handles
the forced normalization keys.

## Initialization and training

Adaptive normalization projections and their gates are zero-initialized. With
normal adaptation, each new residual branch therefore contributes zero at
construction. `ln_no_adaptation=True` switches gates to scalar `1.0` and removes
condition modulation. `drop_prob` is active only with `training=True`; use
`drop_per_sample=False` for one keep/drop decision shared by a whole batch.
