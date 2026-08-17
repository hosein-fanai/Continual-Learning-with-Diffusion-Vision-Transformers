# DiT checkpoint

`model.weights.h5` is a saved unconditional/conditional diffusion-transformer
wrapper checkpoint and `config.yaml` records the experiment overrides used to
create it. Reconstruct the matching raw `DiffusionTransformer` and
`DiffusionModel` before calling `load_weights`; see the parent artifact README
and `diffusion/models/` API documentation. The configuration is historical and
may require translating renamed constructor fields.
