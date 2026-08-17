# DiT classifier checkpoint

`model.weights.h5` is a saved joint diffusion/classification checkpoint and
`config.yaml` records its experiment overrides. Load it into a matching
`DiTClassifier` raw network wrapped by the corresponding diffusion classifier.
The configuration uses historical key casing and may require translating field
names to the current constructor API.
