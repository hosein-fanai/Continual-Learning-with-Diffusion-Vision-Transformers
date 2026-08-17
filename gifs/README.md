# Example animations

This directory contains sample diffusion animations retained for visual
comparison. New training runs normally write GIFs below `results/<project>/`
through `diffusion.callbacks.ImageGeneratorCallback` or
`common.utils.create_gif`.

`create_gif(output_path, images1, images2=None, duration=100, loop=0)` accepts a
sequence of grayscale image batches scaled to `[0, 1]`. Supplying `images2`
places corresponding frames below the first sequence with a separator. See the
function docstring for exact shapes and conversion behavior.
