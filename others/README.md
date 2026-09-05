# Reference material

This directory contains background material used during development. The
diffusion-schedule catalog is conceptual reference material; the executable
and authoritative repository interface is `diffusion.schedulers`, documented
in its module docstrings and `diffusion/README.md`.

The catalog contains two mathematical naming errors: sub-VP diffusion is not
the linear interpolation used in rectified flow, and a scaled-linear beta
schedule is not the cosine schedule. For sub-VP, with
`B(t) = integral_0^t beta(s) ds`, the signal coefficient is `exp(-B(t)/2)`
and the marginal noise standard deviation is `1 - exp(-B(t))`; see the
[authors' SDE implementation](https://github.com/yang-song/score_sde/blob/main/sde_lib.py).
Scaled-linear betas interpolate their square roots and then square the result;
the [Diffusers implementation](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddpm.py)
keeps this separate from its cosine option. Preserve the original PDF as
historical reference, and use these corrected definitions in thesis writing.
