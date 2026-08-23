"""Initialize exploratory notebooks from the repository's ``notebooks`` folder.

Importing this helper changes the process working directory to the repository
root resolved from this file, imports the local :mod:`autoencoder` and
:mod:`diffusion` packages, and calls :func:`common.utils.init`, which attempts
to cap the first TensorFlow GPU logical device at 6,144 MiB. The module takes
no explicit input and returns no value; its effect is entirely process-global.
"""

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
"""Absolute repository root used by notebook kernels."""


os.chdir(REPOSITORY_ROOT) # ../


import autoencoder

import diffusion

from common.utils import init


init()
