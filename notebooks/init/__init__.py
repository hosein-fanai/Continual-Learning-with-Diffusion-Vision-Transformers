"""Initialize exploratory notebooks from the repository's ``notebooks`` folder.

Importing this helper changes the process working directory to the repository
root, imports the local :mod:`autoencoder` and :mod:`diffusion` packages, and
calls :func:`common.utils.init`, which attempts to cap the first TensorFlow GPU
logical device at 6,144 MiB.  The module takes no explicit input and returns no
value; its effect is entirely process-global.  Import it only when the current
directory is ``notebooks`` because ``os.chdir("../")`` is relative to the
caller's current working directory.
"""

import os


os.chdir("../")


import autoencoder

import diffusion

from common.utils import init


init()
