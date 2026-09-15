"""Initialize exploratory notebooks from the repository's ``notebooks`` folder.

Importing this helper changes the process working directory to the repository
root resolved from this file, makes the project and thesis helpers importable,
and calls :func:`common.utils.init`, which attempts
to cap the first TensorFlow GPU logical device at 6,144 MiB. The module takes
no explicit input and returns no value; its effect is entirely process-global.
"""

import os
import sys

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "notebooks" / "thesis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.chdir(REPOSITORY_ROOT)


from common.utils import init as _initialize_runtime


_initialize_runtime()
