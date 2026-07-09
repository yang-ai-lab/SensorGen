"""Make the bundled model-implementation code importable by its module names.

The SensorGen inference code loads the released checkpoints with a model
implementation that lives in a sibling directory and uses bare-name imports
(``from models import ...``, ``from transport import ...``,
``from temporal_embedder import ...``). Putting that directory on ``sys.path``
lets ``sensorgen`` reuse the implementation as-is. This module is internal;
users interact only with :mod:`sensorgen.inference`.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
# Model-implementation directory (top-level, loaded onto sys.path at runtime).
_IMPL_DIR = os.path.join(REPO_ROOT, "Model")


def ensure_backbone_on_path():
    """Prepend the internal implementation dir to ``sys.path`` (idempotent)."""
    if not os.path.isdir(_IMPL_DIR):
        raise RuntimeError(
            f"Could not find the model implementation at {_IMPL_DIR}. "
            "Run sensorgen from inside the repository checkout."
        )
    if _IMPL_DIR not in sys.path:
        sys.path.insert(0, _IMPL_DIR)
    return _IMPL_DIR
