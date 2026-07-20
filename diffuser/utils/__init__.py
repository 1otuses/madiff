from .arrays import *
from .bc_evaluator import BCEvaluator
from .bc_training import *
from .colab import *
from .config import *
from .data_encoder import *
from .evaluator import MADEvaluator
from .mamujoco_rendering import MAMuJoCoRenderer
from .mpe_rendering import MPERenderer
from .offline_evaluator import MADOfflineEvaluator
from .progress import *
from .serialization import *
from .setup import *
from .smac_rendering import SMACRenderer
from .training import *


_LAZY_IMPORTS = {
    "MAHalfCheetahRenderer": (
        "diffuser.utils.mahalfcheetah_rendering",
        "MAHalfCheetahRenderer",
    ),
    "MuJoCoRenderer": ("diffuser.utils.rendering", "MuJoCoRenderer"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)

    import importlib

    module_name, attr_name = _LAZY_IMPORTS[name]
    attr = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = attr
    return attr
