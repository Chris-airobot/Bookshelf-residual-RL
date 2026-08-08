"""Compatibility shims loaded automatically when source/bookshelf is on PYTHONPATH.

This lets SB3 checkpoints saved with newer NumPy load inside older Isaac Sim
containers used on HPC.
"""

import importlib
import sys

try:
    import numpy.core as _np_core

    sys.modules.setdefault("numpy._core", _np_core)

    for _name in [
        "_multiarray_umath",
        "multiarray",
        "umath",
        "numeric",
        "fromnumeric",
        "arrayprint",
        "records",
        "memmap",
        "function_base",
        "_methods",
        "_exceptions",
        "_ufunc_config",
        "getlimits",
        "shape_base",
        "numerictypes",
    ]:
        try:
            _mod = importlib.import_module(f"numpy.core.{_name}")
            sys.modules.setdefault(f"numpy._core.{_name}", _mod)
        except Exception:
            pass
except Exception:
    pass

try:
    import numpy.random._pickle as _np_random_pickle

    _orig_bit_generator_ctor = _np_random_pickle.__bit_generator_ctor

    def _compat_bit_generator_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__

        if isinstance(bit_generator_name, str):
            if "PCG64DXSM" in bit_generator_name:
                bit_generator_name = "PCG64DXSM"
            elif "PCG64" in bit_generator_name:
                bit_generator_name = "PCG64"
            elif "MT19937" in bit_generator_name:
                bit_generator_name = "MT19937"
            elif "Philox" in bit_generator_name:
                bit_generator_name = "Philox"
            elif "SFC64" in bit_generator_name:
                bit_generator_name = "SFC64"

        return _orig_bit_generator_ctor(bit_generator_name)

    _np_random_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
except Exception:
    pass
