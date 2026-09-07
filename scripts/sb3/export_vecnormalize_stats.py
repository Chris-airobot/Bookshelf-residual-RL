#!/usr/bin/env python3
"""Export numeric VecNormalize state while ignoring incompatible RNG pickle state."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


class _DiscardedBitGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _DiscardedGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _discarded_bit_generator_ctor(*args, **kwargs):
    return _DiscardedBitGenerator()


def _discarded_generator_ctor(*args, **kwargs):
    return _DiscardedGenerator()


class _StatsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy.random._pickle" and name == "__generator_ctor":
            return _discarded_generator_ctor
        if module == "sitecustomize" and name == "_compat_bit_generator_ctor":
            return _discarded_bit_generator_ctor
        if module == "numpy.random._pcg64" and name == "PCG64":
            return _DiscardedBitGenerator
        return super().find_class(module, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.input.open("rb") as stream:
        vec = _StatsUnpickler(stream).load()
    payload = {
        "obs_mean": np.asarray(vec.obs_rms.mean, dtype=np.float64),
        "obs_var": np.asarray(vec.obs_rms.var, dtype=np.float64),
        "obs_count": np.asarray(float(vec.obs_rms.count), dtype=np.float64),
    }
    if getattr(vec, "ret_rms", None) is not None:
        payload.update(
            ret_mean=np.asarray(vec.ret_rms.mean, dtype=np.float64),
            ret_var=np.asarray(vec.ret_rms.var, dtype=np.float64),
            ret_count=np.asarray(float(vec.ret_rms.count), dtype=np.float64),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print(f"wrote {args.output} obs_shape={payload['obs_mean'].shape} obs_count={float(payload['obs_count'])}")


if __name__ == "__main__":
    main()
