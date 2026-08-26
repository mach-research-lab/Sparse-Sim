from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _core_module():
    try:
        from . import _core
        return _core
    except ImportError as exc:
        package_root = Path(__file__).resolve().parents[2]
        candidates = sorted((package_root / "build").glob("**/_core*.so"))
        if candidates:
            spec = importlib.util.spec_from_file_location("sparse_sim._core", candidates[0])
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["sparse_sim._core"] = module
                spec.loader.exec_module(module)
                return module

        raise RuntimeError(
            "The sparse_sim C++ extension is not built. Run `cmake -S . -B build && cmake --build build`."
        ) from exc


def run_spgemm(matrix: Any, h3_matrix: str = "modulo") -> dict[str, Any]:
    """Run SpGEMM for ``matrix * matrix.T`` and return JSON-compatible data."""
    core = _core_module()

    m1 = matrix.astype(np.single).tocsr()
    m2 = m1.transpose().tocsr()

    spgemm = core.spgemm(
        np.asarray(m1.indptr, dtype=np.int32),
        np.asarray(m1.indices, dtype=np.int32),
        np.asarray(m1.data, dtype=np.float32),
        tuple(m1.shape),
        np.asarray(m2.indptr, dtype=np.int32),
        np.asarray(m2.indices, dtype=np.int32),
        np.asarray(m2.data, dtype=np.float32),
        tuple(m2.shape),
        h3_matrix=h3_matrix,
    )

    return {
        "rows": int(m1.shape[0]),
        "cols": int(m1.shape[1]),
        "NNZ": int(m1.nnz),
        "SpGEMM": spgemm,
    }
