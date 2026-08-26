from __future__ import annotations

import os
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


class SimpleCsrMatrix:
    def __init__(self, indptr, indices, data, shape):
        self.indptr = np.asarray(indptr, dtype=np.int32)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = tuple(shape)
        self.nnz = int(self.data.size)

    def astype(self, dtype):
        return SimpleCsrMatrix(self.indptr, self.indices, self.data.astype(dtype), self.shape)

    def tocsr(self):
        return self

    def transpose(self):
        rows, cols = self.shape
        counts = np.zeros(cols, dtype=np.int32)
        for col in self.indices:
            counts[col] += 1

        indptr = np.zeros(cols + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])

        next_pos = indptr[:-1].copy()
        indices = np.zeros(self.nnz, dtype=np.int32)
        data = np.zeros(self.nnz, dtype=np.float32)

        for row in range(rows):
            for pos in range(self.indptr[row], self.indptr[row + 1]):
                col = self.indices[pos]
                dst = next_pos[col]
                indices[dst] = row
                data[dst] = self.data[pos]
                next_pos[col] += 1

        return SimpleCsrMatrix(indptr, indices, data, (cols, rows))


def default_scratch_dir() -> Path:
    base = os.environ.get("TMPDIR")
    if base:
        return Path(base) / "sparse_sim"
    return Path("tmp") / "sparse_sim"


def parse_result_stem(stem: str) -> tuple[str | None, str]:
    prefix, sep, name = stem.partition("_")
    if sep and prefix.isdigit():
        return prefix, name
    return None, stem


def resolve_archive(matrix: str, matrix_cache: Path) -> Path:
    matrix_key = Path(matrix).stem
    matrix_id, matrix_name = parse_result_stem(matrix_key)
    cache = Path(matrix_cache)

    candidates: list[Path] = [cache / matrix_key / f"{matrix_key}.tar.gz"]
    if matrix_id:
        candidates.append(cache / f"{matrix_id}_{matrix_name}" / f"{matrix_name}.tar.gz")
    candidates.append(cache / matrix_name / f"{matrix_name}.tar.gz")
    if matrix_key.isdigit():
        candidates.extend(sorted(cache.glob(f"{matrix_key}_*/*.tar.gz")))
    else:
        candidates.extend(sorted(cache.glob(f"*_{matrix_key}/{matrix_key}.tar.gz")))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve {matrix!r} under {cache}")


def _safe_member_path(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    root_resolved = root.resolve()
    if root_resolved != target and root_resolved not in target.parents:
        raise ValueError(f"Refusing to extract unsafe tar member {member_name!r}")
    return target


@contextmanager
def extracted_matrix_market(archive: Path, scratch_dir: Path | None = None) -> Iterator[Path]:
    scratch = Path(scratch_dir) if scratch_dir is not None else default_scratch_dir()
    scratch.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive) as tar:
        members = [member for member in tar.getmembers() if member.name.endswith(".mtx")]
        if not members:
            raise FileNotFoundError(f"No .mtx file found in {archive}")

        member = members[0]
        target = _safe_member_path(scratch, member.name)
        if target.exists():
            target.unlink()
        tar.extract(member, scratch)

    try:
        yield target
    finally:
        try:
            target.unlink()
        except FileNotFoundError:
            pass


def load_matrix_market(path: Path):
    try:
        from scipy.io import mmread
        from scipy.sparse import csr_matrix
    except ImportError:
        return _load_matrix_market_fallback(path)

    return csr_matrix(mmread(path))


def _load_matrix_market_fallback(path: Path) -> SimpleCsrMatrix:
    with Path(path).open("r", encoding="utf-8") as fp:
        header = fp.readline().strip().lower().split()
        if len(header) != 5 or header[:3] != ["%%matrixmarket", "matrix", "coordinate"]:
            raise ValueError(f"Unsupported Matrix Market header in {path}")

        field = header[3]
        if field not in {"real", "integer", "pattern"}:
            raise ValueError(f"Unsupported Matrix Market field {field!r}")

        symmetry = header[4]
        if symmetry not in {"general", "symmetric"}:
            raise ValueError(f"Unsupported Matrix Market symmetry {symmetry!r}")

        for line in fp:
            if not line.startswith("%"):
                rows, cols, _ = (int(x) for x in line.split()[:3])
                break
        else:
            raise ValueError(f"Matrix Market file {path} has no size line")

        entries: list[tuple[int, int, float]] = []
        for line in fp:
            if not line.strip() or line.startswith("%"):
                continue
            parts = line.split()
            if field == "pattern":
                if len(parts) < 2:
                    raise ValueError(f"Malformed Matrix Market entry in {path}: {line.rstrip()!r}")
                row_s, col_s = parts[:2]
                val = 1.0
            else:
                if len(parts) < 3:
                    raise ValueError(f"Malformed Matrix Market entry in {path}: {line.rstrip()!r}")
                row_s, col_s, val_s = parts[:3]
                val = float(val_s)
            row = int(row_s) - 1
            col = int(col_s) - 1
            entries.append((row, col, val))
            if symmetry == "symmetric" and row != col:
                entries.append((col, row, val))

    entries.sort(key=lambda item: (item[0], item[1]))
    indptr = np.zeros(rows + 1, dtype=np.int32)
    for row, _, _ in entries:
        indptr[row + 1] += 1
    np.cumsum(indptr, out=indptr)

    indices = np.fromiter((col for _, col, _ in entries), dtype=np.int32, count=len(entries))
    data = np.fromiter((val for _, _, val in entries), dtype=np.float32, count=len(entries))
    return SimpleCsrMatrix(indptr, indices, data, (rows, cols))
