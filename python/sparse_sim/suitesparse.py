from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


SUITESPARSE_INDEX_URL = "https://sparse.tamu.edu/files/ssstats.csv"


@dataclass(frozen=True)
class SuiteSparseMatrix:
    matrix_id: int
    group: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.matrix_id}_{self.name}"


def suitesparse_archive_url(matrix: SuiteSparseMatrix) -> str:
    return f"https://sparse.tamu.edu/MM/{quote(matrix.group)}/{quote(matrix.name)}.tar.gz"


def matrix_archive_path(matrix_cache: Path, matrix: SuiteSparseMatrix) -> Path:
    return Path(matrix_cache) / matrix.key / f"{matrix.name}.tar.gz"


def download_matrix(matrix_cache: Path, matrix: SuiteSparseMatrix) -> Path:
    target = matrix_archive_path(matrix_cache, matrix)
    if target.exists() and target.stat().st_size > 0:
        print(f"SKIP {matrix.key}: {target}", flush=True)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    url = suitesparse_archive_url(matrix)
    print(f"DOWNLOAD {matrix.key}: {url}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)
    print(f"WROTE {target} ({target.stat().st_size} bytes)", flush=True)
    return target


def resolve_suitesparse_matrix(identifier: str, index_path: Path | None = None) -> SuiteSparseMatrix:
    matrices = load_suitesparse_index(index_path)
    query = identifier.strip()
    if not query:
        raise ValueError("Matrix identifier cannot be empty")

    if query.isdigit():
        matrix_id = int(query)
        for matrix in matrices:
            if matrix.matrix_id == matrix_id:
                return matrix
        raise ValueError(f"No SuiteSparse matrix with id {matrix_id}")

    if "/" in query:
        group, name = query.split("/", 1)
        matches = [
            matrix
            for matrix in matrices
            if matrix.group.lower() == group.lower() and matrix.name.lower() == name.lower()
        ]
    else:
        matches = [matrix for matrix in matrices if matrix.name.lower() == query.lower()]

    if not matches:
        raise ValueError(f"No SuiteSparse matrix matching {identifier!r}")
    if len(matches) > 1:
        options = ", ".join(f"{matrix.group}/{matrix.name} (id {matrix.matrix_id})" for matrix in matches)
        raise ValueError(f"Matrix name {identifier!r} is ambiguous; use one of: {options}")
    return matches[0]


def load_suitesparse_index(index_path: Path | None = None) -> list[SuiteSparseMatrix]:
    if index_path is None:
        lines = _read_url_lines(SUITESPARSE_INDEX_URL)
    else:
        lines = Path(index_path).read_text(encoding="utf-8").splitlines()

    rows = [line for line in lines if line.strip() and not line.startswith("#")]
    if not rows:
        return []
    if rows[0].lower().startswith("id,"):
        return _parse_index_with_id_column(rows)

    if rows[0].isdigit():
        rows = rows[2:]

    matrices: list[SuiteSparseMatrix] = []
    for matrix_id, line in enumerate(rows, start=1):
        parts = line.split(",", 12)
        if len(parts) < 13:
            continue
        group, name, *_ = parts
        matrices.append(SuiteSparseMatrix(matrix_id, group.strip(), name.strip()))
    return matrices


def _parse_index_with_id_column(lines: list[str]) -> list[SuiteSparseMatrix]:
    matrices: list[SuiteSparseMatrix] = []
    for line in lines[1:]:
        parts = line.split(",", 13)
        if len(parts) < 3:
            continue
        matrix_id_s, group, name = parts[:3]
        try:
            matrix_id = int(matrix_id_s.strip())
        except ValueError:
            continue
        matrices.append(SuiteSparseMatrix(matrix_id, group.strip(), name.strip()))
    return matrices


def _read_url_lines(url: str) -> list[str]:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8").splitlines()
