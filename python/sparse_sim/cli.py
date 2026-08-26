from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .matrix_io import extracted_matrix_market, load_matrix_market, resolve_archive
from .runner import run_spgemm
from .suitesparse import download_matrix, resolve_suitesparse_matrix


def _json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4)


def _run(args: argparse.Namespace) -> int:
    matrix_names = args.matrices
    if args.limit is not None:
        matrix_names = matrix_names[: args.limit]

    for matrix_name in matrix_names:
        archive = resolve_archive(matrix_name, Path(args.matrix_cache))
        with extracted_matrix_market(archive, Path(args.scratch_dir) if args.scratch_dir else None) as mtx:
            result = run_spgemm(load_matrix_market(mtx), h3_matrix=args.h3_matrix)
        _json_dump(Path(args.out) / f"{Path(matrix_name).stem}.json", result)
    return 0


def _download_matrix(args: argparse.Namespace) -> int:
    matrix = resolve_suitesparse_matrix(args.matrix, Path(args.index) if args.index else None)
    download_matrix(Path(args.matrix_cache), matrix)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparse-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run SpGEMM simulations")
    run_parser.add_argument("matrices", nargs="+", help="SuiteSparse matrix names, such as 1_1138_bus")
    run_parser.add_argument("--matrix-cache", default="data/matrix_storage")
    run_parser.add_argument("--scratch-dir", default=None)
    run_parser.add_argument("--out", default="results/suitesparse_16")
    run_parser.add_argument("--h3-matrix", choices=("modulo", "ovf", "accel"), default="modulo")
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.set_defaults(func=_run)

    download_parser = subparsers.add_parser("download-matrix", help="download a SuiteSparse Matrix Market archive")
    download_parser.add_argument("matrix", help="SuiteSparse matrix id, name, or Group/name")
    download_parser.add_argument("--index", default=None, help="optional local SuiteSparse ssstats.csv path")
    download_parser.add_argument("--matrix-cache", default="data/matrix_storage")
    download_parser.set_defaults(func=_download_matrix)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
