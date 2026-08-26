# Sparse Sim

Sparse Sim is the trace-based simulator used in **SIMA: Scalable Scratchpad-Based Index-Matching Hardware Accelerator for Sparse Linear Algebra**. It models hash-based hardware temporary accumulators for sparse linear algebra kernels through a small set of index-matching data-movement instructions.

The goal of this simulator is to evaluate the data-driven performance trade-offs of the different memory models without the overheads of full-scale simulators like gem5.

## Repository Layout

```text
cpp/
	include/            Public C++ interfaces for matrices, kernels, models, statistics, configuration, and H3 hashing.
	src/                C++ implementation files.
		models/         SIMA, VIA, InnerSP, and ASA model implementations.
			common/     Shared model infrastructure, configuration, H3 hashing, and model-group logic.
python/
	sparse_sim/         Python CLI, matrix loading, archive extraction, and simulator runner.
lib/
	pybind11/           pybind11 dependency, kept as a project submodule.
```

## Build

Create or activate the Python environment you want to use, then install the package in editable mode:

```bash
python -m pip install -e .
```

This installs the `sparse-sim` command and the runtime dependencies. `numpy` is required by the Python runner. `scipy` is required to load SuiteSparse Matrix Market archives.

Initialize the pybind11 submodule:

```bash
git submodule update --init --recursive lib/pybind11
```

Build the C++ core and Python extension:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(which python)"
cmake --build build
```

Make sure the `python` used by CMake is the same interpreter from the environment where `sparse-sim`, `numpy`, and `scipy` are installed.

```bash
sparse-sim --help
```

## Matrices

Input matrices come from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/). Download matrices in Matrix Market archive form, typically as `.tar.gz` archives containing a `.mtx` file. Place those archives in any local cache directory; the simulator does not require a fixed location.

To download a matrix directly from SuiteSparse:

```bash
sparse-sim download-matrix 1 --matrix-cache data/matrix_storage
```

The downloader accepts a SuiteSparse matrix ID, matrix name, or `Group/name`. Downloaded archives are stored as `<ID>_<name>/<name>.tar.gz`.

The cache path passed to `--matrix-cache` should contain one directory per matrix:

```text
matrix_cache/
	1_1138_bus/
		1138_bus.tar.gz
```

Archives are extracted into a scratch directory while running. Extracted `.mtx` files are temporary artifacts and are deleted after loading.

## Running The Example Kernel

SpGEMM is included as an example kernel that emits the modeled instruction stream. It is useful for exercising the accumulator models on real sparse matrices, but the main simulator interface is the IDM instruction layer described below.

```bash
sparse-sim run \
  --matrix-cache path/to/matrix_cache \
  --scratch-dir tmp/extract \
  --out results/modulo \
  --h3-matrix modulo \
  1_1138_bus
```

Pass multiple matrix names to run a batch.

Each run writes one raw JSON file per matrix. Top-level fields describe the input matrix (`rows`, `cols`, `NNZ`). The `SpGEMM` object contains per-model `cycles`, instruction counts, reads, writes, utilization statistics, InnerSP checks, and the overflow flag.

## IDM Instructions

The modeled hardware mechanism is a hash-based temporary accumulator that stores `(index, value)` pairs. Kernels interact with it through IDM instructions; each model implements the same instruction interface and reports how many cycles it would take.

- `idm_ld`: read index-matched accumulator values. For each lane, return the stored value for that index or zero if the index is not resident.
- `idm_st`: update existing accumulator values or insert missing `(index, value)` pairs. Overflow is raised if the target bucket cannot accept a new pair.
- `idm_init`: bulk initialize the accumulator with `(index, value)` pairs. These lanes are counted as writes.
- `idm_evict_simd`: evict up to one valid pair per logical SIMD lane. Partial eviction order is model-dependent.
- `idm_evict`: evict all valid pairs.
- `idm_rst`: clear accumulator pairs and metadata.
- `idm_val`: return the number of valid pairs currently resident in the accumulator.

All accumulator models implement `IdmModel` in `cpp/include/idm_model.hpp`. Kernels call the wrapper in `cpp/src/models/common/idm_model_group.cpp`; the wrapper forwards each instruction to all active models, compares comparable load results, and accumulates per-model statistics.

## Adding Kernels

New kernels can be implemented as producers of IDM instructions. The intended path is:

1. Add a public kernel entry point in `cpp/include/`, for example `spmv.hpp`.
2. Add the implementation in `cpp/src/`, for example `spmv.cpp`.
3. Inside the kernel, create `KernelStats stats = make_stats_with_models(model_names())` and `IdmModelGroup models(create_default_models(h3_matrix))`.
4. Express the kernel's temporary-accumulator behavior using `models.idm_ld`, `models.idm_st`, `models.idm_init`, `models.idm_evict_simd`, `models.idm_evict`, `models.idm_rst`, and `models.idm_val`.
5. Keep arithmetic and sparse-format traversal in the kernel. Keep architecture-specific timing, banking, hashing, and overflow behavior inside the models.
6. At natural accumulator boundaries, sample utilization and reset model state.
7. Add pybind11 bindings in `cpp/src/bindings.cpp` and a small Python runner/CLI path if the kernel should be callable from Python.

The model registry is shared by all kernels. Adding a kernel should not require changing SIMA, VIA, InnerSP, or ASA unless the kernel needs an IDM instruction that is not currently modeled.

## Models

The active model variants are configured in `cpp/include/model_registry.hpp`:

- `sima_1`, `sima_2`, `sima_4`
- `via_4`, `via_16`
- `innersp_256`
- `asa`

Concrete implementations live in `cpp/src/models/`. Shared geometry and default simulator configuration live in `cpp/include/config.hpp` and `cpp/src/models/common/config.cpp`.

## H3 Hashing

SIMA and InnerSP use an H3 hash function to map sparse indices to accumulator buckets or banks. H3 hashing is binary matrix multiplication over bits; changing the hash function means selecting a different binary matrix.

Available CLI selections are:

- `modulo`: low-bit projection, equivalent to modulo for power-of-two bucket counts.
- `ovf`: mixes a second bucket-sized slice of input bits into the hash.
- `accel`: sliding-window H3 matrix.

The public H3 declarations are in `cpp/include/h3.hpp`. Matrix definitions and selection live in `cpp/src/models/common/h3_matrices.cpp`; the bitwise multiplication itself is implemented in `cpp/src/models/common/h3.cpp`.
