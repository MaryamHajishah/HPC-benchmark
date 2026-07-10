# HPC-benchmark — Parallelization of Matrix Multiplication (OpenMP · MPI · CUDA)

We take the sequential `C = A·B` (n×n doubles, `A=2`, `B=3`, so every
`C[i][j] == 6·n`) and make it fast three ways — **OpenMP** (multicore), **MPI**
(multi-process) and **CUDA** (GPU) — always measuring speedup/efficiency against
the **same best sequential time**. Grading is report-weighted: hot-spot analysis,
the iterative parallelization reasoning, and correct speedup / efficiency /
strong-scaling presentation.

## What each phase does

| Phase | Folder / script | Goal | Expect |
|---|---|---|---|
| 1 Machine specs | `scripts/specs.sh` | document the CPU/GPU | i9-12900K, 16c/24t, AVX2 (no AVX-512), 30 MB L3 |
| 2 Sequential | `seq/`, `run_seq.sh` | fastest 1-core version = baseline | i-k-j ≫ i-j-k; `-xHost` ~2×; one best block |
| 3 OpenMP | `omp/`, `run_omp.sh` | use all cores | near-linear to ~8 cores, then bandwidth wall |
| 4 MPI | `mpi/`, `run_mpi.sh` | use many processes | flattens earlier than OpenMP (comm overhead) |
| 5 CUDA | `cuda/` (Colab) | use the GPU | naive → tiled (big jump) → cuBLAS; FP64 ≈ 1/32 FP32 |


## Layout
```
seq/matmul_seq.c     sequential (i-k-j, -DNAIVE = i-j-k, optional blocking)
omp/matmul_omp.c     OpenMP (parallel outer loop, schedule(runtime))
mpi/matmul_mpi.c     MPI (row-block Scatterv/Bcast/Gatherv; -DP2P = Send/Recv)
cuda/matmul_cuda.cu  CUDA (naive / shared-memory tiled / cuBLAS; FP32 + FP64)
scripts/             specs.sh, run_seq.sh, run_omp.sh, run_mpi.sh
outputs/             redirect command output here (then send/commit)
report/              HPC-Report-TEMPLATE.docx (fill in tables/screenshots)
```
