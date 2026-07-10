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

Correctness is checked on **every** run: `max_err = max|C[i][j] − 6n|` must be `0`.
Only the multiply kernel is timed (`clock_gettime` / `omp_get_wtime` / `MPI_Wtime`
/ cudaEvents — never `clock()`); each run prints `t_min` (used for speedup).

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

## Lab command list (run one by one on the i9-12900K, keep every output)

```bash
# once per session
source /opt/intel/oneapi/setvars.sh        # enables icx / mpiicx (skip -> use gcc)
mkdir -p outputs

bash scripts/specs.sh   >> outputs/specs.txt   2>&1   # Phase 1
bash scripts/run_seq.sh >> outputs/run_seq.txt 2>&1   # Phase 2 (baseline)
bash scripts/run_omp.sh >> outputs/run_omp.txt 2>&1   # Phase 3
bash scripts/run_mpi.sh >> outputs/run_mpi.txt 2>&1   # Phase 4
```
Take a **terminal screenshot** of each run (the professor wants screenshots, not
just the files), then commit `outputs/` or send the `.txt` files.

## CUDA (Colab, not the lab)
Runtime → Change runtime type → **T4 GPU**, upload `cuda/matmul_cuda.cu`, then:
```bash
nvcc -O3 -arch=sm_75 cuda/matmul_cuda.cu -o matmul_cuda -lcublas
./matmul_cuda 5000            # then 10000
```

## Manual build
```bash
make            # icx: seq + omp        (make CC=gcc for gcc)
make mpi        # mpiicx/mpicc: mpi (collective + p2p)
```
