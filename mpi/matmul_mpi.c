/* matmul_mpi.c -- Distributed-memory dense matrix multiplication (HPC project, Phase 4)
 *
 *   C = A x B,  A=2.0, B=3.0  ->  every C[i][j] == 6*n  (oracle max|c-6n|).
 *
 *   Decomposition: 1-D ROW BLOCK.  Rank 0 owns the full A and B, scatters the
 *   rows of A across the ranks (Scatterv, handles n % P != 0), broadcasts the
 *   whole B (Bcast), every rank multiplies its row-stripe of A by B to get its
 *   stripe of C, and rank 0 gathers the stripes back (Gatherv).  The local
 *   multiply is the same cache-friendly i-k-j kernel as the sequential/OpenMP
 *   versions, with optional cache blocking.
 *
 *   Two communication back-ends (for the "collectives vs. point-to-point" report
 *   section) selected at compile time:
 *     default : MPI_Scatterv / MPI_Bcast / MPI_Gatherv     (collectives)
 *     -DP2P   : rank 0 hand-codes the same data movement with MPI_Send/MPI_Recv
 *
 *   Timing (MPI_Wtime, wall clock): per repetition we measure
 *     t_comm  = scatter(A) + bcast(B) + gather(C)
 *     t_comp  = local multiply only
 *     t_tot   = t_comm + t_comp
 *   and report, over `reps`, the MIN t_tot (the honest parallel time used for
 *   speedup) together with the matching comm/comp split.  Only rank 0 prints.
 *
 *   Build (lab, Intel):  mpiicx -O3 -xHost -std=c11 matmul_mpi.c -o matmul_mpi
 *   Build (GCC/OpenMPI): mpicc  -O3 -march=native -std=c11 matmul_mpi.c -o matmul_mpi
 *   P2P variant:         add  -DP2P
 *   Run:  mpirun -np <P> ./matmul_mpi n [block_size] [reps]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

static int cmp_double(const void *pa, const void *pb) {
    double a = *(const double *)pa, b = *(const double *)pb;
    return (a > b) - (a < b);
}

/* Local kernel: C_local[rows][n] = A_local[rows][n] * B[n][n], i-k-j, optional blocking.
 * restrict: Aloc, B, Cloc are separate allocations that never overlap, so icx can
 * vectorize the inner FMA. Without restrict, icx assumes they may alias and the kernel
 * runs ~2x slower than the sequential/OpenMP versions (whose a/b/c are distinct VLAs). */
#ifdef KERNEL_VLA
/* Candidate fix for the remaining ~2x gap vs the sequential kernel under icx:
 * identical loop-nest FORM to matmul_seq.c (VLA 2-D indexing, no per-k hoisted
 * restrict pointers), so icx can apply the same outer-loop transformations
 * (unroll-and-jam over k) it applies to the sequential nest.
 * Build:  MPIXHOST=-xCORE-AVX2 mpiicx -O3 -xCORE-AVX2 -DKERNEL_VLA ...
 * Select this ONLY if scripts/kernel_bisect.c shows mpivla fast and mpiflat slow. */
static void local_mult(int rows, int n, int bs,
                       const double *restrict A, const double *restrict B, double *restrict C) {
    memset(C, 0, (size_t)rows * n * sizeof(double));
    const double (*a)[n] = (const double (*)[n])A;
    const double (*b)[n] = (const double (*)[n])B;
    double (*c)[n] = (double (*)[n])C;
    if (bs <= 0) {                                   /* plain i-k-j */
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < n; k++) {
                const double aik = a[i][k];
                for (int j = 0; j < n; j++)
                    c[i][j] += aik * b[k][j];
            }
    } else {                                         /* tiled i-k-j */
        for (int ii = 0; ii < rows; ii += bs)
            for (int kk = 0; kk < n; kk += bs)
                for (int jj = 0; jj < n; jj += bs) {
                    const int imax = ii + bs < rows ? ii + bs : rows;
                    const int kmax = kk + bs < n ? kk + bs : n;
                    const int jmax = jj + bs < n ? jj + bs : n;
                    for (int i = ii; i < imax; i++)
                        for (int k = kk; k < kmax; k++) {
                            const double aik = a[i][k];
                            for (int j = jj; j < jmax; j++)
                                c[i][j] += aik * b[k][j];
                        }
                }
    }
}
#else
static void local_mult(int rows, int n, int bs,
                       const double *restrict A, const double *restrict B, double *restrict C) {
    memset(C, 0, (size_t)rows * n * sizeof(double));
    if (bs <= 0) {                                   /* plain i-k-j */
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < n; k++) {
                const double aik = A[(size_t)i * n + k];
                const double *restrict Brow = &B[(size_t)k * n];
                double *restrict Crow = &C[(size_t)i * n];
                for (int j = 0; j < n; j++)
                    Crow[j] += aik * Brow[j];
            }
    } else {                                         /* tiled i-k-j */
        for (int ii = 0; ii < rows; ii += bs)
            for (int kk = 0; kk < n; kk += bs)
                for (int jj = 0; jj < n; jj += bs) {
                    const int imax = ii + bs < rows ? ii + bs : rows;
                    const int kmax = kk + bs < n ? kk + bs : n;
                    const int jmax = jj + bs < n ? jj + bs : n;
                    for (int i = ii; i < imax; i++)
                        for (int k = kk; k < kmax; k++) {
                            const double aik = A[(size_t)i * n + k];
                            const double *restrict Brow = &B[(size_t)k * n];
                            double *restrict Crow = &C[(size_t)i * n];
                            for (int j = jj; j < jmax; j++)
                                Crow[j] += aik * Brow[j];
                        }
                }
    }
}
#endif /* KERNEL_VLA */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s n [block_size] [reps]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const int n    = atoi(argv[1]);
    const int bs   = (argc > 2) ? atoi(argv[2]) : 0;
    const int reps = (argc > 3) ? atoi(argv[3]) : 3;
    if (n <= 0 || reps <= 0) {
        if (rank == 0) fprintf(stderr, "invalid arguments\n");
        MPI_Finalize();
        return 1;
    }

    /* Row distribution: rank r gets rows_cnt[r] rows starting at row_displ[r]. */
    int *rows_cnt   = malloc(nprocs * sizeof(int));   /* rows per rank            */
    int *sendcnt    = malloc(nprocs * sizeof(int));   /* elements per rank (rows*n)*/
    int *senddispl  = malloc(nprocs * sizeof(int));   /* element offset per rank  */
    int base = n / nprocs, rem = n % nprocs, off = 0;
    for (int r = 0; r < nprocs; r++) {
        rows_cnt[r]  = base + (r < rem ? 1 : 0);
        sendcnt[r]   = rows_cnt[r] * n;
        senddispl[r] = off;
        off += sendcnt[r];
    }
    const int my_rows = rows_cnt[rank];

    /* Allocations: rank 0 holds full A,B,C; every rank holds full B + local A,C. */
    double *A = NULL, *C = NULL;                       /* full matrices (rank 0)   */
    double *B      = malloc((size_t)n * n * sizeof(double));
    double *Aloc   = malloc((size_t)my_rows * n * sizeof(double));
    double *Cloc   = malloc((size_t)my_rows * n * sizeof(double));
    if (!B || !Aloc || !Cloc) { fprintf(stderr, "rank %d: alloc failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }

    if (rank == 0) {
        A = malloc((size_t)n * n * sizeof(double));
        C = malloc((size_t)n * n * sizeof(double));
        if (!A || !C) { fprintf(stderr, "rank 0: alloc failed (need ~%.1f GB)\n",
                                3.0 * (double)n * n * sizeof(double) / 1e9);
                        MPI_Abort(MPI_COMM_WORLD, 1); }
        for (size_t t = 0; t < (size_t)n * n; t++) { A[t] = 2.0; B[t] = 3.0; }
    }

    double best_tot = 1e30, best_comm = 0.0, best_comp = 0.0;

    for (int r = 0; r < reps; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();

        /* ---- distribute A (rows) and B (whole) ---- */
#ifdef P2P
        if (rank == 0) {
            memcpy(Aloc, A, (size_t)my_rows * n * sizeof(double));   /* own slice */
            for (int p = 1; p < nprocs; p++)
                MPI_Send(A + senddispl[p], sendcnt[p], MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(Aloc, my_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
#else
        MPI_Scatterv(A, sendcnt, senddispl, MPI_DOUBLE,
                     Aloc, my_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        const double t1 = MPI_Wtime();

        /* ---- local multiply ---- */
        local_mult(my_rows, n, bs, Aloc, B, Cloc);

        const double t2 = MPI_Wtime();

        /* ---- gather C ---- */
#ifdef P2P
        if (rank == 0) {
            memcpy(C, Cloc, (size_t)my_rows * n * sizeof(double));
            for (int p = 1; p < nprocs; p++)
                MPI_Recv(C + senddispl[p], sendcnt[p], MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Send(Cloc, my_rows * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
#else
        MPI_Gatherv(Cloc, my_rows * n, MPI_DOUBLE,
                    C, sendcnt, senddispl, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        const double t3 = MPI_Wtime();

        /* reduce the slowest rank's compute time; comm is measured on rank 0 path */
        double comp_local = t2 - t1, comp_max;
        MPI_Reduce(&comp_local, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            const double comm = (t1 - t0) + (t3 - t2);
            const double tot  = comm + comp_max;
            if (tot < best_tot) { best_tot = tot; best_comm = comm; best_comp = comp_max; }
        }
    }

    /* Correctness on rank 0 over the full matrix. */
    if (rank == 0) {
        const double expected = 6.0 * (double)n;
        double max_err = 0.0;
        for (size_t t = 0; t < (size_t)n * n; t++) {
            const double e = fabs(C[t] - expected);
            if (e > max_err) max_err = e;
        }
        const double gflops = 2.0 * (double)n * n * n / best_tot / 1e9;
#ifdef P2P
        const char *mode = "p2p";
#else
        const char *mode = "collective";
#endif
        printf("mpi n=%d block=%d procs=%d mode=%s reps=%d | "
               "t_tot=%.6f s (comm=%.6f comp=%.6f, comm%%=%.1f) | %.2f GFLOP/s | max_err=%.3g\n",
               n, bs, nprocs, mode, reps,
               best_tot, best_comm, best_comp,
               100.0 * best_comm / best_tot, gflops, max_err);
    }

    free(rows_cnt); free(sendcnt); free(senddispl);
    free(B); free(Aloc); free(Cloc);
    if (rank == 0) { free(A); free(C); }
    MPI_Finalize();
    return 0;
}
