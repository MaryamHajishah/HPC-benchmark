/* matmul_omp.c -- OpenMP dense matrix multiplication (HPC project, Phase 3)
 *
 *   C = A x B, A=2.0, B=3.0  ->  every C[i][j] == 6*n (correctness oracle max|c-6n|).
 *
 *   Parallelizes the OUTERMOST loop only (the i-loop, or the ii block-loop): each
 *   thread owns distinct rows of C, so there is NO race and no atomics/locks.
 *   We deliberately do NOT collapse i,k -- multiple k for one i write the same
 *   C[i][j], which would be a race.
 *
 *     block_size == 0 : parallel i-k-j, NO blocking     (the "naive-parallel" step)
 *     block_size  > 0 : parallel cache-blocked i-k-j    (the tuned step)
 *     schedule(runtime): pick static|dynamic|guided via  OMP_SCHEDULE
 *     thread count     : via  OMP_NUM_THREADS
 *
 *   Timing: omp_get_wtime() (wall clock) over the kernel only -- NOT clock(),
 *   which returns summed CPU time and would inflate every parallel run.
 *
 *   Build (lab, Intel):  icx -O3 -xHost -qopenmp -std=c11 matmul_omp.c -o matmul_omp
 *   Build (GCC):         gcc -O3 -march=native -fopenmp -std=c11 matmul_omp.c -o matmul_omp
 *   Run:  OMP_NUM_THREADS=24 OMP_SCHEDULE=static ./matmul_omp n [block_size] [reps]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static int cmp_double(const void *pa, const void *pb) {
    double a = *(const double *)pa, b = *(const double *)pb;
    return (a > b) - (a < b);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s n [block_size] [reps]\n", argv[0]);
        return 1;
    }
    const int n    = 10000;
    const int bs   = (argc > 2) ? atoi(argv[2]) : 0;   /* 0 => no blocking */
    const int reps = (argc > 3) ? atoi(argv[3]) : 3;
    if (n <= 0 || reps <= 0) { fprintf(stderr, "invalid arguments\n"); return 1; }

    const int nthreads = omp_get_max_threads();
    const char *sched  = getenv("OMP_SCHEDULE");

    double (*a)[n] = malloc(sizeof(double[n][n]));
    double (*b)[n] = malloc(sizeof(double[n][n]));
    double (*c)[n] = malloc(sizeof(double[n][n]));
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed (need ~%.1f GB)\n",
                3.0 * (double)n * (double)n * sizeof(double) / 1e9);
        return 1;
    }

    /* Parallel first-touch init: each thread's pages are placed on its own NUMA
       node (matters on multi-socket; harmless on this single-socket i9). */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) { a[i][j] = 2.0; b[i][j] = 3.0; }

    double *times = malloc((size_t)reps * sizeof(double));

    for (int r = 0; r < reps; r++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) c[i][j] = 0.0;   /* not timed below */

        const double t0 = omp_get_wtime();

        if (bs <= 0) {                                   /* parallel i-k-j */
            #pragma omp parallel for schedule(runtime)
            for (int i = 0; i < n; i++)
                for (int k = 0; k < n; k++) {
                    const double aik = a[i][k];
                    for (int j = 0; j < n; j++)
                        c[i][j] += aik * b[k][j];
                }
        } else {                                         /* parallel blocked */
            #pragma omp parallel for schedule(runtime)
            for (int ii = 0; ii < n; ii += bs)
                for (int kk = 0; kk < n; kk += bs)
                    for (int jj = 0; jj < n; jj += bs) {
                        const int imax = ii + bs < n ? ii + bs : n;
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
        times[r] = omp_get_wtime() - t0;
    }

    const double expected = 6.0 * (double)n;
    double max_err = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            const double e = fabs(c[i][j] - expected);
            if (e > max_err) max_err = e;
        }

    qsort(times, (size_t)reps, sizeof(double), cmp_double);
    const double t_min = times[0];
    const double t_med = times[reps / 2];
    const double gflops = 2.0 * (double)n * (double)n * (double)n / t_min / 1e9;

    printf("omp n=%d block=%d threads=%d sched=%s reps=%d | "
           "t_min=%.6f s  t_med=%.6f s | %.2f GFLOP/s | max_err=%.3g\n",
           n, bs, nthreads, sched ? sched : "static(default)", reps,
           t_min, t_med, gflops, max_err);

    free(a); free(b); free(c); free(times);
    return 0;
}
