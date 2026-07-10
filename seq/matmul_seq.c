/* matmul_seq.c -- Sequential dense matrix multiplication (HPC project baseline)
 *
 *   C = A x B,  with A[i][j] = 2.0 and B[i][j] = 3.0.
 *   Built-in correctness oracle: every C[i][j] must equal 6*n exactly
 *   (2*3 accumulated n times), so the code reports max|C[i][j] - 6n|.
 *
 *   Loop order is i-k-j (cache-friendly for row-major C: the inner j-loop is
 *   unit-stride on both C[i][j] and B[k][j], and A[i][k] is loop-invariant,
 *   so -O3 -xHost auto-vectorizes it).  Optional cache blocking (tiling).
 *   Build with -DNAIVE for the reference i-j-k order (column-wise, stride-n
 *   reads of B -> cache-hostile) to demonstrate the loop-order effect.
 *
 *   Only the multiply kernel is timed, with a monotonic wall clock
 *   (clock_gettime, NOT clock(): clock() returns summed CPU time, not wall time).
 *   Allocation, initialization and any I/O are excluded from the measurement.
 *
 *   Usage:  ./matmul_seq n [block_size] [reps]
 *     n           matrix dimension                       (required)
 *     block_size  0 = plain i-k-j (default); >0 = tiled with this tile size
 *     reps        number of timed repetitions            (default 3)
 *                 -> reports the MIN and MEDIAN kernel time
 *
 *   Build (lab i9-12900K, Intel compiler):
 *     icx -O3 -xHost -std=c11 matmul_seq.c -o matmul_seq
 *   Build (GCC):
 *     gcc -O3 -march=native -std=c11 matmul_seq.c -o matmul_seq
 */
#define _POSIX_C_SOURCE 200809L   /* expose clock_gettime / CLOCK_MONOTONIC under -std=c11 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *pa, const void *pb) {
    double a = *(const double *)pa, b = *(const double *)pb;
    return (a > b) - (a < b);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s n [block_size] [reps]\n", argv[0]);
        return 1;
    }
    const int n    = atoi(argv[1]);
    const int bs   = (argc > 2) ? atoi(argv[2]) : 0;   /* 0 => no blocking   */
    const int reps = (argc > 3) ? atoi(argv[3]) : 3;
    if (n <= 0 || reps <= 0) { fprintf(stderr, "invalid arguments\n"); return 1; }

    /* VLA-pointer allocation keeps the same 2-D indexing as the given code. */
    double (*a)[n] = malloc(sizeof(double[n][n]));
    double (*b)[n] = malloc(sizeof(double[n][n]));
    double (*c)[n] = malloc(sizeof(double[n][n]));
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed (need ~%.1f GB of RAM)\n",
                3.0 * (double)n * (double)n * sizeof(double) / 1e9);
        return 1;
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) { a[i][j] = 2.0; b[i][j] = 3.0; }

    double *times = malloc((size_t)reps * sizeof(double));

    for (int r = 0; r < reps; r++) {
        memset(c, 0, sizeof(double[n][n]));            /* C = 0, not timed */
        const double t0 = wtime();

#ifdef NAIVE
        /* Reference i-j-k: inner k-loop makes B[k][j] a stride-n column walk,
           cache-hostile. Built only with -DNAIVE for the loop-order comparison. */
        (void)bs;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    c[i][j] += a[i][k] * b[k][j];
#else
        if (bs <= 0) {                                 /* plain i-k-j */
            for (int i = 0; i < n; i++)
                for (int k = 0; k < n; k++) {
                    const double aik = a[i][k];
                    for (int j = 0; j < n; j++)
                        c[i][j] += aik * b[k][j];
                }
        } else {                                       /* tiled i-k-j */
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
#endif
        times[r] = wtime() - t0;
    }

    /* Correctness: exact integer result 6*n for every element. */
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

    printf("n=%d block=%d reps=%d | t_min=%.6f s  t_med=%.6f s | "
           "%.2f GFLOP/s | max_err=%.3g\n",
           n, bs, reps, t_min, t_med, gflops, max_err);

    free(a); free(b); free(c); free(times);
    return 0;
}
