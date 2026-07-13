/* kernel_bisect.c — find why the MPI local_mult runs at half the speed of the
 * sequential kernel, WITHOUT any MPI: both kernels compiled side by side in one
 * binary, same allocation, same timing.
 *
 *   icx -O3 -xHost -std=c11 kernel_bisect.c -o kernel_bisect     (lab)
 *   gcc -O3 -march=native -std=c11 kernel_bisect.c -o kernel_bisect
 *   ./kernel_bisect <n> <block> <reps>       e.g. ./kernel_bisect 2000 32 3
 *
 * Variants:
 *   seqvla  — the sequential kernel verbatim (VLA 2-D indexing, in a function)
 *   mpiflat — local_mult verbatim from matmul_mpi.c (flat + restrict + hoisted rows)
 *   mpivla  — local_mult but pointers cast to VLA form inside the function
 *   nomemset— mpiflat with the memset hoisted out of the timed region
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double wtime(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }

/* ---- variant 1: sequential kernel, VLA indexing (as in matmul_seq.c) ---- */
static void k_seqvla(int n, int bs, double (*a)[n], double (*b)[n], double (*c)[n]) {
    if (bs <= 0) {
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                const double aik = a[i][k];
                for (int j = 0; j < n; j++) c[i][j] += aik * b[k][j];
            }
    } else {
        for (int ii = 0; ii < n; ii += bs)
            for (int kk = 0; kk < n; kk += bs)
                for (int jj = 0; jj < n; jj += bs) {
                    const int imax = ii+bs<n?ii+bs:n, kmax = kk+bs<n?kk+bs:n, jmax = jj+bs<n?jj+bs:n;
                    for (int i = ii; i < imax; i++)
                        for (int k = kk; k < kmax; k++) {
                            const double aik = a[i][k];
                            for (int j = jj; j < jmax; j++) c[i][j] += aik * b[k][j];
                        }
                }
    }
}

/* ---- variant 2: local_mult VERBATIM from mpi/matmul_mpi.c ---- */
static void k_mpiflat(int rows, int n, int bs,
                      const double *restrict A, const double *restrict B, double *restrict C) {
    memset(C, 0, (size_t)rows * n * sizeof(double));
    if (bs <= 0) {
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < n; k++) {
                const double aik = A[(size_t)i * n + k];
                const double *restrict Brow = &B[(size_t)k * n];
                double *restrict Crow = &C[(size_t)i * n];
                for (int j = 0; j < n; j++) Crow[j] += aik * Brow[j];
            }
    } else {
        for (int ii = 0; ii < rows; ii += bs)
            for (int kk = 0; kk < n; kk += bs)
                for (int jj = 0; jj < n; jj += bs) {
                    const int imax = ii+bs<rows?ii+bs:rows, kmax = kk+bs<n?kk+bs:n, jmax = jj+bs<n?jj+bs:n;
                    for (int i = ii; i < imax; i++)
                        for (int k = kk; k < kmax; k++) {
                            const double aik = A[(size_t)i * n + k];
                            const double *restrict Brow = &B[(size_t)k * n];
                            double *restrict Crow = &C[(size_t)i * n];
                            for (int j = jj; j < jmax; j++) Crow[j] += aik * Brow[j];
                        }
                }
    }
}

/* ---- variant 3: same signature as local_mult, but VLA cast inside ---- */
static void k_mpivla(int rows, int n, int bs,
                     const double *restrict A, const double *restrict B, double *restrict C) {
    memset(C, 0, (size_t)rows * n * sizeof(double));
    const double (*a)[n] = (const double (*)[n])A;
    const double (*b)[n] = (const double (*)[n])B;
    double (*c)[n] = (double (*)[n])C;
    if (bs <= 0) {
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < n; k++) {
                const double aik = a[i][k];
                for (int j = 0; j < n; j++) c[i][j] += aik * b[k][j];
            }
    } else {
        for (int ii = 0; ii < rows; ii += bs)
            for (int kk = 0; kk < n; kk += bs)
                for (int jj = 0; jj < n; jj += bs) {
                    const int imax = ii+bs<rows?ii+bs:rows, kmax = kk+bs<n?kk+bs:n, jmax = jj+bs<n?jj+bs:n;
                    for (int i = ii; i < imax; i++)
                        for (int k = kk; k < kmax; k++) {
                            const double aik = a[i][k];
                            for (int j = jj; j < jmax; j++) c[i][j] += aik * b[k][j];
                        }
                }
    }
}

/* ---- variant 4: mpiflat without the memset inside the timed region ---- */
static void k_nomemset(int rows, int n, int bs,
                       const double *restrict A, const double *restrict B, double *restrict C) {
    if (bs <= 0) {
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < n; k++) {
                const double aik = A[(size_t)i * n + k];
                const double *restrict Brow = &B[(size_t)k * n];
                double *restrict Crow = &C[(size_t)i * n];
                for (int j = 0; j < n; j++) Crow[j] += aik * Brow[j];
            }
    } else {
        for (int ii = 0; ii < rows; ii += bs)
            for (int kk = 0; kk < n; kk += bs)
                for (int jj = 0; jj < n; jj += bs) {
                    const int imax = ii+bs<rows?ii+bs:rows, kmax = kk+bs<n?kk+bs:n, jmax = jj+bs<n?jj+bs:n;
                    for (int i = ii; i < imax; i++)
                        for (int k = kk; k < kmax; k++) {
                            const double aik = A[(size_t)i * n + k];
                            const double *restrict Brow = &B[(size_t)k * n];
                            double *restrict Crow = &C[(size_t)i * n];
                            for (int j = jj; j < jmax; j++) Crow[j] += aik * Brow[j];
                        }
                }
    }
}

static void report(const char *name, int n, double t, const double *C) {
    double maxe = 0, exp6n = 6.0 * n;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        double e = fabs(C[i] - exp6n); if (e > maxe) maxe = e;
    }
    printf("%-9s n=%d | t=%9.4f s | %7.2f GFLOP/s | max_err=%g\n",
           name, n, t, 2.0 * n * n * n / t / 1e9, maxe);
    fflush(stdout);
}

int main(int argc, char **argv) {
    const int n    = argc > 1 ? atoi(argv[1]) : 2000;
    const int bs   = argc > 2 ? atoi(argv[2]) : 32;
    const int reps = argc > 3 ? atoi(argv[3]) : 3;
    double *A = malloc((size_t)n*n*8), *B = malloc((size_t)n*n*8), *C = malloc((size_t)n*n*8);
    if (!A || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (size_t i = 0; i < (size_t)n*n; i++) { A[i] = 2.0; B[i] = 3.0; }
    printf("# n=%d block=%d reps=%d (min over reps reported)\n", n, bs, reps);

    double t;
    t = 1e30; for (int r = 0; r < reps; r++) { memset(C,0,(size_t)n*n*8);
        double t0=wtime(); k_seqvla(n,bs,(double(*)[n])A,(double(*)[n])B,(double(*)[n])C);
        double d=wtime()-t0; if(d<t)t=d; }                       report("seqvla",  n,t,C);
    t = 1e30; for (int r = 0; r < reps; r++) { double t0=wtime();
        k_mpiflat(n,n,bs,A,B,C); double d=wtime()-t0; if(d<t)t=d; } report("mpiflat", n,t,C);
    t = 1e30; for (int r = 0; r < reps; r++) { double t0=wtime();
        k_mpivla(n,n,bs,A,B,C);  double d=wtime()-t0; if(d<t)t=d; } report("mpivla",  n,t,C);
    t = 1e30; for (int r = 0; r < reps; r++) { memset(C,0,(size_t)n*n*8);
        double t0=wtime(); k_nomemset(n,n,bs,A,B,C);
        double d=wtime()-t0; if(d<t)t=d; }                        report("nomemset",n,t,C);
    return 0;
}
