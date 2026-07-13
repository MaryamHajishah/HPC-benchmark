/* matmul_cuda.cu -- GPU dense matrix multiplication (HPC project, Phase 5, Colab T4)
 *
 *   C = A x B,  A=2, B=3  ->  every C[i][j] == 6*n  (oracle max|c-6n|).
 *   Because A and B are constant, C is symmetric, so row-major vs. column-major
 *   (cuBLAS) does not change the 6n oracle -- handy for a clean comparison.
 *
 *   Progression reported (for BOTH float and double):
 *     1. naive kernel            -- one thread per C element, all reads from global
 *                                   memory; swept over block 8x8 / 16x16 / 32x32
 *     2. shared-memory tiled     -- TILE x TILE block cooperatively cached in shared
 *                                   memory and reused (tile 16 and 32)
 *     3. cuBLAS Sgemm/Dgemm      -- vendor library = practical ceiling
 *
 *   Timing: cudaEvent around the kernel only; host<->device copies timed
 *   separately (for a single GEMM the transfer can dominate -- the offload's
 *   Amdahl serial part).  FP64 on the T4 runs at ~1/32 of FP32 (hardware).
 *
 *   Build (Colab / T4):  nvcc -O3 -arch=sm_75 matmul_cuda.cu -o matmul_cuda -lcublas
 *   Run:  ./matmul_cuda n [reps]
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* error-check macro. Internal var is _ck (NOT 'e') so it never collides with the
   cudaEvent_t variables named s/e that we pass into CK(...). */
#define CK(call) do { cudaError_t _ck=(call); if(_ck!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_ck)); exit(1);}}while(0)

/* ---------- kernels ---------- */
template <typename T>
__global__ void matmul_naive(const T* A, const T* B, T* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        T acc = 0;
        for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];
        C[row * n + col] = acc;
    }
}

template <typename T, int TILE>
__global__ void matmul_tiled(const T* A, const T* B, T* C, int n) {
    __shared__ T As[TILE][TILE];
    __shared__ T Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    T acc = 0;
    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : (T)0;
        Bs[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : (T)0;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = acc;
}

/* Register-blocked (2D thread-tiling) kernel. Each thread computes a TM x TN
   micro-tile of C held in REGISTERS. A BM x BK tile of A and BK x BN tile of B
   are staged in shared memory; each K-step loads TM values of A and TN of B into
   registers and does TM*TN FMAs -> reuse ratio (TM*TN)/(TM+TN) FMA per shared load,
   which is how GotoBLAS/CUTLASS approach peak (Goto & van de Geijn; "How To Write
   Fast Numerical Code"). Bounds are handled by zero-padding loads + guarded stores. */
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_reg(const T* A, const T* B, T* C, int n) {
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int threadCol = threadIdx.x;              /* 0 .. BN/TN-1 */
    const int threadRow = threadIdx.y;              /* 0 .. BM/TM-1 */
    const int nThreads  = (BM / TM) * (BN / TN);
    const int tid       = threadRow * (BN / TN) + threadCol;

    T acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (T)0;

    for (int kk = 0; kk < n; kk += BK) {
        for (int idx = tid; idx < BM * BK; idx += nThreads) {
            int r = idx / BK, c = idx % BK, gr = blockRow + r, gc = kk + c;
            As[r][c] = (gr < n && gc < n) ? A[(size_t)gr * n + gc] : (T)0;
        }
        for (int idx = tid; idx < BK * BN; idx += nThreads) {
            int r = idx / BN, c = idx % BN, gr = kk + r, gc = blockCol + c;
            Bs[r][c] = (gr < n && gc < n) ? B[(size_t)gr * n + gc] : (T)0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            T aReg[TM], bReg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[threadRow * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[k][threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gr = blockRow + threadRow * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gc = blockCol + threadCol * TN + j;
            if (gr < n && gc < n) C[(size_t)gr * n + gc] = acc[i][j];
        }
    }
}

/* ---------- cuBLAS overloads (C = A*B, column-major; symmetric so orientation is moot) ---------- */
static void gemm(cublasHandle_t h, int n, const float* A, const float* B, float* C) {
    float a = 1.f, b = 0.f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &a, B, n, A, n, &b, C, n);
}
static void gemm(cublasHandle_t h, int n, const double* A, const double* B, double* C) {
    double a = 1.0, b = 0.0;
    cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &a, B, n, A, n, &b, C, n);
}

/* ---------- helpers ---------- */
template <typename T>
static double check(const T* C, int n) {          /* max|C - 6n| over the whole matrix */
    double expected = 6.0 * n, maxe = 0.0;
    for (size_t t = 0; t < (size_t)n * n; ++t) {
        double e = fabs((double)C[t] - expected);
        if (e > maxe) maxe = e;
    }
    return maxe;
}
static float time_ms(cudaEvent_t s, cudaEvent_t e) { float ms; cudaEventElapsedTime(&ms, s, e); return ms; }

template <typename T>
static void run(int n, int reps, const char* tname) {
    size_t bytes = (size_t)n * n * sizeof(T);
    T *hA = (T*)malloc(bytes), *hB = (T*)malloc(bytes), *hC = (T*)malloc(bytes);
    for (size_t t = 0; t < (size_t)n * n; ++t) { hA[t] = (T)2; hB[t] = (T)3; }

    T *dA, *dB, *dC;
    CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));

    cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));

    /* ---- Host->Device copy (timed) ---- */
    CK(cudaEventRecord(s));
    CK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float h2d = time_ms(s, e);

    const double gf = 2.0 * (double)n * n * n / 1e9;     /* GFLOP (divide by seconds) */

    auto finish = [&](const char* name, float best_ms) {
        CK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        printf("cuda %-6s %-14s n=%d | kernel=%.3f ms | %.2f GFLOP/s | H2D=%.3f ms | max_err=%.3g\n",
               tname, name, n, best_ms, gf / (best_ms / 1e3), h2d, check(hC, n));
    };

    /* ---- naive, block 8/16/32 ---- */
    for (int bsz : {8, 16, 32}) {
        dim3 blk(bsz, bsz), grd((n + bsz - 1) / bsz, (n + bsz - 1) / bsz);
        float best = 1e30f;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(s));
            matmul_naive<T><<<grd, blk>>>(dA, dB, dC, n);
            CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
            CK(cudaGetLastError());
            best = fminf(best, time_ms(s, e));
        }
        char nm[24]; snprintf(nm, sizeof nm, "naive-b%d", bsz);
        finish(nm, best);
    }

    /* ---- tiled, TILE 16 and 32 ---- */
    {
        const int TILE = 16;
        dim3 blk(TILE, TILE), grd((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        float best = 1e30f;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(s));
            matmul_tiled<T, 16><<<grd, blk>>>(dA, dB, dC, n);
            CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
            CK(cudaGetLastError());
            best = fminf(best, time_ms(s, e));
        }
        finish("tiled-16", best);
    }
    {
        const int TILE = 32;
        dim3 blk(TILE, TILE), grd((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        float best = 1e30f;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(s));
            matmul_tiled<T, 32><<<grd, blk>>>(dA, dB, dC, n);
            CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
            CK(cudaGetLastError());
            best = fminf(best, time_ms(s, e));
        }
        finish("tiled-32", best);
    }

    /* ---- register-blocked (2D thread-tiling): micro-tile 4x4 and 8x8 ---- */
    {   /* 4x4: BM=BN=64, BK=8 -> 16x16=256 threads, 16 outputs/thread */
        dim3 blk(16, 16), grd((n + 63) / 64, (n + 63) / 64);
        float best = 1e30f;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(s));
            matmul_reg<T, 64, 64, 8, 4, 4><<<grd, blk>>>(dA, dB, dC, n);
            CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
            CK(cudaGetLastError());
            best = fminf(best, time_ms(s, e));
        }
        finish("reg-4x4", best);
    }
    {   /* 8x8: BM=BN=128, BK=8 -> 16x16=256 threads, 64 outputs/thread */
        dim3 blk(16, 16), grd((n + 127) / 128, (n + 127) / 128);
        float best = 1e30f;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(s));
            matmul_reg<T, 128, 128, 8, 8, 8><<<grd, blk>>>(dA, dB, dC, n);
            CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
            CK(cudaGetLastError());
            best = fminf(best, time_ms(s, e));
        }
        finish("reg-8x8", best);
    }

    /* ---- cuBLAS ---- */
    {
        cublasHandle_t h; cublasCreate(&h);
        float best = 1e30f;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(s));
            gemm(h, n, dA, dB, dC);
            CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
            best = fminf(best, time_ms(s, e));
        }
        cublasDestroy(h);
        finish("cublas", best);
    }

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s n [reps]\n", argv[0]); return 1; }
    int n = atoi(argv[1]);
    int reps = (argc > 2) ? atoi(argv[2]) : 3;
    if (n <= 0 || reps <= 0) { fprintf(stderr, "invalid arguments\n"); return 1; }

    int dev = 0; cudaDeviceProp p;
    CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&p, dev));
    printf("# GPU: %s  CC %d.%d  SMs=%d  globalMem=%.1f GB\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / 1e9);
    printf("# reps=%d  (report min kernel time; H2D shown once per precision)\n\n", reps);

    printf("=== FP32 (float) ===\n");   run<float>(n, reps, "fp32");
    printf("\n=== FP64 (double) ===\n"); run<double>(n, reps, "fp64");
    return 0;
}
