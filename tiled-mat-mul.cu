// Build: nvcc -O3 -std=c++17 tiled-mat-mul.cu -o tmm -lcublas
// Run:   ./tmm

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

// -------------------------
// CUDA / cuBLAS error macros
// -------------------------
#define CUDA_CHECK(stmt)                                                     \
do {                                                                         \
    cudaError_t err = (stmt);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA ERROR %s (%d): %s at %s:%d\n",                 \
                #stmt, int(err), cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

#define CUBLAS_CHECK(stmt)                                                   \
do {                                                                         \
    cublasStatus_t stat = (stmt);                                            \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                     \
        fprintf(stderr, "cuBLAS ERROR %s (%d) at %s:%d\n",                   \
                #stmt, int(stat), __FILE__, __LINE__);                       \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

// -------------------------
// ANSI colors
// -------------------------
#if defined(_WIN32)
  #define ANSI_GREEN ""
  #define ANSI_RED   ""
  #define ANSI_RESET ""
#else
  #define ANSI_GREEN "\x1b[32m"
  #define ANSI_RED   "\x1b[31m"
  #define ANSI_RESET "\x1b[0m"
#endif

// ################### DO NOT CHANGE ANYTHING ABOVE ###################

// ------------------------------------------------------------------
// CPU reference: C[MxN] = A[MxK] * B[KxN]   (row-major)
// ------------------------------------------------------------------
static void cpu_sgemm(float *a, float *b, float *c, const unsigned int M, const unsigned int N, const unsigned int K) {
    for (int m = 0; m < (int)M; m++) {
        for (int n = 0; n < (int)N; n++) {
            float psum = 0.0f;
            for (int i = 0; i < (int)K; i++) {
                psum += a[m * K + i] * b[i * N + n];
            }
            c[m * N + n] = psum;
        }
    }
}

// -------------------------
// Helper to init input data
// -------------------------
static void init_random(float* v, long long n, unsigned long long seed = 42ULL) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (long long i = 0; i < n; i++) v[i] = dist(gen);
}

static int verify_equal(const float* ref, const float* got, int count, float delta) {
    for (int i = 0; i < count; i++) {
        if (std::abs(ref[i] - got[i]) > delta) {
            fprintf(stderr, "Mismatch at %d: CPU=%g, GPU=%g\n", i, ref[i], got[i]);
            return 1;
        }
    }
    return 0;
}

// ------------------------------------------------------------------
// Simple naive GEMM: one C element per thread (row-major)
// ------------------------------------------------------------------
__global__ void simple_gemm(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* C,
                            const unsigned int M, const unsigned int N, const unsigned int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= (int)M || col >= (int)N) return;

    float sum = 0.0f;
    for (int i = 0; i < (int)K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

// ------------------------------------------------------------------
// Tiled shared-memory GEMM (row-major)
// ------------------------------------------------------------------
#define TILE 32
//tile_width?
__global__ void tiled_gemm_sm(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* C,
                              const unsigned int M, const unsigned int N, const unsigned int K) {

	__shared__ float A_Shared[TILE][TILE];
	__shared__ float B_Shared[TILE][TILE];

	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int tIdx = threadIdx.x;
	int tIdy = threadIdx.y;
// now block location anf thread location are managed

	int Row = blockY * blockDim.y + tIdy;
	int Column = blockX * blockDim.x + tIdx;

//same read-in location calculation from Matrix Addition

	float MatMulSum = 0.0;

	for (int p = 0; p < (K + TILE - 1)/ TILE; ++p){
        //loading data
	    //bounds checking
        if (Row < M && (p*TILE+tIdx) < K){
            A_Shared[tIdy][tIdx] = A[Row*K +p * TILE + tIdx];
        }
        else{
            A_Shared[tIdy][tIdx] = 0.0f;
        }

        if (Column < N && (p*TILE+tIdy) < K){
            B_Shared[tIdy][tIdx] = B[(p*TILE+tIdy)* N+Column];
        }
        else{
            B_Shared[tIdy][tIdx] = 0.0f;
        }
        __syncthreads();
		for (int k = 0; k < TILE; ++k){
		MatMulSum += A_Shared[tIdy][k] * B_Shared[k][tIdx];
	}
	__syncthreads();
	}

	if(Row < M&& Column< N) {
    C[Row * N + Column] = MatMulSum;
    }
       	// TODO: Implement tiled shared-memory matrix multiplication
}

// ------------------------------------------------------------------
// cuBLAS SGEMM (row-major) via transpose trick
// ------------------------------------------------------------------
static inline void cublas_sgemm_rowmajor(cublasHandle_t handle,
                                         const float* dA, const float* dB, float* dC,
                                         int M, int N, int K,
                                         float alpha, float beta) {
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        /*m=*/N, /*n=*/M, /*k=*/K,
        &alpha,
        /*A=*/dB, /*lda=*/N,
        /*B=*/dA, /*ldb=*/K,
        &beta,
        /*C=*/dC, /*ldc=*/N));
}

// ##################################################################

int main() {
    const int TESTNUM = 6;
    const int VERSIONS = 4; // CPU_ms, simple_gpu_ms, tiled_gpu_ms, cublas_ms
    const int repeat = 10;

    const unsigned int M_list[10] = {
      512, 129, 191, 255, 383, 511, 769,
      1025, 1537, 2049
    };

    const unsigned int N_list[10] = {
      512, 131, 193, 257, 385, 513, 771,
      1027, 1539, 2051
    };

    const unsigned int K_list[10] = {
      512, 1001, 1023, 1025, 1151, 1277,
      1537, 1799, 2049, 2305
    };

    double times_ms[TESTNUM][VERSIONS];
    std::vector<std::string> names;

    const int tile_dim = TILE; // fixed

    int rights = 0;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < TESTNUM; i++) {
        const unsigned int M = M_list[i];
        const unsigned int N = N_list[i];
        const unsigned int K = K_list[i];

        std::string name = std::to_string(M) + "-" + std::to_string(K) + "-" + std::to_string(N);
        names.push_back(name);

        // Host
        float *hA = new float[M * K];
        float *hB = new float[K * N];
        float *hC_ref = new float[M * N];
        float *hC_simple = new float[M * N];
        float *hC_tiled  = new float[M * N];
        float *hC_cublas = new float[M * N];

        init_random(hA, (long long)M * (long long)K, 1234ULL);
        init_random(hB, (long long)K * (long long)N, 5678ULL);

        // Device
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaMalloc((void **) &dB, K * N * sizeof(float));
        cudaMalloc((void **) &dA, M * K *sizeof(float));
        cudaMalloc((void **) &dC, M * N * sizeof(float));
        // TODO: Allocate device memory for dA, dB, dC
        // ...
        // TODO: Copy hA, hB to device memory
        // ...
       cudaMemcpy(dB, hB,  K * N * sizeof(float), cudaMemcpyHostToDevice);
       cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);

        // CPU reference
        auto cpu_start = std::chrono::steady_clock::now();
        cpu_sgemm(hA, hB, hC_ref, M, N, K);
        auto cpu_end = std::chrono::steady_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // TODO: Configure grid and block dimensions, use dim3 for block and grid
        // dim3 block(...);
        // dim3 grid(...); // Hint: you will need block.x and block.y
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

        // Simple kernel
        CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));
        for (int run = 0; run < repeat; run++) {
            simple_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float simple_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&simple_ms, start, stop));
        simple_ms /= repeat;
        // TODO: Copy dC to hC_simple
        // ...
        cudaMemcpy(hC_simple, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        // Tiled shared-memory kernel
        CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));
        for (int run = 0; run < repeat; run++) {
            tiled_gemm_sm<<<grid, block>>>(dA, dB, dC, M, N, K);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float tiled_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, start, stop));
        tiled_ms /= repeat;
        // TODO: Copy dC to hC_tiled
        cudaMemcpy(hC_tiled, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        // ...

        // cuBLAS
        cublas_sgemm_rowmajor(handle, dA, dB, dC, M, N, K, 1.0f, 0.0f); // warmup
        CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));
        for (int run = 0; run < repeat; run++) {
            cublas_sgemm_rowmajor(handle, dA, dB, dC, M, N, K, 1.0f, 0.0f);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float cublas_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
        cublas_ms /= repeat;
        CUDA_CHECK(cudaMemcpy(hC_cublas, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify
        int err_simple = verify_equal(hC_ref, hC_simple, (int)(M * N), 1e-3f);
        int err_tiled  = verify_equal(hC_ref, hC_tiled,  (int)(M * N), 1e-3f);
        int err_cublas = verify_equal(hC_ref, hC_cublas, (int)(M * N), 1e-3f);
        printf("[%s]: simple=%s, tiled=%s, cuBLAS=%s\n", name.c_str(),
               err_simple == 0 ? ANSI_GREEN "PASS" ANSI_RESET : ANSI_RED "FAIL" ANSI_RESET,
               err_tiled  == 0 ? ANSI_GREEN "PASS" ANSI_RESET : ANSI_RED "FAIL" ANSI_RESET,
               err_cublas == 0 ? ANSI_GREEN "PASS" ANSI_RESET : ANSI_RED "FAIL" ANSI_RESET);
        if (err_simple == 0 && err_tiled == 0 && err_cublas == 0) rights++;

        // Store times
        times_ms[i][0] = cpu_ms;
        times_ms[i][1] = simple_ms;
        times_ms[i][2] = tiled_ms;
        times_ms[i][3] = cublas_ms;


        // TODO: Free device memory for dA, dB, dC
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        // ...
        delete[] hA; delete[] hB; delete[] hC_ref; delete[] hC_simple; delete[] hC_tiled; delete[] hC_cublas;
    }

    printf("[%d/%d] %s\n", rights, TESTNUM, rights==TESTNUM?ANSI_GREEN "PASS" ANSI_RESET:ANSI_RED "FAIL" ANSI_RESET);
    printf("    %-16s %12s %14s %14s %12s\n",
       "M-K-N", "cpu_ms", "simple_gpu_ms", "tiled_gpu_ms", "cublas_ms");
    for (int i = 0; i < TESTNUM; ++i) {
        printf("    %-16s %12.3f %14.3f %14.3f %12.3f\n",
            names[i].c_str(),
            times_ms[i][0], times_ms[i][1], times_ms[i][2], times_ms[i][3]);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return EXIT_SUCCESS;
}
