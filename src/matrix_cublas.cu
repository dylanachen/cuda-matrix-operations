#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cublas_v2.h>

// Set up and launch kernel, measure execution time
int main(int argc, char **argv) {
    // Accepting matrix size from CLI argument    
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N <= 0) {
        fprintf(stderr, "Error: Matrix size N must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    // Check for CUDA-capable device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    // Get and print device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s\n", prop.name);
    printf("Matrix size: %d x %d\n", N, N);

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Set seed for reproducibility
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Create cuBLAS handle (tracks CUDA stream and state)
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS matrix multiplication: C = A * B
    float alpha = 1.0f;
    float beta = 0.0f;

    // Timing CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run and measure time
    cudaEventRecord(start);
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N);
    
    // Check for kernel launch errors if present
    cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate time elapsed, copy result back to host
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print execution time in seconds
    printf("cuBLAS execution time (N=%d): %.6f seconds (%.3f ms)\n", N, milliseconds / 1000.0f, milliseconds);

    // Clean up
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
