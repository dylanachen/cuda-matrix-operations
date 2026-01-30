#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

// GPU Matrix Multiplication using Tiling for better performance
#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    // Declare shared memory for tiles
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Calculate row and column indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calc row and column of elements + initialize sum
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0.0f;

    // Loop over the tiles of the input matrices
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        // Load tile from matrix A
        if (Row < N && m * TILE_WIDTH + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[Row * N + m * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        // Load tile from matrix B
        if (Col < N && m * TILE_WIDTH + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * N + Col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Make sure tiles are loaded
        __syncthreads();
        // Multiply the two tiles together
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        // Make sure computation is done before loading new tiles
        __syncthreads();
    }
    // Write the result to matrix C
    if (Row < N && Col < N) {
        C[Row * N + Col] = Pvalue;
    }
}

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
    
    // Allocate memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // Set seed for reproducibility
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }
    
    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Kernel configuration
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    
    // Timing CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Run and measure time
    cudaEventRecord(start);
    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    // Check for kernel launch errors if present
    cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate time elapsed, copy result back to host
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Print execution time in seconds
    printf("Tiled CUDA execution time (N=%d): %.6f seconds (%.3f ms)\n", N, milliseconds / 1000.0f, milliseconds);
    
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
