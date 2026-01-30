#include <stdio.h>
#include <cuda_runtime.h>

// GPU Matrix Multiplication Kernel using Tiling, exposed to library
#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    // Declare shared memory for tiles
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

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
            ds_A[threadIdx.y][threadIdx.x] = A[Row * N + m * TILE_WIDTH + threadIdx.x];
        else
            ds_A[threadIdx.y][threadIdx.x] = 0.0f;
        // Load tile from matrix B
        if (Col < N && m * TILE_WIDTH + threadIdx.y < N)
            ds_B[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * N + Col];
        else
            ds_B[threadIdx.y][threadIdx.x] = 0.0f;

        // Make sure tiles are loaded
        __syncthreads();
        // Multiply the two tiles together
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += ds_A[threadIdx.y][k] * ds_B[k][threadIdx.x];
        }
        // Make sure computation is done before loading new tiles
        __syncthreads();
    }
    // Write the result to matrix C
    if (Row < N && Col < N) {
        C[Row * N + Col] = Pvalue;
    }
}

// Exposed C function for Python
extern "C" __declspec(dllexport) void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch  tiled matrix multiplication kernel, synchronize
    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
