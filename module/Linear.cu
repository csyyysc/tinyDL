#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "Linear.cuh"

#define BLOCK_SIZE 32

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < M && t * BLOCK_SIZE + tx < N)
            shared_A[ty][tx] = A[row * N + t * BLOCK_SIZE + tx];
        else
            shared_A[ty][tx] = 0.0f;

        if (col < K && t * BLOCK_SIZE + ty < N)
            shared_B[ty][tx] = B[(t * BLOCK_SIZE + ty) * K + col];
        else
            shared_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
            sum += shared_A[ty][i] * shared_B[i][tx];

        __syncthreads();
    }

    if (row < M && col < K)
        C[row * K + col] = sum;
}



Linear::Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
    cudaMalloc(&d_weight, in_f * out_f * sizeof(float));
    cudaMalloc(&d_bias, out_f * sizeof(float));
    cudaMalloc(&d_grad_weight, in_f * out_f * sizeof(float));
    cudaMalloc(&d_grad_bias, out_f * sizeof(float));

    float* h_weight = (float*)malloc(in_f * out_f * sizeof(float));
    float* h_bias = (float*)malloc(out_f * sizeof(float));
    for (int i = 0; i < in_f * out_f; ++i) h_weight[i] = 1.0f;
    for (int i = 0; i < out_f; ++i) h_bias[i] = 0.0f;
    cudaMemcpy(d_weight, h_weight, in_f * out_f * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, out_f * sizeof(float), cudaMemcpyHostToDevice);
    free(h_weight);
    free(h_bias);
}

void Linear::forward(float* input, float* output, int batch_size) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * in_features * sizeof(float));
    cudaMalloc(&d_output, batch_size * out_features * sizeof(float));

    cudaMemcpy(d_input, input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_input, d_weight, d_output, batch_size, in_features, out_features);

    cudaMemcpy(output, d_output, batch_size * out_features * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void Linear::backward(float* input, float* grad_output, int batch_size) {
    float *d_input, *d_grad_output;
    cudaMalloc(&d_input, batch_size * in_features * sizeof(float));
    cudaMalloc(&d_grad_output, batch_size * out_features * sizeof(float));

    cudaMemcpy(d_input, input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, grad_output, batch_size * out_features * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSizeW((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridSizeB((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    // d_grad_weight = input^T * grad_output
    matrixMultiplyKernel<<<gridSizeW, blockSize>>>(d_input, d_grad_output, d_grad_weight, in_features, batch_size, out_features);

    // 簡化處理：將 d_grad_bias 設為 grad_output 的列總和（這裡略，需另外寫 kernel 做 sum reduction）

    cudaFree(d_input);
    cudaFree(d_grad_output);
}

Linear::~Linear() {
}