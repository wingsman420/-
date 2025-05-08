#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// 定义矩阵大小
#define N 1024
#define TILE_SIZE 16

// CUDA 核函数，利用Tensor Cores执行矩阵乘法
__global__ void matrixMultiply(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;

        // 使用Tensor Cores进行矩阵乘法
        #pragma unroll
        for (int i = 0; i < n; i += 8) {
            float a = __ldg(&A[row * n + i]);
            float b = __ldg(&B[i * n + col]);
            sum += a * b;
        }

        // 将结果写入全局内存
        C[row * n + col] = sum;
    }
}

int main() {
    // 分配和初始化矩阵 A, B, C
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    int matrixSize = N * N * sizeof(float);

    h_A = (float*)malloc(matrixSize);
    h_B = (float*)malloc(matrixSize);
    h_C = (float*)malloc(matrixSize);

    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);

    // 初始化矩阵数据（这里简化为随机数据）
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // 定义网格和块大小
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    // 启动CUDA核函数
    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    // 打印部分结果进行验证
    for (int i = 0; i < std::min(5, N); ++i) {
        for (int j = 0; j < std::min(5, N); ++j) {
            std::cout << h_C[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
