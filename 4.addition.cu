// nvcc file_name.cu
// ./a.out

#include <iostream>
#include <cuda_runtime.h>

using namespace std;  // Added this line

#define N 1000000  // Size of the vectors

// CUDA Kernel Function for Vector Addition
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        C[index] = A[index] + B[index];
    }
}

int main() {
    // Host pointers
    float *h_A, *h_B, *h_C;

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on host
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    // Copy input vectors from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with appropriate number of blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the first 10 results
    for (int i = 0; i < 10; i++) {
        cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << endl;  // No need for std::
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
