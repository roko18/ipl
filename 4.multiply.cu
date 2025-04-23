// nvcc file_name.cu
// ./a.out

#include<iostream>
#include<cuda_runtime.h>
using namespace std;

__global__ void matrixMultiply(int *A, int *B, int *C, int rowsA, int colsA, int colsB) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(row < rowsA && col < colsB) {
		int sum = 0;
		for(int k = 0; k < colsA; k++) {
			sum += A[row * colsA + k] * B[k * colsB + col];
		}
		C[row * colsB + col] = sum;
	}
}

int main() {
	int rowsA = 10;
	int colsA = 10;
	int rowsB = colsA;
	int colsB = 10;
	
	int n_A = rowsA * colsA;
	int n_B = rowsB * colsB;
	int n_C = rowsA * colsB;
	
	int *h_A = new int[n_A];
	int *h_B = new int[n_B];
	int *h_C = new int[n_C];
	
	for(int i = 0; i < n_A; i++) h_A[i] = i;
	for(int i = 0; i < n_B; i++) h_B[i] = 2*i;
	
	int *d_A, *d_B, *d_C;
	size_t sizeA = n_A * sizeof(int);
	size_t sizeB = n_B * sizeof(int);
	size_t sizeC = n_C * sizeof(int);
	
	cudaMalloc(&d_A, sizeA);
	cudaMalloc(&d_B, sizeB);
	cudaMalloc(&d_C, sizeC);
	
	cudaMemcpy(d_A,h_A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B, sizeB, cudaMemcpyHostToDevice);
	
	dim3 blockDim(16,16);
	dim3 gridDim (
	(colsB + blockDim.x -1)/blockDim.x,
	(rowsA + blockDim.y -1)/blockDim.y
	);
	matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
	
	cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < 10; i++) {
		for(int j = 0; j < 10; j++) {
			cout << h_C[i * colsB + j] << " ";
		}
		cout << endl;
	}
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	
}