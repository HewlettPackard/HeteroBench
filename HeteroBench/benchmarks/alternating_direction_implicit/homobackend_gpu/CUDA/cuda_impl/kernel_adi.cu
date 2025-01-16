/*
 * (C) Copyright [2024] Hewlett Packard Enterprise Development LP
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the Software),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
 
#include "cuda_impl.h"
#include "omp.h"

using namespace std;

// CUDA Kernel to adi
__global__ void adi_loop1_kernel(double *d_X, double *d_A, double *d_B)
{
  int c2 = blockIdx.x * blockDim.x + threadIdx.x;

  if(c2 < N) {
    for (int c8 = 1; c8 <= N - 1; c8++) {
      d_B[c2*N+c8] -= d_A[c2*N+c8] * d_A[c2*N+c8] / d_B[c2*N+c8 - 1];
    }
    for (int c8 = 1; c8 <= N - 1; c8++) {
      d_X[c2*N+c8] -= d_X[c2*N+(c8 - 1)] * d_A[c2*N+c8] / d_B[c2*N+(c8 - 1)];
    }
    for (int c8 = 0; c8 <= N - 3; c8++) {
      d_X[c2*N+(N - c8 - 2)] = (d_X[c2*N+(N - 2 - c8)] - d_X[c2*N+(N - 2 - c8 - 1)] * d_A[c2*N+(N - c8 - 3)]) / d_B[c2*N+(N - 3 - c8)];
    }
  }
}

__global__ void adi_loop2_kernel(double *d_X, double *d_B)
{
  int c2 = blockIdx.x * blockDim.x + threadIdx.x;

  if(c2 < N) {
    d_X[c2*N+(N - 1)] = d_X[c2*N+(N - 1)] / d_B[c2*N+(N - 1)];
  }
}

__global__ void adi_loop3_kernel(double *d_X, double *d_A, double *d_B)
{
  int c2 = blockIdx.x * blockDim.x + threadIdx.x;

  if(c2 < N) {
    for (int c8 = 1; c8 < N; c8++) {
      d_B[c8*N+c2] -= d_A[c8*N+c2] * d_A[c8*N+c2] / d_B[(c8 - 1)*N+c2];
    }
    for (int c8 = 1; c8 < N; c8++) {
      d_X[c8*N+c2] -= d_X[(c8 - 1)*N+c2] * d_A[c8*N+c2] / d_B[(c8 - 1)*N+c2];
    }
    for (int c8 = 0; c8 < N - 2; c8++) {
      d_X[(N - 2 - c8)*N+c2] = (d_X[(N - 2 - c8)*N+c2] - d_X[(N - c8 - 3)*N+c2] * d_A[(N - 3 - c8)*N+c2]) / d_B[(N - 2 - c8)*N+c2];
    }
  }
}

__global__ void adi_loop4_kernel(double *d_X, double *d_B)
{
  int c2 = blockIdx.x * blockDim.x + threadIdx.x;

  if(c2 < N) {
    d_X[(N - 1)*N+c2] = d_X[(N - 1)*N+c2] / d_B[(N - 1)*N+c2];
  }
}


// Host function to launch the CUDA kernel
void kernel_adi(int tsteps,int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0])
{
  double *d_X;
  double *d_A;
  double *d_B;

  int size = N*N*sizeof(double);

  cudaMalloc(&d_X, size);
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);

  cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  for (int c0 = 0; c0 <= TSTEPS; c0++) {
    adi_loop1_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_A, d_B);

    cudaDeviceSynchronize();

    adi_loop2_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_B);
    
    cudaDeviceSynchronize();

    adi_loop3_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_A, d_B);
    
    cudaDeviceSynchronize();

    adi_loop4_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_B);
    
    cudaDeviceSynchronize();
  }
  
  cudaMemcpy(X, d_X, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

  cudaFree(d_X);
  cudaFree(d_A);
  cudaFree(d_B);
}