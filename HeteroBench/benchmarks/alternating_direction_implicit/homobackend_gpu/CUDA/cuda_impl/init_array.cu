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

// CUDA Kernel to init_array
__global__ void init_array_kernel(int n, double *d_X, double *d_A, double *d_B)
{
  int c1 = blockIdx.x * blockDim.x + threadIdx.x;
  int c2 = blockIdx.y * blockDim.y + threadIdx.y;

  if(c1 < n && c2 < n) {
      d_X[c1*n+c2] = (((double )c1) * (c2 + 1) + 1) / n;
      d_A[c1*n+c2] = (((double )c1) * (c2 + 2) + 2) / n;
      d_B[c1*n+c2] = (((double )c1) * (c2 + 3) + 3) / n;
  }

}

// Host function to launch the CUDA kernel
void init_array(int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0])
{
  if (n >= 1) {
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
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                        (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    init_array_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_X, d_A, d_B);

    cudaMemcpy(X, d_X, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_A);
    cudaFree(d_B);
  }
}