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
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel to perform 3mm
__global__ void kernel_3m_0_kernel(double *d_A, double *d_B, double *d_E)
{
  int c1 = blockIdx.x * blockDim.x + threadIdx.x;
  int c2 = blockIdx.y * blockDim.y + threadIdx.y;

  if(c1 < NI && c2 < NJ) {
    for (int c5 = 0; c5 < NK; c5++) {
      d_E[c1 * NJ + c2] += d_A[c1 * NK + c5] * d_B[c5 * NJ + c2];
    }
  }
}

// Host function to launch the CUDA kernel
void kernel_3m_0(double A[NI][NK], double B[NK][NJ], double E[NI][NJ])
{
  double *d_A;
  double *d_B;
  double *d_E;

  cudaMalloc((void**)&d_A, NI * NK * sizeof(double));
  cudaMalloc((void**)&d_B, NK * NJ * sizeof(double));
  cudaMalloc((void**)&d_E, NI * NJ * sizeof(double));

  cudaMemcpy(d_A, A, NI * NK * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, NK * NJ * sizeof(double), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((NI + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (NJ + threadsPerBlock.y - 1) / threadsPerBlock.y);

  kernel_3m_0_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_E);

  cudaMemcpy(E, d_E, NI * NJ * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_E);
}