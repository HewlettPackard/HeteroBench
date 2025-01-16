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
__global__ void kernel_3m_1_kernel(double *d_C, double *d_D, double *d_F)
{
  int c1 = blockIdx.x * blockDim.x + threadIdx.x;
  int c2 = blockIdx.y * blockDim.y + threadIdx.y;

  if(c1 < NJ && c2 < NL)
  {
    for (int c5 = 0; c5 < NM; c5++) {
      d_F[c1 * NL + c2] += d_C[c1 * NM + c5] * d_D[c5 * NL + c2];
    }
  }
}

// Host function to launch the CUDA kernel
void kernel_3m_1(double C[NJ][NM], double D[NM][NL], double F[NJ][NL])
{
  double *d_C;
  double *d_D;
  double *d_F;

  cudaMalloc((void**)&d_C, NJ * NM * sizeof(double));
  cudaMalloc((void**)&d_D, NM * NL * sizeof(double));
  cudaMalloc((void**)&d_F, NJ * NL * sizeof(double));

  cudaMemcpy(d_C, C, NJ * NM * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, D, NM * NL * sizeof(double), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((NJ + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (NL + threadsPerBlock.y - 1) / threadsPerBlock.y);

  kernel_3m_1_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_D, d_F);

  cudaMemcpy(F, d_F, NJ * NL * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_C);
  cudaFree(d_D);
  cudaFree(d_F);
}
