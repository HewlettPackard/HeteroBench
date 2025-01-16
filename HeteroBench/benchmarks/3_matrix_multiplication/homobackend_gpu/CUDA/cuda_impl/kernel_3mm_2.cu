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
__global__ void kernel_3m_2_kernel(double *d_E, double *d_F, double *d_G)
{
  int c1 = blockIdx.x * blockDim.x + threadIdx.x;
  int c2 = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(c1 < NI && c2 < NJ) 
  {
    for (int c6 = 0; c6 < NL; c6++) {
      d_G[c1 * NL + c2] += d_E[c1 * NJ + c6] * d_F[c6 * NL + c2];
    }
  }
}

// Host function to launch the CUDA kernel
void kernel_3m_2(double E[NI][NJ], double F[NJ][NL], double G[NI][NL])
{
  double *d_E;
  double *d_F;
  double *d_G;

  cudaMalloc((void**)&d_E, NI * NJ * sizeof(double));
  cudaMalloc((void**)&d_F, NJ * NL * sizeof(double));
  cudaMalloc((void**)&d_G, NI * NL * sizeof(double));

  cudaMemcpy(d_E, E, NI * NJ * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_F, F, NJ * NL * sizeof(double), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((NI + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (NL + threadsPerBlock.y - 1) / threadsPerBlock.y);

  kernel_3m_2_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_E, d_F, d_G);
  
  cudaMemcpy(G, d_G, NI * NL * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_E);
  cudaFree(d_F);
  cudaFree(d_G);
}
