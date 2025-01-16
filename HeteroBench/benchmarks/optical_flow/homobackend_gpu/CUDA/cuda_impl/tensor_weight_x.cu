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
#include <cstdio>
#include <cuda_runtime.h>

// CUDA Kernel to compute tensor weight x
__global__ void tensor_weight_x_kernel(tensor_t *d_tensor_y, pixel_t d_tensor_filter[3], tensor_t *d_tensor)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;

  if(r >= 0 && r < MAX_HEIGHT && c >= 0 && c < MAX_WIDTH + 1) {
    tensor_t acc;
    for(int k = 0; k < 6; k++)
    {
      acc.val[k] = 0;
    }
    if (c >= 2 && c < MAX_WIDTH) 
    {
      for (int i = 0; i < 3; i ++)
      {
        for (int component = 0; component < 6; component ++)
        {
          acc.val[component] += d_tensor_y[r*MAX_HEIGHT+c-i].val[component] * d_tensor_filter[i];
        }
      }
    }
    if (c >= 1)
    {
      d_tensor[r*MAX_HEIGHT+c-1] = acc;
    }
  }
}

// Host function to launch the CUDA kernel
void tensor_weight_x(tensor_t *tensor_y,
                     tensor_t *tensor)
{
  tensor_t *d_tensor_y;
  tensor_t *d_tensor;
  pixel_t *d_tensor_filter;

  int size = MAX_HEIGHT*MAX_WIDTH*sizeof(tensor_t);

  cudaMalloc(&d_tensor_y, size);
  cudaMalloc(&d_tensor, size);
  cudaMalloc(&d_tensor_filter, 3*sizeof(pixel_t));

  cudaMemcpy(d_tensor_y, tensor_y, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_tensor_filter, TENSOR_FILTER, 3*sizeof(pixel_t), cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((MAX_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (MAX_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

  tensor_weight_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tensor_y, d_tensor_filter, d_tensor);
  
  cudaMemcpy(tensor, d_tensor, size, cudaMemcpyDeviceToHost);

  cudaFree(d_tensor_y);
  cudaFree(d_tensor);
  cudaFree(d_tensor_filter);
}