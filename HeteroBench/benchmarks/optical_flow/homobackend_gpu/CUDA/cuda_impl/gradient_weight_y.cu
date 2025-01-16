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

// CUDA Kernel to compute y weight
__global__ void gradient_weight_y_kernel(pixel_t *d_gradient_x, pixel_t *d_gradient_y, pixel_t *d_gradient_z, const pixel_t d_grad_filter[7], gradient_t *d_y_filtered)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;

  if(r >= 0 && r < MAX_HEIGHT + 3 && c >= 0 && c < MAX_WIDTH) {
    gradient_t acc;
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    if (r >= 6 && r < MAX_HEIGHT)
    { 
      for (int i = 0; i < 7; i ++)
      {
        acc.x += d_gradient_x[(r-i)*MAX_HEIGHT+c] * d_grad_filter[i];
        acc.y += d_gradient_y[(r-i)*MAX_HEIGHT+c] * d_grad_filter[i];
        acc.z += d_gradient_z[(r-i)*MAX_HEIGHT+c] * d_grad_filter[i];
      }
      d_y_filtered[(r-3)*MAX_HEIGHT+c] = acc;            
    }
    else if (r >= 3)
    {
      d_y_filtered[(r-3)*MAX_HEIGHT+c] = acc;
    }
  }
}

// Host function to launch the CUDA kernel
void gradient_weight_y(pixel_t *gradient_x, pixel_t *gradient_y, pixel_t *gradient_z, gradient_t *y_filtered)
{
  pixel_t *d_gradient_x;
  pixel_t *d_gradient_y;
  pixel_t *d_gradient_z;
  gradient_t *d_y_filtered;
  pixel_t *d_grad_filter;

  int size = MAX_HEIGHT*MAX_WIDTH*sizeof(pixel_t);
  int sizeFiltered = MAX_HEIGHT*MAX_WIDTH*sizeof(gradient_t);

  cudaMalloc(&d_gradient_x, size);
  cudaMalloc(&d_gradient_y, size);
  cudaMalloc(&d_gradient_z, size);
  cudaMalloc(&d_y_filtered, sizeFiltered);
  cudaMalloc(&d_grad_filter, 7*sizeof(pixel_t));

  cudaMemcpy(d_gradient_x, gradient_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gradient_y, gradient_y, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gradient_z, gradient_z, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_grad_filter, GRAD_FILTER, 7*sizeof(pixel_t), cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((MAX_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (MAX_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

  gradient_weight_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_gradient_x, d_gradient_y, d_gradient_z, d_grad_filter, d_y_filtered);
  
  cudaMemcpy(y_filtered, d_y_filtered, sizeFiltered, cudaMemcpyDeviceToHost);

  cudaFree(d_gradient_x);
  cudaFree(d_gradient_y);
  cudaFree(d_gradient_z);
  cudaFree(d_y_filtered);
  cudaFree(d_grad_filter);
}
