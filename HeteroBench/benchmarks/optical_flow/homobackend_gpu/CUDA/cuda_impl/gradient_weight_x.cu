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

// CUDA Kernel to compute x weight
__global__ void gradient_weight_x_kernel(gradient_t *d_y_filtered, const pixel_t d_grad_filter[7], gradient_t *d_filtered_gradient)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;

  if(r >= 0 && r < MAX_HEIGHT && c >= 0 && c < MAX_WIDTH + 3) {
    gradient_t acc;
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    if (c >= 6 && c < MAX_WIDTH)
    {
      for (int i = 0; i < 7; i++)
      {
        acc.x += d_y_filtered[r*MAX_HEIGHT+c-i].x * d_grad_filter[i];
        acc.y += d_y_filtered[r*MAX_HEIGHT+c-i].y * d_grad_filter[i];
        acc.z += d_y_filtered[r*MAX_HEIGHT+c-i].z * d_grad_filter[i];
      }
      d_filtered_gradient[r*MAX_HEIGHT+c-3] = acc;
    }
    else if (c >= 3)
    {
      d_filtered_gradient[r*MAX_HEIGHT+c-3] = acc;
    }
  }
}

// Host function to launch the CUDA kernel
void gradient_weight_x(gradient_t *y_filtered, gradient_t *filtered_gradient)
{
  gradient_t *d_y_filtered;
  gradient_t *d_filtered_gradient;
  pixel_t *d_grad_filter; 

  int size = MAX_HEIGHT * MAX_WIDTH * sizeof(gradient_t);

  cudaMalloc(&d_y_filtered, size);
  cudaMalloc(&d_filtered_gradient, size);
  cudaMalloc(&d_grad_filter, 7*sizeof(pixel_t));

  cudaMemcpy(d_y_filtered, y_filtered, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_grad_filter, GRAD_FILTER, 7*sizeof(pixel_t), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((MAX_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (MAX_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

  gradient_weight_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_y_filtered, d_grad_filter, d_filtered_gradient);

  cudaMemcpy(filtered_gradient, d_filtered_gradient, size, cudaMemcpyDeviceToHost);

  cudaFree(d_y_filtered);
  cudaFree(d_filtered_gradient);
}