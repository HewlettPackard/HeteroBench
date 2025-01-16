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

// CUDA Kernel to compute x, y gradient
__global__ void gradient_xy_calc_kernel(pixel_t *d_frame2, pixel_t *d_gradient_x, int d_grad_weights[7], pixel_t *d_gradient_y)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;

  if(r >= 0 && r < MAX_HEIGHT + 2 && c >= 0 && c < MAX_WIDTH + 2) {
    pixel_t x_grad = 0;
    pixel_t y_grad = 0;
    if (r >= 4 && r < MAX_HEIGHT && c >= 4 && c < MAX_WIDTH)
    {
      for (int i = 0; i < 5; i++)
      {
        x_grad += d_frame2[(r-2)*MAX_HEIGHT+c-i] * d_grad_weights[4-i];
        y_grad += d_frame2[(r-i)*MAX_HEIGHT+c-2] * d_grad_weights[4-i];
      }
      d_gradient_x[(r-2)*MAX_HEIGHT+c-2] = x_grad / 12;
      d_gradient_y[(r-2)*MAX_HEIGHT+c-2] = y_grad / 12;
    }
    else if (r >= 2 && c >= 2)
    {
      d_gradient_x[(r-2)*MAX_HEIGHT+c-2] = 0;
      d_gradient_y[(r-2)*MAX_HEIGHT+c-2] = 0;
    }
  }
}


// Host function to launch the CUDA kernel
void gradient_xy_calc(pixel_t *frame2, pixel_t *gradient_x, pixel_t *gradient_y)
{
  pixel_t *d_frame2;
  pixel_t *d_gradient_x;
  pixel_t *d_gradient_y;
  int *d_grad_weights;

  int size = MAX_HEIGHT * MAX_WIDTH * sizeof(pixel_t);

  cudaMalloc(&d_frame2, size);
  cudaMalloc(&d_gradient_x, size);
  cudaMalloc(&d_gradient_y, size);
  cudaMalloc(&d_grad_weights, 5*sizeof(int));

  cudaMemcpy(d_frame2, frame2, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gradient_x, gradient_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_grad_weights, GRAD_WEIGHTS, size, cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((MAX_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (MAX_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

  gradient_xy_calc_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_frame2, d_gradient_x, d_grad_weights, d_gradient_y);
  
  cudaMemcpy(gradient_y, d_gradient_y, size, cudaMemcpyDeviceToHost);

  cudaFree(d_frame2);
  cudaFree(d_gradient_x);
  cudaFree(d_gradient_y);
  cudaFree(d_grad_weights);
}
