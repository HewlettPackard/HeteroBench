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

// CUDA Kernel to compute compute z gradient
__global__ void gradient_z_calc_kernel(pixel_t *d_frame0, pixel_t *d_frame1, pixel_t *d_frame2, pixel_t *d_frame3, pixel_t *d_frame4, int grad_weights[5], pixel_t *d_gradient_z)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;

  if(r >= 0 && r < MAX_HEIGHT && c >= 0 && c < MAX_WIDTH) {
    d_gradient_z[r*MAX_HEIGHT+c] = 0.0f;
    d_gradient_z[r*MAX_HEIGHT+c] += d_frame0[r*MAX_HEIGHT+c] * grad_weights[0]; 
    d_gradient_z[r*MAX_HEIGHT+c] += d_frame1[r*MAX_HEIGHT+c] * grad_weights[1]; 
    d_gradient_z[r*MAX_HEIGHT+c] += d_frame2[r*MAX_HEIGHT+c] * grad_weights[2]; 
    d_gradient_z[r*MAX_HEIGHT+c] += d_frame3[r*MAX_HEIGHT+c] * grad_weights[3]; 
    d_gradient_z[r*MAX_HEIGHT+c] += d_frame4[r*MAX_HEIGHT+c] * grad_weights[4]; 
    d_gradient_z[r*MAX_HEIGHT+c] /= 12.0f;
  }
}

// Host function to launch the CUDA kernel
void gradient_z_calc(pixel_t *frame0, pixel_t *frame1, pixel_t *frame2, pixel_t *frame3, pixel_t *frame4, pixel_t *gradient_z)
{
  #pragma omp target data map(to: frame0[0:MAX_HEIGHT*MAX_WIDTH]) \
                          map(to: frame1[0:MAX_HEIGHT*MAX_WIDTH]) \
                          map(to: frame2[0:MAX_HEIGHT*MAX_WIDTH]) \
                          map(to: frame3[0:MAX_HEIGHT*MAX_WIDTH]) \
                          map(to: frame4[0:MAX_HEIGHT*MAX_WIDTH]) \
                          map(to: GRAD_WEIGHTS[0:5]) \
                          map(alloc: gradient_z[0:MAX_HEIGHT*MAX_WIDTH]) \
                          map(from: gradient_z[0:MAX_HEIGHT*MAX_WIDTH])
  pixel_t *d_frame0;
  pixel_t *d_frame1;
  pixel_t *d_frame2;
  pixel_t *d_frame3;
  pixel_t *d_frame4;
  pixel_t *d_gradient_z;
  int *d_grad_weights;

  int size = MAX_HEIGHT * MAX_WIDTH * sizeof(pixel_t);

  cudaMalloc(&d_frame0, size);
  cudaMalloc(&d_frame1, size);
  cudaMalloc(&d_frame2, size);
  cudaMalloc(&d_frame3, size);
  cudaMalloc(&d_frame4, size);
  cudaMalloc(&d_gradient_z, size);
  cudaMalloc(&d_grad_weights, 5*sizeof(int));

  cudaMemcpy(d_frame0, frame0, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_frame1, frame1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_frame2, frame2, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_frame3, frame3, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_frame4, frame4, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_grad_weights, GRAD_WEIGHTS, 5*sizeof(int), cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((MAX_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (MAX_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

  gradient_z_calc_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_frame0, d_frame1, d_frame2, d_frame3, d_frame4, d_grad_weights, d_gradient_z);
  
  cudaMemcpy(gradient_z, d_gradient_z, size, cudaMemcpyDeviceToHost);

  cudaFree(d_frame0);
  cudaFree(d_frame1);
  cudaFree(d_frame2);
  cudaFree(d_frame3);
  cudaFree(d_frame4);
  cudaFree(d_gradient_z);
  cudaFree(d_grad_weights);
}