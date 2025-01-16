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

// CUDA Kernel to compute outer product
__global__ void outer_product_kernel(gradient_t *d_filtered_gradient, outer_t *d_out_product)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;

  if(r >= 0 && r < MAX_HEIGHT && c >= 0 && c < MAX_WIDTH) {
    gradient_t grad = d_filtered_gradient[r*MAX_HEIGHT+c];
    outer_t out;
    out.val[0] = grad.x * grad.x;
    out.val[1] = grad.y * grad.y;
    out.val[2] = grad.z * grad.z;
    out.val[3] = grad.x * grad.y;
    out.val[4] = grad.x * grad.z;
    out.val[5] = grad.y * grad.z;
    d_out_product[r*MAX_HEIGHT+c] = out;
  }

}

// Host function to launch the CUDA kernel
void outer_product(gradient_t *filtered_gradient, outer_t *out_product)
{
  gradient_t *d_filtered_gradient;
  outer_t *d_out_product;

  int sizeGradient = MAX_HEIGHT*MAX_WIDTH*sizeof(gradient_t);
  int sizeOuter = MAX_HEIGHT*MAX_WIDTH*sizeof(outer_t);

  cudaMalloc(&d_filtered_gradient, sizeGradient);
  cudaMalloc(&d_out_product, sizeOuter);

  cudaMemcpy(d_filtered_gradient, filtered_gradient, sizeGradient, cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((MAX_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (MAX_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

  outer_product_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_filtered_gradient, d_out_product);
  
  cudaMemcpy(out_product, d_out_product, sizeOuter, cudaMemcpyDeviceToHost);

  cudaFree(d_filtered_gradient);
  cudaFree(d_out_product);
}
