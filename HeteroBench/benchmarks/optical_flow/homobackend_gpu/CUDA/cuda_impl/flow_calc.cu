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

// CUDA Kernel to compute flow
__global__ void flow_calc_kernel(tensor_t *d_tensor, velocity_t *d_outputs)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;

  if(r >= 0 && r < MAX_HEIGHT && c >= 0 && c < MAX_WIDTH) {
    if (r >= 2 && r < MAX_HEIGHT - 2 && c >= 2 && c < MAX_WIDTH - 2)
    {
      pixel_t denom = d_tensor[r*MAX_HEIGHT+c].val[0] * d_tensor[r*MAX_HEIGHT+c].val[1] -
                      d_tensor[r*MAX_HEIGHT+c].val[3] * d_tensor[r*MAX_HEIGHT+c].val[3];
      d_outputs[r*MAX_HEIGHT+c].x = (d_tensor[r*MAX_HEIGHT+c].val[5] * d_tensor[r*MAX_HEIGHT+c].val[3] -
                        d_tensor[r*MAX_HEIGHT+c].val[4] * d_tensor[r*MAX_HEIGHT+c].val[1]) / denom;
      d_outputs[r*MAX_HEIGHT+c].y = (d_tensor[r*MAX_HEIGHT+c].val[4] * d_tensor[r*MAX_HEIGHT+c].val[3] -
                        d_tensor[r*MAX_HEIGHT+c].val[5] * d_tensor[r*MAX_HEIGHT+c].val[0]) / denom;
    }
    else
    {
      d_outputs[r*MAX_HEIGHT+c].x = 0;
      d_outputs[r*MAX_HEIGHT+c].y = 0;
    }
  }
}

// Host function to launch the CUDA kernel
void flow_calc(tensor_t *tensor, velocity_t *outputs)
{
  tensor_t *d_tensor;
  velocity_t *d_outputs;

  int tensorSize = MAX_HEIGHT*MAX_WIDTH*sizeof(tensor_t);
  int outSize = MAX_HEIGHT*MAX_WIDTH*sizeof(velocity_t);

  cudaMalloc(&d_tensor, tensorSize);
  cudaMalloc(&d_outputs, outSize);

  cudaMemcpy(d_tensor, tensor, tensorSize, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((MAX_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (MAX_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

  flow_calc_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tensor, d_outputs);

  cudaMemcpy(outputs, d_outputs, outSize, cudaMemcpyDeviceToHost);

  cudaFree(d_tensor);
  cudaFree(d_outputs);
}