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
#include "math.h"

__global__ void updateParameter_kernel(FeatureType *d_param, FeatureType *d_grad, FeatureType step_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < NUM_FEATURES) {
    d_param[i] += step_size * d_grad[i];
  }
}

// Update the parameter vector
void updateParameter(
    FeatureType param[NUM_FEATURES],
    FeatureType grad[NUM_FEATURES],
    FeatureType step_size)
{
  FeatureType *d_param;
  FeatureType *d_grad;
  
  int size = NUM_FEATURES*sizeof(FeatureType);

  cudaMalloc(&d_param, size);
  cudaMalloc(&d_grad, size);

  cudaMemcpy(d_grad, grad, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (NUM_FEATURES + threadsPerBlock - 1) / threadsPerBlock;

  updateParameter_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_param, d_grad, step_size);

  cudaMemcpy(param, d_param, size, cudaMemcpyDeviceToHost);

  cudaFree(d_param);
  cudaFree(d_grad);
}
