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

// Compute the gradient of the cost function
__global__ void computeGradient_kernel(FeatureType *d_grad, DataType *d_feature, FeatureType d_scale)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < NUM_FEATURES) {
    d_grad[i] = d_scale * d_feature[i];
  }
}

// Host function to launch the CUDA kernel
void computeGradient(
    FeatureType grad[NUM_FEATURES],
    DataType    feature[NUM_FEATURES],
    FeatureType scale)
{
  #pragma acc data copyin(feature[0:NUM_FEATURES], scale) \
                   create(grad[0:NUM_FEATURES]) \
                   copyout(grad[0:NUM_FEATURES])
  FeatureType *d_grad;
  DataType *d_feature;

  cudaMalloc(&d_grad, NUM_FEATURES*sizeof(FeatureType));
  cudaMalloc(&d_feature, NUM_FEATURES*sizeof(DataType));

  cudaMemcpy(d_feature, feature, NUM_FEATURES*sizeof(DataType), cudaMemcpyHostToDevice);

  int threadsPerBlock = 128;
  int blocksPerGrid = (NUM_FEATURES + threadsPerBlock - 1) / threadsPerBlock;

  computeGradient_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_grad, d_feature, scale);

  cudaMemcpy(grad, d_grad, NUM_FEATURES*sizeof(FeatureType), cudaMemcpyDeviceToHost);

  cudaFree(d_grad);
  cudaFree(d_feature);
}
