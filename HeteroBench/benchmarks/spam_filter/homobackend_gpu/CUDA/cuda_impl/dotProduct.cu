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

__global__ void dotProduct_kernel(FeatureType *d_param, DataType *d_feature, FeatureType *d_result)
{
  __shared__ FeatureType shared_sum[256];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  shared_sum[tid] = 0;

  if (i < NUM_FEATURES) {
    shared_sum[tid] = d_param[i] * d_feature[i];
  }
  __syncthreads();

  // Perform reduction to accumulate results in the block
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  // The first thread in each block adds the block's result to the global result
  if (tid == 0) {
    atomicAdd(d_result, shared_sum[0]);
  }
}


// Function to compute the dot product of data (feature) vector and parameter vector
FeatureType dotProduct(FeatureType param[NUM_FEATURES],
                       DataType    feature[NUM_FEATURES])
{
  FeatureType result = 0;
  
  FeatureType* d_param;
  DataType* d_feature;
  FeatureType* d_result;

  cudaMalloc(&d_param, NUM_FEATURES * sizeof(FeatureType));
  cudaMalloc(&d_feature, NUM_FEATURES * sizeof(DataType));
  cudaMalloc(&d_result, sizeof(FeatureType));

  cudaMemcpy(d_param, param, NUM_FEATURES * sizeof(FeatureType), cudaMemcpyHostToDevice);
  cudaMemcpy(d_feature, feature, NUM_FEATURES * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, &result, sizeof(FeatureType), cudaMemcpyHostToDevice);  
 
  int threadsPerBlock = 256;
  int blocksPerGrid = (NUM_FEATURES + threadsPerBlock - 1) / threadsPerBlock;

  dotProduct_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_param, d_feature, d_result);

  cudaDeviceSynchronize();

  cudaMemcpy(&result, d_result, sizeof(FeatureType), cudaMemcpyDeviceToHost);

  cudaFree(d_param);
  cudaFree(d_feature);
  cudaFree(d_result);

  return result;
}

