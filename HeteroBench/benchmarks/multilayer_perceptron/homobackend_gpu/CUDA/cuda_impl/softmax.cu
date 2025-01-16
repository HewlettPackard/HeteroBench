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
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
def softmax(input_0):
  exp_input_0 = np.exp(input_0)
  sum_total_0 = np.sum(exp_input_0)
  output_0 = exp_input_0 / sum_total_0
  return output_0
*/

// CUDA kernel to compute exponentials
__global__ void softmax_exp_kernel(double *d_softmax_input, double *d_exp_results, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_exp_results[i] = exp(d_softmax_input[i]);
    }
}

// CUDA kernel to perform reduction (sum) using atomic addition
__global__ void reduction_kernel(double *d_exp_results, double *d_sum_total, int size) {
    __shared__ double partial_sum[256]; // Adjust based on your block size
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    partial_sum[tid] = 0.0;
    if(i < size) {
      partial_sum[tid] = d_exp_results[i];
    }
    __syncthreads();

    // Perform parallel reduction within the block
    for (int j = blockDim.x / 2; j > 0; j >>= 1) {
        if (tid < j) {
            partial_sum[tid] += partial_sum[tid + j];
        }
        __syncthreads();
    }

    // Sum the results across all blocks using atomic addition
    if (tid == 0) {
        atomicAdd(d_sum_total, partial_sum[0]);
    }
}

// CUDA kernel to softmax_kernel
__global__ void softmax_kernel(double *d_softmax_output, double *d_exp_results, double *d_sum_total, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
      d_softmax_output[i] = d_exp_results[i] / *d_sum_total;
    }
}

// Host function to launch the CUDA kernel
void softmax(double *softmax_input, double *exp_results, double *softmax_output, int size) {
  double sum_total_0 = 0;

  double *d_softmax_input;
  double *d_exp_results;
  double *d_softmax_output;
  double *d_sum_total;

  cudaMalloc(&d_softmax_input, size * sizeof(double));
  cudaMalloc(&d_exp_results, size * sizeof(double));
  cudaMalloc(&d_softmax_output, size * sizeof(double));
  cudaMalloc(&d_sum_total, sizeof(double));

  cudaMemcpy(d_softmax_input, softmax_input, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sum_total, &sum_total_0, sizeof(double), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  
  softmax_exp_kernel<<<numBlocks, blockSize>>>(d_softmax_input, d_exp_results, size);
  cudaDeviceSynchronize();

  reduction_kernel<<<numBlocks, blockSize>>>(d_exp_results, d_sum_total, size);
  cudaDeviceSynchronize();
  
  softmax_kernel<<<numBlocks, blockSize>>>(d_softmax_output, d_exp_results, d_sum_total, size);
  cudaDeviceSynchronize();

  cudaMemcpy(softmax_output, d_softmax_output, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(d_softmax_input);
  cudaFree(d_exp_results);
  cudaFree(d_softmax_output);
  cudaFree(d_sum_total);

}