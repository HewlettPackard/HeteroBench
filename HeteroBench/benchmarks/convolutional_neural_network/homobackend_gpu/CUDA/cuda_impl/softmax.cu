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
// #include <math.h>
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

__global__ void softmax_kernel1(double *softmax_input, double *exp_results, double *sum_total, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;

    if (i < size) {
        exp_results[i] = exp(softmax_input[i]);
        local_sum = exp_results[i];
    }

    atomicAdd(sum_total, local_sum);
}

__global__ void softmax_kernel2(double *exp_results, double *softmax_output, double sum_total, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        softmax_output[i] = exp_results[i] / sum_total;
    }
}

void softmax(double *softmax_input, double *exp_results, double *softmax_output, int size) 
{
    double *d_softmax_input;
    double *d_exp_results;
    double *d_sum_total;
    double *d_softmax_output;
    double h_sum_total = 0.0;

    cudaMalloc(&d_softmax_input, size * sizeof(double));
    cudaMalloc(&d_exp_results, size * sizeof(double));
    cudaMalloc(&d_sum_total, sizeof(double));
    cudaMalloc(&d_softmax_output, size * sizeof(double));

    cudaMemcpy(d_sum_total, &h_sum_total, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_softmax_input, softmax_input, size * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel1<<<blocksPerGrid, threadsPerBlock>>>(d_softmax_input, d_exp_results, d_sum_total, size);

    cudaDeviceSynchronize();
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error softmax_kernel1: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    softmax_kernel2<<<blocksPerGrid, threadsPerBlock>>>(d_exp_results, d_softmax_output, h_sum_total, size);
    cudaMemcpy(softmax_output, d_softmax_output, size*sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error softmax_kernel2: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaFree(d_softmax_input);
    cudaFree(d_exp_results);
    cudaFree(d_sum_total);
}