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
def pad_input(input, padding):
  if padding == 0:
    return input
  padded_input = np.zeros((input.shape[0] + 2*padding, input.shape[1] + 2*padding))
  for i in range(input.shape[0]):
    for j in range(input.shape[1]):
      padded_input[i + padding][j + padding] = input[i][j]
  return padded_input
*/

__global__ void pad_input_kernel(double *d_input, double *d_output, int input_h, int input_w, int padded_h, int padded_w, int padding) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < padded_h && j < padded_w) {
        if (i < padding || j < padding || i >= input_h + padding || j >= input_w + padding) {
            d_output[i * padded_w + j] = 0.0;
        } else {
            d_output[i * padded_w + j] = d_input[(i - padding) * input_w + (j - padding)];
        }
    }
}

void pad_input(double *pad_input_input, double *pad_input_output, int input_h, int input_w, int padding) 
{
  int padded_h = input_h + 2 * padding;
  int padded_w = input_w + 2 * padding;

  double *d_input;
  double *d_output;

  cudaMalloc((void **)&d_input, input_h * input_w * sizeof(double));
  cudaMalloc((void **)&d_output, padded_h * padded_w * sizeof(double));

  cudaMemcpy(d_input, pad_input_input, input_h * input_w * sizeof(double), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((padded_w + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (padded_h + threadsPerBlock.y - 1) / threadsPerBlock.y);

  pad_input_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, input_h, input_w, padded_h, padded_w, padding);

  cudaMemcpy(pad_input_output, d_output, padded_h * padded_w * sizeof(double), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error pad_input_kernel: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaFree(d_input);
  cudaFree(d_output);
}