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
def max_pooling(input, pool_size, pool_stride=2):
  output_height = (input.shape[0] - pool_size) // pool_stride + 1
  output_width = (input.shape[1] - pool_size) // pool_stride + 1
  output = np.zeros((output_height, output_width))
  for i in range(0, output_height):
    for j in range(0, output_width):
      region = input[i*pool_stride:i*pool_stride+pool_size, j*pool_stride:j*pool_stride+pool_size]
      output[i, j] = np.max(region)
  return output
*/

__global__ void max_pooling_kernel(double *d_max_pooling_input, int pool_size, int pool_stride, int output_h, int output_w, int input_w, double *d_max_pooling_output)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < output_h && j < output_w) {
    double max_val = 0;
    for (int k = 0; k < pool_size; k++) {
      for (int l = 0; l < pool_size; l++) {
        max_val = max(max_val, d_max_pooling_input[(i * pool_stride + k) * input_w + j * pool_stride + l]);
      }
    }
    d_max_pooling_output[i * output_w + j] = max_val;
  }
}

void max_pooling(double *max_pooling_input, int pool_size, int pool_stride, int input_h, int input_w, double *max_pooling_output)
{
  int output_h = (input_h - pool_size) / pool_stride + 1;
  int output_w = (input_w - pool_size) / pool_stride + 1;
  
  double *d_max_pooling_input;
  double *d_max_pooling_output;

  cudaMalloc(&d_max_pooling_input, input_h*input_w*sizeof(double));
  cudaMalloc(&d_max_pooling_output, input_h*input_w*sizeof(double));

  cudaMemcpy(d_max_pooling_input, max_pooling_input, input_h*input_w*sizeof(double), cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((output_w + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                     (output_h + threadsPerBlock.y - 1) / threadsPerBlock.y);

  max_pooling_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_max_pooling_input, pool_size, pool_stride, output_h, output_w, input_w, d_max_pooling_output);

  cudaMemcpy(max_pooling_output, d_max_pooling_output, input_h*input_w*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error max_pooling_kernel: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaFree(d_max_pooling_input);
  cudaFree(d_max_pooling_output);
}