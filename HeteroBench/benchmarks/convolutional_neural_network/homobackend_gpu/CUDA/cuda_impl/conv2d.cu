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
def conv2d(input, kernel, bias, stride, padding):
  input_padded = pad_input(input, padding)
  kernel_height, kernel_width = kernel.shape
  output_height = (input_padded.shape[0] - kernel_height) // stride + 1
  output_width = (input_padded.shape[1] - kernel_width) // stride + 1
  conv2d_output = np.zeros((output_height, output_width))
  for i in range(0, output_height):
    for j in range(0, output_width):
      region = input_padded[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
      conv2d_output[i, j] = np.sum(region * kernel) + bias
  return conv2d_output
*/
__global__ void conv2d_cuda_kernel(double *d_conv2d_kernel, double *d_input_padded, double conv2d_bias,
                              int stride, int kernel_h, int kernel_w, int output_h,
                              int output_w, int padded_w, double *d_conv2d_output)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < output_h && j < output_w) {
    double tmp = 0;
    for (int k = 0; k < kernel_h; k++) {
      for (int l = 0; l < kernel_w; l++) {
        tmp += d_input_padded[(i * stride + k) * padded_w + j * stride + l] * d_conv2d_kernel[k * kernel_w + l];
      }
    }
    d_conv2d_output[i * output_w + j] = tmp + conv2d_bias;
  }
}

void conv2d(double *conv2d_input, double *conv2d_kernel, double *input_padded, double conv2d_bias, int stride, int padding, int input_h, int input_w, int kernel_h, int kernel_w, double *conv2d_output) 
{
  pad_input(conv2d_input, input_padded, input_h, input_w, padding);

  int padded_h = input_h + 2 * padding;
  int padded_w = input_w + 2 * padding;
  int output_h = (padded_h - kernel_h) / stride + 1;
  int output_w = (padded_w - kernel_w) / stride + 1;

  double *d_conv2d_kernel;
  double *d_input_padded;
  double *d_conv2d_output;

  cudaMalloc(&d_conv2d_kernel, kernel_h*kernel_w*sizeof(double));
  cudaMalloc(&d_input_padded, padded_h*padded_w*sizeof(double));
  cudaMalloc(&d_conv2d_output, output_h*output_w*sizeof(double));

  cudaMemcpy(d_conv2d_kernel, conv2d_kernel, kernel_h*kernel_w*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_padded, input_padded, padded_h*padded_w*sizeof(double), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((output_w + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                     (output_h + threadsPerBlock.y - 1) / threadsPerBlock.y);

  conv2d_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_conv2d_kernel, d_input_padded, conv2d_bias, stride, kernel_h, kernel_w, output_h, output_w, padded_w, d_conv2d_output);

  cudaMemcpy(conv2d_output, d_conv2d_output, output_h*output_w*sizeof(double), cudaMemcpyDeviceToHost);
  
  cudaDeviceSynchronize();
  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error conv2d_cuda_kernel: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaFree(d_conv2d_kernel);
  cudaFree(d_input_padded);
  cudaFree(d_conv2d_output);
}