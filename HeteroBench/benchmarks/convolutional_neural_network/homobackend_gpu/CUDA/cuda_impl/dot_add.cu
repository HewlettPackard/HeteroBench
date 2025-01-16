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
  def dot_add(x, W, b):
    mm = np.dot(x, W) + b
    return mm
*/

__global__ void dot_add_kernel(double *d_dot_add_input_x, double *d_dot_add_input_W, double *d_dot_add_input_b, double *d_dot_add_output, int x_h, int x_w, int W_h, int W_w) 
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < x_h && j < W_w) {
    double tmp = 0;
    for (int k = 0; k < x_w; k++) {
      tmp += d_dot_add_input_x[i * x_w + k] * d_dot_add_input_W[k * W_w + j];
    }
    d_dot_add_output[i * W_w + j] = tmp + d_dot_add_input_b[j];
  }
}

void dot_add(double *dot_add_input_x, double *dot_add_input_W, double *dot_add_input_b, double *dot_add_output, int x_h, int x_w, int W_h, int W_w)
{
  double *d_dot_add_input_x;
  double *d_dot_add_input_W;
  double *d_dot_add_input_b;
  double *d_dot_add_output;

  cudaMalloc(&d_dot_add_input_x, x_h*x_w*sizeof(double));
  cudaMalloc(&d_dot_add_input_W, x_w*W_w*sizeof(double));
  cudaMalloc(&d_dot_add_input_b, W_w*sizeof(double));
  cudaMalloc(&d_dot_add_output, x_h*W_w*sizeof(double));

  cudaMemcpy(d_dot_add_input_x, dot_add_input_x, x_h*x_w*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dot_add_input_W, dot_add_input_W, x_w*W_w*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dot_add_input_b, dot_add_input_b, W_w*sizeof(double), cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((W_w + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                     (x_h + threadsPerBlock.y - 1) / threadsPerBlock.y);

  dot_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_dot_add_input_x, d_dot_add_input_W, d_dot_add_input_b, d_dot_add_output, x_h, x_w, W_h, W_w);

  cudaMemcpy(dot_add_output, d_dot_add_output, x_h*W_w*sizeof(double), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error dot_add_kernel: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaFree(d_dot_add_input_x);
  cudaFree(d_dot_add_input_W);
  cudaFree(d_dot_add_input_b);
  cudaFree(d_dot_add_output);
}