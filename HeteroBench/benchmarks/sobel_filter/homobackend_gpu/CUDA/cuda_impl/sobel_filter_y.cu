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
#include "omp.h"
#include <cstring>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel for sobel filter y
__global__ void sobel_filter_y_kernel(const uint8_t *d_input_image, int height, int width, const int d_kernel_y[3][3], double *d_sobel_y) 
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if(row > 0 && row < height - 1 && col > 0 && col < width - 1) {
    double gy = 0;
    for (int krow = -1; krow <= 1; ++krow) {
      for (int kcol = -1; kcol <= 1; ++kcol) {
        int pixel_val = d_input_image[(row + krow) * width + (col + kcol)];
        gy += pixel_val * d_kernel_y[krow + 1][kcol + 1];
      }
    }
    d_sobel_y[row * width + col] = gy;
  }
}

// Host function to launch the CUDA kernel
void sobel_filter_y(const uint8_t *input_image, int height, int width, double *sobel_y) {
  const int kernel_y[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
  };

  int imgSize = height * width * sizeof(uint8_t);
  int outSize = height * width * sizeof(double);

  uint8_t *d_input_image;
  double *d_sobel_y;
  int (*d_kernel_y)[3];
  
  cudaMalloc(&d_input_image, imgSize);
  cudaMalloc(&d_sobel_y, outSize);
  cudaMalloc(&d_kernel_y, 3*3*sizeof(int));

  cudaMemcpy(d_input_image, input_image, imgSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel_y, kernel_y, 3*3*sizeof(int), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  sobel_filter_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input_image, height, width, d_kernel_y, d_sobel_y);

  cudaMemcpy(sobel_y, d_sobel_y, outSize, cudaMemcpyDeviceToHost);

  cudaFree(d_input_image);
  cudaFree(d_sobel_y);
  cudaFree(d_kernel_y);
}
