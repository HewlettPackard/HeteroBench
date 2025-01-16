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

// CUDA Kernel for Gaussian blur
__global__ void gaussian_filter_kernel(const uint8_t *d_inImage, int height, int width,
                                   uint8_t *d_outImage, const double *d_kernel) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= OFFSET && col < width - OFFSET && row >= OFFSET && row < height - OFFSET) {
    double outIntensity = 0;
    int kIndex = 0;
    int pxIndex = col + (row * width);
    for (int krow = -OFFSET; krow <= OFFSET; krow++) {
      for (int kcol = -OFFSET; kcol <= OFFSET; kcol++) {
        outIntensity +=
          d_inImage[pxIndex + (kcol + (krow * width))] * d_kernel[kIndex];
        kIndex++;
      }
    }
    d_outImage[pxIndex] = (uint8_t)(outIntensity);
  }
}

// Host function to launch the CUDA kernel
void gaussian_filter(const uint8_t *inImage, int height, int width, uint8_t *outImage) {
  // Gaussian kernel
  const double kernel[9] = {0.0625, 0.125, 0.0625, 0.1250, 0.250, 0.1250, 0.0625, 0.125, 0.0625};
  uint8_t *d_inImage, *d_outImage;
  double *d_kernel;
  size_t imgSize = width * height * sizeof(uint8_t);
  size_t krnlSize = 9 * sizeof(double);

  cudaMalloc(&d_inImage, imgSize);
  cudaMalloc(&d_outImage, imgSize);
  cudaMalloc(&d_kernel, krnlSize);

  cudaMemcpy(d_inImage, inImage, imgSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, krnlSize, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  gaussian_filter_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_inImage, height, width,
                                                             d_outImage, d_kernel);

  cudaMemcpy(outImage, d_outImage, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_inImage);
  cudaFree(d_outImage);
  cudaFree(d_kernel);
}
