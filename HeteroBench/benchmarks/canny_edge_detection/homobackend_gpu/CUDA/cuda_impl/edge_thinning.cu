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

// CUDA Kernel for non-maximum suppression
__global__ void edge_thinning_kernel(double *d_intensity, uint8_t *d_direction,
                                        int height, int width, double *d_outImage) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= OFFSET && col < width - OFFSET && row >= OFFSET && row < height - OFFSET) {
    int pxIndex = col + (row * width);

    // Unconditionally suppress border pixels
    if (row == OFFSET || col == OFFSET || col == width - OFFSET - 1 || row == height - OFFSET - 1) {
      d_outImage[pxIndex] = 0;
      return;
    }

    switch (d_direction[pxIndex]) {
    case 1:
      if (d_intensity[pxIndex - 1] >= d_intensity[pxIndex] ||
        d_intensity[pxIndex + 1] > d_intensity[pxIndex])
        d_outImage[pxIndex] = 0;
      break;
    case 2:
      if (d_intensity[pxIndex - (width - 1)] >= d_intensity[pxIndex] ||
        d_intensity[pxIndex + (width - 1)] > d_intensity[pxIndex])
        d_outImage[pxIndex] = 0;
      break;
    case 3:
      if (d_intensity[pxIndex - width] >= d_intensity[pxIndex] ||
        d_intensity[pxIndex + width] > d_intensity[pxIndex])
        d_outImage[pxIndex] = 0;
      break;
    case 4:
      if (d_intensity[pxIndex - (width + 1)] >= d_intensity[pxIndex] ||
        d_intensity[pxIndex + (width + 1)] > d_intensity[pxIndex])
        d_outImage[pxIndex] = 0;
      break;
    default:
      d_outImage[pxIndex] = 0;
      break;
    }
  }
}

// Host function to launch the CUDA kernel
void edge_thinning(double *intensity, uint8_t *direction, int height, int width,
                         double *outImage) {
  double *d_intensity, *d_outImage;
  uint8_t *d_direction;
  size_t imgSize = width * height * sizeof(double);
  size_t dirSize = width * height * sizeof(uint8_t);

  cudaMalloc(&d_intensity, imgSize);
  cudaMalloc(&d_outImage, imgSize);
  cudaMalloc(&d_direction, dirSize);

  cudaMemcpy(d_intensity, intensity, imgSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_direction, direction, dirSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_outImage, intensity, imgSize, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  edge_thinning_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_intensity, d_direction, height, width, d_outImage);

  cudaMemcpy(outImage, d_outImage, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_intensity);
  cudaFree(d_outImage);
  cudaFree(d_direction);
}