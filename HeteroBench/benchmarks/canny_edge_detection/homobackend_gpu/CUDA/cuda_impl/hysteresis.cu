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

// CUDA Kernel for hysteresis
__global__ void hysteresis_kernel(uint8_t *d_inImage, int height, int width,
                                  uint8_t *d_outImage) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= OFFSET && col < width - OFFSET && row >= OFFSET && row < height - OFFSET) {
    int pxIndex = col + (row * width);
    if (d_outImage[pxIndex] == 100) {
      if (d_outImage[pxIndex - 1] == 255 ||
          d_outImage[pxIndex + 1] == 255 ||
          d_outImage[pxIndex - width] == 255 ||
          d_outImage[pxIndex + width] == 255 ||
          d_outImage[pxIndex - width - 1] == 255 ||
          d_outImage[pxIndex - width + 1] == 255 ||
          d_outImage[pxIndex + width - 1] == 255 ||
          d_outImage[pxIndex + width + 1] == 255)
        atomicExch((unsigned int*)&d_outImage[pxIndex], (unsigned int)255);
      else
        atomicExch((unsigned int*)&d_outImage[pxIndex], (unsigned int)0);
    }
  }
}

// Host function to launch the CUDA kernel
void hysteresis(uint8_t *inImage, int height, int width,
                uint8_t *outImage) {
  memcpy(outImage, inImage, width * height * sizeof(uint8_t));
  
  uint8_t *d_inImage, *d_outImage;
  size_t imgSize = width * height * sizeof(uint8_t);

  cudaMalloc(&d_inImage, imgSize);
  cudaMalloc(&d_outImage, imgSize);

  cudaMemcpy(d_inImage, inImage, imgSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_outImage, inImage, imgSize, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  hysteresis_kernel<<<1, 1>>>(d_inImage, height, width, d_outImage);

  cudaMemcpy(outImage, d_outImage, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_inImage);
  cudaFree(d_outImage);
}