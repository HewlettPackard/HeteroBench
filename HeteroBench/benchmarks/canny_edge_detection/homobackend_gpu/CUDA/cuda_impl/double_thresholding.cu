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

// CUDA Kernel to perform thresholding
__global__ void double_thresholding_kernel(double *d_suppressed_image, int height, int width, 
                                   int high_threshold, int low_threshold, uint8_t *outImage) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pxIndex = col + (row * width);
        if (d_suppressed_image[pxIndex] > high_threshold) {
            outImage[pxIndex] = 255;   // Strong edge
        } else if (d_suppressed_image[pxIndex] > low_threshold) {
            outImage[pxIndex] = 100;   // Weak edge
        } else {
            outImage[pxIndex] = 0;     // Not an edge
        }
    }
}

// Host function to launch the CUDA kernel
void double_thresholding(double *suppressed_image, int height, int width,
                  int high_threshold, int low_threshold,
                  uint8_t *outImage) {
    double *d_suppressed_image;
    uint8_t *d_outImage;
    size_t imgSize = height * width * sizeof(double);
    size_t outSize = height * width * sizeof(uint8_t);

    cudaMalloc(&d_suppressed_image, imgSize);
    cudaMalloc(&d_outImage, outSize);

    cudaMemcpy(d_suppressed_image, suppressed_image, imgSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    double_thresholding_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_suppressed_image, height, width, 
                                                            high_threshold, low_threshold, d_outImage);

    cudaMemcpy(outImage, d_outImage, outSize, cudaMemcpyDeviceToHost);

    cudaFree(d_suppressed_image);
    cudaFree(d_outImage);
}