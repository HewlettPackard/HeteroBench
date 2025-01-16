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

// CUDA Kernel for computing gradient intensity and direction
__global__ void gradient_intensity_direction_kernel(const uint8_t *d_inImage, int height, int width,
                                                 double *d_intensity, uint8_t *d_direction,
                                                 const int8_t *d_Gx, const int8_t *d_Gy) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= OFFSET && col < width - OFFSET && row >= OFFSET && row < height - OFFSET) {
        double Gx_sum = 0.0;
        double Gy_sum = 0.0;
        int kIndex = 0;
        int pxIndex = col + (row * width);

        for (int krow = -OFFSET; krow <= OFFSET; krow++) {
            for (int kcol = -OFFSET; kcol <= OFFSET; kcol++) {
                Gx_sum += d_inImage[pxIndex + (kcol + (krow * width))] * d_Gx[kIndex];
                Gy_sum += d_inImage[pxIndex + (kcol + (krow * width))] * d_Gy[kIndex];
                kIndex++;
            }
        }

        if (Gx_sum == 0.0 || Gy_sum == 0.0) {
            d_intensity[pxIndex] = 0;
        } else {
            d_intensity[pxIndex] = sqrt((Gx_sum * Gx_sum) + (Gy_sum * Gy_sum));
            double theta = atan2(Gy_sum, Gx_sum);
            theta = theta * (360.0 / (2.0 * M_PI));

            if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) || (theta >= 157.5))
                d_direction[pxIndex] = 1;   // horizontal -
            else if ((theta > 22.5 && theta <= 67.5) || (theta > -157.5 && theta <= -112.5))
                d_direction[pxIndex] = 2;   // north-east -> south-west /
            else if ((theta > 67.5 && theta <= 112.5) || (theta >= -112.5 && theta < -67.5))
                d_direction[pxIndex] = 3;   // vertical |
            else if ((theta >= -67.5 && theta < -22.5) || (theta > 112.5 && theta < 157.5))
                d_direction[pxIndex] = 4;   // north-west -> south-east \'
        }
    }
}

// Host function to launch the CUDA kernel
void gradient_intensity_direction(const uint8_t *inImage, int height, int width,
                                  double *intensity, uint8_t *direction) {
    const int8_t Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int8_t Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    uint8_t *d_inImage;
    double *d_intensity;
    uint8_t *d_direction;
    int8_t *d_Gx, *d_Gy;
    size_t imgSize = width * height * sizeof(uint8_t);
    size_t intensitySize = width * height * sizeof(double);
    size_t krnlSize = 9 * sizeof(int8_t);

    cudaMalloc(&d_inImage, imgSize);
    cudaMalloc(&d_intensity, intensitySize);
    cudaMalloc(&d_direction, imgSize);
    cudaMalloc(&d_Gx, krnlSize);
    cudaMalloc(&d_Gy, krnlSize);

    cudaMemcpy(d_inImage, inImage, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Gx, Gx, krnlSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Gy, Gy, krnlSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gradient_intensity_direction_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_inImage, height, width,
                                                                            d_intensity, d_direction,
                                                                            d_Gx, d_Gy);

    cudaMemcpy(intensity, d_intensity, intensitySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(direction, d_direction, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inImage);
    cudaFree(d_intensity);
    cudaFree(d_direction);
    cudaFree(d_Gx);
    cudaFree(d_Gy);
}