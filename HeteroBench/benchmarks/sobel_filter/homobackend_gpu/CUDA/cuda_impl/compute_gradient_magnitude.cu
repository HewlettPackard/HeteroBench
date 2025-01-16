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

// CUDA Kernel for computing gradient magnitude
__global__ void compute_gradient_magnitude_kernel(const double *d_sobel_x, const double *d_sobel_y, int height, int width, double *d_gradient_magnitude)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int imgSize = height*width;

    if(i < imgSize) {
        d_gradient_magnitude[i] = sqrt(d_sobel_x[i] * d_sobel_x[i] + d_sobel_y[i] * d_sobel_y[i]);
    }

}

// Host function to launch the CUDA kernel
void compute_gradient_magnitude(const double *sobel_x, const double *sobel_y, int height, int width, double *gradient_magnitude) {
    int imgSize = height*width*sizeof(double);
    double *d_sobel_x;
    double *d_sobel_y;
    double *d_gradient_magnitude;

    cudaMalloc(&d_sobel_x, imgSize);
    cudaMalloc(&d_sobel_y, imgSize);
    cudaMalloc(&d_gradient_magnitude, imgSize);

    cudaMemcpy(d_sobel_x, sobel_x, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sobel_y, sobel_y, imgSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    compute_gradient_magnitude_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_sobel_x, d_sobel_y, height, width, d_gradient_magnitude);
    
    cudaMemcpy(gradient_magnitude, d_gradient_magnitude, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_sobel_x);
    cudaFree(d_sobel_y);
    cudaFree(d_gradient_magnitude);
}
