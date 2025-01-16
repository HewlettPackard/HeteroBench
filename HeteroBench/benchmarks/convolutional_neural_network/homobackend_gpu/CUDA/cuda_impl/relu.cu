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
def relu(x):
  return np.maximum(0, x)
*/

__global__ void relu_kernel(double *relu_input, double *relu_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        relu_output[i] = fmax(0.0, relu_input[i]);
    }
}

void relu(double *relu_input, double *relu_output, int size)
{
    double *d_relu_input;
    double *d_relu_output;

    cudaMalloc((void **)&d_relu_input, size * sizeof(double));
    cudaMalloc((void **)&d_relu_output, size * sizeof(double));

    cudaMemcpy(d_relu_input, relu_input, size * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_relu_input, d_relu_output, size);

    cudaMemcpy(relu_output, d_relu_output, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error relu_kernel: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaFree(d_relu_input);
    cudaFree(d_relu_output);
}