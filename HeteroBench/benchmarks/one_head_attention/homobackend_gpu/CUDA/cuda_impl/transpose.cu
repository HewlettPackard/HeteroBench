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
/* Here the input and output are 3D arraies */
/*
    def transpose(x, dim0=-2, dim1=-1):
        # implement the x.swapaxes(-2, -1) manually
        shape = list(x.shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        transpose_x = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[dim0]):
                for k in range(shape[dim1]):
                    transpose_x[i, j, k] = x[i, k, j]

        return transpose_x
*/

__global__ void transpose_kernel(double *d_transpose_x, double *d_transpose_output, 
                int batch_size, int input_h, int input_w) 
{
    int i = blockIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i>=0 && i < batch_size && j>=0 && j < input_h && k>=0 && k < input_w) {
            d_transpose_output[i * input_h * input_w + k * input_h + j] = 
                d_transpose_x[i * input_h * input_w + j * input_w + k];
    }
}

void transpose(double *transpose_x, double *transpose_output, 
                int batch_size, int input_h, int input_w, int dim0, int dim1) {
    if (dim0 == -2 && dim1 == -1) {
        double *d_transpose_x;
        double *d_transpose_output;

        cudaMalloc(&d_transpose_x, batch_size * input_h * input_w * sizeof(double));
        cudaMalloc(&d_transpose_output, batch_size * input_h * input_w * sizeof(double));

        cudaMemcpy(d_transpose_x, transpose_x, batch_size * input_h * input_w, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((input_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (input_h + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           batch_size);

        transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_transpose_x, d_transpose_output, batch_size, input_h, input_w);

        cudaMemcpy(transpose_output, d_transpose_output, batch_size * input_h * input_w, cudaMemcpyDeviceToHost);
        
        cudaFree(d_transpose_x);
        cudaFree(d_transpose_output);
    } else {
        cout << "Not implemented yet for dim0 != -2 or dim1 != -1" << endl;
    }
}