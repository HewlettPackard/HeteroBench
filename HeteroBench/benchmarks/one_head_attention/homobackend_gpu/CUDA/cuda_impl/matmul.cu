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
    def matmul(x, y):
        if len(x.shape) == 3 and len(y.shape) == 3:
            output = np.zeros((x.shape[0], x.shape[1], y.shape[2]))
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(y.shape[2]):
                        for l in range(x.shape[2]):
                            output[i, j, k] += x[i, j, l] * y[i, l, k]
            return output
        else:
            raise NotImplementedError("Not implemented yet for x.shape != 3 or y.shape != 3")
*/

__global__ void assign_zero_kernel(double *d_matmul_output, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        d_matmul_output[idx] = 0;
    }
}

__global__ void matmul_kernel(double *matmul_x, double *matmul_y, double *matmul_output,
                              int batch_size, int input_h, int input_w, int output_w) {
    int i = blockIdx.x; // Batch index
    int j = blockIdx.y; // Input height index
    int k = threadIdx.x; // Output width index

    if (i < batch_size && j < input_h && k < output_w) {
        for (int l = 0; l < input_w; l++) {
            matmul_output[i * input_h * output_w + j * output_w + k] += 
                        matmul_x[i * input_h * input_w + j * input_w + l] * 
                        matmul_y[i * input_w * output_w + l * output_w + k];
        }
    }
}

void matmul(double *matmul_x, double *matmul_y, double *matmul_output, 
            int batch_size, int input_h, int input_w, int output_w) 
{
    double *d_matmul_x;
    double *d_matmul_y;
    double *d_matmul_output;

    cudaMalloc(&d_matmul_x, batch_size * input_h * input_w * sizeof(double));
    cudaMalloc(&d_matmul_y, batch_size * input_w * output_w * sizeof(double));
    cudaMalloc(&d_matmul_output, batch_size * input_h * output_w * sizeof(double));

    cudaMemcpy(d_matmul_x, matmul_x, batch_size * input_h * input_w * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matmul_y, matmul_y, batch_size * input_w * output_w * sizeof(double), cudaMemcpyHostToDevice);
    
    int total_size = batch_size * input_h * output_w;
    int block_size_zero = 256;
    int grid_size_zero = (total_size + block_size_zero - 1) / block_size_zero;
    assign_zero_kernel<<<grid_size_zero, block_size_zero>>>(d_matmul_output, total_size);

    dim3 blockDim(output_w); // Threads for output width
    dim3 gridDim(batch_size, input_h); // Blocks for batch size and input height
    matmul_kernel<<<gridDim, blockDim>>>(d_matmul_x, d_matmul_y, d_matmul_output, batch_size, input_h, input_w, output_w);

    cudaMemcpy(matmul_output, d_matmul_output, batch_size * input_h * output_w * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_matmul_x);
    cudaFree(d_matmul_y);
    cudaFree(d_matmul_output);
}
