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
    def get_max(x, axis=-1, keepdims=True):
        if axis == -1 and keepdims == True:
            # init the max_x with np.-inf with the size of shape(x) except the last dimension (axis = -1)
            max_x = np.full(x.shape[:-1], -np.inf)
            # iterate over the last dimension of x
            for i in range(x.shape[-1]):
                max_x = np.maximum(max_x, x[..., i])
            # add the last dimension to max_x
            max_x = np.expand_dims(max_x, axis=-1)
        else:
            raise NotImplementedError("Not implemented yet  for axis != -1 or keepdims != True")
        return max_x
*/

__global__ void init_softmax_m_kernel(double *d_softmax_m, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_softmax_m[idx] = -INFINITY;
    }
}

__global__ void get_max_kernel(double *d_softmax_x, double *d_softmax_m, int batch_size, int input_h, int input_w)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    if (i < batch_size && j < input_h) {
        for (int k = 0; k < input_w; k++) {
            d_softmax_m[i * input_h + j] = max(d_softmax_m[i * input_h + j], d_softmax_x[i * input_h * input_w + j * input_w + k]);
        }
    }
}

void get_max(double *softmax_x, double *softmax_m, 
                int batch_size, int input_h, int input_w, int axis, bool keepdims)
{
    
    double *d_softmax_x;
    double *d_softmax_m;

    cudaMalloc(&d_softmax_x, batch_size * input_h * input_w * sizeof(double));
    cudaMalloc(&d_softmax_m, batch_size * input_h * sizeof(double));

    cudaMemcpy(d_softmax_x, softmax_x, batch_size * input_h * input_w * sizeof(double), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size * input_h + threadsPerBlock - 1) / threadsPerBlock;
    init_softmax_m_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_softmax_m, batch_size * input_h);

    
    dim3 grid(batch_size, input_h);
    get_max_kernel<<<grid, 1>>>(d_softmax_x, d_softmax_m, batch_size, input_h, input_w);

    cudaMemcpy(softmax_m, d_softmax_m, batch_size * input_h * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_softmax_x);
    cudaFree(d_softmax_m);
}

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 3D arraies */
/*
    def get_sum(x, axis=-1, keepdims=True):
        if axis == -1 and keepdims == True:
            sum_x = np.zeros(x.shape[:-1])
            for i in range(x.shape[-1]):
                sum_x += x[..., i]
            sum_x = np.expand_dims(sum_x, axis=-1)
        else:
            raise NotImplementedError("Not implemented yet  for axis != -1 or keepdims != True")
        return sum_x
*/

__global__ void init_softmax_l_kernel(double *d_softmax_l, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_softmax_l[idx] = 0;
    }
}

__global__ void get_sum_kernel(double *d_softmax_exp_result, double *d_softmax_l, int batch_size, int input_h, int input_w)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    if (i < batch_size && j < input_h) {
        for (int k = 0; k < input_w; k++) {
            d_softmax_l[i * input_h + j] += d_softmax_exp_result[i * input_h * input_w + j * input_w + k];
        }
    }
}

void get_sum(double *softmax_exp_result, double *softmax_l, 
                int batch_size, int input_h, int input_w, int axis, bool keepdims) 
{
    double *d_softmax_exp_result, *d_softmax_l;
    cudaMalloc(&d_softmax_exp_result, batch_size * input_h * input_w * sizeof(double));
    cudaMalloc(&d_softmax_l, batch_size * input_h * sizeof(double));

    cudaMemcpy(d_softmax_exp_result, softmax_exp_result, batch_size * input_h * input_w * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size * input_h + threadsPerBlock - 1) / threadsPerBlock;
    init_softmax_l_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_softmax_l, batch_size * input_h);

    dim3 grid(batch_size, input_h);
    get_sum_kernel<<<grid, 1>>>(d_softmax_exp_result, d_softmax_l, batch_size, input_h, input_w);

    cudaMemcpy(softmax_l, d_softmax_l, batch_size * input_h * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_softmax_exp_result);
    cudaFree(d_softmax_l);
}

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 3D arraies */
/*
    def get_exp(x):
        exp_x = np.zeros(x.shape)
        for i in range(x.shape[-1]):
            exp_x[..., i] = np.exp(x[..., i])
        return exp_x
*/
// get_exp(x_minus_m, softmax_exp_result, batch_size, input_h, input_w);

__global__ void get_exp_kernel(double *x_minus_m, double *softmax_exp_result, int batch_size, int input_h, int input_w) 
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    if (i < batch_size && j < input_h) {
        for (int k = threadIdx.x; k < input_w; k += blockDim.x) {
            softmax_exp_result[i * input_h * input_w + j * input_w + k] = exp(x_minus_m[i * input_h * input_w + j * input_w + k]);
        }
    }
}

void get_exp(double *x_minus_m, double *softmax_exp_result, int batch_size, int input_h, int input_w) 
{
    double *d_x_minus_m;
    double *d_softmax_exp_result;

    cudaMalloc(&d_x_minus_m, batch_size * input_h * input_w * sizeof(double));
    cudaMalloc(&d_softmax_exp_result, batch_size * input_h * input_w * sizeof(double));

    cudaMemcpy(d_x_minus_m, x_minus_m, batch_size * input_h * input_w * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    dim3 blocksPerGrid(batch_size, input_h);

    get_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x_minus_m, d_softmax_exp_result, batch_size, input_h, input_w);

    cudaMemcpy(softmax_exp_result, d_softmax_exp_result, batch_size * input_h * input_w * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x_minus_m);
    cudaFree(d_softmax_exp_result);
}

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 3D arraies */
/*
    def softmax(x, axis=-1):
        m = get_max(x, axis=axis, keepdims=True)
        exp_result = get_exp(x - m)
        l = get_sum(exp_result, axis=axis, keepdims=True)
        s = exp_result / l
        return s
*/

__global__ void x_minus_m_kernel(double *d_softmax_x, double *d_softmax_m, double *d_x_minus_m, int batch_size, int input_h, int input_w) {
    int i = blockIdx.x;
    int j = blockIdx.y;

    if (i < batch_size && j < input_h) {
        for (int k = threadIdx.x; k < input_w; k += blockDim.x) {
            d_x_minus_m[i * input_h * input_w + j * input_w + k] = d_softmax_x[i * input_h * input_w + j * input_w + k] - d_softmax_m[i * input_h + j];
        }
    }
}

__global__ void softmax_output_kernel(double *d_softmax_exp_result, double *d_softmax_l, double *d_softmax_output, int batch_size, int input_h, int input_w) {
    int i = blockIdx.x;
    int j = blockIdx.y;

    if (i < batch_size && j < input_h) {
        for (int k = threadIdx.x; k < input_w; k += blockDim.x) {
            d_softmax_output[i * input_h * input_w + j * input_w + k] =
                d_softmax_exp_result[i * input_h * input_w + j * input_w + k] / d_softmax_l[i * input_h + j];
        }
    }
}

void softmax(double *softmax_x, double *softmax_output, 
             int batch_size, int input_h, int input_w, int axis) {
    double *softmax_m = new double[batch_size * input_h];
    double *x_minus_m = new double[batch_size * input_h * input_w];
    double *softmax_exp_result = new double[batch_size * input_h * input_w];
    double *softmax_l = new double[batch_size * input_h];

    get_max(softmax_x, softmax_m, batch_size, input_h, input_w, axis, true);

    double *d_softmax_x;
    double *d_softmax_m;
    double *d_x_minus_m;
    
    cudaMalloc(&d_softmax_x, batch_size * input_h * input_w * sizeof(double));
    cudaMalloc(&d_softmax_m, batch_size * input_h * sizeof(double));
    cudaMalloc(&d_x_minus_m, batch_size * input_h * input_w * sizeof(double));
    
    cudaMemcpy(d_softmax_x, softmax_x, batch_size * input_h * input_w * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_softmax_m, softmax_m, batch_size * input_h * sizeof(double), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    dim3 blocksPerGrid(batch_size, input_h);
    x_minus_m_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_softmax_x, d_softmax_m, d_x_minus_m, batch_size, input_h, input_w);

    cudaMemcpy(x_minus_m, d_x_minus_m, batch_size * input_h * input_w * sizeof(double), cudaMemcpyDeviceToHost);

    get_exp(x_minus_m, softmax_exp_result, batch_size, input_h, input_w);
    get_sum(softmax_exp_result, softmax_l, batch_size, input_h, input_w, axis, true);

    double *d_softmax_exp_result;
    double *d_softmax_l;
    double *d_softmax_output;

    cudaMalloc(&d_softmax_exp_result, batch_size * input_h * input_w * sizeof(double));
    cudaMalloc(&d_softmax_l, batch_size * input_h * sizeof(double));
    cudaMalloc(&d_softmax_output, batch_size * input_h * input_w * sizeof(double));
    
    cudaMemcpy(d_softmax_exp_result, softmax_exp_result, batch_size * input_h * input_w * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_softmax_l, softmax_l, batch_size * input_h * sizeof(double), cudaMemcpyHostToDevice);

    softmax_output_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_softmax_exp_result, d_softmax_l, d_softmax_output, batch_size, input_h, input_w);

    
    cudaMemcpy(softmax_output, d_softmax_output, batch_size * input_h * input_w * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_softmax_x);
    cudaFree(d_softmax_m);
    cudaFree(d_x_minus_m);
    cudaFree(d_softmax_exp_result);
    cudaFree(d_softmax_l);
    cudaFree(d_softmax_output);

    delete[] softmax_m;
    delete[] x_minus_m;
    delete[] softmax_exp_result;
    delete[] softmax_l;
}
