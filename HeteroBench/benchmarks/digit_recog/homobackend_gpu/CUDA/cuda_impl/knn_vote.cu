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
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel for knn_vote_count
__global__ void knn_vote_count_kernel(int *labels, int* votes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < K_CONST) {
        atomicAdd(&votes[labels[idx]], 1);
    }
}

__global__ void find_max_vote_kernel(int* votes, LabelType* max_label, int* max_vote) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int shared_max_vote;
    __shared__ LabelType shared_max_label;

    if (threadIdx.x == 0) {
        shared_max_vote = 0;
        shared_max_label = 0;
    }
    __syncthreads();

    if (idx < 10) {
        if (votes[idx] > shared_max_vote) {
            atomicMax(&shared_max_vote, votes[idx]);
            shared_max_label = idx;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        *max_vote = shared_max_vote;
        *max_label = shared_max_label;
    }
}

// Host function to launch the CUDA kernel
void knn_vote(int labels[K_CONST], LabelType* max_label)
{    
    int* d_labels;
    int* d_votes;
    LabelType* d_max_label;
    int* d_max_vote;

    cudaMalloc((void**)&d_labels, K_CONST * sizeof(int));
    cudaMalloc((void**)&d_votes, 10 * sizeof(int));
    cudaMalloc((void**)&d_max_label, sizeof(LabelType));
    cudaMalloc((void**)&d_max_vote, sizeof(int));
    
    cudaMemcpy(d_labels, labels, K_CONST * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_votes, 0, 10 * sizeof(int));
    
    int threadsPerBlock = 128;
    int blocksPerGrid = (K_CONST + threadsPerBlock - 1) / threadsPerBlock;
    
    knn_vote_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_labels, d_votes);
    cudaDeviceSynchronize();
    
    find_max_vote_kernel<<<1, 10>>>(d_votes, d_max_label, d_max_vote);
    cudaDeviceSynchronize();

    cudaMemcpy(max_label, d_max_label, sizeof(LabelType), cudaMemcpyDeviceToHost);

    cudaFree(d_labels);
    cudaFree(d_votes);
    cudaFree(d_max_label);
    cudaFree(d_max_vote);
}