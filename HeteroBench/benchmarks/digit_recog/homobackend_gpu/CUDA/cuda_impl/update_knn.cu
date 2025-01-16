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

__device__ void popcount(DigitType diff, int* popcount_result)
{
    diff -= (diff >> 1) & m1;             //put count of each 2 bits into those 2 bits
    diff = (diff & m2) + ((diff >> 2) & m2); //put count of each 4 bits into those 4 bits 
    diff = (diff + (diff >> 4)) & m4;        //put count of each 8 bits into those 8 bits 
    diff += diff >>  8;  //put count of each 16 bits into their lowest 8 bits
    diff += diff >> 16;  //put count of each 32 bits into their lowest 8 bits
    diff += diff >> 32;  //put count of each 64 bits into their lowest 8 bits
    *popcount_result = diff & 0x7f;
}

__device__ void update(const DigitType* d_training_set, const DigitType* d_test_set, int d_dists[K_CONST], int d_labels[K_CONST], int label)
{
  int dist = 0;

  for (int i = 0; i < DIGIT_WIDTH; i++)
  {
    DigitType diff = d_test_set[i] ^ d_training_set[i];
    // dist += popcount(diff);
    int popcount_result = 0;
    popcount(diff, &popcount_result);
    dist += popcount_result;
  }

  int max_dist = 0;
  int max_dist_id = K_CONST + 1;

  // Find the max distance
  //#pragma omp atomic capture
  for (int k = 0; k < K_CONST; ++k)
  {
    if (d_dists[k] > max_dist)
    {
      max_dist = d_dists[k];
      max_dist_id = k;
    }
  }

  // Replace the entry with the max distance
  if (dist < max_dist)
  {
    d_dists[max_dist_id] = dist;
    d_labels[max_dist_id] = label;
  }
}

// CUDA Kernel for update_knn
__global__ void update_knn_kernel(const DigitType *d_training_set, const DigitType *d_test_set, int *d_dists, int *d_labels)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < NUM_TRAINING) {
    int label = i / CLASS_SIZE;
    update(&d_training_set[i * DIGIT_WIDTH], d_test_set, d_dists, d_labels, label);
  }
}

// Host function to launch the CUDA kernel
void update_knn(const DigitType *training_set, const DigitType *test_set, int *dists, int *labels)
{
  // #pragma acc data copyin(training_set[0:NUM_TRAINING * DIGIT_WIDTH], test_set[0:DIGIT_WIDTH]) \
                   copyin(dists[0:K_CONST], labels[0:K_CONST]) \
                   copyout(dists[0:K_CONST], labels[0:K_CONST])

  DigitType *d_training_set;
  DigitType *d_test_set;
  int *d_dists;
  int *d_labels;

  int sizeTraining = NUM_TRAINING * DIGIT_WIDTH * sizeof(DigitType);
  int sizeTest = DIGIT_WIDTH * sizeof(DigitType);
  int size = K_CONST * sizeof(int);
  cudaMalloc(&d_training_set, sizeTraining);
  cudaMalloc(&d_test_set, sizeTest);
  cudaMalloc(&d_dists, size);
  cudaMalloc(&d_labels, size);

  cudaMemcpy(d_training_set, training_set, sizeTraining, cudaMemcpyHostToDevice);
  cudaMemcpy(d_test_set, test_set, sizeTest, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dists, dists, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_labels, labels, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (NUM_TRAINING + threadsPerBlock - 1) / threadsPerBlock;

  update_knn_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_training_set, d_test_set, d_dists, d_labels);
  
  cudaMemcpy(dists, d_dists, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(labels, d_labels, size, cudaMemcpyDeviceToHost);

  cudaFree(d_training_set);
  cudaFree(d_test_set);
  cudaFree(d_dists);
  cudaFree(d_labels);
}