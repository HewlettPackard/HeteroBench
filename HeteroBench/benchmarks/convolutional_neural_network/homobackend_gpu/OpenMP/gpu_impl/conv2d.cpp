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
 
#include "gpu_impl.h"
#include <iostream>
#include <math.h>

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
def conv2d(input, kernel, bias, stride, padding):
  input_padded = pad_input(input, padding)
  kernel_height, kernel_width = kernel.shape
  output_height = (input_padded.shape[0] - kernel_height) // stride + 1
  output_width = (input_padded.shape[1] - kernel_width) // stride + 1
  conv2d_output = np.zeros((output_height, output_width))
  for i in range(0, output_height):
    for j in range(0, output_width):
      region = input_padded[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
      conv2d_output[i, j] = np.sum(region * kernel) + bias
  return conv2d_output
*/

void conv2d(double *conv2d_input, double *conv2d_kernel, double *input_padded, double conv2d_bias, int stride, int padding, int input_h, int input_w, int kernel_h, int kernel_w, double *conv2d_output) 
{
  pad_input(conv2d_input, input_padded, input_h, input_w, padding);

  #pragma omp target enter data map(to: conv2d_input[0:input_h*input_w])
  #pragma omp target enter data map(to: conv2d_kernel[0:kernel_h*kernel_w])
  #pragma omp target enter data map(alloc: input_padded[0:(input_h + 2 * padding) * (input_w + 2 * padding)])
  #pragma omp target enter data map(to: conv2d_bias)
  #pragma omp target enter data map(to: stride)
  #pragma omp target enter data map(to: padding)
  #pragma omp target enter data map(to: input_h)
  #pragma omp target enter data map(to: input_w)
  #pragma omp target enter data map(to: kernel_h)
  #pragma omp target enter data map(to: kernel_w)
  #pragma omp target enter data map(alloc: conv2d_output[0:((input_h + 2 * padding - kernel_h) / stride + 1) * ((input_w + 2 * padding - kernel_w) / stride + 1)])

  int output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
  int output_w = (input_w + 2 * padding - kernel_w) / stride + 1;

  double tmp = 0;
  #pragma omp target enter data map(to: output_h)
  #pragma omp target enter data map(to: output_w)
  #pragma omp target enter data map(to: tmp)
  #pragma omp target teams distribute parallel for collapse(2) reduction(+:tmp)
  for (int i = 0; i < output_h; i++) {
    for (int j = 0; j < output_w; j++) {
      tmp = 0;
      for (int k = 0; k < kernel_h; k++) {
        for (int l = 0; l < kernel_w; l++) {
          tmp += input_padded[(i * stride + k) * (input_w + 2 * padding) + j * stride + l] * conv2d_kernel[k * kernel_w + l];
        }
      }
      conv2d_output[i * output_w + j] = tmp + conv2d_bias;
    }
  }
  #pragma omp target exit data map(from: conv2d_output[0:((input_h + 2 * padding - kernel_h) / stride + 1) * ((input_w + 2 * padding - kernel_w) / stride + 1)])
  #pragma omp target exit data map(release: output_h)
  #pragma omp target exit data map(release: output_w)
  #pragma omp target exit data map(release: tmp)
  #pragma omp target exit data map(release: conv2d_input)
  #pragma omp target exit data map(release: conv2d_kernel)
  #pragma omp target exit data map(release: input_padded)
  #pragma omp target exit data map(release: conv2d_bias)
  #pragma omp target exit data map(release: stride)
  #pragma omp target exit data map(release: padding)
  #pragma omp target exit data map(release: input_h)
  #pragma omp target exit data map(release: input_w)
  #pragma omp target exit data map(release: kernel_h)
  #pragma omp target exit data map(release: kernel_w)
  
}