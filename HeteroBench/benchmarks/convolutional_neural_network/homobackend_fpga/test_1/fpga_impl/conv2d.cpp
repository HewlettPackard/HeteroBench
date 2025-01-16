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
 
#include "fpga_impl.h"
#include <iostream>
#include <cstdio>
#include <hls_math.h>

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
def pad_input(input, padding):
  if padding == 0:
    return input
  padded_input = np.zeros((input.shape[0] + 2*padding, input.shape[1] + 2*padding))
  for i in range(input.shape[0]):
    for j in range(input.shape[1]):
      padded_input[i + padding][j + padding] = input[i][j]
  return padded_input
*/

void pad_input(double *pad_input_input, double *pad_input_output, int input_h, int input_w, int padding) {
  if (padding == 0) {
    for (int i = 0; i < input_h; i++) {
      for (int j = 0; j < input_w; j++) {
        #pragma HLS pipeline II=1
        pad_input_output[i * input_w + j] = pad_input_input[i * input_w + j];
      }
    }
    return;
  }
  for (int i = 0; i < input_h + 2 * padding; i++) {
    for (int j = 0; j < input_w + 2 * padding; j++) {
      #pragma HLS pipeline II=1
      pad_input_output[i * (input_w + 2 * padding) + j] = 0;
    }
  }
  for (int i = 0; i < input_h; i++) {
    for (int j = 0; j < input_w; j++) {
      #pragma HLS pipeline II=1
      pad_input_output[(i + padding) * (input_w + 2 * padding) + j + padding] = pad_input_input[i * input_w + j];
    }
  }
}

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
  #pragma HLS interface m_axi offset=slave bundle=double_in port=conv2d_input max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=conv2d_input

  #pragma HLS interface m_axi offset=slave bundle=double_in port=conv2d_kernel max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=conv2d_kernel

  #pragma HLS interface m_axi offset=slave bundle=double_in port=input_padded max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=input_padded

  #pragma HLS interface s_axilite bundle=control port=conv2d_bias
  #pragma HLS interface s_axilite bundle=control port=stride
  #pragma HLS interface s_axilite bundle=control port=padding
  #pragma HLS interface s_axilite bundle=control port=input_h
  #pragma HLS interface s_axilite bundle=control port=input_w
  #pragma HLS interface s_axilite bundle=control port=kernel_h
  #pragma HLS interface s_axilite bundle=control port=kernel_w

  #pragma HLS interface m_axi offset=slave bundle=double_out port=conv2d_output max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=conv2d_output

  #pragma HLS interface s_axilite bundle=control port=return

  pad_input(conv2d_input, input_padded, input_h, input_w, padding);
  int output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
  int output_w = (input_w + 2 * padding - kernel_w) / stride + 1;

  double tmp = 0;
  for (int i = 0; i < output_h; i++) {
    for (int j = 0; j < output_w; j++) {
      tmp = 0;
      for (int k = 0; k < kernel_h; k++) {
        for (int l = 0; l < kernel_w; l++) {
          #pragma HLS pipeline II=1
          tmp += input_padded[(i * stride + k) * (input_w + 2 * padding) + j * stride + l] * conv2d_kernel[k * kernel_w + l];
        }
      }
      conv2d_output[i * output_w + j] = tmp + conv2d_bias;
    }
  }
}