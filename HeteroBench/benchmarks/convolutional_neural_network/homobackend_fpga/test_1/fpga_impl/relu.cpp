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
def relu(x):
  return np.maximum(0, x)
*/

void relu(double *relu_input, double *relu_output, int size) {
  #pragma HLS interface m_axi offset=slave bundle=double_in port=relu_input max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=relu_input

  #pragma HLS interface m_axi offset=slave bundle=double_out port=relu_output max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=relu_output

  #pragma HLS interface s_axilite bundle=control port=size

  #pragma HLS interface s_axilite bundle=control port=return

  for (int i = 0; i < size; i++) {
    #pragma HLS pipeline II=1
    relu_output[i] = hls::max(0.0, relu_input[i]);
  }
}