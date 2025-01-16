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
void dot_add2(double* dot_add_input_x, double* dot_add_input_W, double* dot_add_input_b, double* dot_add_output) {

#pragma HLS interface m_axi offset=slave bundle=double_inout port=dot_add_input_x max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=dot_add_input_x

#pragma HLS interface m_axi offset=slave bundle=double_inout  port=dot_add_input_W max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=dot_add_input_W

#pragma HLS interface m_axi offset=slave bundle=double_inout  port=dot_add_input_b max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=dot_add_input_b

#pragma HLS interface m_axi offset=slave bundle=double_inout port=dot_add_output max_write_burst_length=16 num_write_outstanding=64
#pragma HLS interface s_axilite bundle=control port=dot_add_output

//#pragma HLS interface s_axilite bundle=control port=x_h
//#pragma HLS interface s_axilite bundle=control port=x_w
//#pragma HLS interface s_axilite bundle=control port=W_h
//#pragma HLS interface s_axilite bundle=control port=W_w

#pragma HLS interface s_axilite bundle=control port=return

  double tmp = 0;
  for (int i = 0; i < x_h2; i++) {

    for (int j = 0; j < W_w2; j++) {
      tmp = 0;
      for (int k = 0; k < x_w2; k++) {
#pragma HLS pipeline II=1
        tmp += dot_add_input_x[i * x_w2 + k] * dot_add_input_W[k * W_w2 + j];
      }
      dot_add_output[i * W_w2 + j] = tmp;
    }
  }
  for (int i = 0; i < x_h2; i++) {

    for (int j = 0; j < W_w2; j++) {
#pragma HLS pipeline II=1
      dot_add_output[i * W_w2 + j] = dot_add_output[i * W_w2 + j] + dot_add_input_b[j];
    }
  }
}
