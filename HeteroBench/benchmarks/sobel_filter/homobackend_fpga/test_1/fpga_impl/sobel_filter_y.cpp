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
#include <cstdio>

using namespace std;

void sobel_filter_y(const uint8_t *input_image, int height, int width, double *sobel_y) {
  
  #pragma HLS interface m_axi offset=slave bundle=uint8_t_in port=input_image max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=input_image

  #pragma HLS interface m_axi offset=slave bundle=double_out port=sobel_y max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=sobel_y

  #pragma HLS interface s_axilite bundle=control port=height
  #pragma HLS interface s_axilite bundle=control port=width

  #pragma HLS interface s_axilite bundle=control port=return
  
  const int kernel_y[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
  };
  
  for (int row = 1; row < height - 1; ++row) {
    for (int col = 1; col < width - 1; ++col) {
      #pragma HLS pipeline II=1
      double gy = 0;
      for (int krow = -1; krow <= 1; ++krow) {
        for (int kcol = -1; kcol <= 1; ++kcol) {
          int pixel_val = input_image[(row + krow) * width + (col + kcol)];
          gy += pixel_val * kernel_y[krow + 1][kcol + 1];
        }
      }
      sobel_y[row * width + col] = gy;
    }
  }
}