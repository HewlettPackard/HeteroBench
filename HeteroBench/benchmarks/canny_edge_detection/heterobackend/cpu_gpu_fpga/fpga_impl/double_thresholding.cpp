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
#include <cstring>
#include <iostream>
#include <math.h>

using namespace std;

void double_thresholding(double *suppressed_image, int height, int width,
                  int high_threshold, int low_threshold,
                  uint8_t *outImage) {

  #pragma HLS interface m_axi offset=slave bundle=double_in port=suppressed_image max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=suppressed_image

  #pragma HLS interface s_axilite bundle=control port=height
  #pragma HLS interface s_axilite bundle=control port=width
  #pragma HLS interface s_axilite bundle=control port=high_threshold
  #pragma HLS interface s_axilite bundle=control port=low_threshold

  #pragma HLS interface m_axi offset=slave bundle=uint8_out port=outImage max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=outImage

  #pragma HLS interface s_axilite bundle=control port=return

  for (int col = 0; col < width; col++) {
    for (int row = 0; row < height; row++) {
      #pragma HLS PIPELINE
      int pxIndex = col + (row * width);
      if (suppressed_image[pxIndex] > high_threshold)
        outImage[pxIndex] = 255;   // Strong edge
      else if (suppressed_image[pxIndex] > low_threshold)
        outImage[pxIndex] = 100;   // Weak edge
      else
        outImage[pxIndex] = 0;     // Not an edge
    }
  }
}
