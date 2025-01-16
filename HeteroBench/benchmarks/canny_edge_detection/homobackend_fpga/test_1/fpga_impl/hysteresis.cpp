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

using namespace std;

void hysteresis(uint8_t *inImage, int height, int width,
                uint8_t *outImage)
{ 
  #pragma HLS interface m_axi offset=slave bundle=uint8_in port=inImage max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=inImage

  #pragma HLS interface s_axilite bundle=control port=height
  #pragma HLS interface s_axilite bundle=control port=width

  #pragma HLS interface m_axi offset=slave bundle=uint8_out port=outImage max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=outImage

  #pragma HLS interface s_axilite bundle=control port=return

  // memcpy(outImage, inImage, width * height * sizeof(uint8_t));
  for (int i = 0; i < width * height; i++) {
    // #pragma HLS pipeline II=1
    outImage[i] = inImage[i];
  }

  for (int col = OFFSET; col < width - OFFSET; col++) {
    for (int row = OFFSET; row < height - OFFSET; row++) {
      #pragma HLS pipeline II=1
      int pxIndex = col + (row * width);
      if (outImage[pxIndex] == 100) {
        if (outImage[pxIndex - 1] == 255 ||
            outImage[pxIndex + 1] == 255 ||
            outImage[pxIndex - width] == 255 ||
            outImage[pxIndex + width] == 255 ||
            outImage[pxIndex - width - 1] == 255 ||
            outImage[pxIndex - width + 1] == 255 ||
            outImage[pxIndex + width - 1] == 255 ||
            outImage[pxIndex + width + 1] == 255)
          outImage[pxIndex] = 255;
        else
          outImage[pxIndex] = 0;
      }
    }
  }
}
