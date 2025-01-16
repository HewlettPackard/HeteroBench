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

// Compute tensor weight using streams and a line buffer
void tensor_weight_x_st(stream<tensor_t>& tensor_y_stream, stream<tensor_t>& tensor_stream) {

  tensor_t line_buffer[3];
#pragma HLS ARRAY_PARTITION variable=line_buffer type=complete
#pragma HLS BIND_STORAGE variable=line_buffer type=RAM_2P impl=AUTO
  for (int r = 0; r < MAX_HEIGHT; r++) {


    // Initialize the line buffer with the first two elements
    for (int i = 1; i >= 0; i--) {
      line_buffer[i] = tensor_y_stream.read();
    }
    for (int c = 0; c < MAX_WIDTH + 1; c++) {
#pragma HLS pipeline II=1
      tensor_t acc;
      for (int k = 0; k < 6; k++) {
        acc.val[k] = 0;
      }

      if (c >= 2 && c < MAX_WIDTH) {

        tensor_t temp;
        // Shift the line buffer
        for (int i = 2; i > 0; i--) {
          line_buffer[i] = line_buffer[i - 1];
        }
        // Shift line buffer and read new element
        temp = tensor_y_stream.read();
        line_buffer[0] = temp;

        for (int i = 0; i < 3; i++)
        {
          for (int component = 0; component < 6; component++)
          {
            acc.val[component] += line_buffer[i].val[component] * TENSOR_FILTER[i];
          }
        }

      }
      if (c >= 1) {
        tensor_stream.write(acc);
      }
    }
  }
}
