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

// Compute x weight using streams
void gradient_weight_x_st(stream<gradient_t>& y_filt, stream<gradient_t>& filt_grad) {
  gradient_t line_buffer[7];
  // #pragma HLS BIND_STORAGE variable=line_buffer type=RAM_S2P impl=AUTO
  for (int r = 0; r < MAX_HEIGHT; r++) {

    // Initialize the line buffer
    for (int i = 6 - 1; i >= 0; i--) {
      line_buffer[i] = y_filt.read();
    }

    for (int c = 0; c < MAX_WIDTH + 3; c++) {
#pragma HLS pipeline II=1
      gradient_t acc = { 0, 0, 0 };

      if (c >= 6 && c < MAX_WIDTH) {
        gradient_t temp;

        // Shift the line buffer
        for (int i = 6; i > 0; i--) {
          line_buffer[i] = line_buffer[i - 1];
        }
        temp = y_filt.read();
        line_buffer[0] = temp;

        // Compute the weighted sum
        for (int i = 0; i < 7; i++) {
          acc.x += line_buffer[i].x * GRAD_FILTER[i];
          acc.y += line_buffer[i].y * GRAD_FILTER[i];
          acc.z += line_buffer[i].z * GRAD_FILTER[i];
        }
        filt_grad.write(acc);
      }
      else if (c >= 3) {
        filt_grad.write(acc);
      }
    }
  }
}