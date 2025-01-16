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

// compute y weight
void gradient_weight_y_st(
  stream<pixel_t>& gradient_x_stream,
  stream<pixel_t>& gradient_y_stream,
  stream<pixel_t>& gradient_z_stream,
  stream<gradient_t>& filt_grad_stream)
{
  // Initialize buffers
  pixel_t buffer_x[GRAD_FILTER_SIZE][MAX_WIDTH];
  pixel_t buffer_y[GRAD_FILTER_SIZE][MAX_WIDTH];
  pixel_t buffer_z[GRAD_FILTER_SIZE][MAX_WIDTH];


  for (int r = 0; r < MAX_HEIGHT + 3; r++)
  {
    for (int c = 0; c < MAX_WIDTH; c++)
    {
#pragma HLS pipeline II=1

      // Shift the buffer for each column
      for (int i = 0; i < GRAD_FILTER_SIZE - 1; i++)
      {
        buffer_x[i][c] = buffer_x[i + 1][c];
        buffer_y[i][c] = buffer_y[i + 1][c];
        buffer_z[i][c] = buffer_z[i + 1][c];
      }

      // Load new value into buffer's last row for each column
      buffer_x[GRAD_FILTER_SIZE - 1][c] = (r < MAX_HEIGHT) ? gradient_x_stream.read() : 0;
      buffer_y[GRAD_FILTER_SIZE - 1][c] = (r < MAX_HEIGHT) ? gradient_y_stream.read() : 0;
      buffer_z[GRAD_FILTER_SIZE - 1][c] = (r < MAX_HEIGHT) ? gradient_z_stream.read() : 0;

      gradient_t acc;
      acc.x = 0;
      acc.y = 0;
      acc.z = 0;
      if (r >= 6 && r < MAX_HEIGHT)
      {
        for (int i = 0; i < GRAD_FILTER_SIZE; i++)
        {
          acc.x += buffer_x[6 - i][c] * GRAD_FILTER[i];
          acc.y += buffer_y[6 - i][c] * GRAD_FILTER[i];
          acc.z += buffer_z[6 - i][c] * GRAD_FILTER[i];
        }
        filt_grad_stream.write(acc);
      }
      else if (r >= 3)
      {
        filt_grad_stream.write(acc);
      }
    }
  }
}