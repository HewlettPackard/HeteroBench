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

// compute x, y gradient
void gradient_xy_calc_st(pixel_t frame[MAX_HEIGHT][MAX_WIDTH],
  stream<pixel_t>& gradient_x,
  stream<pixel_t>& gradient_y)
{
  pixel_t x_grad, y_grad;
  pixel_t buffer[GRAD_WEIGHTS_SIZE][MAX_WIDTH + 2];



  for (int r = 0; r < MAX_HEIGHT + 2; r++)
  {
    for (int c = 0; c < MAX_WIDTH + 2; c++)
    {
      // Shift the buffer for each column
      for (int i = 0; i < GRAD_WEIGHTS_SIZE - 1; i++)
      {
        buffer[i][c] = buffer[i + 1][c];
      }

      // Load new value into buffer
      buffer[GRAD_WEIGHTS_SIZE - 1][c] = (r < MAX_HEIGHT && c < MAX_WIDTH) ? frame[r][c] : 0;

      x_grad = 0;
      y_grad = 0;
      if (r >= 4 && r < MAX_HEIGHT && c >= 4 && c < MAX_WIDTH)
      {
        for (int i = 0; i < GRAD_WEIGHTS_SIZE; i++)
        {
          x_grad += buffer[4 - 2][c - i] * GRAD_WEIGHTS[4 - i];
          y_grad += buffer[4 - i][c - 2] * GRAD_WEIGHTS[4 - i];
        }
        gradient_x.write(x_grad / 12);
        gradient_y.write(y_grad / 12);
      }
      else if (r >= 2 && c >= 2)
      {
        gradient_x.write(0);
        gradient_y.write(0);
      }
    }
  }
}