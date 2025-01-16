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

// compute z gradient using streams
void gradient_z_calc_st(
  pixel_t frame0[MAX_HEIGHT][MAX_WIDTH],
  pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
  pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
  pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
  pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
  stream<pixel_t>& gradient_z)
{
  for (int r = 0; r < MAX_HEIGHT; r++)
  {
    for (int c = 0; c < MAX_WIDTH; c++)
    {
#pragma HLS pipeline II=1
      pixel_t grad_z = 0.0f;
      grad_z += frame0[r][c] * GRAD_WEIGHTS[0];
      grad_z += frame1[r][c] * GRAD_WEIGHTS[1];
      grad_z += frame2[r][c] * GRAD_WEIGHTS[2];
      grad_z += frame3[r][c] * GRAD_WEIGHTS[3];
      grad_z += frame4[r][c] * GRAD_WEIGHTS[4];
      grad_z /= 12.0f;

      gradient_z.write(grad_z);
    }
  }
}