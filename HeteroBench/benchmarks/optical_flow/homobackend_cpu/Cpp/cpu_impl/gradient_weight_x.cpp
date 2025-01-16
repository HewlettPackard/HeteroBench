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
 
#include "cpu_impl.h"
#include <cstdio>

// compute x weight
void gradient_weight_x(gradient_t y_filt[MAX_HEIGHT][MAX_WIDTH],
                       gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH])
{
  for (int r = 0; r < MAX_HEIGHT; r ++)
  {
    for (int c = 0; c < MAX_WIDTH + 3; c ++)
    {
      gradient_t acc;
      acc.x = 0;
      acc.y = 0;
      acc.z = 0;
      if (c >= 6 && c < MAX_WIDTH)
      {
        for (int i = 0; i < 7; i ++)
        {
          acc.x += y_filt[r][c-i].x * GRAD_FILTER[i];
          acc.y += y_filt[r][c-i].y * GRAD_FILTER[i];
          acc.z += y_filt[r][c-i].z * GRAD_FILTER[i];
        }
        filt_grad[r][c-3] = acc;
      }
      else if (c >= 3)
      {
        filt_grad[r][c-3] = acc;
      }
    }
  }
}