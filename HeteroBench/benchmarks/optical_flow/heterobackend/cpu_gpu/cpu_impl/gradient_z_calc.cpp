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

// compute z gradient
void gradient_z_calc(pixel_t *frame0, 
                     pixel_t *frame1,
                     pixel_t *frame2,
                     pixel_t *frame3,
                     pixel_t *frame4,
                     pixel_t *gradient_z)
{
  #pragma omp parallel for collapse(2)
  for (int r = 0; r < MAX_HEIGHT; r ++)
  {
    for (int c = 0; c < MAX_WIDTH; c ++)
    {
      gradient_z[r*MAX_HEIGHT+c] = 0.0f;
      gradient_z[r*MAX_HEIGHT+c] += frame0[r*MAX_HEIGHT+c] * GRAD_WEIGHTS[0]; 
      gradient_z[r*MAX_HEIGHT+c] += frame1[r*MAX_HEIGHT+c] * GRAD_WEIGHTS[1]; 
      gradient_z[r*MAX_HEIGHT+c] += frame2[r*MAX_HEIGHT+c] * GRAD_WEIGHTS[2]; 
      gradient_z[r*MAX_HEIGHT+c] += frame3[r*MAX_HEIGHT+c] * GRAD_WEIGHTS[3]; 
      gradient_z[r*MAX_HEIGHT+c] += frame4[r*MAX_HEIGHT+c] * GRAD_WEIGHTS[4]; 
      gradient_z[r*MAX_HEIGHT+c] /= 12.0f;
    }
  }
}