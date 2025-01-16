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

// compute flow
void flow_calc(tensor_t *tensor, velocity_t *outputs)
{
  #pragma omp parallel for collapse(2)
  for(int r = 0; r < MAX_HEIGHT; r ++)
  {
    for(int c = 0; c < MAX_WIDTH; c ++)
    {
      if (r >= 2 && r < MAX_HEIGHT - 2 && c >= 2 && c < MAX_WIDTH - 2)
      {
        pixel_t denom = tensor[r*MAX_HEIGHT+c].val[0] * tensor[r*MAX_HEIGHT+c].val[1] -
                        tensor[r*MAX_HEIGHT+c].val[3] * tensor[r*MAX_HEIGHT+c].val[3];
        outputs[r*MAX_HEIGHT+c].x = (tensor[r*MAX_HEIGHT+c].val[5] * tensor[r*MAX_HEIGHT+c].val[3] -
                          tensor[r*MAX_HEIGHT+c].val[4] * tensor[r*MAX_HEIGHT+c].val[1]) / denom;
        outputs[r*MAX_HEIGHT+c].y = (tensor[r*MAX_HEIGHT+c].val[4] * tensor[r*MAX_HEIGHT+c].val[3] -
                          tensor[r*MAX_HEIGHT+c].val[5] * tensor[r*MAX_HEIGHT+c].val[0]) / denom;
      }
      else
      {
        outputs[r*MAX_HEIGHT+c].x = 0;
        outputs[r*MAX_HEIGHT+c].y = 0;
      }
    }
  }
}