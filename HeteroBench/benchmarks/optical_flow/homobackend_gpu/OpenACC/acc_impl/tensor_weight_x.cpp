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
 
#include "acc_impl.h"
#include <cstdio>

// tensor weight x
void tensor_weight_x(tensor_t *tensor_y,
                     tensor_t *tensor)
{
  #pragma acc data copyin(tensor_y[0:MAX_HEIGHT*MAX_WIDTH]) \
                   create(tensor[0:MAX_HEIGHT*MAX_WIDTH]) \
                   copyin(TENSOR_FILTER[0:3]) \
                   copyout(tensor[0:MAX_HEIGHT*MAX_WIDTH])
  {
    #pragma acc parallel loop collapse(2)
    for (int r = 0; r < MAX_HEIGHT; r ++)
    {
      for (int c = 0; c < MAX_WIDTH + 1; c ++)
      {
        tensor_t acc;
        for(int k = 0; k < 6; k++)
        {
          acc.val[k] = 0;
        }
        if (c >= 2 && c < MAX_WIDTH) 
        {
          for (int i = 0; i < 3; i ++)
          {
            for (int component = 0; component < 6; component ++)
            {
              acc.val[component] += tensor_y[r*MAX_HEIGHT+c-i].val[component] * TENSOR_FILTER[i];
            }
          }
        }
        if (c >= 1)
        {
          tensor[r*MAX_HEIGHT+c-1] = acc;
        }
      }
    }
  }
}