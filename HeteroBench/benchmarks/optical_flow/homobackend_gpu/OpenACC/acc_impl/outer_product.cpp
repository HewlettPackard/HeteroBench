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

// outer product
void outer_product(gradient_t *filtered_gradient,
                   outer_t *out_product)
{ 
  #pragma acc data copyin(filtered_gradient[0:MAX_HEIGHT*MAX_WIDTH]) \
                   create(out_product[0:MAX_HEIGHT*MAX_WIDTH]) \
                   copyout(out_product[0:MAX_HEIGHT*MAX_WIDTH])
  {
    #pragma acc parallel loop collapse(2)
    for (int r = 0; r < MAX_HEIGHT; r ++)
    {
      for (int c = 0; c < MAX_WIDTH; c ++)
      {
        gradient_t grad = filtered_gradient[r*MAX_HEIGHT+c];
        outer_t out;
        out.val[0] = grad.x * grad.x;
        out.val[1] = grad.y * grad.y;
        out.val[2] = grad.z * grad.z;
        out.val[3] = grad.x * grad.y;
        out.val[4] = grad.x * grad.z;
        out.val[5] = grad.y * grad.z;
        out_product[r*MAX_HEIGHT+c] = out;
      }
    }
  }
}
