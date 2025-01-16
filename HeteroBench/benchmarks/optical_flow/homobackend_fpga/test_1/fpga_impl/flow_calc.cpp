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

// compute flow
void flow_calc_st(stream<tensor_t>& tensors,
  stream<velocity_t>& outputs_st)
{


  for (int r = 0; r < MAX_HEIGHT; r++)
  {
    for (int c = 0; c < MAX_WIDTH; c++)
    {
#pragma HLS pipeline II=1
      tensor_t temp_in;
      velocity_t temp_out;
      temp_in = tensors.read();
      if (r >= 2 && r < MAX_HEIGHT - 2 && c >= 2 && c < MAX_WIDTH - 2)
      {


        pixel_t denom = temp_in.val[0] * temp_in.val[1] -
          temp_in.val[3] * temp_in.val[3];
        temp_out.x = (temp_in.val[5] * temp_in.val[3] -
          temp_in.val[4] * temp_in.val[1]) / denom;
        temp_out.y = (temp_in.val[4] * temp_in.val[3] -
          temp_in.val[5] * temp_in.val[0]) / denom;
        outputs_st.write(temp_out);
      }
      else
      {
        temp_out.x = 0;
        temp_out.y = 0;
        outputs_st.write(temp_out);

      }

    }
  }
}