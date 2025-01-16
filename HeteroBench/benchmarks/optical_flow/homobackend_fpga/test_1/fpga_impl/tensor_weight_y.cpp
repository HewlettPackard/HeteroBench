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

void tensor_weight_y_st(stream<outer_t>& outer_st, stream<tensor_t>& tensor_y)
{
  // Initialize buffers
  outer_t buffer[TENSOR_FILTER_SIZE][MAX_WIDTH];

  for (int r = 0; r < MAX_HEIGHT + 1; r++)
  {
    for (int c = 0; c < MAX_WIDTH; c++)
    {
#pragma HLS pipeline II=1

      // Shift the buffer for each column
      for (int i = 0; i < TENSOR_FILTER_SIZE - 1; i++)
      {
        buffer[i][c] = buffer[i + 1][c];
      }

      // Load new value into buffer's last row for each column
      if (r < MAX_HEIGHT)
      {
        outer_t temp = outer_st.read();
        buffer[TENSOR_FILTER_SIZE - 1][c] = temp;
      }
      else
      {
        for (int k = 0; k < COMPONENT_SIZE; k++)
        {
          buffer[TENSOR_FILTER_SIZE - 1][c].val[k] = 0;
        }
      }

      tensor_t acc;
      for (int k = 0; k < COMPONENT_SIZE; k++)
      {
        acc.val[k] = 0;
      }

      if (r >= 2 && r < MAX_HEIGHT)
      {
        for (int i = 0; i < TENSOR_FILTER_SIZE; i++)
        {
          for (int component = 0; component < COMPONENT_SIZE; component++)
          {
            acc.val[component] += buffer[TENSOR_FILTER_SIZE - 1 - i][c].val[component] * TENSOR_FILTER[i];
          }
        }
      }
      if (r >= 1)
      {
        tensor_y.write(acc);
      }
    }
  }
}