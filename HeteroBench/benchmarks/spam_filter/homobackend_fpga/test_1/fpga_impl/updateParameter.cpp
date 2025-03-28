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
#include "math.h"

// Update the parameter vector
void updateParameter(
    FeatureType param[NUM_FEATURES],
    FeatureType grad[NUM_FEATURES],
    FeatureType step_size)
{
  #pragma HLS interface m_axi offset=slave bundle=FeatureType_in port=param max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=param

  #pragma HLS interface m_axi offset=slave bundle=FeatureType_in port=grad max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=grad

  #pragma HLS interface s_axilite bundle=control port=step_size

  #pragma HLS interface s_axilite bundle=control port=return

  for (int i = 0; i < NUM_FEATURES; i++) {
    #pragma HLS pipeline II=1
    param[i] += step_size * grad[i];
  }
}
