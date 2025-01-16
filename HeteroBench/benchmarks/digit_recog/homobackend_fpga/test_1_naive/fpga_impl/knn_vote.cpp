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

using namespace std;

void knn_vote(int labels[K_CONST], LabelType* max_label) 
{
  #pragma HLS interface m_axi offset=slave bundle=int_in port=labels max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=labels

  #pragma HLS interface m_axi offset=slave bundle=LabelType_out port=max_label
  #pragma HLS interface s_axilite bundle=control port=max_label

  #pragma HLS interface s_axilite bundle=control port=return
  
  int max_vote = 0;
  
  int votes[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  
  for (int i = 0; i < K_CONST; i ++ )
  {
    #pragma HLS pipeline II=1
    votes[labels[i]] ++;
  }
  
  for (int i = 0; i < 10; i ++ ) 
  {
    #pragma HLS pipeline II=1
    if (votes[i] > max_vote)
    {
      max_vote = votes[i];
      *max_label = i;
    }
  }
  

  return;

}