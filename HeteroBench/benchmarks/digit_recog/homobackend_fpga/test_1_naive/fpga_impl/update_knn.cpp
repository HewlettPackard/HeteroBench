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

using namespace std;

void popcount(DigitType diff, int* popcount_result)
{
    #pragma HLS inline

    diff -= (diff >> 1) & m1;             //put count of each 2 bits into those 2 bits
    diff = (diff & m2) + ((diff >> 2) & m2); //put count of each 4 bits into those 4 bits 
    diff = (diff + (diff >> 4)) & m4;        //put count of each 8 bits into those 8 bits 
    diff += diff >>  8;  //put count of each 16 bits into their lowest 8 bits
    diff += diff >> 16;  //put count of each 32 bits into their lowest 8 bits
    diff += diff >> 32;  //put count of each 64 bits into their lowest 8 bits
    *popcount_result = diff & 0x7f;
    return;
}

void update_knn( const DigitType* training_set, const DigitType* test_set, int dists[K_CONST], int labels[K_CONST], int label)
{
  #pragma HLS interface m_axi offset=slave bundle=DigitType_in port=training_set max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=training_set

  #pragma HLS interface m_axi offset=slave bundle=DigitType_in port=test_set max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=test_set

  #pragma HLS interface m_axi offset=slave bundle=int_out port=dists max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=dists

  #pragma HLS interface m_axi offset=slave bundle=int_out port=labels max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=labels

  #pragma HLS interface s_axilite bundle=control port=label

  #pragma HLS interface s_axilite bundle=control port=return


  int dist = 0;
  
  for (int i = 0; i < DIGIT_WIDTH; i ++ )
  {
    #pragma HLS pipeline II=1
    DigitType diff = test_set[i] ^ training_set[i];
    // dist += popcount(diff);
    int popcount_result = 0;
    popcount(diff, &popcount_result);
    dist += popcount_result;
  }

  int max_dist = 0;
  int max_dist_id = K_CONST+1;

  // Find the max distance
  for ( int k = 0; k < K_CONST; ++k ) 
  {
    #pragma HLS pipeline II=1
    if ( dists[k] > max_dist ) 
    {
      max_dist = dists[k];
      max_dist_id = k;
    }
  }

  // Replace the entry with the max distance
  if ( dist < max_dist )
  {
    dists[max_dist_id] = dist;
    labels[max_dist_id] = label;
  }

  return;
}