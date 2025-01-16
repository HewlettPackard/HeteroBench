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

using namespace std;

void update( const DigitType* training_set, const DigitType* test_set, int dists[K_CONST], int labels[K_CONST], int label)
{
  int dist = 0;
  
  for (int i = 0; i < DIGIT_WIDTH; i ++ )
  {
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

void update_knn(const DigitType training_set[NUM_TRAINING * DIGIT_WIDTH], 
                const DigitType test_set[DIGIT_WIDTH],
                int dists[K_CONST], int labels[K_CONST])
{
  // #pragma omp parallel for
  for ( int i = 0; i < NUM_TRAINING; ++i ) 
    {
      int label = i / CLASS_SIZE;
      update(&training_set[i * DIGIT_WIDTH], test_set, dists, labels, label);
    }
}