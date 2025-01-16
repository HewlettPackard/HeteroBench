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

void knn_vote(int labels[K_CONST], LabelType* max_label) 
{
  int max_vote = 0;
  
  int votes[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // #pragma omp parallel for reduction(+:votes)
  for (int i = 0; i < K_CONST; i ++ )
  {
    int index;
    #pragma omp atomic read
    index = labels[i];

    #pragma omp atomic
    votes[index]++;
  }

  // #pragma omp parallel for
  for (int i = 0; i < 10; i ++ ) 
  {
    int current_votes;
    #pragma omp atomic read
    current_votes = votes[i];

    if (current_votes > max_vote) 
    {
      #pragma omp atomic write
      max_vote = current_votes;

      #pragma omp atomic write
      *max_label = i;
    }
  }

  return;

}