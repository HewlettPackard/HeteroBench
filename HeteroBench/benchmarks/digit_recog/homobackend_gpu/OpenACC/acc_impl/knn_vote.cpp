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

using namespace std;

void knn_vote(int labels[K_CONST], LabelType* max_label)
{
    int votes[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    #pragma acc data copyin(labels[0:K_CONST], votes[0:10]) create(max_label[0:1]) copyout(max_label[0:1])
    {
        int max_vote = 0;

        // First parallel loop to count votes
        #pragma acc parallel loop present(labels, votes)
        for (int i = 0; i < K_CONST; i++) {
            #pragma acc atomic
            votes[labels[i]]++;
        }

        // Second parallel loop to find the label with the maximum votes
        #pragma acc parallel loop shared(max_vote) present(votes, max_label)
        for (int i = 0; i < 10; i++) {
            if (votes[i] > max_vote) {
                #pragma acc atomic
                {
                    max_vote = votes[i];
                    *max_label = i;
                }
            }
        }
    }
}