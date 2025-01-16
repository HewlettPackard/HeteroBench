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

void kernel_3m_1(double C[NJ + 0][NM + 0], double D[NM + 0][NL + 0], double F[NJ + 0][NL + 0])
{
  int c1;
  int c2;
  int c5;

  // #pragma omp target enter data map(to: C[0:NJ][0:NM])
  // #pragma omp target enter data map(to: D[0:NM][0:NL])
  // #pragma omp target enter data map(to: F[0:NJ][0:NL])
  #pragma acc data copyin(C[0:NJ][0:NM], D[0:NM][0:NL]) copy(F[0:NJ][0:NL])
  {
    #pragma acc parallel loop private(c5, c2)
    for (c1 = 0; c1 <= NJ - 1; c1++) {
      for (c2 = 0; c2 <= NL - 1; c2++) {
        for (c5 = 0; c5 <= NM - 1; c5++) {
          F[c1][c2] += C[c1][c5] * D[c5][c2];
        }
      }
    }
  }
  // #pragma omp target exit data map(from: F[0:NJ][0:NL])
  // #pragma omp target exit data map(release: C[0:NJ][0:NM])
  // #pragma omp target exit data map(release: D[0:NM][0:NL])
  // #pragma omp target exit data map(release: F[0:NJ][0:NL])
}
