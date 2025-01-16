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
 
#include "gpu_impl.h"

using namespace std;

void kernel_3m_2(double E[NI + 0][NJ + 0], double F[NJ + 0][NL + 0], double G[NI + 0][NL + 0])
{
  int c1;
  int c2;
  int c6;

  #pragma omp target enter data map(to: E[0:NI][0:NJ])
  #pragma omp target enter data map(to: F[0:NJ][0:NL])
  #pragma omp target enter data map(to: G[0:NI][0:NL])

  #pragma omp target teams distribute parallel for private(c6, c2)
  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NJ - 1; c2++) {
      for (c6 = 0; c6 <= NL - 1; c6++) {
        G[c1][c6] += E[c1][c2] * F[c2][c6];
      }
    }
  }
  #pragma omp target exit data map(from: G[0:NI][0:NL])
  #pragma omp target exit data map(release: E[0:NI][0:NJ])
  #pragma omp target exit data map(release: F[0:NJ][0:NL])
  #pragma omp target exit data map(release: G[0:NI][0:NL])
}
