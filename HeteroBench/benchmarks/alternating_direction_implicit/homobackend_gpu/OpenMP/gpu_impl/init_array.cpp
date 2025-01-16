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
#include "omp.h"

using namespace std;

void init_array(int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0])
{
  //int i;
  //int j;
  {
    int c1;
    int c2;
    #pragma omp target enter data map(alloc: X[0:N][0:N])
    #pragma omp target enter data map(alloc: A[0:N][0:N])
    #pragma omp target enter data map(alloc: B[0:N][0:N])
    if (n >= 1) {
    #pragma omp target teams distribute parallel for private(c2)
      for (c1 = 0; c1 <= n + -1; c1++) {
        for (c2 = 0; c2 <= n + -1; c2++) {
          X[c1][c2] = (((double )c1) * (c2 + 1) + 1) / n;
          A[c1][c2] = (((double )c1) * (c2 + 2) + 2) / n;
          B[c1][c2] = (((double )c1) * (c2 + 3) + 3) / n;
        }
      }
    }
    #pragma omp target exit data map(from: X[0:N][0:N])
    #pragma omp target exit data map(from: A[0:N][0:N])
    #pragma omp target exit data map(from: B[0:N][0:N])
    #pragma omp target exit data map(release: X[0:N][0:N])
    #pragma omp target exit data map(release: A[0:N][0:N])
    #pragma omp target exit data map(release: B[0:N][0:N])
  }
}