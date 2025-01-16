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

void kernel_adi(int tsteps,int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0])
{
  //int t;
  //int i1;
  //int i2;
  
  //#pragma scop
  {
    int c0;
    int c2;
    int c8;
    for (c0 = 0; c0 <= TSTEPS; c0++) {
      #pragma omp parallel for private(c8)
      for (c2 = 0; c2 <= N - 1; c2++) {
        for (c8 = 1; c8 <= N - 1; c8++) {
          B[c2][c8] = B[c2][c8] - A[c2][c8] * A[c2][c8] / B[c2][c8 - 1];
        }
        for (c8 = 1; c8 <= N - 1; c8++) {
          X[c2][c8] = X[c2][c8] - X[c2][c8 - 1] * A[c2][c8] / B[c2][c8 - 1];
        }
        for (c8 = 0; c8 <= N - 3; c8++) {
          X[c2][N - c8 - 2] = (X[c2][N - 2 - c8] - X[c2][N - 2 - c8 - 1] * A[c2][N - c8 - 3]) / B[c2][N - 3 - c8];
        }
      }
      #pragma omp parallel for
      for (c2 = 0; c2 <= N - 1; c2++) {
        X[c2][N - 1] = X[c2][N - 1] / B[c2][N - 1];
      }
      #pragma omp parallel for private(c8)
      for (c2 = 0; c2 <= N - 1; c2++) {
        for (c8 = 1; c8 <= N - 1; c8++) {
          B[c8][c2] = B[c8][c2] - A[c8][c2] * A[c8][c2] / B[c8 - 1][c2];
        }
        for (c8 = 1; c8 <= N - 1; c8++) {
          X[c8][c2] = X[c8][c2] - X[c8 - 1][c2] * A[c8][c2] / B[c8 - 1][c2];
        }
        for (c8 = 0; c8 <= N - 3; c8++) {
          X[N - 2 - c8][c2] = (X[N - 2 - c8][c2] - X[N - c8 - 3][c2] * A[N - 3 - c8][c2]) / B[N - 2 - c8][c2];
        }
      }
      #pragma omp parallel for
      for (c2 = 0; c2 <= N - 1; c2++) {
        X[N - 1][c2] = X[N - 1][c2] / B[N - 1][c2];
      }
    }
  }
  
//#pragma endscop
}