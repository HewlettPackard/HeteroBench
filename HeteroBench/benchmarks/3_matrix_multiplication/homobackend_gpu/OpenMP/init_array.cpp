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
 
#include "init_array.h"
#include <time.h>
#include <stdlib.h>

using namespace std;

void init_array(int ni,int nj,int nk,int nl,int nm,double A[NI + 0][NK + 0],double B[NK + 0][NJ + 0],double C[NJ + 0][NM + 0],double D[NM + 0][NL + 0], double E[NI + 0][NJ + 0], double F[NJ + 0][NL + 0],double G[NI + 0][NL + 0])
{
  int c1, c2;

  // random initialization of A,B,C,D
  long seed = time(NULL);
  srand48(seed);
  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NK - 1; c2++) {
      A[c1][c2] = drand48();
    }
  }

  for (c1 = 0; c1 <= NK - 1; c1++) {
    for (c2 = 0; c2 <= NJ - 1; c2++) {
      B[c1][c2] = drand48();
    }
  }

  for (c1 = 0; c1 <= NJ - 1; c1++) {
    for (c2 = 0; c2 <= NM - 1; c2++) {
      C[c1][c2] = drand48();
    }
  }

  for (c1 = 0; c1 <= NM - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      D[c1][c2] = drand48();
    }
  }

  // zero initialization of E,F,G
  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      G[c1][c2] = 0;
    }
  }

  for (c1 = 0; c1 <= NJ - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      F[c1][c2] = 0;
    }
  }

  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NJ - 1; c2++) {
      E[c1][c2] = 0;
    }
  }

}