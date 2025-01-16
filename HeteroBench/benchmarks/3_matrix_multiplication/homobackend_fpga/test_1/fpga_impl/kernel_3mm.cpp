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

void kernel_3m_0(double A[NI + 0][NK + 0], double B[NK + 0][NJ + 0], double E[NI + 0][NJ + 0])
{
  int c1;
  int c2;
  int c5;

  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NJ - 1; c2++) {
      double sum = 0;
      for (c5 = 0; c5 <= NK - 1; c5++) {
        #pragma HLS PIPELINE
        sum += A[c1][c5] * B[c5][c2];
      }
      E[c1][c2] = sum;
    }
  }
}

void kernel_3m_1(double C[NJ + 0][NM + 0], double D[NM + 0][NL + 0], double F[NJ + 0][NL + 0])
{
  int c1;
  int c2;
  int c5;

  for (c1 = 0; c1 <= NJ - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      double sum = 0;
      for (c5 = 0; c5 <= NM - 1; c5++) {
        #pragma HLS PIPELINE
        sum += C[c1][c5] * D[c5][c2];
      }
      F[c1][c2] = sum;
    }
  }
}

void kernel_3m_2(double E[NI + 0][NJ + 0], double F[NJ + 0][NL + 0], double G[NI + 0][NL + 0])
{
  int c1;
  int c2;
  int c6;

  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NJ - 1; c2++) {
      double sum = 0;
      for (c6 = 0; c6 <= NL - 1; c6++) {
        #pragma HLS PIPELINE
        sum += E[c1][c2] * F[c2][c6];
      }
      G[c1][c2] = sum;
    }
  }
}

void kernel_3mm(double E[NI + 0][NJ + 0],double A[NI + 0][NK + 0],double B[NK + 0][NJ + 0],double F[NJ + 0][NL + 0],double C[NJ + 0][NM + 0],double D[NM + 0][NL + 0],double G[NI + 0][NL + 0])
{
  #pragma HLS interface m_axi offset=slave bundle=double_in_0 port=A max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=A

  #pragma HLS interface m_axi offset=slave bundle=double_in_0 port=B max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=B

  #pragma HLS interface m_axi offset=slave bundle=double_in_1 port=C max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=C

  #pragma HLS interface m_axi offset=slave bundle=double_in_1 port=D max_read_burst_length=16 num_read_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=D

  #pragma HLS interface m_axi offset=slave bundle=double_out_0 port=E max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=E

  #pragma HLS interface m_axi offset=slave bundle=double_out_1 port=F max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=F

  #pragma HLS interface m_axi offset=slave bundle=double_out_2 port=G max_write_burst_length=16 num_write_outstanding=64
  #pragma HLS interface s_axilite bundle=control port=G

  #pragma HLS interface s_axilite bundle=control port=return

  #pragma HLS dataflow
  
  kernel_3m_0(A, B, E);
  kernel_3m_1(C, D, F);
  kernel_3m_2(E, F, G);
}
