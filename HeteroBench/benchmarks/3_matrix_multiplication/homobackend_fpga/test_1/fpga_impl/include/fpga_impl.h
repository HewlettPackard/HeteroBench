#ifndef __FPGA_IMPL_H__
#define __FPGA_IMPL_H__
#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// #define NI 1024
// #define NJ 1024
// #define NK 1024
// #define NL 1024
// #define NM 1024

void kernel_3mm(double E[NI + 0][NJ + 0],double A[NI + 0][NK + 0],double B[NK + 0][NJ + 0],double F[NJ + 0][NL + 0],double C[NJ + 0][NM + 0],double D[NM + 0][NL + 0],double G[NI + 0][NL + 0]);

#endif