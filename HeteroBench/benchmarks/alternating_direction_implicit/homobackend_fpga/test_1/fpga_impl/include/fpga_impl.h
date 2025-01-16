#ifndef __FPGA_IMPL_H__
#define __FPGA_IMPL_H__
#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// #define TSTEPS 50
// #define N 1024

void init_array(int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0]);
void kernel_adi(int tsteps,int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0]);

#endif