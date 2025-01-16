#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

// #define NI 1024
// #define NJ 1024
// #define NK 1024
// #define NL 1024
// #define NM 1024

void kernel_3m_0(double A[NI + 0][NK + 0],double B[NK + 0][NJ + 0],double E[NI + 0][NJ + 0]);
void kernel_3m_1(double C[NJ + 0][NM + 0],double D[NM + 0][NL + 0],double F[NJ + 0][NL + 0]);
void kernel_3m_2(double E[NI + 0][NJ + 0], double F[NJ + 0][NL + 0], double G[NI + 0][NL + 0]);