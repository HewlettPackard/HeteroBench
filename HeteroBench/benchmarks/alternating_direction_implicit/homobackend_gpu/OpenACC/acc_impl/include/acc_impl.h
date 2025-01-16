#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

// #define TSTEPS 50
// #define N 1024

void init_array(int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0]);
void kernel_adi(int tsteps,int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0]);