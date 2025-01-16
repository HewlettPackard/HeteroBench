#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define NUM_LAYER 4

// #define L0_H1 3072
// #define L0_W1 2048
// #define L0_W2 4096
#define SIZE_A0 L0_H1 * L0_W1
#define SIZE_Z0 L0_H1 * L0_W1
#define SIZE_W0 L0_W1 * L0_W2
#define SIZE_B0 L0_H1 * L0_W2

// #define L1_H1 3072
// #define L1_W1 4096
// #define L1_W2 4096
#define SIZE_A1 L1_H1 * L1_W1
#define SIZE_Z1 L1_H1 * L1_W1
#define SIZE_W1 L1_W1 * L1_W2
#define SIZE_B1 L1_H1 * L1_W2

// #define L2_H1 3072
// #define L2_W1 4096
// #define L2_W2 4096
#define SIZE_A2 L2_H1 * L2_W1
#define SIZE_Z2 L2_H1 * L2_W1
#define SIZE_W2 L2_W1 * L2_W2
#define SIZE_B2 L2_H1 * L2_W2

// #define L3_H1 3072
// #define L3_W1 4096
// #define L3_W2 1024
#define SIZE_A3 L3_H1 * L3_W1
#define SIZE_Z3 L3_H1 * L3_W1
#define SIZE_W3 L3_W1 * L3_W2
#define SIZE_B3 L3_H1 * L3_W2

#define SIZE_A4 L3_H1 * L3_W2
#define SIZE_Z4 L3_H1 * L3_W2

void sigmoid(double *sigmoid_input, double *sigmoid_output, int size);
void softmax(double *softmax_input, double *exp_results, double *softmax_output, int size);
void dot_add(double *dot_add_input_x, double *dot_add_input_W, double *dot_add_input_b, double *dot_add_output, int x_h, int x_w, int W_h, int W_w);