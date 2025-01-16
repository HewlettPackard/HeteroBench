#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

// #define CONV2D_STRIDE 1
// #define CONV2D_PADDING 1
// #define CONV2D_BIAS 0.1
// #define POOLING_SIZE 2
// #define POOLING_STRIDE 2
// #define INPUT_SIZE_H 1024
// #define INPUT_SIZE_W 2048
// #define CONV_KERNEL_SIZE_H 3
// #define CONV_KERNEL_SIZE_W 3

#define CONV_OUTPUT_HEIGHT ((INPUT_SIZE_H - CONV_KERNEL_SIZE_H + 2 * CONV2D_PADDING) / CONV2D_STRIDE + 1)
#define CONV_OUTPUT_WIDTH ((INPUT_SIZE_W - CONV_KERNEL_SIZE_W + 2 * CONV2D_PADDING) / CONV2D_STRIDE + 1)
#define POOLING_OUTPUT_HEIGHT ((CONV_OUTPUT_HEIGHT - POOLING_SIZE) / POOLING_STRIDE + 1)
#define POOLING_OUTPUT_WIDTH ((CONV_OUTPUT_WIDTH - POOLING_SIZE) / POOLING_STRIDE + 1)

#define FLATTENED_OUTPUT_SIZE (POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH)

#define FULL_CONNECT_LAYER_SIZE_H FLATTENED_OUTPUT_SIZE
// #define FULL_CONNECT_LAYER_SIZE_W 2048

void conv2d(double *conv2d_input, double *conv2d_kernel, double *input_padded, double conv2d_bias, int stride, int padding, int input_h, int input_w, int kernel_h, int kernel_w, double *conv2d_output);
void relu(double *relu_input, double *relu_output, int size);
void max_pooling(double *max_pooling_input, int pool_size, int pool_stride, int input_h, int input_w, double *max_pooling_output);
void pad_input(double *pad_input_input, double *pad_input_output, int input_h, int input_w, int padding);
void dot_add(double *dot_add_input_x, double *dot_add_input_W, double *dot_add_input_b, double *dot_add_output, int x_h, int x_w, int W_h, int W_w);
void softmax(double *softmax_input, double *exp_results, double *softmax_output, int size);