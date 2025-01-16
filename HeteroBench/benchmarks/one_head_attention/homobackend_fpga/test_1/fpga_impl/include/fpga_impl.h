#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// #define BATCH_SIZE; 
// #define INPUT_H; 
// #define INPUT_W;
// #define OUTPUT_W;

void transpose(double transpose_x[BATCH_SIZE*INPUT_H*INPUT_W], double transpose_output[BATCH_SIZE*INPUT_H*INPUT_W], int dim0, int dim1);
void matmul0(double matmul_x[BATCH_SIZE * INPUT_H * INPUT_W], double matmul_y[BATCH_SIZE * INPUT_W * OUTPUT_W], double matmul_output[BATCH_SIZE * INPUT_H * OUTPUT_W]);
void matmul1(double matmul_x[BATCH_SIZE * INPUT_H * OUTPUT_W], double matmul_y[BATCH_SIZE * INPUT_W * OUTPUT_W], double matmul_output[BATCH_SIZE * INPUT_H * INPUT_W]);
void softmax(double softmax_x[BATCH_SIZE * INPUT_H * INPUT_H], double softmax_output[BATCH_SIZE * INPUT_H * INPUT_H], int axis);