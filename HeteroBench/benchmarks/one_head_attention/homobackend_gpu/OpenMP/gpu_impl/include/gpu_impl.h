#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

void transpose(double *transpose_x, double *transpose_output, 
                int batch_size, int input_h, int input_w, int dim0, int dim1);
void matmul(double *matmul_x, double *matmul_y, double *matmul_output, 
                int batch_size, int input_h, int input_w, int output_w);
void softmax(double *softmax_x, double *softmax_output, 
             int batch_size, int input_h, int input_w, int axis);