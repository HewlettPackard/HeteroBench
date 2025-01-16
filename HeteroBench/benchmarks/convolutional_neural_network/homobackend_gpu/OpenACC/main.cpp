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
 
#include "acc_impl.h"
#include "omp.h"

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

bool checkFileExistence(
  const std::string& filePath
) 
{
  std::ifstream file(filePath);
  if (!file) {
    std::cout << "Error: File " << filePath << " does not exist or cannot be opened." << std::endl;
    return false;
  }
  return true;
}

/* Read data from file*/
void readData(const char* file_path_mat, double* data_mat, size_t size_mat) {
  // std::cout << "Reading " << file_path_mat << " ... ";
  if (!checkFileExistence(file_path_mat)) {
    exit(1);
  }
  std::ifstream in_mat(file_path_mat, std::ios::in | std::ios::binary);
  in_mat.read((char *)data_mat, sizeof(double) * size_mat);
  in_mat.close();
  // std::cout << "Done" << std::endl;
}

/* Check results */
void checkResult(double* data, double* data_golden, size_t size) {
  int error = 0;
  for (int i = 0; i < size; i++) {
    if (abs(data[i] - data_golden[i]) > 1e-3) {
      error++;
    }
  }
  if (error == 0) {
    std::cout << "Correct" << std::endl;
  } else {
    std::cout << "Wrong" << std::endl;
    std::cout << "error: " << error << std::endl;
    
    // print the first 10 elements of computed results
    std::cout << "First 10 elements of computed results: ";
    for (int i = 0; i < 10; i++) {
      std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // print the first 10 elements of golden results
    std::cout << "First 10 elements of golden results: ";
    for (int i = 0; i < 10; i++) {
      std::cout << data_golden[i] << " ";
    }
    std::cout << std::endl;
  }

}

void cnn_forward(string input_path, string output_path)
{
  double *input_image = new double [INPUT_SIZE_H * INPUT_SIZE_W];
  double *input_padded = new double [(INPUT_SIZE_H + 2 * CONV2D_PADDING) * (INPUT_SIZE_W + 2 * CONV2D_PADDING)];
  double *W_conv = new double [CONV_KERNEL_SIZE_H * CONV_KERNEL_SIZE_W];
  double *W_fc = new double [FULL_CONNECT_LAYER_SIZE_H * FULL_CONNECT_LAYER_SIZE_W];
  double *b_fc = new double [FULL_CONNECT_LAYER_SIZE_W];
  double *softmax_exp_results = new double [FULL_CONNECT_LAYER_SIZE_W];

  double *conv_output = new double [CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
  double *relu_output = new double [CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
  double *pool_output = new double [POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH];
  double *flattened_output = new double [FLATTENED_OUTPUT_SIZE];
  double *fc_output = new double [FULL_CONNECT_LAYER_SIZE_W];
  double *softmax_output = new double [FULL_CONNECT_LAYER_SIZE_W];

  // readData("../../dataset/input_image.bin", input_image, INPUT_SIZE_H * INPUT_SIZE_W);
  // readData("../../dataset/W_conv.bin", W_conv, CONV_KERNEL_SIZE_H * CONV_KERNEL_SIZE_W);
  // readData("../../dataset/W_fc.bin", W_fc, FULL_CONNECT_LAYER_SIZE_H * FULL_CONNECT_LAYER_SIZE_W);
  // readData("../../dataset/b_fc.bin", b_fc, FULL_CONNECT_LAYER_SIZE_W);
  readData((input_path + "/input_image.bin").c_str(), input_image, INPUT_SIZE_H * INPUT_SIZE_W);
  readData((input_path + "/W_conv.bin").c_str(), W_conv, CONV_KERNEL_SIZE_H * CONV_KERNEL_SIZE_W);
  readData((input_path + "/W_fc.bin").c_str(), W_fc, FULL_CONNECT_LAYER_SIZE_H * FULL_CONNECT_LAYER_SIZE_W);
  readData((input_path + "/b_fc.bin").c_str(), b_fc, FULL_CONNECT_LAYER_SIZE_W);
  std::cout << "read data done" << std::endl;

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;
  conv2d(input_image, W_conv, input_padded, CONV2D_BIAS, CONV2D_STRIDE, CONV2D_PADDING, INPUT_SIZE_H, INPUT_SIZE_W, CONV_KERNEL_SIZE_H, CONV_KERNEL_SIZE_W, conv_output);
  relu(conv_output, relu_output, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  max_pooling(relu_output, POOLING_SIZE, POOLING_STRIDE, CONV_OUTPUT_HEIGHT, CONV_OUTPUT_WIDTH, pool_output);
  // flattened_output = pooled_output.flatten()
  for (int i = 0; i < POOLING_OUTPUT_HEIGHT; i++) {
    for (int j = 0; j < POOLING_OUTPUT_WIDTH; j++) {
      flattened_output[i * POOLING_OUTPUT_WIDTH + j] = pool_output[i * POOLING_OUTPUT_WIDTH + j];
    }
  }
  // Note here the size of flattened_output is 1 x FLATTENED_OUTPUT_SIZE
  dot_add(flattened_output, W_fc, b_fc, fc_output, 1, FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_W);
  for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
    fc_output[i] = fc_output[i] / 15000;
  }
  softmax(fc_output, softmax_exp_results, softmax_output, FULL_CONNECT_LAYER_SIZE_W);
  for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
    softmax_output[i] = softmax_output[i] * 10000;
  }
  std::cout << "Done" << std::endl;

  // Check results
  std::cout << "Check results ..." << std::endl;
  double *conv_output_golden = new double [CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
  double *relu_output_golden = new double [CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
  double *pool_output_golden = new double [POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH];
  double *flattened_output_golden = new double [FLATTENED_OUTPUT_SIZE];
  double *fc_output_golden = new double [FULL_CONNECT_LAYER_SIZE_W];
  double *softmax_output_golden = new double [FULL_CONNECT_LAYER_SIZE_W];
/*
  // readData("../../dataset/conv_output.bin", conv_output_golden, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  // readData("../../dataset/relu_output.bin", relu_output_golden, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  // readData("../../dataset/pooled_output.bin", pool_output_golden, POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH);
  // readData("../../dataset/flattened_output.bin", flattened_output_golden, FLATTENED_OUTPUT_SIZE);
  // readData("../../dataset/fc_output.bin", fc_output_golden, FULL_CONNECT_LAYER_SIZE_W);
  // readData("../../dataset/softmax_output.bin", softmax_output_golden, FULL_CONNECT_LAYER_SIZE_W);
  readData((output_path + "/conv_output.bin").c_str(), conv_output_golden, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  readData((output_path + "/relu_output.bin").c_str(), relu_output_golden, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  readData((output_path + "/pooled_output.bin").c_str(), pool_output_golden, POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH);
  readData((output_path + "/flattened_output.bin").c_str(), flattened_output_golden, FLATTENED_OUTPUT_SIZE);
  readData((output_path + "/fc_output.bin").c_str(), fc_output_golden, FULL_CONNECT_LAYER_SIZE_W);
  readData((output_path + "/softmax_output.bin").c_str(), softmax_output_golden, FULL_CONNECT_LAYER_SIZE_W);

  std::cout << "checking conv_output ... ";
  checkResult(conv_output, conv_output_golden, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  std::cout << "checking relu_output ... ";
  checkResult(relu_output, relu_output_golden, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  std::cout << "checking pool_output ... ";
  checkResult(pool_output, pool_output_golden, POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH);
  std::cout << "checking flattened_output ... ";
  checkResult(flattened_output, flattened_output_golden, FLATTENED_OUTPUT_SIZE);
  std::cout << "checking fc_output ... ";
  checkResult(fc_output, fc_output_golden, FULL_CONNECT_LAYER_SIZE_W);
  std::cout << "checking softmax_output ... ";
  checkResult(softmax_output, softmax_output_golden, FULL_CONNECT_LAYER_SIZE_W);
*/

  // multi iterations
  int iterations = ITERATIONS;
  std::cout << "Running " << iterations << " iterations ..." << std::endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double conv2d_time = 0;
  double relu_time = 0;
  double max_pooling_time = 0;
  double dot_add_time = 0;
  double softmax_time = 0;

  for (int i = 0; i < iterations; i++) {
    // std::cout << "Iteration " << i << std::endl;

    start_iteration_time = omp_get_wtime();
    conv2d(input_image, W_conv, input_padded, CONV2D_BIAS, CONV2D_STRIDE, CONV2D_PADDING, INPUT_SIZE_H, INPUT_SIZE_W, CONV_KERNEL_SIZE_H, CONV_KERNEL_SIZE_W, conv_output);
    conv2d_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    relu(conv_output, relu_output, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
    relu_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    max_pooling(relu_output, POOLING_SIZE, POOLING_STRIDE, CONV_OUTPUT_HEIGHT, CONV_OUTPUT_WIDTH, pool_output);
    max_pooling_time += omp_get_wtime() - start_iteration_time;

    for (int i = 0; i < POOLING_OUTPUT_HEIGHT; i++) {
      for (int j = 0; j < POOLING_OUTPUT_WIDTH; j++) {
        flattened_output[i * POOLING_OUTPUT_WIDTH + j] = pool_output[i * POOLING_OUTPUT_WIDTH + j];
      }
    }

    start_iteration_time = omp_get_wtime();
    dot_add(flattened_output, W_fc, b_fc, fc_output, 1, FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_W);
    dot_add_time += omp_get_wtime() - start_iteration_time;

    for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
      fc_output[i] = fc_output[i] / 15000;
    }

    start_iteration_time = omp_get_wtime();
    softmax(fc_output, softmax_exp_results, softmax_output, FULL_CONNECT_LAYER_SIZE_W);
    softmax_time += omp_get_wtime() - start_iteration_time;

    for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
      softmax_output[i] = softmax_output[i] * 10000;
    }
  }
  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations << " iterations " << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "conv2d kernel time: " << conv2d_time / iterations * 1000 << " ms" << endl;
  cout << "relu kernel time: " << relu_time / iterations * 1000 << " ms" << endl;
  cout << "max_pooling kernel time: " << max_pooling_time / iterations * 1000 << " ms" << endl;
  cout << "dot_add kernel time: " << dot_add_time / iterations * 1000 << " ms" << endl;
  cout << "softmax kernel time: " << softmax_time / iterations * 1000 << " ms" << endl;

  delete[] input_image;
  delete[] input_padded;
  delete[] W_conv;
  delete[] W_fc;
  delete[] b_fc;
  delete[] conv_output;
  delete[] relu_output;
  delete[] pool_output;
  delete[] flattened_output;
  delete[] fc_output;
  delete[] softmax_output;
  delete[] softmax_exp_results;
  delete[] conv_output_golden;
  delete[] relu_output_golden;
  delete[] pool_output_golden;
  delete[] flattened_output_golden;
  delete[] fc_output_golden;
  delete[] softmax_output_golden;

  return;
}

int main(int argc, char *argv[])
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running cnn benchmark C++ OpenACC GPU" << std::endl;
  std::cout << "=======================================" << std::endl;
  
  string input_path;
  string output_path;
  if (argc == 3) {
    input_path = argv[1];
    output_path = argv[2];
  } else {
    printf("Usage: ./cnn_sw <input_path> <output_path>\n");
    exit(-1);
  }
  cnn_forward(input_path, output_path);
  return 0;
}