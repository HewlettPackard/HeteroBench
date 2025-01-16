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
#include "omp.h"
#include <iostream>

#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
// #include <ap_int.h>
#include <ctime>
#include <stdlib.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <experimental/xrt_xclbin.h>
#include <xrt/xrt_kernel.h>
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"

using namespace std;

#define DEVICE_ID 0

#define conv2d_ptr_input 0
#define conv2d_ptr_kernel 1
#define conv2d_ptr_input_padded 2
#define conv2d_ptr_bias 3
#define conv2d_ptr_stride 4
#define conv2d_ptr_padding 5
#define conv2d_ptr_input_h 6
#define conv2d_ptr_input_w 7
#define conv2d_ptr_kernel_h 8
#define conv2d_ptr_kernel_w 9
#define conv2d_ptr_output 10

#define relu_ptr_input 0
#define relu_ptr_output 1
#define relu_ptr_size 2

#define max_pooling_ptr_input 0
#define max_pooling_ptr_pool_size 1
#define max_pooling_ptr_pool_stride 2
#define max_pooling_ptr_input_h 3
#define max_pooling_ptr_input_w 4
#define max_pooling_ptr_output 5

#define dot_add_ptr_input_x 0
#define dot_add_ptr_input_W 1
#define dot_add_ptr_input_b 2
#define dot_add_ptr_output 3

#define softmax_ptr_input 0
#define softmax_ptr_exp_results 1
#define softmax_ptr_output 2
#define softmax_ptr_size 3

bool checkFileExistence(
  const std::string& filePath
) 
{
  std::ifstream file(filePath);
  if (!file) {
    std::cout << "Error: File " << filePath << " does not exist or cannot be opened." << std::endl;
    std::cout << "You may need to run the Python version of this benchmark to generate the data file first." << std::endl;
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
  
  // Load xclbin
  std::string xclbin_file = "overlay_hw.xclbin";
  std::cout << "Loading: " << xclbin_file << std::endl;
  xrt::device device = xrt::device(DEVICE_ID);
  xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
  std::cout << "Loaded xclbin: " << xclbin_file << std::endl;

  // create kernel object
  xrt::kernel kernel_conv2d = xrt::kernel(device, xclbin_uuid, "conv2d");
  xrt::kernel kernel_relu = xrt::kernel(device, xclbin_uuid, "relu");
  xrt::kernel kernel_max_pooling = xrt::kernel(device, xclbin_uuid, "max_pooling");
  xrt::kernel kernel_dot_add = xrt::kernel(device, xclbin_uuid, "dot_add");
  xrt::kernel kernel_softmax = xrt::kernel(device, xclbin_uuid, "softmax");

  std::cout << "Kernel object created" << std::endl;

  // create memory groups
  xrtMemoryGroup bank_grp_kernel_conv2d_input = kernel_conv2d.group_id(conv2d_ptr_input);
  xrtMemoryGroup bank_grp_kernel_conv2d_kernel = kernel_conv2d.group_id(conv2d_ptr_kernel);
  xrtMemoryGroup bank_grp_kernel_conv2d_input_padded = kernel_conv2d.group_id(conv2d_ptr_input_padded); 
  xrtMemoryGroup bank_grp_kernel_conv2d_output = kernel_conv2d.group_id(conv2d_ptr_output);

  xrtMemoryGroup bank_grp_kernel_relu_input = kernel_relu.group_id(relu_ptr_input);
  xrtMemoryGroup bank_grp_kernel_relu_output = kernel_relu.group_id(relu_ptr_output);

  xrtMemoryGroup bank_grp_kernel_max_pooling_input = kernel_max_pooling.group_id(max_pooling_ptr_input);
  xrtMemoryGroup bank_grp_kernel_max_pooling_output = kernel_max_pooling.group_id(max_pooling_ptr_output);

  xrtMemoryGroup bank_grp_kernel_dot_add_input_x = kernel_dot_add.group_id(dot_add_ptr_input_x);
  xrtMemoryGroup bank_grp_kernel_dot_add_input_W = kernel_dot_add.group_id(dot_add_ptr_input_W);
  xrtMemoryGroup bank_grp_kernel_dot_add_input_b = kernel_dot_add.group_id(dot_add_ptr_input_b);
  xrtMemoryGroup bank_grp_kernel_dot_add_output = kernel_dot_add.group_id(dot_add_ptr_output);

  xrtMemoryGroup bank_grp_kernel_softmax_input = kernel_softmax.group_id(softmax_ptr_input);
  xrtMemoryGroup bank_grp_kernel_softmax_exp_results = kernel_softmax.group_id(softmax_ptr_exp_results);
  xrtMemoryGroup bank_grp_kernel_softmax_output = kernel_softmax.group_id(softmax_ptr_output);

  // create buffer objects
  xrt::bo data_buffer_kernel_conv2d_input = \
  xrt::bo(device, sizeof(double) * INPUT_SIZE_H * INPUT_SIZE_W, xrt::bo::flags::normal, bank_grp_kernel_conv2d_input);
  xrt::bo data_buffer_kernel_conv2d_kernel = \
  xrt::bo(device, sizeof(double) * CONV_KERNEL_SIZE_H * CONV_KERNEL_SIZE_W, xrt::bo::flags::normal, bank_grp_kernel_conv2d_kernel);
  xrt::bo data_buffer_kernel_conv2d_input_padded = \
  xrt::bo(device, sizeof(double) * (INPUT_SIZE_H + 2 * CONV2D_PADDING) * (INPUT_SIZE_W + 2 * CONV2D_PADDING), xrt::bo::flags::normal, bank_grp_kernel_conv2d_input_padded);
  xrt::bo data_buffer_kernel_conv2d_output = \
  xrt::bo(device, sizeof(double) * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH, xrt::bo::flags::normal, bank_grp_kernel_conv2d_output);

  xrt::bo data_buffer_kernel_relu_input = \
  xrt::bo(device, sizeof(double) * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH, xrt::bo::flags::normal, bank_grp_kernel_relu_input);
  xrt::bo data_buffer_kernel_relu_output = \
  xrt::bo(device, sizeof(double) * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH, xrt::bo::flags::normal, bank_grp_kernel_relu_output);

  xrt::bo data_buffer_kernel_max_pooling_input = \
  xrt::bo(device, sizeof(double) * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH, xrt::bo::flags::normal, bank_grp_kernel_max_pooling_input);
  xrt::bo data_buffer_kernel_max_pooling_output = \
  xrt::bo(device, sizeof(double) * POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH, xrt::bo::flags::normal, bank_grp_kernel_max_pooling_output);

  xrt::bo data_buffer_kernel_dot_add_input_x = \
  xrt::bo(device, sizeof(double) * FLATTENED_OUTPUT_SIZE, xrt::bo::flags::normal, bank_grp_kernel_dot_add_input_x);
  xrt::bo data_buffer_kernel_dot_add_input_W = \
  xrt::bo(device, sizeof(double) * FULL_CONNECT_LAYER_SIZE_H * FULL_CONNECT_LAYER_SIZE_W, xrt::bo::flags::normal, bank_grp_kernel_dot_add_input_W);
  xrt::bo data_buffer_kernel_dot_add_input_b = \
  xrt::bo(device, sizeof(double) * FULL_CONNECT_LAYER_SIZE_W, xrt::bo::flags::normal, bank_grp_kernel_dot_add_input_b);
  xrt::bo data_buffer_kernel_dot_add_output = \
  xrt::bo(device, sizeof(double) * FULL_CONNECT_LAYER_SIZE_W, xrt::bo::flags::normal, bank_grp_kernel_dot_add_output);

  xrt::bo data_buffer_kernel_softmax_input = \
  xrt::bo(device, sizeof(double) * FULL_CONNECT_LAYER_SIZE_W, xrt::bo::flags::normal, bank_grp_kernel_softmax_input);
  xrt::bo data_buffer_kernel_softmax_exp_results = \
  xrt::bo(device, sizeof(double) * FULL_CONNECT_LAYER_SIZE_W, xrt::bo::flags::normal, bank_grp_kernel_softmax_exp_results);
  xrt::bo data_buffer_kernel_softmax_output = \
  xrt::bo(device, sizeof(double) * FULL_CONNECT_LAYER_SIZE_W, xrt::bo::flags::normal, bank_grp_kernel_softmax_output);

  std::cout << "Buffer object created" << std::endl;

  // create kernel runner
  xrt::run run_kernel_conv2d(kernel_conv2d);
  xrt::run run_kernel_relu(kernel_relu);
  xrt::run run_kernel_max_pooling(kernel_max_pooling);
  xrt::run run_kernel_dot_add(kernel_dot_add);
  xrt::run run_kernel_softmax(kernel_softmax);

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;
  // conv2d(input_image, W_conv, input_padded, CONV2D_BIAS, CONV2D_STRIDE, CONV2D_PADDING, INPUT_SIZE_H, INPUT_SIZE_W, CONV_KERNEL_SIZE_H, CONV_KERNEL_SIZE_W, conv_output);
  // write data to buffer of the conv2d kernel
  data_buffer_kernel_conv2d_input.write(input_image);
  data_buffer_kernel_conv2d_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_conv2d_kernel.write(W_conv);
  data_buffer_kernel_conv2d_kernel.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_conv2d_input_padded.write(input_padded);
  data_buffer_kernel_conv2d_input_padded.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  // set arguments of the conv2d kernel
  run_kernel_conv2d.set_arg(conv2d_ptr_input, data_buffer_kernel_conv2d_input);
  run_kernel_conv2d.set_arg(conv2d_ptr_kernel, data_buffer_kernel_conv2d_kernel);
  run_kernel_conv2d.set_arg(conv2d_ptr_input_padded, data_buffer_kernel_conv2d_input_padded);
  run_kernel_conv2d.set_arg(conv2d_ptr_bias, CONV2D_BIAS);
  run_kernel_conv2d.set_arg(conv2d_ptr_stride, CONV2D_STRIDE);
  run_kernel_conv2d.set_arg(conv2d_ptr_padding, CONV2D_PADDING);
  run_kernel_conv2d.set_arg(conv2d_ptr_input_h, INPUT_SIZE_H);
  run_kernel_conv2d.set_arg(conv2d_ptr_input_w, INPUT_SIZE_W);
  run_kernel_conv2d.set_arg(conv2d_ptr_kernel_h, CONV_KERNEL_SIZE_H);
  run_kernel_conv2d.set_arg(conv2d_ptr_kernel_w, CONV_KERNEL_SIZE_W);
  run_kernel_conv2d.set_arg(conv2d_ptr_output, data_buffer_kernel_conv2d_output);

  std::cout << "set arguments of the conv2d kernel done" << std::endl;

  // run the conv2d kernel
  run_kernel_conv2d.start();
  run_kernel_conv2d.wait();

  std::cout << "Run conv2d kernel done" << std::endl;

  // Read the result back from the buffer
  data_buffer_kernel_conv2d_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_conv2d_output.read(conv_output);
  std::cout << "conv2d kernel done" << std::endl;

  // relu(conv_output, relu_output, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  // write data to buffer of the relu kernel
  data_buffer_kernel_relu_input.write(conv_output);
  data_buffer_kernel_relu_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of the relu kernel
  run_kernel_relu.set_arg(relu_ptr_input, data_buffer_kernel_relu_input);
  run_kernel_relu.set_arg(relu_ptr_output, data_buffer_kernel_relu_output);
  run_kernel_relu.set_arg(relu_ptr_size, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);

  std::cout << "set arguments of the relu kernel done" << std::endl;

  // run the relu kernel
  run_kernel_relu.start();
  run_kernel_relu.wait();

  std::cout << "Run relu kernel done" << std::endl;

  // Read the result back from the buffer
  data_buffer_kernel_relu_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_relu_output.read(relu_output);
  std::cout << "relu kernel done" << std::endl;

  // max_pooling(relu_output, POOLING_SIZE, POOLING_STRIDE, CONV_OUTPUT_HEIGHT, CONV_OUTPUT_WIDTH, pool_output);
  // write data to buffer of the max_pooling kernel
  data_buffer_kernel_max_pooling_input.write(relu_output);
  data_buffer_kernel_max_pooling_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of the max_pooling kernel
  run_kernel_max_pooling.set_arg(max_pooling_ptr_input, data_buffer_kernel_max_pooling_input);
  run_kernel_max_pooling.set_arg(max_pooling_ptr_pool_size, POOLING_SIZE);
  run_kernel_max_pooling.set_arg(max_pooling_ptr_pool_stride, POOLING_STRIDE);
  run_kernel_max_pooling.set_arg(max_pooling_ptr_input_h, CONV_OUTPUT_HEIGHT);
  run_kernel_max_pooling.set_arg(max_pooling_ptr_input_w, CONV_OUTPUT_WIDTH);
  run_kernel_max_pooling.set_arg(max_pooling_ptr_output, data_buffer_kernel_max_pooling_output);

  std::cout << "set arguments of the max_pooling kernel done" << std::endl;

  // run the max_pooling kernel
  run_kernel_max_pooling.start();
  run_kernel_max_pooling.wait();

  std::cout << "Run max_pooling kernel done" << std::endl;

  // Read the result back from the buffer
  data_buffer_kernel_max_pooling_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_max_pooling_output.read(pool_output);
  std::cout << "max_pooling kernel done" << std::endl;

  // flattened_output = pooled_output.flatten()
  for (int i = 0; i < POOLING_OUTPUT_HEIGHT; i++) {
    for (int j = 0; j < POOLING_OUTPUT_WIDTH; j++) {
      flattened_output[i * POOLING_OUTPUT_WIDTH + j] = pool_output[i * POOLING_OUTPUT_WIDTH + j];
    }
  }
  std::cout << "flattened_output kernel done" << std::endl;

  // Note here the size of flattened_output is 1 x FLATTENED_OUTPUT_SIZE
  // dot_add(flattened_output, W_fc, b_fc, fc_output, 1, FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_W);
  // write data to buffer of the dot_add kernel
  data_buffer_kernel_dot_add_input_x.write(flattened_output);
  data_buffer_kernel_dot_add_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_dot_add_input_W.write(W_fc);
  data_buffer_kernel_dot_add_input_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_dot_add_input_b.write(b_fc);
  data_buffer_kernel_dot_add_input_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of the dot_add kernel
  run_kernel_dot_add.set_arg(dot_add_ptr_input_x, data_buffer_kernel_dot_add_input_x);
  run_kernel_dot_add.set_arg(dot_add_ptr_input_W, data_buffer_kernel_dot_add_input_W);
  run_kernel_dot_add.set_arg(dot_add_ptr_input_b, data_buffer_kernel_dot_add_input_b);
  run_kernel_dot_add.set_arg(dot_add_ptr_output, data_buffer_kernel_dot_add_output);

  std::cout << "set arguments of the dot_add kernel done" << std::endl;

  // run the dot_add kernel
  run_kernel_dot_add.start();
  run_kernel_dot_add.wait();

  std::cout << "Run dot_add kernel done" << std::endl;

  // Read the result back from the buffer
  data_buffer_kernel_dot_add_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_dot_add_output.read(fc_output);
  std::cout << "dot_add kernel done" << std::endl;

  for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
    fc_output[i] = fc_output[i] / 15000;
  }

  // softmax(fc_output, softmax_exp_results, softmax_output, FULL_CONNECT_LAYER_SIZE_W);
  // write data to buffer of the softmax kernel
  data_buffer_kernel_softmax_input.write(fc_output);
  data_buffer_kernel_softmax_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_softmax_exp_results.write(softmax_exp_results);
  data_buffer_kernel_softmax_exp_results.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of the softmax kernel
  run_kernel_softmax.set_arg(softmax_ptr_input, data_buffer_kernel_softmax_input);
  run_kernel_softmax.set_arg(softmax_ptr_exp_results, data_buffer_kernel_softmax_exp_results);
  run_kernel_softmax.set_arg(softmax_ptr_output, data_buffer_kernel_softmax_output);
  run_kernel_softmax.set_arg(softmax_ptr_size, FULL_CONNECT_LAYER_SIZE_W);

  std::cout << "set arguments of the softmax kernel done" << std::endl;

  // run the softmax kernel
  run_kernel_softmax.start();
  run_kernel_softmax.wait();

  std::cout << "Run softmax kernel done" << std::endl;

  // Read the result back from the buffer
  data_buffer_kernel_softmax_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_softmax_output.read(softmax_output);
  std::cout << "softmax kernel done" << std::endl;

  for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
    softmax_output[i] = softmax_output[i] * 10000;
  }
  std::cout << "1 warm up iteration done" << std::endl;

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
    std::cout << "Iteration " << i << std::endl;
    
    start_iteration_time = omp_get_wtime();
    // conv2d(input_image, W_conv, input_padded, CONV2D_BIAS, CONV2D_STRIDE, CONV2D_PADDING, INPUT_SIZE_H, INPUT_SIZE_W, CONV_KERNEL_SIZE_H, CONV_KERNEL_SIZE_W, conv_output);
    // write data to buffer of the conv2d kernel
    data_buffer_kernel_conv2d_input.write(input_image);
    data_buffer_kernel_conv2d_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_conv2d_kernel.write(W_conv);
    data_buffer_kernel_conv2d_kernel.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_conv2d_input_padded.write(input_padded);
    data_buffer_kernel_conv2d_input_padded.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // set arguments of the conv2d kernel
    run_kernel_conv2d.set_arg(conv2d_ptr_input, data_buffer_kernel_conv2d_input);
    run_kernel_conv2d.set_arg(conv2d_ptr_kernel, data_buffer_kernel_conv2d_kernel);
    run_kernel_conv2d.set_arg(conv2d_ptr_input_padded, data_buffer_kernel_conv2d_input_padded);
    run_kernel_conv2d.set_arg(conv2d_ptr_bias, CONV2D_BIAS);
    run_kernel_conv2d.set_arg(conv2d_ptr_stride, CONV2D_STRIDE);
    run_kernel_conv2d.set_arg(conv2d_ptr_padding, CONV2D_PADDING);
    run_kernel_conv2d.set_arg(conv2d_ptr_input_h, INPUT_SIZE_H);
    run_kernel_conv2d.set_arg(conv2d_ptr_input_w, INPUT_SIZE_W);
    run_kernel_conv2d.set_arg(conv2d_ptr_kernel_h, CONV_KERNEL_SIZE_H);
    run_kernel_conv2d.set_arg(conv2d_ptr_kernel_w, CONV_KERNEL_SIZE_W);
    run_kernel_conv2d.set_arg(conv2d_ptr_output, data_buffer_kernel_conv2d_output);

    std::cout << "    set arguments of the conv2d kernel done" << std::endl;

    // run the conv2d kernel
    run_kernel_conv2d.start();
    run_kernel_conv2d.wait();

    std::cout << "    Run conv2d kernel done" << std::endl;

    // Read the result back from the buffer
    data_buffer_kernel_conv2d_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_conv2d_output.read(conv_output);
    std::cout << "conv2d kernel done" << std::endl;

    conv2d_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    // relu(conv_output, relu_output, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
    // write data to buffer of the relu kernel
    data_buffer_kernel_relu_input.write(conv_output);
    data_buffer_kernel_relu_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of the relu kernel
    run_kernel_relu.set_arg(relu_ptr_input, data_buffer_kernel_relu_input);
    run_kernel_relu.set_arg(relu_ptr_output, data_buffer_kernel_relu_output);
    run_kernel_relu.set_arg(relu_ptr_size, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);

    std::cout << "    set arguments of the relu kernel done" << std::endl;

    // run the relu kernel
    run_kernel_relu.start();
    run_kernel_relu.wait();

    std::cout << "    Run relu kernel done" << std::endl;

    // Read the result back from the buffer
    data_buffer_kernel_relu_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_relu_output.read(relu_output);
    std::cout << "    relu kernel done" << std::endl;

    relu_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    // max_pooling(relu_output, POOLING_SIZE, POOLING_STRIDE, CONV_OUTPUT_HEIGHT, CONV_OUTPUT_WIDTH, pool_output);
    // write data to buffer of the max_pooling kernel
    data_buffer_kernel_max_pooling_input.write(relu_output);
    data_buffer_kernel_max_pooling_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of the max_pooling kernel
    run_kernel_max_pooling.set_arg(max_pooling_ptr_input, data_buffer_kernel_max_pooling_input);
    run_kernel_max_pooling.set_arg(max_pooling_ptr_pool_size, POOLING_SIZE);
    run_kernel_max_pooling.set_arg(max_pooling_ptr_pool_stride, POOLING_STRIDE);
    run_kernel_max_pooling.set_arg(max_pooling_ptr_input_h, CONV_OUTPUT_HEIGHT);
    run_kernel_max_pooling.set_arg(max_pooling_ptr_input_w, CONV_OUTPUT_WIDTH);
    run_kernel_max_pooling.set_arg(max_pooling_ptr_output, data_buffer_kernel_max_pooling_output);

    std::cout << "    set arguments of the max_pooling kernel done" << std::endl;

    // run the max_pooling kernel
    run_kernel_max_pooling.start();
    run_kernel_max_pooling.wait();

    std::cout << "    Run max_pooling kernel done" << std::endl;

    // Read the result back from the buffer
    data_buffer_kernel_max_pooling_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_max_pooling_output.read(pool_output);
    std::cout << "    max_pooling kernel done" << std::endl;

    max_pooling_time += omp_get_wtime() - start_iteration_time;

    // flattened_output = pooled_output.flatten()
    for (int i = 0; i < POOLING_OUTPUT_HEIGHT; i++) {
      for (int j = 0; j < POOLING_OUTPUT_WIDTH; j++) {
        flattened_output[i * POOLING_OUTPUT_WIDTH + j] = pool_output[i * POOLING_OUTPUT_WIDTH + j];
      }
    }
    std::cout << "    flattened_output kernel done" << std::endl;

    start_iteration_time = omp_get_wtime();
    // Note here the size of flattened_output is 1 x FLATTENED_OUTPUT_SIZE
    // dot_add(flattened_output, W_fc, b_fc, fc_output, 1, FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_W);
    // write data to buffer of the dot_add kernel
    data_buffer_kernel_dot_add_input_x.write(flattened_output);
    data_buffer_kernel_dot_add_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_dot_add_input_W.write(W_fc);
    data_buffer_kernel_dot_add_input_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_dot_add_input_b.write(b_fc);
    data_buffer_kernel_dot_add_input_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of the dot_add kernel
    run_kernel_dot_add.set_arg(dot_add_ptr_input_x, data_buffer_kernel_dot_add_input_x);
    run_kernel_dot_add.set_arg(dot_add_ptr_input_W, data_buffer_kernel_dot_add_input_W);
    run_kernel_dot_add.set_arg(dot_add_ptr_input_b, data_buffer_kernel_dot_add_input_b);
    run_kernel_dot_add.set_arg(dot_add_ptr_output, data_buffer_kernel_dot_add_output);

    std::cout << "    set arguments of the dot_add kernel done" << std::endl;

    // run the dot_add kernel
    run_kernel_dot_add.start();
    run_kernel_dot_add.wait();

    std::cout << "    Run dot_add kernel done" << std::endl;

    // Read the result back from the buffer
    data_buffer_kernel_dot_add_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_dot_add_output.read(fc_output);
    std::cout << "    dot_add kernel done" << std::endl;

    dot_add_time += omp_get_wtime() - start_iteration_time;

    for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
      fc_output[i] = fc_output[i] / 15000;
    }

    start_iteration_time = omp_get_wtime();
    // softmax(fc_output, softmax_exp_results, softmax_output, FULL_CONNECT_LAYER_SIZE_W);
    // write data to buffer of the softmax kernel
    data_buffer_kernel_softmax_input.write(fc_output);
    data_buffer_kernel_softmax_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_softmax_exp_results.write(softmax_exp_results);
    data_buffer_kernel_softmax_exp_results.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of the softmax kernel
    run_kernel_softmax.set_arg(softmax_ptr_input, data_buffer_kernel_softmax_input);
    run_kernel_softmax.set_arg(softmax_ptr_exp_results, data_buffer_kernel_softmax_exp_results);
    run_kernel_softmax.set_arg(softmax_ptr_output, data_buffer_kernel_softmax_output);
    run_kernel_softmax.set_arg(softmax_ptr_size, FULL_CONNECT_LAYER_SIZE_W);

    std::cout << "    set arguments of the softmax kernel done" << std::endl;

    // run the softmax kernel
    run_kernel_softmax.start();
    run_kernel_softmax.wait();

    std::cout << "    Run softmax kernel done" << std::endl;

    // Read the result back from the buffer
    data_buffer_kernel_softmax_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_softmax_output.read(softmax_output);
    std::cout << "    softmax kernel done" << std::endl;

    softmax_time += omp_get_wtime() - start_iteration_time;
    
    for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
      softmax_output[i] = softmax_output[i] * 10000;
    }
  }
  std::cout << "Run " << iterations << " iterations done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations << " iterations " << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "conv2d kernel time: " << conv2d_time / iterations * 1000 << " ms" << endl;
  cout << "relu kernel time: " << relu_time / iterations * 1000 << " ms" << endl;
  cout << "max_pooling kernel time: " << max_pooling_time / iterations * 1000 << " ms" << endl;
  cout << "dot_add kernel time: " << dot_add_time / iterations * 1000 << " ms" << endl;
  cout << "softmax kernel time: " << softmax_time / iterations * 1000 << " ms" << endl;

  delete[] input_image;
  delete[] W_conv;
  delete[] W_fc;
  delete[] b_fc;
  delete[] conv_output;
  delete[] relu_output;
  delete[] pool_output;
  delete[] flattened_output;
  delete[] fc_output;
  delete[] softmax_output;
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
  std::cout << "Running cnn benchmark C++ HLS" << std::endl;
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