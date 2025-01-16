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

#define dot_add_ptr_input_x 0
#define dot_add_ptr_input_W 1
#define dot_add_ptr_input_b 2
#define dot_add_ptr_output 3
#define dot_add_ptr_x_h 4
#define dot_add_ptr_x_w 5
#define dot_add_ptr_W_h 6
#define dot_add_ptr_W_w 7

#define sigmoid_ptr_input 0
#define sigmoid_ptr_output 1
#define sigmoid_ptr_size 2

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
    if (abs(data[i] - data_golden[i]) > 1e-2) {
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

void mlp_forward(string input_path, string output_path)
{
  double* data_a0 = new double[SIZE_A0];
  double* data_a1 = new double[SIZE_A1];
  double* data_z1 = new double[SIZE_Z1];
  double* data_a2 = new double[SIZE_A2];
  double* data_z2 = new double[SIZE_Z2];
  double* data_a3 = new double[SIZE_A3];
  double* data_z3 = new double[SIZE_Z3];
  double* data_a4 = new double[SIZE_A4];
  double* data_a4_exp = new double[SIZE_A4];
  double* data_z4 = new double[SIZE_Z4];

  double* data_w0 = new double[SIZE_W0];
  double* data_w1 = new double[SIZE_W1];
  double* data_w2 = new double[SIZE_W2];
  double* data_w3 = new double[SIZE_W3];

  double* data_b0 = new double[SIZE_B0];
  double* data_b1 = new double[SIZE_B1];
  double* data_b2 = new double[SIZE_B2];
  double* data_b3 = new double[SIZE_B3];

  // readData("../../dataset/a0.bin", data_a0, SIZE_A0);
  // readData("../../dataset/w0.bin", data_w0, SIZE_W0);
  // readData("../../dataset/b0.bin", data_b0, SIZE_B0);
  // readData("../../dataset/w1.bin", data_w1, SIZE_W1);
  // readData("../../dataset/b1.bin", data_b1, SIZE_B1);
  // readData("../../dataset/w2.bin", data_w2, SIZE_W2);
  // readData("../../dataset/b2.bin", data_b2, SIZE_B2);
  // readData("../../dataset/w3.bin", data_w3, SIZE_W3);
  // readData("../../dataset/b3.bin", data_b3, SIZE_B3);
  readData((input_path + "/a0.bin").c_str(), data_a0, SIZE_A0);
  readData((input_path + "/w0.bin").c_str(), data_w0, SIZE_W0);
  readData((input_path + "/b0.bin").c_str(), data_b0, SIZE_B0);
  readData((input_path + "/w1.bin").c_str(), data_w1, SIZE_W1);
  readData((input_path + "/b1.bin").c_str(), data_b1, SIZE_B1);
  readData((input_path + "/w2.bin").c_str(), data_w2, SIZE_W2);
  readData((input_path + "/b2.bin").c_str(), data_b2, SIZE_B2);
  readData((input_path + "/w3.bin").c_str(), data_w3, SIZE_W3);
  readData((input_path + "/b3.bin").c_str(), data_b3, SIZE_B3);
  
  // Load xclbin
  std::string xclbin_file = "overlay_hw.xclbin";
  std::cout << "Loading: " << xclbin_file << std::endl;
  xrt::device device = xrt::device(DEVICE_ID);
  xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
  std::cout << "Loaded xclbin: " << xclbin_file << std::endl;
  
  // create kernel object
  xrt::kernel kernel_dot_add0 = xrt::kernel(device, xclbin_uuid, "dot_add0");
  xrt::kernel kernel_dot_add1 = xrt::kernel(device, xclbin_uuid, "dot_add1");
  xrt::kernel kernel_dot_add2 = xrt::kernel(device, xclbin_uuid, "dot_add2");
  xrt::kernel kernel_dot_add3 = xrt::kernel(device, xclbin_uuid, "dot_add3");
  std::cout << "Created kernel: " << "dot_adds" << std::endl;
  xrt::kernel kernel_sigmoid = xrt::kernel(device, xclbin_uuid, "sigmoid");
  std::cout << "Created kernel: " << "sigmoid" << std::endl;
  xrt::kernel kernel_softmax = xrt::kernel(device, xclbin_uuid, "softmax");
  std::cout << "Created kernel: " << "softmax" << std::endl;

  // create memory groups
  xrtMemoryGroup bank_grp_kernel_dot_add0_input_x = kernel_dot_add0.group_id(dot_add_ptr_input_x);
  xrtMemoryGroup bank_grp_kernel_dot_add0_input_W = kernel_dot_add0.group_id(dot_add_ptr_input_W);
  xrtMemoryGroup bank_grp_kernel_dot_add0_input_b = kernel_dot_add0.group_id(dot_add_ptr_input_b);
  xrtMemoryGroup bank_grp_kernel_dot_add0_output = kernel_dot_add0.group_id(dot_add_ptr_output);

  xrtMemoryGroup bank_grp_kernel_dot_add1_input_x = kernel_dot_add1.group_id(dot_add_ptr_input_x);
  xrtMemoryGroup bank_grp_kernel_dot_add1_input_W = kernel_dot_add1.group_id(dot_add_ptr_input_W);
  xrtMemoryGroup bank_grp_kernel_dot_add1_input_b = kernel_dot_add1.group_id(dot_add_ptr_input_b);
  xrtMemoryGroup bank_grp_kernel_dot_add1_output = kernel_dot_add1.group_id(dot_add_ptr_output);

  xrtMemoryGroup bank_grp_kernel_dot_add2_input_x = kernel_dot_add2.group_id(dot_add_ptr_input_x);
  xrtMemoryGroup bank_grp_kernel_dot_add2_input_W = kernel_dot_add2.group_id(dot_add_ptr_input_W);
  xrtMemoryGroup bank_grp_kernel_dot_add2_input_b = kernel_dot_add2.group_id(dot_add_ptr_input_b);
  xrtMemoryGroup bank_grp_kernel_dot_add2_output = kernel_dot_add2.group_id(dot_add_ptr_output);

  xrtMemoryGroup bank_grp_kernel_dot_add3_input_x = kernel_dot_add3.group_id(dot_add_ptr_input_x);
  xrtMemoryGroup bank_grp_kernel_dot_add3_input_W = kernel_dot_add3.group_id(dot_add_ptr_input_W);
  xrtMemoryGroup bank_grp_kernel_dot_add3_input_b = kernel_dot_add3.group_id(dot_add_ptr_input_b);
  xrtMemoryGroup bank_grp_kernel_dot_add3_output = kernel_dot_add3.group_id(dot_add_ptr_output);

  xrtMemoryGroup bank_grp_kernel_sigmoid_input = kernel_sigmoid.group_id(sigmoid_ptr_input);
  xrtMemoryGroup bank_grp_kernel_sigmoid_output = kernel_sigmoid.group_id(sigmoid_ptr_output);

  xrtMemoryGroup bank_grp_kernel_softmax_input = kernel_softmax.group_id(softmax_ptr_input);
  xrtMemoryGroup bank_grp_kernel_softmax_exp_results = kernel_softmax.group_id(softmax_ptr_exp_results);
  xrtMemoryGroup bank_grp_kernel_softmax_output = kernel_softmax.group_id(softmax_ptr_output);

  // create buffer objects
  xrt::bo data_buffer_kernel_dot_add0_input_x = \
    xrt::bo(device, SIZE_A0 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add0_input_x);
  xrt::bo data_buffer_kernel_dot_add0_input_W = \
    xrt::bo(device, SIZE_W0 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add0_input_W);
  xrt::bo data_buffer_kernel_dot_add0_input_b = \
    xrt::bo(device, SIZE_B0 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add0_input_b);
  xrt::bo data_buffer_kernel_dot_add0_output = \
    xrt::bo(device, SIZE_B0 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add0_output);

  xrt::bo data_buffer_kernel_dot_add1_input_x = \
    xrt::bo(device, SIZE_A1 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add1_input_x);
  xrt::bo data_buffer_kernel_dot_add1_input_W = \
    xrt::bo(device, SIZE_W1 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add1_input_W);
  xrt::bo data_buffer_kernel_dot_add1_input_b = \
    xrt::bo(device, SIZE_B1 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add1_input_b);
  xrt::bo data_buffer_kernel_dot_add1_output = \
    xrt::bo(device, SIZE_B1 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add1_output);

  xrt::bo data_buffer_kernel_dot_add2_input_x = \
    xrt::bo(device, SIZE_A2 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add2_input_x);
  xrt::bo data_buffer_kernel_dot_add2_input_W = \
    xrt::bo(device, SIZE_W2 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add2_input_W);
  xrt::bo data_buffer_kernel_dot_add2_input_b = \
    xrt::bo(device, SIZE_B2 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add2_input_b);
  xrt::bo data_buffer_kernel_dot_add2_output = \
    xrt::bo(device, SIZE_B2 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add2_output);

  xrt::bo data_buffer_kernel_dot_add3_input_x = \
    xrt::bo(device, SIZE_A3 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add3_input_x);
  xrt::bo data_buffer_kernel_dot_add3_input_W = \
    xrt::bo(device, SIZE_W3 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add3_input_W);
  xrt::bo data_buffer_kernel_dot_add3_input_b = \
    xrt::bo(device, SIZE_B3 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add3_input_b);
  xrt::bo data_buffer_kernel_dot_add3_output = \
    xrt::bo(device, SIZE_B3 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_dot_add3_output);

  xrt::bo data_buffer_kernel_sigmoid_input = \
    xrt::bo(device, SIZE_A1 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_sigmoid_input);
  xrt::bo data_buffer_kernel_sigmoid_output = \
    xrt::bo(device, SIZE_Z1 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_sigmoid_output);

  xrt::bo data_buffer_kernel_softmax_input = \
    xrt::bo(device, SIZE_A4 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_softmax_input);
  xrt::bo data_buffer_kernel_softmax_exp_results = \
    xrt::bo(device, SIZE_Z4 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_softmax_exp_results);
  xrt::bo data_buffer_kernel_softmax_output = \
    xrt::bo(device, SIZE_Z4 * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_softmax_output);

  std::cout << "Created buffer objects" << std::endl;

  // create kernel runner
  xrt::run run_kernel_dot_add0(kernel_dot_add0);
  xrt::run run_kernel_dot_add1(kernel_dot_add1);
  xrt::run run_kernel_dot_add2(kernel_dot_add2);
  xrt::run run_kernel_dot_add3(kernel_dot_add3);
  xrt::run run_kernel_sigmoid(kernel_sigmoid);
  xrt::run run_kernel_softmax(kernel_softmax);

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;
  // dot_add(data_a0, data_w0, data_b0, data_a1, L0_H1, L0_W1, L0_W1, L0_W2);

  // write data to buffer of dot_add kernel
  data_buffer_kernel_dot_add0_input_x.write(data_a0);
  data_buffer_kernel_dot_add0_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_dot_add0_input_W.write(data_w0);
  data_buffer_kernel_dot_add0_input_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_dot_add0_input_b.write(data_b0);
  data_buffer_kernel_dot_add0_input_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of dot_add kernel
  run_kernel_dot_add0.set_arg(dot_add_ptr_input_x, data_buffer_kernel_dot_add0_input_x);
  run_kernel_dot_add0.set_arg(dot_add_ptr_input_W, data_buffer_kernel_dot_add0_input_W);
  run_kernel_dot_add0.set_arg(dot_add_ptr_input_b, data_buffer_kernel_dot_add0_input_b);
  run_kernel_dot_add0.set_arg(dot_add_ptr_output, data_buffer_kernel_dot_add0_output);

  std::cout << "Set arguments of dot_add0 kernel" << std::endl;

  // run dot_add kernel
  run_kernel_dot_add0.start();
  run_kernel_dot_add0.wait();

  std::cout << "Run dot_add0 kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_dot_add0_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_dot_add0_output.read(data_a1);

  std::cout << "Read the result back" << std::endl;

  #pragma omp parallel for
  for (int i = 0; i < SIZE_A1; i++) {
    data_a1[i] = data_a1[i] / 500;
  }
  // sigmoid(data_a1, data_z1, SIZE_Z1);

  // write data to buffer of sigmoid kernel
  data_buffer_kernel_sigmoid_input.write(data_a1);
  data_buffer_kernel_sigmoid_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of sigmoid kernel
  run_kernel_sigmoid.set_arg(sigmoid_ptr_input, data_buffer_kernel_sigmoid_input);
  run_kernel_sigmoid.set_arg(sigmoid_ptr_output, data_buffer_kernel_sigmoid_output);
  run_kernel_sigmoid.set_arg(sigmoid_ptr_size, SIZE_Z1);

  std::cout << "Set arguments of sigmoid kernel" << std::endl;

  // run sigmoid kernel
  run_kernel_sigmoid.start();
  run_kernel_sigmoid.wait();

  std::cout << "Run sigmoid kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_sigmoid_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_sigmoid_output.read(data_z1);

  std::cout << "Read the result back" << std::endl;

  // dot_add(data_z1, data_w1, data_b1, data_a2, L1_H1, L1_W1, L1_W1, L1_W2);

  // write data to buffer of dot_add kernel
  data_buffer_kernel_dot_add1_input_x.write(data_z1);
  data_buffer_kernel_dot_add1_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_dot_add1_input_W.write(data_w1);
  data_buffer_kernel_dot_add1_input_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_dot_add1_input_b.write(data_b1);
  data_buffer_kernel_dot_add1_input_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of dot_add kernel
  run_kernel_dot_add1.set_arg(dot_add_ptr_input_x, data_buffer_kernel_dot_add1_input_x);
  run_kernel_dot_add1.set_arg(dot_add_ptr_input_W, data_buffer_kernel_dot_add1_input_W);
  run_kernel_dot_add1.set_arg(dot_add_ptr_input_b, data_buffer_kernel_dot_add1_input_b);
  run_kernel_dot_add1.set_arg(dot_add_ptr_output, data_buffer_kernel_dot_add1_output);

  std::cout << "Set arguments of dot_add1 kernel" << std::endl;

  // run dot_add kernel
  run_kernel_dot_add1.start();
  run_kernel_dot_add1.wait();

  std::cout << "Run dot_add1 kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_dot_add1_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_dot_add1_output.read(data_a2);

  std::cout << "Read the result back" << std::endl;
  
  #pragma omp parallel for
  for (int i = 0; i < SIZE_A2; i++) {
    data_a2[i] = data_a2[i] / 1500;
  }
  // sigmoid(data_a2, data_z2, SIZE_Z2);

  // write data to buffer of sigmoid kernel
  data_buffer_kernel_sigmoid_input.write(data_a2);
  data_buffer_kernel_sigmoid_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of sigmoid kernel
  run_kernel_sigmoid.set_arg(sigmoid_ptr_input, data_buffer_kernel_sigmoid_input);
  run_kernel_sigmoid.set_arg(sigmoid_ptr_output, data_buffer_kernel_sigmoid_output);
  run_kernel_sigmoid.set_arg(sigmoid_ptr_size, SIZE_Z2);

  std::cout << "Set arguments of sigmoid kernel" << std::endl;

  // run sigmoid kernel
  run_kernel_sigmoid.start();
  run_kernel_sigmoid.wait();

  std::cout << "Run sigmoid kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_sigmoid_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_sigmoid_output.read(data_z2);

  std::cout << "Read the result back" << std::endl;

  // dot_add(data_z2, data_w2, data_b2, data_a3, L2_H1, L2_W1, L2_W1, L2_W2);

  // write data to buffer of dot_add kernel
  data_buffer_kernel_dot_add2_input_x.write(data_z2);
  data_buffer_kernel_dot_add2_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_dot_add2_input_W.write(data_w2);
  data_buffer_kernel_dot_add2_input_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_dot_add2_input_b.write(data_b2);
  data_buffer_kernel_dot_add2_input_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of dot_add kernel
  run_kernel_dot_add2.set_arg(dot_add_ptr_input_x, data_buffer_kernel_dot_add2_input_x);
  run_kernel_dot_add2.set_arg(dot_add_ptr_input_W, data_buffer_kernel_dot_add2_input_W);
  run_kernel_dot_add2.set_arg(dot_add_ptr_input_b, data_buffer_kernel_dot_add2_input_b);
  run_kernel_dot_add2.set_arg(dot_add_ptr_output, data_buffer_kernel_dot_add2_output);


  std::cout << "Set arguments of dot_add2 kernel" << std::endl;

  // run dot_add kernel
  run_kernel_dot_add2.start();
  run_kernel_dot_add2.wait();

  std::cout << "Run dot_add2 kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_dot_add2_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_dot_add2_output.read(data_a3);

  std::cout << "Read the result back" << std::endl;
  
  #pragma omp parallel for
  for (int i = 0; i < SIZE_A3; i++) {
    data_a3[i] = data_a3[i] / 1500;
  }
  // sigmoid(data_a3, data_z3, SIZE_Z3);

  // write data to buffer of sigmoid kernel
  data_buffer_kernel_sigmoid_input.write(data_a3);
  data_buffer_kernel_sigmoid_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of sigmoid kernel
  run_kernel_sigmoid.set_arg(sigmoid_ptr_input, data_buffer_kernel_sigmoid_input);
  run_kernel_sigmoid.set_arg(sigmoid_ptr_output, data_buffer_kernel_sigmoid_output);
  run_kernel_sigmoid.set_arg(sigmoid_ptr_size, SIZE_Z3);

  std::cout << "Set arguments of sigmoid kernel" << std::endl;

  // run sigmoid kernel
  run_kernel_sigmoid.start();
  run_kernel_sigmoid.wait();

  std::cout << "Run sigmoid kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_sigmoid_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_sigmoid_output.read(data_z3);

  std::cout << "Read the result back" << std::endl;

  // dot_add(data_z3, data_w3, data_b3, data_a4, L3_H1, L3_W1, L3_W1, L3_W2);

  // write data to buffer of dot_add kernel
  data_buffer_kernel_dot_add3_input_x.write(data_z3);
  data_buffer_kernel_dot_add3_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_dot_add3_input_W.write(data_w3);
  data_buffer_kernel_dot_add3_input_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_dot_add3_input_b.write(data_b3);
  data_buffer_kernel_dot_add3_input_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of dot_add kernel
  run_kernel_dot_add3.set_arg(dot_add_ptr_input_x, data_buffer_kernel_dot_add3_input_x);
  run_kernel_dot_add3.set_arg(dot_add_ptr_input_W, data_buffer_kernel_dot_add3_input_W);
  run_kernel_dot_add3.set_arg(dot_add_ptr_input_b, data_buffer_kernel_dot_add3_input_b);
  run_kernel_dot_add3.set_arg(dot_add_ptr_output, data_buffer_kernel_dot_add3_output);


  std::cout << "Set arguments of dot_add3 kernel" << std::endl;

  //run dot_add kernel
  run_kernel_dot_add3.start();
  run_kernel_dot_add3.wait();

  std::cout << "Run dot_add3 kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_dot_add3_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_dot_add3_output.read(data_a4);

  std::cout << "Read the result back" << std::endl;

  #pragma omp parallel for
  for (int i = 0; i < SIZE_A4; i++) {
    data_a4[i] = data_a4[i] / 1500;
  }
  // softmax(data_a4, data_a4_exp, data_z4, SIZE_Z4);

  // write data to buffer of softmax kernel
  data_buffer_kernel_softmax_input.write(data_a4);
  data_buffer_kernel_softmax_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_softmax_exp_results.write(data_a4_exp);
  data_buffer_kernel_softmax_exp_results.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of softmax kernel
  run_kernel_softmax.set_arg(softmax_ptr_input, data_buffer_kernel_softmax_input);
  run_kernel_softmax.set_arg(softmax_ptr_exp_results, data_buffer_kernel_softmax_exp_results);
  run_kernel_softmax.set_arg(softmax_ptr_output, data_buffer_kernel_softmax_output);
  run_kernel_softmax.set_arg(softmax_ptr_size, SIZE_Z4);

  std::cout << "Set arguments of softmax kernel" << std::endl;

  // run softmax kernel
  run_kernel_softmax.start();
  run_kernel_softmax.wait();

  // Read the result back
  data_buffer_kernel_softmax_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_softmax_output.read(data_z4);

  std::cout << "Read the result back" << std::endl;

  #pragma omp parallel for
  for (int i = 0; i < SIZE_A4; i++) {
    data_z4[i] = data_z4[i] * 1000000;
  }

  std::cout << "1 warm up iteration done" << std::endl;

  // check the result
  double *data_a1_golden = new double [SIZE_A1];
  double *data_z1_golden = new double [SIZE_Z1];
  double *data_a2_golden = new double [SIZE_A2];
  double *data_z2_golden = new double [SIZE_Z2];
  double *data_a3_golden = new double [SIZE_A3];
  double *data_z3_golden = new double [SIZE_Z3];
  double *data_a4_golden = new double [SIZE_A4];
  double *data_z4_golden = new double [SIZE_Z4];
/*
  // readData("../../dataset/a1.bin", data_a1_golden, SIZE_A1);
  // readData("../../dataset/z1.bin", data_z1_golden, SIZE_Z1);
  // readData("../../dataset/a2.bin", data_a2_golden, SIZE_A2);
  // readData("../../dataset/z2.bin", data_z2_golden, SIZE_Z2);
  // readData("../../dataset/a3.bin", data_a3_golden, SIZE_A3);
  // readData("../../dataset/z3.bin", data_z3_golden, SIZE_Z3);
  // readData("../../dataset/a4.bin", data_a4_golden, SIZE_A4);
  // readData("../../dataset/z4.bin", data_z4_golden, SIZE_Z4);
  readData((output_path + "/a1.bin").c_str(), data_a1_golden, SIZE_A1);
  readData((output_path + "/z1.bin").c_str(), data_z1_golden, SIZE_Z1);
  readData((output_path + "/a2.bin").c_str(), data_a2_golden, SIZE_A2);
  readData((output_path + "/z2.bin").c_str(), data_z2_golden, SIZE_Z2);
  readData((output_path + "/a3.bin").c_str(), data_a3_golden, SIZE_A3);
  readData((output_path + "/z3.bin").c_str(), data_z3_golden, SIZE_Z3);
  readData((output_path + "/a4.bin").c_str(), data_a4_golden, SIZE_A4);
  readData((output_path + "/z4.bin").c_str(), data_z4_golden, SIZE_Z4);

  // check a1 
  std::cout << "check a1 ... ";
  checkResult(data_a1, data_a1_golden, SIZE_A1);
  // check z1
  std::cout << "check z1 ... ";
  checkResult(data_z1, data_z1_golden, SIZE_Z1);
  // check a2
  std::cout << "check a2 ... ";
  checkResult(data_a2, data_a2_golden, SIZE_A2);
  // check z2
  std::cout << "check z2 ... ";
  checkResult(data_z2, data_z2_golden, SIZE_Z2);
  // check a3
  std::cout << "check a3 ... ";
  checkResult(data_a3, data_a3_golden, SIZE_A3);
  // check z3
  std::cout << "check z3 ... ";
  checkResult(data_z3, data_z3_golden, SIZE_Z3);
  // check a4
  std::cout << "check a4 ... ";
  checkResult(data_a4, data_a4_golden, SIZE_A4);
  // check z4
  std::cout << "check z4 ... ";
  checkResult(data_z4, data_z4_golden, SIZE_Z4);
*/

  // multi iterations
  int iterations = ITERATIONS;
  std::cout << "Running " << iterations << " iterations ..." << std::endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double layer_0_time = 0;
  double layer_1_time = 0;
  double layer_2_time = 0;
  double layer_3_time = 0;

  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    // dot_add(data_a0, data_w0, data_b0, data_a1, L0_H1, L0_W1, L0_W1, L0_W2);

    // write data to buffer of dot_add kernel
    data_buffer_kernel_dot_add0_input_x.write(data_a0);
    data_buffer_kernel_dot_add0_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_dot_add0_input_W.write(data_w0);
    data_buffer_kernel_dot_add0_input_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_dot_add0_input_b.write(data_b0);
    data_buffer_kernel_dot_add0_input_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of dot_add kernel
    run_kernel_dot_add0.set_arg(dot_add_ptr_input_x, data_buffer_kernel_dot_add0_input_x);
    run_kernel_dot_add0.set_arg(dot_add_ptr_input_W, data_buffer_kernel_dot_add0_input_W);
    run_kernel_dot_add0.set_arg(dot_add_ptr_input_b, data_buffer_kernel_dot_add0_input_b);
    run_kernel_dot_add0.set_arg(dot_add_ptr_output, data_buffer_kernel_dot_add0_output);

    // run dot_add kernel
    run_kernel_dot_add0.start();
    run_kernel_dot_add0.wait();

    // Read the result back
    data_buffer_kernel_dot_add0_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_dot_add0_output.read(data_a1);

    #pragma omp parallel for
    for (int i = 0; i < SIZE_A1; i++) {
      data_a1[i] = data_a1[i] / 500;
    }
    // sigmoid(data_a1, data_z1, SIZE_Z1);

    // write data to buffer of sigmoid kernel
    data_buffer_kernel_sigmoid_input.write(data_a1);
    data_buffer_kernel_sigmoid_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of sigmoid kernel
    run_kernel_sigmoid.set_arg(sigmoid_ptr_input, data_buffer_kernel_sigmoid_input);
    run_kernel_sigmoid.set_arg(sigmoid_ptr_output, data_buffer_kernel_sigmoid_output);
    run_kernel_sigmoid.set_arg(sigmoid_ptr_size, SIZE_Z1);

    // run sigmoid kernel
    run_kernel_sigmoid.start();
    run_kernel_sigmoid.wait();

    // Read the result back
    data_buffer_kernel_sigmoid_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_sigmoid_output.read(data_z1);

    layer_0_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    // dot_add(data_z1, data_w1, data_b1, data_a2, L1_H1, L1_W1, L1_W1, L1_W2);

    // write data to buffer of dot_add kernel
    data_buffer_kernel_dot_add1_input_x.write(data_z1);
    data_buffer_kernel_dot_add1_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_dot_add1_input_W.write(data_w1);
    data_buffer_kernel_dot_add1_input_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_dot_add1_input_b.write(data_b1);
    data_buffer_kernel_dot_add1_input_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of dot_add kernel
    run_kernel_dot_add1.set_arg(dot_add_ptr_input_x, data_buffer_kernel_dot_add1_input_x);
    run_kernel_dot_add1.set_arg(dot_add_ptr_input_W, data_buffer_kernel_dot_add1_input_W);
    run_kernel_dot_add1.set_arg(dot_add_ptr_input_b, data_buffer_kernel_dot_add1_input_b);
    run_kernel_dot_add1.set_arg(dot_add_ptr_output, data_buffer_kernel_dot_add1_output);

    // run dot_add kernel
    run_kernel_dot_add1.start();
    run_kernel_dot_add1.wait();

    // Read the result back
    data_buffer_kernel_dot_add1_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_dot_add1_output.read(data_a2);
    
    #pragma omp parallel for
    for (int i = 0; i < SIZE_A2; i++) {
      data_a2[i] = data_a2[i] / 1500;
    }
    // sigmoid(data_a2, data_z2, SIZE_Z2);

    // write data to buffer of sigmoid kernel
    data_buffer_kernel_sigmoid_input.write(data_a2);
    data_buffer_kernel_sigmoid_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of sigmoid kernel
    run_kernel_sigmoid.set_arg(sigmoid_ptr_input, data_buffer_kernel_sigmoid_input);
    run_kernel_sigmoid.set_arg(sigmoid_ptr_output, data_buffer_kernel_sigmoid_output);
    run_kernel_sigmoid.set_arg(sigmoid_ptr_size, SIZE_Z2);

    // run sigmoid kernel
    run_kernel_sigmoid.start();
    run_kernel_sigmoid.wait();

    // Read the result back
    data_buffer_kernel_sigmoid_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_sigmoid_output.read(data_z2);

    layer_1_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    // dot_add(data_z2, data_w2, data_b2, data_a3, L2_H1, L2_W1, L2_W1, L2_W2);

    // write data to buffer of dot_add kernel
    data_buffer_kernel_dot_add2_input_x.write(data_z2);
    data_buffer_kernel_dot_add2_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_dot_add2_input_W.write(data_w2);
    data_buffer_kernel_dot_add2_input_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_dot_add2_input_b.write(data_b2);
    data_buffer_kernel_dot_add2_input_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of dot_add kernel
    run_kernel_dot_add2.set_arg(dot_add_ptr_input_x, data_buffer_kernel_dot_add2_input_x);
    run_kernel_dot_add2.set_arg(dot_add_ptr_input_W, data_buffer_kernel_dot_add2_input_W);
    run_kernel_dot_add2.set_arg(dot_add_ptr_input_b, data_buffer_kernel_dot_add2_input_b);
    run_kernel_dot_add2.set_arg(dot_add_ptr_output, data_buffer_kernel_dot_add2_output);

    // run dot_add kernel
    run_kernel_dot_add2.start();
    run_kernel_dot_add2.wait();

    // Read the result back
    data_buffer_kernel_dot_add2_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_dot_add2_output.read(data_a3);
    
    #pragma omp parallel for
    for (int i = 0; i < SIZE_A3; i++) {
      data_a3[i] = data_a3[i] / 1500;
    }
    // sigmoid(data_a3, data_z3, SIZE_Z3);

    // write data to buffer of sigmoid kernel
    data_buffer_kernel_sigmoid_input.write(data_a3);
    data_buffer_kernel_sigmoid_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of sigmoid kernel
    run_kernel_sigmoid.set_arg(sigmoid_ptr_input, data_buffer_kernel_sigmoid_input);
    run_kernel_sigmoid.set_arg(sigmoid_ptr_output, data_buffer_kernel_sigmoid_output);
    run_kernel_sigmoid.set_arg(sigmoid_ptr_size, SIZE_Z3);

    // run sigmoid kernel
    run_kernel_sigmoid.start();
    run_kernel_sigmoid.wait();

    // Read the result back
    data_buffer_kernel_sigmoid_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_sigmoid_output.read(data_z3);

    layer_2_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    // dot_add(data_z3, data_w3, data_b3, data_a4, L3_H1, L3_W1, L3_W1, L3_W2);

    // write data to buffer of dot_add kernel
    data_buffer_kernel_dot_add3_input_x.write(data_z3);
    data_buffer_kernel_dot_add3_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_dot_add3_input_W.write(data_w3);
    data_buffer_kernel_dot_add3_input_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_dot_add3_input_b.write(data_b3);
    data_buffer_kernel_dot_add3_input_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of dot_add kernel
    run_kernel_dot_add3.set_arg(dot_add_ptr_input_x, data_buffer_kernel_dot_add3_input_x);
    run_kernel_dot_add3.set_arg(dot_add_ptr_input_W, data_buffer_kernel_dot_add3_input_W);
    run_kernel_dot_add3.set_arg(dot_add_ptr_input_b, data_buffer_kernel_dot_add3_input_b);
    run_kernel_dot_add3.set_arg(dot_add_ptr_output, data_buffer_kernel_dot_add3_output);

    // run dot_add kernel
    run_kernel_dot_add3.start();
    run_kernel_dot_add3.wait();

    // Read the result back
    data_buffer_kernel_dot_add3_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_dot_add3_output.read(data_a4);

    #pragma omp parallel for
    for (int i = 0; i < SIZE_A4; i++) {
      data_a4[i] = data_a4[i] / 1500;
    }
    // softmax(data_a4, data_a4_exp, data_z4, SIZE_Z4);

    // write data to buffer of softmax kernel
    data_buffer_kernel_softmax_input.write(data_a4);
    data_buffer_kernel_softmax_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_softmax_exp_results.write(data_a4_exp);
    data_buffer_kernel_softmax_exp_results.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of softmax kernel
    run_kernel_softmax.set_arg(softmax_ptr_input, data_buffer_kernel_softmax_input);
    run_kernel_softmax.set_arg(softmax_ptr_exp_results, data_buffer_kernel_softmax_exp_results);
    run_kernel_softmax.set_arg(softmax_ptr_output, data_buffer_kernel_softmax_output);
    run_kernel_softmax.set_arg(softmax_ptr_size, SIZE_Z4);

    // run softmax kernel
    run_kernel_softmax.start();
    run_kernel_softmax.wait();

    // Read the result back
    data_buffer_kernel_softmax_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_softmax_output.read(data_z4);
    
    #pragma omp parallel for
    for (int i = 0; i < SIZE_A4; i++) {
      data_z4[i] = data_z4[i] * 1000000;
    }
    layer_3_time += omp_get_wtime() - start_iteration_time;
  }

  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations << " iterations " << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "Layer 0 time: " << (layer_0_time / iterations) * 1000 << " ms" << endl;
  cout << "Layer 1 time: " << (layer_1_time / iterations) * 1000 << " ms" << endl;
  cout << "Layer 2 time: " << (layer_2_time / iterations) * 1000 << " ms" << endl;
  cout << "Layer 3 time: " << (layer_3_time / iterations) * 1000 << " ms" << endl;

  // check the result
  delete[] data_a0;
  delete[] data_a1;
  delete[] data_z1;
  delete[] data_a2;
  delete[] data_z2;
  delete[] data_a3;
  delete[] data_z3;
  delete[] data_a4;
  delete[] data_a4_exp;
  delete[] data_z4;
  delete[] data_w0;
  delete[] data_w1;
  delete[] data_w2;
  delete[] data_w3;
  delete[] data_b0;
  delete[] data_b1;
  delete[] data_b2;
  delete[] data_b3;
  delete[] data_a1_golden;
  delete[] data_z1_golden;
  delete[] data_a2_golden;
  delete[] data_z2_golden;
  delete[] data_a3_golden;
  delete[] data_z3_golden;
  delete[] data_a4_golden;
  delete[] data_z4_golden;

}

int main(int argc, char *argv[])
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running mlp benchmark C++ HLS" << std::endl;
  std::cout << "=======================================" << std::endl;

  string input_path;
  string output_path;
  if (argc == 3) {
    input_path = argv[1];
    output_path = argv[2];
  } else {
    printf("Usage: ./mlp_sw <input_path> <output_path>\n");
    exit(-1);
  }
  mlp_forward(input_path, output_path);
  return 0;
}