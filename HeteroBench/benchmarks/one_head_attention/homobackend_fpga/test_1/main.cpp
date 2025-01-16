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

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

#define DEVICE_ID 0
#define transpose_ptr_transpose_x 0
#define transpose_ptr_transpose_output 1
#define transpose_ptr_dim0 2
#define transpose_ptr_dim1 3



#define matmul0_ptr_matmul_x 0
#define matmul0_ptr_matmul_y 1
#define matmul0_ptr_matmul_output 2

#define matmul1_ptr_matmul_x 0
#define matmul1_ptr_matmul_y 1
#define matmul1_ptr_matmul_output 2

#define softmax_ptr_softmax_x 0
#define softmax_ptr_softmax_output 1
#define softmax_ptr_axis 3


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
  in_mat.read((char*)data_mat, sizeof(double) * size_mat);
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
  }
  else {
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

void one_head_attention(string input_path, string output_path)
{
  double* query = new double[BATCH_SIZE * N * D];
  double* key = new double[BATCH_SIZE * N * D];
  double* value = new double[BATCH_SIZE * N * D];
  double* transpose_key = new double[BATCH_SIZE * D * N];
  double* matmul_query_key = new double[BATCH_SIZE * N * N];
  double* softmax_result = new double[BATCH_SIZE * N * N];
  double* matmul_softmax_value = new double[BATCH_SIZE * N * D];

  readData((input_path + "/query.bin").c_str(), query, BATCH_SIZE * N * D);
  readData((input_path + "/key.bin").c_str(), key, BATCH_SIZE * N * D);
  readData((input_path + "/value.bin").c_str(), value, BATCH_SIZE * N * D);

  std::cout << "read data done" << std::endl;

  // Load xclbin
  std::string xclbin_file = "overlay_hw.xclbin";
  std::cout << "Loading: " << xclbin_file << std::endl;
  xrt::device device = xrt::device(DEVICE_ID);
  xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
  std::cout << "Loaded xclbin: " << xclbin_file << std::endl;

  // create kernel object
  std::cout << "Created kernel: " << "transpose" << std::endl;
  xrt::kernel kernel_transpose = xrt::kernel(device, xclbin_uuid, "transpose");
  std::cout << "Created kernel: " << "matmul0" << std::endl;
  xrt::kernel kernel_matmul0 = xrt::kernel(device, xclbin_uuid, "matmul0");
  std::cout << "Created kernel: " << "matmul1" << std::endl;
  xrt::kernel kernel_matmul1 = xrt::kernel(device, xclbin_uuid, "matmul1");
  std::cout << "Created kernel: " << "softmax" << std::endl;
  xrt::kernel kernel_softmax = xrt::kernel(device, xclbin_uuid, "softmax");


  // create memory groups
  xrtMemoryGroup bank_grp_kernel_transpose_transpose_x = kernel_transpose.group_id(transpose_ptr_transpose_x);
  xrtMemoryGroup bank_grp_kernel_transpose_transpose_output = kernel_transpose.group_id(transpose_ptr_transpose_output);

  xrtMemoryGroup bank_grp_kernel_matmul0_matmul_x = kernel_matmul0.group_id(matmul0_ptr_matmul_x);
  xrtMemoryGroup bank_grp_kernel_matmul0_matmul_y = kernel_matmul0.group_id(matmul0_ptr_matmul_y);
  xrtMemoryGroup bank_grp_kernel_matmul0_matmul_output = kernel_matmul0.group_id(matmul0_ptr_matmul_output);

  xrtMemoryGroup bank_grp_kernel_matmul1_matmul_x = kernel_matmul1.group_id(matmul1_ptr_matmul_x);
  xrtMemoryGroup bank_grp_kernel_matmul1_matmul_y = kernel_matmul1.group_id(matmul1_ptr_matmul_y);
  xrtMemoryGroup bank_grp_kernel_matmul1_matmul_output = kernel_matmul1.group_id(matmul1_ptr_matmul_output);

  xrtMemoryGroup bank_grp_kernel_softmax_softmax_x = kernel_softmax.group_id(softmax_ptr_softmax_x);
  xrtMemoryGroup bank_grp_kernel_softmax_softmax_output = kernel_softmax.group_id(softmax_ptr_softmax_output);

  // create buffer objects
  xrt::bo data_buffer_kernel_transpose_transpose_x = \
    xrt::bo(device, BATCH_SIZE * INPUT_H * INPUT_W * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_transpose_transpose_x);
  xrt::bo data_buffer_kernel_transpose_transpose_output = \
    xrt::bo(device, BATCH_SIZE * INPUT_H * INPUT_W * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_transpose_transpose_output);

  xrt::bo data_buffer_kernel_matmul0_matmul_x = \
    xrt::bo(device, BATCH_SIZE * INPUT_H * INPUT_W * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_matmul0_matmul_x);
  xrt::bo data_buffer_kernel_matmul0_matmul_y = \
    xrt::bo(device, BATCH_SIZE * INPUT_W * OUTPUT_W * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_matmul0_matmul_y);
  xrt::bo data_buffer_kernel_matmul0_matmul_output = \
    xrt::bo(device, BATCH_SIZE * INPUT_H * OUTPUT_W * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_matmul0_matmul_output);

  xrt::bo data_buffer_kernel_matmul1_matmul_x = \
    xrt::bo(device, BATCH_SIZE * INPUT_H * OUTPUT_W * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_matmul1_matmul_x);
  xrt::bo data_buffer_kernel_matmul1_matmul_y = \
    xrt::bo(device, BATCH_SIZE * INPUT_W * OUTPUT_W * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_matmul1_matmul_y);
  xrt::bo data_buffer_kernel_matmul1_matmul_output = \
    xrt::bo(device, BATCH_SIZE * INPUT_H * INPUT_W * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_matmul1_matmul_output);

  xrt::bo data_buffer_kernel_softmax_softmax_x = \
    xrt::bo(device, BATCH_SIZE * INPUT_H * INPUT_H * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_softmax_softmax_x);
  xrt::bo data_buffer_kernel_softmax_softmax_output = \
    xrt::bo(device, BATCH_SIZE * INPUT_H * INPUT_H * sizeof(double), xrt::bo::flags::normal, bank_grp_kernel_softmax_softmax_output);


  std::cout << "Created buffer objects" << std::endl;

  // create kernel runner
  xrt::run run_kernel_transpose(kernel_transpose);
  xrt::run run_kernel_softmax(kernel_softmax);
  xrt::run run_kernel_matmul0(kernel_matmul0);
  xrt::run run_kernel_matmul1(kernel_matmul1);

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;
  // dot_add(data_a0, data_w0, data_b0, data_a1, L0_H1, L0_W1, L0_W1, L0_W2);

  // write data to buffer of dot_add kernel
  data_buffer_kernel_transpose_transpose_x.write(key);
  data_buffer_kernel_transpose_transpose_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_transpose_transpose_output.write(transpose_key);
  data_buffer_kernel_transpose_transpose_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);


  // set arguments of dot_add kernel
  run_kernel_transpose.set_arg(transpose_ptr_transpose_x, data_buffer_kernel_transpose_transpose_x);
  run_kernel_transpose.set_arg(transpose_ptr_transpose_output, data_buffer_kernel_transpose_transpose_output);
  run_kernel_transpose.set_arg(transpose_ptr_dim0, -2);
  run_kernel_transpose.set_arg(transpose_ptr_dim1, -1);

  std::cout << "Set arguments of transpose kernel" << std::endl;

  // run dot_add kernel
  run_kernel_transpose.start();
  run_kernel_transpose.wait();

  std::cout << "Run transpose kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_transpose_transpose_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_transpose_transpose_output.read(transpose_key);

  std::cout << "Read the result back" << std::endl;


  data_buffer_kernel_matmul0_matmul_x.write(query);
  data_buffer_kernel_matmul0_matmul_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_matmul0_matmul_y.write(transpose_key);
  data_buffer_kernel_matmul0_matmul_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_matmul0_matmul_output.write(matmul_query_key);
  data_buffer_kernel_matmul0_matmul_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of sigmoid kernel
  run_kernel_matmul0.set_arg(matmul0_ptr_matmul_x, data_buffer_kernel_matmul0_matmul_x);
  run_kernel_matmul0.set_arg(matmul0_ptr_matmul_y, data_buffer_kernel_matmul0_matmul_y);
  run_kernel_matmul0.set_arg(matmul0_ptr_matmul_output, data_buffer_kernel_matmul0_matmul_output);

  std::cout << "Set matmul of matmul0 kernel" << std::endl;

  // run sigmoid kernel
  run_kernel_matmul0.start();
  run_kernel_matmul0.wait();

  std::cout << "Run matmul kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_matmul0_matmul_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_matmul0_matmul_output.read(matmul_query_key);

  std::cout << "Read the result back" << std::endl;

  // dot_add(data_z1, data_w1, data_b1, data_a2, L1_H1, L1_W1, L1_W1, L1_W2);

  // write data to buffer of dot_add kernel
  data_buffer_kernel_softmax_softmax_x.write(matmul_query_key);
  data_buffer_kernel_softmax_softmax_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_softmax_softmax_output.write(softmax_result);
  data_buffer_kernel_softmax_softmax_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);


  // set arguments of dot_add kernel
  run_kernel_softmax.set_arg(softmax_ptr_softmax_x, data_buffer_kernel_softmax_softmax_x);
  run_kernel_softmax.set_arg(softmax_ptr_softmax_output, data_buffer_kernel_softmax_softmax_output);
  run_kernel_softmax.set_arg(softmax_ptr_axis, -1);


  std::cout << "Set arguments of softmx kernel" << std::endl;

  // run dot_add kernel
  run_kernel_softmax.start();
  run_kernel_softmax.wait();

  std::cout << "Run softmax kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_softmax_softmax_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_softmax_softmax_output.read(softmax_result);

  std::cout << "Read the result back" << std::endl;
  data_buffer_kernel_matmul1_matmul_x.write(softmax_result);
  data_buffer_kernel_matmul1_matmul_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_matmul1_matmul_y.write(value);
  data_buffer_kernel_matmul1_matmul_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_matmul1_matmul_output.write(matmul_softmax_value);
  data_buffer_kernel_matmul1_matmul_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of sigmoid kernel
  run_kernel_matmul1.set_arg(matmul1_ptr_matmul_x, data_buffer_kernel_matmul1_matmul_x);
  run_kernel_matmul1.set_arg(matmul1_ptr_matmul_y, data_buffer_kernel_matmul1_matmul_y);
  run_kernel_matmul1.set_arg(matmul1_ptr_matmul_output, data_buffer_kernel_matmul1_matmul_output);

  std::cout << "Set matmul of matmul1 kernel" << std::endl;

  // run sigmoid kernel
  run_kernel_matmul1.start();
  run_kernel_matmul1.wait();

  std::cout << "Run matmul1 kernel" << std::endl;

  // Read the result back
  data_buffer_kernel_matmul1_matmul_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_matmul1_matmul_output.read(matmul_softmax_value);

  std::cout << "1 warm up iteration done" << std::endl;

/*
  // Check results
  std::cout << "Check results ..." << std::endl;
  double* transpose_key_golden = new double[BATCH_SIZE * D * N];
  double* matmul_query_key_golden = new double[BATCH_SIZE * N * N];
  double* softmax_result_golden = new double[BATCH_SIZE * N * N];
  double* matmul_softmax_value_golden = new double[BATCH_SIZE * N * D];
  readData((output_path + "/transpose_key_golden.bin").c_str(), transpose_key_golden, BATCH_SIZE * D * N);
  readData((output_path + "/matmul_query_key_golden.bin").c_str(), matmul_query_key_golden, BATCH_SIZE * N * N);
  readData((output_path + "/softmax_result_golden.bin").c_str(), softmax_result_golden, BATCH_SIZE * N * N);
  readData((output_path + "/matmul_softmax_value_golden.bin").c_str(), matmul_softmax_value_golden, BATCH_SIZE * N * D);

  std::cout << "checking transpose_key ... ";
  checkResult(transpose_key, transpose_key_golden, BATCH_SIZE * D * N);
  std::cout << "checking matmul_query_key ... ";
  checkResult(matmul_query_key, matmul_query_key_golden, BATCH_SIZE * N * N);
  std::cout << "checking softmax_result ... ";
  checkResult(softmax_result, softmax_result_golden, BATCH_SIZE * N * N);
  std::cout << "checking matmul_softmax_value ... ";
  checkResult(matmul_softmax_value, matmul_softmax_value_golden, BATCH_SIZE * N * D);
*/

  // multi iterations
  int iterations = ITERATIONS;
  std::cout << "Running " << iterations << " iterations ..." << std::endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double transpose_time = 0;
  double matmul_query_key_time = 0;
  double softmax_time = 0;
  double matmul_softmax_value_time = 0;

  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    // write data to buffer of dot_add kernel
    data_buffer_kernel_transpose_transpose_x.write(key);
    data_buffer_kernel_transpose_transpose_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_transpose_transpose_output.write(transpose_key);
    data_buffer_kernel_transpose_transpose_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);


    // set arguments of dot_add kernel
    run_kernel_transpose.set_arg(transpose_ptr_transpose_x, data_buffer_kernel_transpose_transpose_x);
    run_kernel_transpose.set_arg(transpose_ptr_transpose_output, data_buffer_kernel_transpose_transpose_output);
    run_kernel_transpose.set_arg(transpose_ptr_dim0, -2);
    run_kernel_transpose.set_arg(transpose_ptr_dim1, -1);

    std::cout << "Set arguments of transpose kernel" << std::endl;

    // run dot_add kernel
    run_kernel_transpose.start();
    run_kernel_transpose.wait();

    std::cout << "Run transpose kernel" << std::endl;

    // Read the result back
    data_buffer_kernel_transpose_transpose_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_transpose_transpose_output.read(transpose_key);

    std::cout << "Read the result back" << std::endl;
    transpose_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    data_buffer_kernel_matmul0_matmul_x.write(query);
    data_buffer_kernel_matmul0_matmul_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_matmul0_matmul_y.write(transpose_key);
    data_buffer_kernel_matmul0_matmul_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_matmul0_matmul_output.write(matmul_query_key);
    data_buffer_kernel_matmul0_matmul_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of sigmoid kernel
    run_kernel_matmul0.set_arg(matmul0_ptr_matmul_x, data_buffer_kernel_matmul0_matmul_x);
    run_kernel_matmul0.set_arg(matmul0_ptr_matmul_y, data_buffer_kernel_matmul0_matmul_y);
    run_kernel_matmul0.set_arg(matmul0_ptr_matmul_output, data_buffer_kernel_matmul0_matmul_output);

    std::cout << "Set matmul of matmul0 kernel" << std::endl;

    // run sigmoid kernel
    run_kernel_matmul0.start();
    run_kernel_matmul0.wait();

    std::cout << "Run matmul kernel" << std::endl;

    // Read the result back
    data_buffer_kernel_matmul0_matmul_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_matmul0_matmul_output.read(matmul_query_key);

    std::cout << "Read the result back" << std::endl;
    matmul_query_key_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    data_buffer_kernel_softmax_softmax_x.write(matmul_query_key);
    data_buffer_kernel_softmax_softmax_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_softmax_softmax_output.write(softmax_result);
    data_buffer_kernel_softmax_softmax_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);


    // set arguments of dot_add kernel
    run_kernel_softmax.set_arg(softmax_ptr_softmax_x, data_buffer_kernel_softmax_softmax_x);
    run_kernel_softmax.set_arg(softmax_ptr_softmax_output, data_buffer_kernel_softmax_softmax_output);
    run_kernel_softmax.set_arg(softmax_ptr_axis, -1);


    std::cout << "Set arguments of softmx kernel" << std::endl;

    // run dot_add kernel
    run_kernel_softmax.start();
    run_kernel_softmax.wait();

    std::cout << "Run softmax kernel" << std::endl;

    // Read the result back
    data_buffer_kernel_softmax_softmax_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_softmax_softmax_output.read(softmax_result);

    std::cout << "Read the result back" << std::endl;
    softmax_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    data_buffer_kernel_matmul1_matmul_x.write(softmax_result);
    data_buffer_kernel_matmul1_matmul_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_matmul1_matmul_y.write(value);
    data_buffer_kernel_matmul1_matmul_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_matmul1_matmul_output.write(matmul_softmax_value);
    data_buffer_kernel_matmul1_matmul_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of sigmoid kernel
    run_kernel_matmul1.set_arg(matmul1_ptr_matmul_x, data_buffer_kernel_matmul1_matmul_x);
    run_kernel_matmul1.set_arg(matmul1_ptr_matmul_y, data_buffer_kernel_matmul1_matmul_y);
    run_kernel_matmul1.set_arg(matmul1_ptr_matmul_output, data_buffer_kernel_matmul1_matmul_output);

    std::cout << "Set matmul of matmul1 kernel" << std::endl;

    // run sigmoid kernel
    run_kernel_matmul1.start();
    run_kernel_matmul1.wait();

    std::cout << "Run matmul1 kernel" << std::endl;

    // Read the result back
    data_buffer_kernel_matmul1_matmul_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_matmul1_matmul_output.read(matmul_softmax_value);

    matmul_softmax_value_time += omp_get_wtime() - start_iteration_time;

  }
  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations << " iterations " << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "transpose time: " << (transpose_time / iterations) * 1000 << " ms" << endl;
  cout << "matmul_1 time: " << (matmul_query_key_time / iterations) * 1000 << " ms" << endl;
  cout << "softmax time: " << (softmax_time / iterations) * 1000 << " ms" << endl;
  cout << "matmul_2 time: " << (matmul_softmax_value_time / iterations) * 1000 << " ms" << endl;

  delete[] query;
  delete[] key;
  delete[] value;
  delete[] transpose_key;
  delete[] matmul_query_key;
  delete[] softmax_result;
  delete[] matmul_softmax_value;
  delete[] transpose_key_golden;
  delete[] matmul_query_key_golden;
  delete[] softmax_result_golden;
  delete[] matmul_softmax_value_golden;

  return;

}

int main(int argc, char* argv[])
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running one_head_attention benchmark benchmark C++ HLS" << std::endl;
  std::cout << "=======================================" << std::endl;

  string input_path;
  string output_path;
  if (argc == 3) {
    input_path = argv[1];
    output_path = argv[2];
  }
  else {
    printf("Usage: ./one_head_attention_sw <input_path> <output_path>\n");
    exit(-1);
  }
  one_head_attention(input_path, output_path);
  return 0;
}