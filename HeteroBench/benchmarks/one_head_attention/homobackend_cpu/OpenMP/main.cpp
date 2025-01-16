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
 
#include "cpu_impl.h"
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

void one_head_attention(string input_path, string output_path)
{
  double *query = new double [BATCH_SIZE * N * D];
  double *key = new double [BATCH_SIZE * N * D];
  double *value = new double [BATCH_SIZE * N * D];
  double *transpose_key = new double [BATCH_SIZE * D * N];
  double *matmul_query_key = new double [BATCH_SIZE * N * N];
  double *softmax_result = new double [BATCH_SIZE * N * N];
  double *matmul_softmax_value = new double [BATCH_SIZE * N * D];
  
  readData((input_path + "/query.bin").c_str(), query, BATCH_SIZE * N * D);
  readData((input_path + "/key.bin").c_str(), key, BATCH_SIZE * N * D);
  readData((input_path + "/value.bin").c_str(), value, BATCH_SIZE * N * D);

  std::cout << "read data done" << std::endl;

  // // init matmul_query_key to 0
  // for (int i = 0; i < BATCH_SIZE * N * N; i++) {
  //   matmul_query_key[i] = 0;
  // }

  // // init matmul_softmax_value to 0
  // for (int i = 0; i < BATCH_SIZE * N * D; i++) {
  //   matmul_softmax_value[i] = 0;
  // }

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;
  transpose(key, transpose_key, BATCH_SIZE, N, D, -2, -1);
  matmul(query, transpose_key, matmul_query_key, BATCH_SIZE, N, D, N);
  softmax(matmul_query_key, softmax_result, BATCH_SIZE, N, N, -1);
  matmul(softmax_result, value, matmul_softmax_value, BATCH_SIZE, N, N, D);
  std::cout << "Done" << std::endl;

  // Check results
  std::cout << "Check results ..." << std::endl;
  double *transpose_key_golden = new double [BATCH_SIZE * D * N];
  double *matmul_query_key_golden = new double [BATCH_SIZE * N * N];
  double *softmax_result_golden = new double [BATCH_SIZE * N * N];
  double *matmul_softmax_value_golden = new double [BATCH_SIZE * N * D];
/*
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
    transpose(key, transpose_key, BATCH_SIZE, N, D, -2, -1);
    transpose_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    matmul(query, transpose_key, matmul_query_key, BATCH_SIZE, N, D, N);
    matmul_query_key_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    softmax(matmul_query_key, softmax_result, BATCH_SIZE, N, N, -1);
    softmax_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    matmul(softmax_result, value, matmul_softmax_value, BATCH_SIZE, N, N, D);
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

int main(int argc, char *argv[])
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running one_head_attention benchmark C++ OpenMP" << std::endl;
  std::cout << "=======================================" << std::endl;
  
  string input_path;
  string output_path;
  
  if (argc == 3) {
    input_path = argv[1];
    output_path = argv[2];
  } else {
    printf("Usage: ./oha_sw <input_path> <output_path>\n");
    exit(-1);
  }
  one_head_attention(input_path, output_path);
  return 0;
}