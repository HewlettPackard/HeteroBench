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
 
#include "cuda_impl.h"
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

void mlp_forward(string input_path, string output_path)
{
  double *data_a0 = new double [SIZE_A0];
  double *data_a1 = new double [SIZE_A1];
  double *data_z1 = new double [SIZE_Z1];
  double *data_a2 = new double [SIZE_A2];
  double *data_z2 = new double [SIZE_Z2];
  double *data_a3 = new double [SIZE_A3];
  double *data_z3 = new double [SIZE_Z3];
  double *data_a4 = new double [SIZE_A4];
  double *data_a4_exp = new double [SIZE_A4];
  double *data_z4 = new double [SIZE_Z4];

  double *data_w0 = new double [SIZE_W0];
  double *data_w1 = new double [SIZE_W1];
  double *data_w2 = new double [SIZE_W2];
  double *data_w3 = new double [SIZE_W3];

  double *data_b0 = new double [SIZE_B0];
  double *data_b1 = new double [SIZE_B1];
  double *data_b2 = new double [SIZE_B2];
  double *data_b3 = new double [SIZE_B3];

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

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;
  dot_add(data_a0, data_w0, data_b0, data_a1, L0_H1, L0_W1, L0_W1, L0_W2);
  #pragma omp target enter data map(to: data_a1[0:SIZE_A1])
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < SIZE_A1; i++) {
    data_a1[i] = data_a1[i] / 500;
  }
  #pragma omp target exit data map(from: data_a1[0:SIZE_A1])
  sigmoid(data_a1, data_z1, SIZE_Z1);

  dot_add(data_z1, data_w1, data_b1, data_a2, L1_H1, L1_W1, L1_W1, L1_W2);
  #pragma omp target enter data map(to: data_a2[0:SIZE_A2])
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < SIZE_A2; i++) {
    data_a2[i] = data_a2[i] / 1500;
  }
  #pragma omp target exit data map(from: data_a2[0:SIZE_A2])
  sigmoid(data_a2, data_z2, SIZE_Z2);

  dot_add(data_z2, data_w2, data_b2, data_a3, L2_H1, L2_W1, L2_W1, L2_W2);
  #pragma omp target enter data map(to: data_a3[0:SIZE_A3])
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < SIZE_A3; i++) {
    data_a3[i] = data_a3[i] / 1500;
  }
  #pragma omp target exit data map(from: data_a3[0:SIZE_A3])
  sigmoid(data_a3, data_z3, SIZE_Z3);

  dot_add(data_z3, data_w3, data_b3, data_a4, L3_H1, L3_W1, L3_W1, L3_W2);
  #pragma omp target enter data map(to: data_a4[0:SIZE_A4])
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < SIZE_A4; i++) {
    data_a4[i] = data_a4[i] / 1500;
  }
  #pragma omp target exit data map(from: data_a4[0:SIZE_A4])
  softmax(data_a4, data_a4_exp, data_z4, SIZE_Z4);
  #pragma omp target enter data map(to: data_a4[0:SIZE_A4])
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < SIZE_A4; i++) {
    data_z4[i] = data_z4[i] * 1000000;
  }
  #pragma omp target exit data map(from: data_a4[0:SIZE_A4])

  std::cout << "Done" << std::endl;

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
  double start_sigmoid;
  double start_softmax;

  double start_iteration_time;
  double layer_0_time = 0;
  double layer_1_time = 0;
  double layer_2_time = 0;
  double layer_3_time = 0;

  double dot_add_1_time = 0;
  double dot_add_2_time = 0;
  double dot_add_3_time = 0;
  double dot_add_4_time = 0;

  double sigmoid_1_time = 0;
  double sigmoid_2_time = 0;
  double sigmoid_3_time = 0;

  double softmax_time = 0;

  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    dot_add(data_a0, data_w0, data_b0, data_a1, L0_H1, L0_W1, L0_W1, L0_W2);
    dot_add_1_time += omp_get_wtime() - start_iteration_time;
    #pragma omp target enter data map(to: data_a1[0:SIZE_A1])
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < SIZE_A1; i++) {
      data_a1[i] = data_a1[i] / 500;
    }
    #pragma omp target exit data map(from: data_a1[0:SIZE_A1])
    start_sigmoid = omp_get_wtime();
    sigmoid(data_a1, data_z1, SIZE_Z1);
    sigmoid_1_time += omp_get_wtime() - start_sigmoid;
    layer_0_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    dot_add(data_z1, data_w1, data_b1, data_a2, L1_H1, L1_W1, L1_W1, L1_W2);
    dot_add_2_time += omp_get_wtime() - start_iteration_time;
    #pragma omp target enter data map(to: data_a2[0:SIZE_A2])
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < SIZE_A2; i++) {
      data_a2[i] = data_a2[i] / 1500;
    }
    #pragma omp target exit data map(from: data_a2[0:SIZE_A2])
    start_sigmoid = omp_get_wtime();
    sigmoid(data_a2, data_z2, SIZE_Z2);
    sigmoid_2_time += omp_get_wtime() - start_sigmoid;
    layer_1_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    dot_add(data_z2, data_w2, data_b2, data_a3, L2_H1, L2_W1, L2_W1, L2_W2);
    dot_add_3_time += omp_get_wtime() - start_iteration_time;
    #pragma omp target enter data map(to: data_a3[0:SIZE_A3])
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < SIZE_A3; i++) {
      data_a3[i] = data_a3[i] / 1500;
    }
    #pragma omp target exit data map(from: data_a3[0:SIZE_A3])
    start_sigmoid = omp_get_wtime();
    sigmoid(data_a3, data_z3, SIZE_Z3);
    sigmoid_3_time += omp_get_wtime() - start_sigmoid;
    layer_2_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    dot_add(data_z3, data_w3, data_b3, data_a4, L3_H1, L3_W1, L3_W1, L3_W2);
    dot_add_4_time += omp_get_wtime() - start_iteration_time;
    #pragma omp target enter data map(to: data_a4[0:SIZE_A4])
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < SIZE_A4; i++) {
      data_a4[i] = data_a4[i] / 1500;
    }
    #pragma omp target exit data map(from: data_a4[0:SIZE_A4])
    start_softmax = omp_get_wtime();
    softmax(data_a4, data_a4_exp, data_z4, SIZE_Z4);
    softmax_time += omp_get_wtime() - start_softmax;
    #pragma omp target enter data map(to: data_a4[0:SIZE_A4])
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < SIZE_A4; i++) {
      data_z4[i] = data_z4[i] * 1000000;
    }
    #pragma omp target exit data map(from: data_a4[0:SIZE_A4])
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

  cout << "Dot add 1 time: " << (dot_add_1_time / iterations) * 1000 << " ms" << endl;
  cout << "Dot add 2 time: " << (dot_add_2_time / iterations) * 1000 << " ms" << endl;
  cout << "Dot add 3 time: " << (dot_add_3_time / iterations) * 1000 << " ms" << endl;
  cout << "Dot add 4 time: " << (dot_add_4_time / iterations) * 1000 << " ms" << endl;

  cout << "Sigmoid 1 time: " << (sigmoid_1_time / iterations) * 1000 << " ms" << endl;
  cout << "Sigmoid 2 time: " << (sigmoid_2_time / iterations) * 1000 << " ms" << endl;
  cout << "Sigmoid 3 time: " << (sigmoid_3_time / iterations) * 1000 << " ms" << endl;

  cout << "Softmax time: " << (softmax_time / iterations) * 1000 << " ms" << endl;

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
  std::cout << "Running mlp benchmark C++ CUDA GPU" << std::endl;
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