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
 
// standard C/C++ headers
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <cmath>
#include <time.h>
#include <fstream>
#include <sys/time.h>
#include "omp.h"

// fpga related headers
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

// other headers
#include "fpga_impl.h"

using namespace std;

#define DEVICE_ID 0

#define update_knn_ptr_training_set 0
#define update_knn_ptr_test_set 1
#define update_knn_ptr_dists 2
#define update_knn_ptr_labels 3
#define update_knn_ptr_label 4

#define knn_vote_ptr_labels 0
#define knn_vote_ptr_max_label 1

void check_results(LabelType* result, const LabelType* expected, int cnt)
{
  int correct_cnt = 0;

  std::ofstream ofile;
  ofile.open("outputs.txt");
  if (ofile.is_open())
  {
    for (int i = 0; i < cnt; i++)
    {
      if (result[i] != expected[i])
        ofile << "Test " << i << ": expected = " << int(expected[i]) << ", result = " << int(result[i]) << std::endl;
      else
        correct_cnt++;
    }

    ofile << "\n\t " << correct_cnt << " / " << cnt << " correct!" << std::endl;
    ofile.close();
  }
  else
  {
    std::cout << "Failed to create output file!" << std::endl;
  }


}

DigitType testing_data[NUM_TEST * DIGIT_WIDTH] = {
  #include "../../196data/test_set.dat"
};

LabelType expected[NUM_TEST] = {
  #include "../../196data/expected.dat"
};

DigitType training_data[NUM_TRAINING * DIGIT_WIDTH] = {
  #include "../../196data/training_set_0.dat" 
  #include "../../196data/training_set_1.dat" 
  #include "../../196data/training_set_2.dat" 
  #include "../../196data/training_set_3.dat" 
  #include "../../196data/training_set_4.dat" 
  #include "../../196data/training_set_5.dat" 
  #include "../../196data/training_set_6.dat" 
  #include "../../196data/training_set_7.dat" 
  #include "../../196data/training_set_8.dat" 
  #include "../../196data/training_set_9.dat"
};

void DigitRec_sw(DigitType training_set[NUM_TRAINING * DIGIT_WIDTH],
  DigitType test_set[NUM_TEST * DIGIT_WIDTH],
  LabelType results[NUM_TEST])
{

  // nearest neighbor set
  int dists[K_CONST];
  int labels[K_CONST];
  // Load xclbin
  std::string xclbin_file = "overlay_hw.xclbin";
  std::cout << "Loading: " << xclbin_file << std::endl;
  xrt::device device = xrt::device(DEVICE_ID);
  xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
  std::cout << "Loaded xclbin: " << xclbin_file << std::endl;

  // create kernel object
  xrt::kernel update_knn_kernel = xrt::kernel(device, xclbin_uuid, "update_knn");
  xrt::kernel knn_vote_kernel = xrt::kernel(device, xclbin_uuid, "knn_vote");

  // create memory groups
  xrtMemoryGroup bank_grp_update_knn_training_set = update_knn_kernel.group_id(update_knn_ptr_training_set);
  xrtMemoryGroup bank_grp_update_knn_test_set = update_knn_kernel.group_id(update_knn_ptr_test_set);
  xrtMemoryGroup bank_grp_update_knn_dists = update_knn_kernel.group_id(update_knn_ptr_dists);
  xrtMemoryGroup bank_grp_update_knn_labels = update_knn_kernel.group_id(update_knn_ptr_labels);

  xrtMemoryGroup bank_grp_knn_vote_labels = knn_vote_kernel.group_id(knn_vote_ptr_labels);
  xrtMemoryGroup bank_grp_knn_vote_max_label = knn_vote_kernel.group_id(knn_vote_ptr_max_label);

  // create buffer objects
  xrt::bo data_buffer_update_knn_training_set = xrt::bo(device, DIGIT_WIDTH * sizeof(DigitType), xrt::bo::flags::normal, bank_grp_update_knn_training_set);
  xrt::bo data_buffer_update_knn_test_set = xrt::bo(device, DIGIT_WIDTH * sizeof(DigitType), xrt::bo::flags::normal, bank_grp_update_knn_test_set);
  xrt::bo data_buffer_update_knn_dists = xrt::bo(device, K_CONST * sizeof(int), xrt::bo::flags::normal, bank_grp_update_knn_dists);
  xrt::bo data_buffer_update_knn_labels = xrt::bo(device, K_CONST * sizeof(int), xrt::bo::flags::normal, bank_grp_update_knn_labels);

  xrt::bo data_buffer_knn_vote_labels = xrt::bo(device, K_CONST * sizeof(int), xrt::bo::flags::normal, bank_grp_knn_vote_labels);
  xrt::bo data_buffer_knn_vote_max_label = xrt::bo(device, sizeof(LabelType), xrt::bo::flags::normal, bank_grp_knn_vote_max_label);

  // create kernel runner
  xrt::run run_update_knn(update_knn_kernel);
  xrt::run run_knn_vote(knn_vote_kernel);


  double start_whole_time = omp_get_wtime();

  double start_iteration_time;
  double start_update_knn_time = 0;
  double start_knn_vote_time = 0;
  // multi iterations
  std::cout << "Running " << NUM_TEST << " iterations ..." << std::endl;
  // loop through test set
  for (int t = 0; t < NUM_TEST; ++t)
  {
    // Initialize the neighbor set
    for (int i = 0; i < K_CONST; ++i)
    {
      // Note that the max distance is 256
      dists[i] = 256;
      labels[i] = 0;
    }
    // for each training instance, compare it with the test instance, and update the nearest neighbor set
    start_iteration_time = omp_get_wtime();
    for (int i = 0; i < NUM_TRAINING; ++i)
    {
      int label = i / CLASS_SIZE;

      DigitType* base_ptr1 = test_set;
      DigitType* base_ptr0 = training_set;
      DigitType* element_ptr0 = &training_set[i * DIGIT_WIDTH];
      DigitType* element_ptr1 = &test_set[t * DIGIT_WIDTH];



      data_buffer_update_knn_training_set.write(element_ptr0);
      data_buffer_update_knn_training_set.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      data_buffer_update_knn_test_set.write(element_ptr1);
      data_buffer_update_knn_test_set.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      data_buffer_update_knn_dists.write(dists);
      data_buffer_update_knn_dists.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      data_buffer_update_knn_labels.write(labels);
      data_buffer_update_knn_labels.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      // set kernel arguments
      run_update_knn.set_arg(update_knn_ptr_training_set, data_buffer_update_knn_training_set);
      run_update_knn.set_arg(update_knn_ptr_test_set, data_buffer_update_knn_test_set);
      run_update_knn.set_arg(update_knn_ptr_dists, data_buffer_update_knn_dists);
      run_update_knn.set_arg(update_knn_ptr_labels, data_buffer_update_knn_labels);
      run_update_knn.set_arg(update_knn_ptr_label, label);

      // run kernel
      run_update_knn.start();
      run_update_knn.wait();

      // read data from buffer objects
      data_buffer_update_knn_dists.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      data_buffer_update_knn_dists.read(dists);
      data_buffer_update_knn_labels.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      data_buffer_update_knn_labels.read(labels);
 
    }
    start_update_knn_time += omp_get_wtime() - start_iteration_time;

    // Compute the final output
    start_iteration_time = omp_get_wtime();
    LabelType max_label = 0;
    // knn_vote(labels, &max_label);
    // write data to buffer objects
    data_buffer_knn_vote_labels.write(labels);
    data_buffer_knn_vote_labels.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_knn_vote_max_label.write(&max_label);
    data_buffer_knn_vote_max_label.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set kernel arguments
    run_knn_vote.set_arg(knn_vote_ptr_labels, data_buffer_knn_vote_labels);
    run_knn_vote.set_arg(knn_vote_ptr_max_label, data_buffer_knn_vote_max_label);

    // run kernel
    run_knn_vote.start();
    run_knn_vote.wait();

    // read data from buffer objects
    data_buffer_knn_vote_max_label.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_knn_vote_max_label.read(&max_label);

    results[t] = max_label;
    start_knn_vote_time += omp_get_wtime() - start_iteration_time;
  }  
  
  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "Total " << NUM_TEST << " iterations " << endl;
  cout << "Single iteration time: " << (run_whole_time / NUM_TEST) * 1000 << " ms" << endl;
  cout << "Update knn time: " << (start_update_knn_time / NUM_TEST) * 1000 << " ms" << endl;
  cout << "Knn vote time: " << (start_knn_vote_time / NUM_TEST) * 1000 << " ms" << endl;

}

int main(int argc, char** argv)
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running digit_recog benchmark C++ HLS (no optimization)" << std::endl;
  std::cout << "=======================================" << std::endl;

  // sw version host code
  // create space for the result
  LabelType* result = new LabelType[NUM_TEST];

  // software version
  DigitRec_sw(training_data, testing_data, result);

/*
  // check results
  printf("Checking results:\n");
  check_results(result, expected, NUM_TEST);
  printf("Done\n");
*/

  delete[]result;

  return EXIT_SUCCESS;

}
