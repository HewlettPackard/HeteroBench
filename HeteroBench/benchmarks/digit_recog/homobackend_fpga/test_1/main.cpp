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

#define DigitRec_hw_ptr_training_set 0
#define DigitRec_hw_ptr_test_set 1
#define DigitRec_hw_ptr_results 2

void check_results(LabelType* result, const LabelType* expected, int cnt)
{
  int correct_cnt = 0;

  std::ofstream ofile;
  ofile.open("outputs.txt");
  if (ofile.is_open())
  {
    for (int i = 0; i < cnt; i ++ )
    {
      if (result[i] != expected[i])
        ofile << "Test " << i << ": expected = " << int(expected[i]) << ", result = " << int(result[i]) << std::endl;
      else
        correct_cnt ++;
    }

    ofile << "\n\t " << correct_cnt << " / " << cnt << " correct!" << std::endl;
    ofile.close();
  }
  else
  {
    std::cout << "Failed to create output file!" << std::endl;
  }
      

}

const DigitType testing_data[NUM_TEST * DIGIT_WIDTH] = {
  #include "../../196data/test_set.dat"
};

const LabelType expected[NUM_TEST] = {
  #include "../../196data/expected.dat"
};

const DigitType training_data[NUM_TRAINING * DIGIT_WIDTH] = {
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

int main(int argc, char ** argv) 
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running digit_recog benchmark C++ HLS" << std::endl;
  std::cout << "=======================================" << std::endl;


  // hw version host code
  // create space for the result
  LabelType result[NUM_TEST] = { 0 };
  WholeDigitType training_in[NUM_TRAINING] = { 0 };
  WholeDigitType test_in[NUM_TEST] = { 0 };

  // pack the data into a wide datatype
  for (int i = 0; i < NUM_TRAINING; i++)
  {
    training_in[i].range(63, 0) = training_data[i * DIGIT_WIDTH + 0];
    training_in[i].range(127, 64) = training_data[i * DIGIT_WIDTH + 1];
    training_in[i].range(191, 128) = training_data[i * DIGIT_WIDTH + 2];
    training_in[i].range(255, 192) = training_data[i * DIGIT_WIDTH + 3];
  }
  for (int i = 0; i < NUM_TEST; i++)
  {
    test_in[i].range(63, 0) = testing_data[i * DIGIT_WIDTH + 0];
    test_in[i].range(127, 64) = testing_data[i * DIGIT_WIDTH + 1];
    test_in[i].range(191, 128) = testing_data[i * DIGIT_WIDTH + 2];
    test_in[i].range(255, 192) = testing_data[i * DIGIT_WIDTH + 3];
  }


  // Load xclbin
  std::string xclbin_file = "overlay_hw.xclbin";
  std::cout << "Loading: " << xclbin_file << std::endl;
  xrt::device device = xrt::device(DEVICE_ID);
  xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
  std::cout << "Loaded xclbin: " << xclbin_file << std::endl;

  // create kernel object
  xrt::kernel DigitRec_hw_kernel = xrt::kernel(device, xclbin_uuid, "DigitRec_hw");

  // create memory groups
  xrtMemoryGroup bank_grp_DigitRec_hw_training_set = DigitRec_hw_kernel.group_id(DigitRec_hw_ptr_training_set);
  xrtMemoryGroup bank_grp_DigitRec_hw_test_set = DigitRec_hw_kernel.group_id(DigitRec_hw_ptr_test_set);
  xrtMemoryGroup bank_grp_DigitRec_hw_results = DigitRec_hw_kernel.group_id(DigitRec_hw_ptr_results);

  // create buffer objects
  xrt::bo data_buffer_DigitRec_hw_training_set = xrt::bo(device, sizeof(training_data), xrt::bo::flags::normal, bank_grp_DigitRec_hw_training_set);
  xrt::bo data_buffer_DigitRec_hw_test_set = xrt::bo(device, sizeof(testing_data), xrt::bo::flags::normal, bank_grp_DigitRec_hw_test_set);
  xrt::bo data_buffer_DigitRec_hw_results = xrt::bo(device, sizeof(result), xrt::bo::flags::normal, bank_grp_DigitRec_hw_results);

  // create kernel runner
  xrt::run run_DigitRec_hw(DigitRec_hw_kernel);

  // multi iterations
  std::cout << "Running " << NUM_TEST << " iterations ..." << std::endl;

  double start_whole_time = omp_get_wtime();

  // write data to buffer objects
  data_buffer_DigitRec_hw_training_set.write(training_data);
  data_buffer_DigitRec_hw_training_set.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_DigitRec_hw_test_set.write(testing_data);
  data_buffer_DigitRec_hw_test_set.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set kernel arguments
  run_DigitRec_hw.set_arg(DigitRec_hw_ptr_training_set, data_buffer_DigitRec_hw_training_set);
  run_DigitRec_hw.set_arg(DigitRec_hw_ptr_test_set, data_buffer_DigitRec_hw_test_set);
  run_DigitRec_hw.set_arg(DigitRec_hw_ptr_results, data_buffer_DigitRec_hw_results);

  // run kernel
  run_DigitRec_hw.start();
  run_DigitRec_hw.wait();

  // read data from buffer objects
  data_buffer_DigitRec_hw_results.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_DigitRec_hw_results.read(result);

  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "Total " << NUM_TEST << " iterations " << endl;
  cout << "Single iteration time: " << (run_whole_time / NUM_TEST) * 1000 << " ms" << endl;

/*
  // check results
  std::cout << "Checking results ..." << std::endl;
  check_results( result, expected, NUM_TEST );
  std::cout << "Done" << std::endl;
*/

  delete []result;

  return EXIT_SUCCESS;

}
