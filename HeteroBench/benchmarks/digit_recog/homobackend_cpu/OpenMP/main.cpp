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

void DigitRec_sw(const DigitType training_set[NUM_TRAINING * DIGIT_WIDTH], 
                 const DigitType test_set[NUM_TEST * DIGIT_WIDTH], 
                 LabelType results[NUM_TEST]) 
{
  // multi iterations
  std::cout << "Running " << NUM_TEST << " iterations ..." << std::endl;

  double start_whole_time = omp_get_wtime();

  double start_iteration_time;
  double start_update_knn_time = 0;
  double start_knn_vote_time = 0;

  // loop through test set
  for (int t = 0; t < NUM_TEST; ++t) 
  {
    // nearest neighbor set
    int dists[K_CONST];
    int labels[K_CONST];
    // Initialize the neighbor set
    for ( int i = 0; i < K_CONST; ++i ) 
    {
      // Note that the max distance is 256
      dists[i] = 256;
      labels[i] = 0;
    }

    // for each training instance, compare it with the test instance, and update the nearest neighbor set
    start_iteration_time = omp_get_wtime();
    update_knn(training_set, &test_set[t * DIGIT_WIDTH], dists, labels);
    start_update_knn_time += omp_get_wtime() - start_iteration_time;

    // Compute the final output
    start_iteration_time = omp_get_wtime();
    LabelType max_label = 0;
    knn_vote(labels, &max_label);
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

int main(int argc, char ** argv) 
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running digit_recog benchmark C++ OpenMP" << std::endl;
  std::cout << "=======================================" << std::endl;


  // sw version host code
  // create space for the result
  LabelType* result = new LabelType[NUM_TEST];

  // software version
  DigitRec_sw(training_data, testing_data, result);

/*
  // check results
  std::cout << "Checking results ..." << std::endl;
  check_results( result, expected, NUM_TEST );
  std::cout << "Done" << std::endl;
*/

  delete []result;

  return EXIT_SUCCESS;

}
