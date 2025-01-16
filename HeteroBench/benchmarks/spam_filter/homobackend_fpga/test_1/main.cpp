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

#define dotProduct_ptr_param 0
#define dotProduct_ptr_feature 1
#define dotProduct_ptr_result 2

#define Sigmoid_ptr_exponent 0
#define Sigmoid_ptr_result 1

#define computeGradient_ptr_grad 0
#define computeGradient_ptr_feature 1
#define computeGradient_ptr_scale 2

#define updateParameter_ptr_param 0
#define updateParameter_ptr_grad 1
#define updateParameter_ptr_step_size 2

void print_usage(char* filename)
{
    printf("usage: %s <options>\n", filename);
    printf("  -f [kernel file]\n");
    printf("  -p [path to data]\n");
}

void parse_sdaccel_command_line_args(
    int argc,
    char** argv,
    std::string& kernelFile,
    std::string& path_to_data) 
{

  int c = 0;

  while ((c = getopt(argc, argv, "f:p:")) != -1) 
  {
    switch (c) 
    {
      case 'f':
        kernelFile = optarg;
        break;
      case 'p':
        path_to_data = optarg;
        break;
      default:
      {
        print_usage(argv[0]);
        exit(-1);
      }
    } // matching on arguments
  } // while args present
}

void parse_sdsoc_command_line_args(
    int argc,
    char** argv,
    std::string& path_to_data) 
{

  int c = 0;

  while ((c = getopt(argc, argv, "f:p:")) != -1) 
  {
    switch (c) 
    {
      case 'p':
        path_to_data = optarg;
        break;
      default:
      {
        print_usage(argv[0]);
        exit(-1);
      }
    } // matching on arguments
  } // while args present
}

// data structure only used in this file
typedef struct DataSet_s 
{
  DataType*    data_points;
  LabelType*   labels;
  FeatureType* parameter_vector;
  size_t num_data_points;
  size_t num_features;
} DataSet;


// sub-functions for result checking
// dot product
float dotProduct_host(FeatureType* param_vector, DataType* data_point_i, const size_t num_features)
{
  FeatureType result = 0.0f;

  for (int i = 0; i < num_features; i ++ )
    result += param_vector[i] * data_point_i[i];

  return result;
}

// predict
LabelType getPrediction(FeatureType* parameter_vector, DataType* data_point_i, size_t num_features, const double treshold = 0) 
{
  float parameter_vector_dot_x_i = dotProduct_host(parameter_vector, data_point_i, num_features);
  return (parameter_vector_dot_x_i > treshold) ? 1 : 0;
}

// compute error rate
double computeErrorRate(
    DataSet data_set,
    double& cumulative_true_positive_rate,
    double& cumulative_false_positive_rate,
    double& cumulative_error)
{

  size_t true_positives = 0, true_negatives = 0, false_positives = 0, false_negatives = 0;

  for (size_t i = 0; i < data_set.num_data_points; i++) 
  {
    LabelType prediction = getPrediction(data_set.parameter_vector, &data_set.data_points[i * data_set.num_features], data_set.num_features);
    if (prediction != data_set.labels[i])
    {
      if (prediction == 1)
        false_positives++;
      else
        false_negatives++;
    } 
    else 
    {
      if (prediction == 1)
        true_positives++;
      else
        true_negatives++;
    }
  }

  double error_rate = (double)(false_positives + false_negatives) / data_set.num_data_points;

  cumulative_true_positive_rate += (double)true_positives / (true_positives + false_negatives);
  cumulative_false_positive_rate += (double)false_positives / (false_positives + true_negatives);
  cumulative_error += error_rate;

  return error_rate;
}

// check results
void check_results(FeatureType* param_vector, DataType* data_points, LabelType* labels)
{
  std::ofstream ofile;
  ofile.open("output.txt");
  if (ofile.is_open())
  {
    ofile << "\nmain parameter vector: \n";
    for(int i = 0; i < 30; i ++ )
      ofile << "m[" << i << "]: " << param_vector[i] << " | ";
    ofile << std::endl;

    // Initialize benchmark variables
    double training_tpr = 0.0;
    double training_fpr = 0.0;
    double training_error = 0.0;
    double testing_tpr = 0.0;
    double testing_fpr = 0.0;
    double testing_error = 0.0;

    // Get Training error
    DataSet training_set;
    training_set.data_points = data_points;
    training_set.labels = labels;
    training_set.num_data_points = NUM_TRAINING;
    training_set.num_features = NUM_FEATURES;
    training_set.parameter_vector = param_vector;
    computeErrorRate(training_set, training_tpr, training_fpr, training_error);

    // Get Testing error
    DataSet testing_set;
    testing_set.data_points = &data_points[NUM_FEATURES * NUM_TRAINING];
    testing_set.labels = &labels[NUM_TRAINING];
    testing_set.num_data_points = NUM_TESTING;
    testing_set.num_features = NUM_FEATURES;
    testing_set.parameter_vector = param_vector;
    computeErrorRate(testing_set, testing_tpr, testing_fpr, testing_error);

    training_tpr *= 100.0;
    training_fpr *= 100.0;
    training_error *= 100.0;
    testing_tpr *= 100.0;
    testing_fpr *= 100.0;
    testing_error *= 100.0;

    ofile << "Training TPR: " << training_tpr << std::endl; 
    ofile << "Training FPR: " << training_fpr << std::endl; 
    ofile << "Training Error: " << training_error << std::endl; 
    ofile << "Testing TPR: " << testing_tpr << std::endl; 
    ofile << "Testing FPR: " << testing_fpr << std::endl; 
    ofile << "Testing Error: " << testing_error << std::endl; 
  }
  else
  {
    std::cout << "Failed to create output file!" << std::endl;
  }
}

// top-level function 
void SgdLR_sw( DataType    top_feature[NUM_FEATURES * NUM_TRAINING],
               LabelType   label[NUM_TRAINING],
               FeatureType param[NUM_FEATURES])
{
  // intermediate variable for storing gradient
  FeatureType grad[NUM_FEATURES];

  // Load xclbin
  std::string xclbin_file = "overlay_hw.xclbin";
  std::cout << "Loading: " << xclbin_file << std::endl;
  xrt::device device = xrt::device(DEVICE_ID);
  xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
  std::cout << "Loaded xclbin: " << xclbin_file << std::endl;

  // create kernel object
  xrt::kernel dotProduct_kernel = xrt::kernel(device, xclbin_uuid, "dotProduct");
  xrt::kernel Sigmoid_kernel = xrt::kernel(device, xclbin_uuid, "Sigmoid");
  xrt::kernel computeGradient_kernel = xrt::kernel(device, xclbin_uuid, "computeGradient");
  xrt::kernel updateParameter_kernel = xrt::kernel(device, xclbin_uuid, "updateParameter");

  // create memory groups
  xrtMemoryGroup bank_grp_dotProduct_param = dotProduct_kernel.group_id(dotProduct_ptr_param);
  xrtMemoryGroup bank_grp_dotProduct_feature = dotProduct_kernel.group_id(dotProduct_ptr_feature);
  xrtMemoryGroup bank_grp_dotProduct_result = dotProduct_kernel.group_id(dotProduct_ptr_result);

  xrtMemoryGroup bank_grp_sigmoid_exponent = Sigmoid_kernel.group_id(Sigmoid_ptr_exponent);
  xrtMemoryGroup bank_grp_sigmoid_result = Sigmoid_kernel.group_id(Sigmoid_ptr_result);

  xrtMemoryGroup bank_grp_computeGradient_grad = computeGradient_kernel.group_id(computeGradient_ptr_grad);
  xrtMemoryGroup bank_grp_computeGradient_feature = computeGradient_kernel.group_id(computeGradient_ptr_feature);

  xrtMemoryGroup bank_grp_updateParameter_param = updateParameter_kernel.group_id(updateParameter_ptr_param);
  xrtMemoryGroup bank_grp_updateParameter_grad = updateParameter_kernel.group_id(updateParameter_ptr_grad);

  // create buffer objects
  xrt::bo data_buffer_dotProduct_param = xrt::bo(device, sizeof(FeatureType) * NUM_FEATURES, xrt::bo::flags::normal, bank_grp_dotProduct_param);
  xrt::bo data_buffer_dotProduct_feature = xrt::bo(device, sizeof(DataType) * NUM_FEATURES, xrt::bo::flags::normal, bank_grp_dotProduct_feature);
  xrt::bo data_buffer_dotProduct_result = xrt::bo(device, sizeof(DataType), xrt::bo::flags::normal, bank_grp_dotProduct_result);

  xrt::bo data_buffer_sigmoid_exponent = xrt::bo(device, sizeof(FeatureType), xrt::bo::flags::normal, bank_grp_sigmoid_exponent);
  xrt::bo data_buffer_sigmoid_result = xrt::bo(device, sizeof(DataType), xrt::bo::flags::normal, bank_grp_sigmoid_result);

  xrt::bo data_buffer_computeGradient_grad = xrt::bo(device, sizeof(FeatureType) * NUM_FEATURES, xrt::bo::flags::normal, bank_grp_computeGradient_grad);
  xrt::bo data_buffer_computeGradient_feature = xrt::bo(device, sizeof(DataType) * NUM_FEATURES, xrt::bo::flags::normal, bank_grp_computeGradient_feature);

  xrt::bo data_buffer_updateParameter_param = xrt::bo(device, sizeof(FeatureType) * NUM_FEATURES, xrt::bo::flags::normal, bank_grp_updateParameter_param);
  xrt::bo data_buffer_updateParameter_grad = xrt::bo(device, sizeof(FeatureType) * NUM_FEATURES, xrt::bo::flags::normal, bank_grp_updateParameter_grad);

  // create kernel runner
  xrt::run run_dotProduct(dotProduct_kernel);
  xrt::run run_Sigmoid(Sigmoid_kernel);
  xrt::run run_computeGradient(computeGradient_kernel);
  xrt::run run_updateParameter(updateParameter_kernel);

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ...";
  for (int training_id = 0; training_id < NUM_TRAINING; training_id ++ )
  {
    // dot product between parameter vector and data sample 
    DataType* feature = &top_feature[NUM_FEATURES * training_id];
    FeatureType exponent[1] = { 0 };
    // dotProduct(param, feature, exponent);
    // write data to buffer objects
    data_buffer_dotProduct_param.write(param);
    data_buffer_dotProduct_param.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_dotProduct_feature.write(feature);
    data_buffer_dotProduct_feature.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_dotProduct_result.write(exponent);
    data_buffer_dotProduct_result.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // set kernel arguments
    run_dotProduct.set_arg(dotProduct_ptr_param, data_buffer_dotProduct_param);
    run_dotProduct.set_arg(dotProduct_ptr_feature, data_buffer_dotProduct_feature);
    run_dotProduct.set_arg(dotProduct_ptr_result, data_buffer_dotProduct_result);

    // run kernel
    run_dotProduct.start();
    run_dotProduct.wait();
    data_buffer_dotProduct_result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_dotProduct_result.read(exponent);
    //cout<<exponent[0]<<endl;

    // read data from buffer objects
    // There is no need to read the result back to the host since result is only a scalar

    // // sigmoid
    FeatureType prob[1] = { 0 };
    // Sigmoid(exponent, prob);
    // set kernel arguments
    data_buffer_sigmoid_result.write(prob);
    data_buffer_sigmoid_result.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_sigmoid_exponent.write(exponent);
    data_buffer_sigmoid_exponent.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    run_Sigmoid.set_arg(Sigmoid_ptr_exponent, data_buffer_sigmoid_exponent);
    run_Sigmoid.set_arg(Sigmoid_ptr_result, data_buffer_sigmoid_result);

    // run kernel
    run_Sigmoid.start();
    run_Sigmoid.wait();

    data_buffer_sigmoid_result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_sigmoid_result.read(prob);
    // cout<<prob[0]<<endl;
      // // read data from buffer objects
      // // There is no need to read the result back to the host since result is only a scalar

      // // compute gradient
    FeatureType scale = prob[0] - label[training_id];
    feature = &top_feature[NUM_FEATURES * training_id];
    // computeGradient(grad, feature, scale);
    // write data to buffer objects
    data_buffer_computeGradient_grad.write(grad);
    data_buffer_computeGradient_grad.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_computeGradient_feature.write(feature);
    data_buffer_computeGradient_feature.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set kernel arguments
    run_computeGradient.set_arg(computeGradient_ptr_grad, data_buffer_computeGradient_grad);
    run_computeGradient.set_arg(computeGradient_ptr_feature, data_buffer_computeGradient_feature);
    run_computeGradient.set_arg(computeGradient_ptr_scale, scale);

    // run kernel
    run_computeGradient.start();
    run_computeGradient.wait();

    // read data from buffer objects
    data_buffer_computeGradient_grad.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_computeGradient_grad.read(grad); // read is not necessary

    // update parameter vector
    FeatureType step_size = -STEP_SIZE;
    // updateParameter(param, grad, step_size);
    // write data to buffer objects
    data_buffer_updateParameter_param.write(param);
    data_buffer_updateParameter_param.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_updateParameter_grad.write(grad);
    data_buffer_updateParameter_grad.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set kernel arguments
    run_updateParameter.set_arg(updateParameter_ptr_param, data_buffer_updateParameter_param);
    run_updateParameter.set_arg(updateParameter_ptr_grad, data_buffer_updateParameter_grad);
    run_updateParameter.set_arg(updateParameter_ptr_step_size, step_size);

    // run kernel
    run_updateParameter.start();
    run_updateParameter.wait();

    // read data from buffer objects
    data_buffer_updateParameter_param.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_updateParameter_param.read(param);

    cout << exponent[0] << " " << prob[0] << " " << param[1] << " " << feature[0] << endl;
  }
  std::cout << "Done" << std::endl;

/*
  // check results
  std::cout << "Checking results ...";
  check_results( theta, data, label );
  std::cout << "Done" << std::endl;
*/

  // multi iterations
  int iterations = ITERATIONS;
  std::cout << "Running " << iterations << " iterations ...";

  double start_whole_time = omp_get_wtime();

  double start_iteration_time;
  double dotProduct_time = 0;
  double Sigmoid_time = 0;
  double computeGradient_time = 0;
  double updateParameter_time = 0;

  // runs for multiple epochs
  for (int epoch = 0; epoch < iterations; epoch ++) 
  {
    // in each epoch, go through each training instance in sequence
    for( int training_id = 0; training_id < NUM_TRAINING; training_id ++ )
    { 
      start_iteration_time = omp_get_wtime();
      // dot product between parameter vector and data sample 
    // dot product between parameter vector and data sample 
      DataType* feature = &top_feature[NUM_FEATURES * training_id];
      FeatureType exponent[1] = { 0 };
      // dotProduct(param, feature, exponent);
      // write data to buffer objects
      data_buffer_dotProduct_param.write(param);
      data_buffer_dotProduct_param.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      data_buffer_dotProduct_feature.write(feature);
      data_buffer_dotProduct_feature.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      data_buffer_dotProduct_result.write(exponent);
      data_buffer_dotProduct_result.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      // set kernel arguments
      run_dotProduct.set_arg(dotProduct_ptr_param, data_buffer_dotProduct_param);
      run_dotProduct.set_arg(dotProduct_ptr_feature, data_buffer_dotProduct_feature);
      run_dotProduct.set_arg(dotProduct_ptr_result, data_buffer_dotProduct_result);

      // run kernel
      run_dotProduct.start();
      run_dotProduct.wait();
      data_buffer_dotProduct_result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      data_buffer_dotProduct_result.read(exponent);
      dotProduct_time += omp_get_wtime() - start_iteration_time;

      start_iteration_time = omp_get_wtime();

      // sigmoid
      FeatureType prob[1] = { 0 };
      // Sigmoid(exponent, prob);
      // set kernel arguments
      data_buffer_sigmoid_result.write(prob);
      data_buffer_sigmoid_result.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      data_buffer_sigmoid_exponent.write(exponent);
      data_buffer_sigmoid_exponent.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      run_Sigmoid.set_arg(Sigmoid_ptr_exponent, data_buffer_sigmoid_exponent);
      run_Sigmoid.set_arg(Sigmoid_ptr_result, data_buffer_sigmoid_result);

      // run kernel
      run_Sigmoid.start();
      run_Sigmoid.wait();

      data_buffer_sigmoid_result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      data_buffer_sigmoid_result.read(prob);

      // read data from buffer objects
      // There is no need to read the result back to the host since result is only a scalar
      Sigmoid_time += omp_get_wtime() - start_iteration_time;

      start_iteration_time = omp_get_wtime();

      // compute gradient
      FeatureType scale = prob[0] - label[training_id];
      feature = &top_feature[NUM_FEATURES * training_id];
      // computeGradient(grad, feature, scale);
      // write data to buffer objects
      data_buffer_computeGradient_grad.write(grad);
      data_buffer_computeGradient_grad.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      data_buffer_computeGradient_feature.write(feature);
      data_buffer_computeGradient_feature.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      // set kernel arguments
      run_computeGradient.set_arg(computeGradient_ptr_grad, data_buffer_computeGradient_grad);
      run_computeGradient.set_arg(computeGradient_ptr_feature, data_buffer_computeGradient_feature);
      run_computeGradient.set_arg(computeGradient_ptr_scale, scale);

      // run kernel
      run_computeGradient.start();
      run_computeGradient.wait();

      // read data from buffer objects
      data_buffer_computeGradient_grad.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      data_buffer_computeGradient_grad.read(grad); // read is not necessary
      computeGradient_time += omp_get_wtime() - start_iteration_time;

      start_iteration_time = omp_get_wtime();

      // update parameter vector
      FeatureType step_size = -STEP_SIZE;
      // updateParameter(param, grad, step_size);
      // write data to buffer objects
      data_buffer_updateParameter_param.write(param);
      data_buffer_updateParameter_param.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      data_buffer_updateParameter_grad.write(grad);
      data_buffer_updateParameter_grad.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      // set kernel arguments
      run_updateParameter.set_arg(updateParameter_ptr_param, data_buffer_updateParameter_param);
      run_updateParameter.set_arg(updateParameter_ptr_grad, data_buffer_updateParameter_grad);
      run_updateParameter.set_arg(updateParameter_ptr_step_size, step_size);

      // run kernel
      run_updateParameter.start();
      run_updateParameter.wait();

      // read data from buffer objects
      data_buffer_updateParameter_param.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      data_buffer_updateParameter_param.read(param);
      updateParameter_time += omp_get_wtime() - start_iteration_time;
    }
  }
  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations  << " iterations" << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "dotProduct time: " << (dotProduct_time / iterations) * 1000 << " ms" << endl;
  cout << "Sigmoid time: " << (Sigmoid_time / iterations) * 1000 << " ms" << endl;
  cout << "computeGradient time: " << (computeGradient_time / iterations) * 1000 << " ms" << endl;
  cout << "updateParameter time: " << (updateParameter_time / iterations) * 1000 << " ms" << endl;
}

int main(int argc, char ** argv) 
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running spam_filter benchmark C++ HLS" << std::endl;
  std::cout << "=======================================" << std::endl;
  
  setbuf(stdout, NULL);

  // parse command line arguments
  std::string path_to_data("");
  // sdaccel version and sdsoc/sw version have different command line options
  parse_sdsoc_command_line_args(argc, argv, path_to_data);

  // allocate space
  // for software verification
  DataType*    data_points  = new DataType[DATA_SET_SIZE];
  LabelType*   labels       = new LabelType  [NUM_SAMPLES];
  FeatureType* param_vector = new FeatureType[NUM_FEATURES];

  // read in dataset
  std::string str_points_filepath = path_to_data + "/shuffledfeats.dat";
  std::string str_labels_filepath = path_to_data + "/shuffledlabels.dat";

  FILE* data_file;
  FILE* label_file;

  data_file = fopen(str_points_filepath.c_str(), "r");
  if (!data_file)
  {
    printf("Failed to open data file %s!\n", str_points_filepath.c_str());
    return EXIT_FAILURE;
  }
  for (int i = 0; i < DATA_SET_SIZE; i ++ )
  {
    float tmp;
    fscanf(data_file, "%f", &tmp);
    data_points[i] = tmp;
  }
  fclose(data_file);

  label_file = fopen(str_labels_filepath.c_str(), "r");
  if (!label_file)
  {
    printf("Failed to open label file %s!\n", str_labels_filepath.c_str());
    return EXIT_FAILURE;
  }
  for (int i = 0; i < NUM_SAMPLES; i ++ )
  {
    int tmp;
    fscanf(label_file, "%d", &tmp);
    labels[i] = tmp;
  }
  fclose(label_file);

  // reset parameter vector
  for (size_t i = 0; i < NUM_FEATURES; i++)
    param_vector[i] = 0;
    
  SgdLR_sw(data_points, labels, param_vector);

  delete []data_points;
  delete []labels;
  delete []param_vector;

  return EXIT_SUCCESS;

}
