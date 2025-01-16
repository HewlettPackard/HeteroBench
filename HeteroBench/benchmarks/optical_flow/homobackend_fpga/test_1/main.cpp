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

#include <unistd.h>
// other headers
#include "fpga_impl.h"
#include "../../imageLib/imageLib.h"

using namespace std;

#define DEVICE_ID 0


#define optical_flow_hw_ptr_frame0 0
#define optical_flow_hw_ptr_frame1 1
#define optical_flow_hw_ptr_frame2 2
#define optical_flow_hw_ptr_frame3 3
#define optical_flow_hw_ptr_frame4 4
#define optical_flow_hw_ptr_outputs 5



void write_results(velocity_t output[MAX_HEIGHT][MAX_WIDTH], CFloatImage refFlow, std::string outFile)
{
  // copy the output into the float image
  CFloatImage outFlow(MAX_WIDTH, MAX_HEIGHT, 2);
  for (int i = 0; i < MAX_HEIGHT; i++) 
  {
    for (int j = 0; j < MAX_WIDTH; j++) 
    {
      double out_x = output[i][j].x;
      double out_y = output[i][j].y;

      if (out_x * out_x + out_y * out_y > 25.0) 
      {
        outFlow.Pixel(j, i, 0) = 1e10;
        outFlow.Pixel(j, i, 1) = 1e10;
      } 
      else 
      {
        outFlow.Pixel(j, i, 0) = out_x;
        outFlow.Pixel(j, i, 1) = out_y;
      }
    }
  }

  std::cout << "Writing output flow file..." << std::endl;
  WriteFlowFile(outFlow, outFile.c_str());
  std::cout << "Output flow file written to " << outFile << std::endl;

}


void print_usage(char* filename)
{
  printf("usage: %s <options>\n", filename);
  printf("  -f [kernel file]\n");
  printf("  -p [path to data]\n");
  printf("  -o [path to output]\n");
}

void parse_sdsoc_command_line_args(
  int argc,
  char** argv,
  std::string& dataPath,
  std::string& outFile)
{

  int c = 0;

  while ((c = getopt(argc, argv, "p:o:")) != -1)
  {
    switch (c)
    {
    case 'p':
      dataPath = optarg;
      break;
    case 'o':
      outFile = optarg;
      break;
    default:
    {
      print_usage(argv[0]);
      exit(-1);
    }
    } // matching on arguments
  } // while args present
}
void inverse_transform_velocity(pixel_t transformed[MAX_HEIGHT][2 * MAX_WIDTH], velocity_t velocities[MAX_HEIGHT][MAX_WIDTH]) {
  for (int i = 0; i < MAX_HEIGHT; ++i) {
    for (int j = 0; j < MAX_WIDTH; ++j) {

      velocities[i][j].x = transformed[i][2 * j];
      velocities[i][j].y = transformed[i][2 * j + 1];
    }
  }
}
void print_frame(const char* name, pixel_t frame[MAX_HEIGHT][MAX_WIDTH]) {
  printf("%s:\n", name);
  for (int i = 0; i < MAX_HEIGHT; i++) {
    for (int j = 0; j < MAX_WIDTH; j++) {
      printf("%f ", frame[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

void print_outputs(const char* name, pixel_t outputs[MAX_HEIGHT][2 * MAX_WIDTH]) {
  printf("%s:\n", name);
  for (int i = 0; i < MAX_HEIGHT; i++) {
    for (int j = 0; j < 2 * MAX_WIDTH; j++) {
      printf("%f ", outputs[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

// top-level sw function
void optical_flow_hw(pixel_t frame0[MAX_HEIGHT][MAX_WIDTH],
  pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
  pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
  pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
  pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
  pixel_t outputs[MAX_HEIGHT][2 * MAX_WIDTH])
{


  // Load xclbin
  std::string xclbin_file = "overlay_hw.xclbin";
  std::cout << "Loading: " << xclbin_file << std::endl;
  xrt::device device = xrt::device(DEVICE_ID);
  std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
  std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";
  xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
  std::cout << "Loaded xclbin: " << xclbin_file << std::endl;

  // create kernel object
  xrt::kernel optical_flow_hw_kernel = xrt::kernel(device, xclbin_uuid, "optical_flow_hw");

  // create memory groups
  xrtMemoryGroup bank_grp_ops_frame0 = optical_flow_hw_kernel.group_id(optical_flow_hw_ptr_frame0);
  xrtMemoryGroup bank_grp_ops_frame1 = optical_flow_hw_kernel.group_id(optical_flow_hw_ptr_frame1);
  xrtMemoryGroup bank_grp_ops_frame2 = optical_flow_hw_kernel.group_id(optical_flow_hw_ptr_frame2);
  xrtMemoryGroup bank_grp_ops_frame3 = optical_flow_hw_kernel.group_id(optical_flow_hw_ptr_frame3);
  xrtMemoryGroup bank_grp_ops_frame4 = optical_flow_hw_kernel.group_id(optical_flow_hw_ptr_frame4);
  xrtMemoryGroup bank_grp_ops_outputs = optical_flow_hw_kernel.group_id(optical_flow_hw_ptr_outputs);

  // create buffer objects
  xrt::bo data_buffer_ops_frame0 = xrt::bo(device, sizeof(pixel_t) * MAX_HEIGHT * MAX_WIDTH, xrt::bo::flags::normal, bank_grp_ops_frame0);
  xrt::bo data_buffer_ops_frame1 = xrt::bo(device, sizeof(pixel_t) * MAX_HEIGHT * MAX_WIDTH, xrt::bo::flags::normal, bank_grp_ops_frame1);
  xrt::bo data_buffer_ops_frame2 = xrt::bo(device, sizeof(pixel_t) * MAX_HEIGHT * MAX_WIDTH, xrt::bo::flags::normal, bank_grp_ops_frame2);
  xrt::bo data_buffer_ops_frame3 = xrt::bo(device, sizeof(pixel_t) * MAX_HEIGHT * MAX_WIDTH, xrt::bo::flags::normal, bank_grp_ops_frame3);
  xrt::bo data_buffer_ops_frame4 = xrt::bo(device, sizeof(pixel_t) * MAX_HEIGHT * MAX_WIDTH, xrt::bo::flags::normal, bank_grp_ops_frame4);
  xrt::bo data_buffer_ops_outputs = xrt::bo(device, 2 * sizeof(pixel_t) * MAX_HEIGHT * MAX_WIDTH, xrt::bo::flags::normal, bank_grp_ops_outputs);

  std::cout << "Created buffer objects" << std::endl;

  // create kernel runner
  xrt::run run_ops(optical_flow_hw_kernel);
  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;

  // gradient_z_calc(frame0, frame1, frame2, frame3, frame4, gradient_z);
  // write data to buffer objects
  data_buffer_ops_frame0.write(frame0);
  data_buffer_ops_frame0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_ops_frame1.write(frame1);
  data_buffer_ops_frame1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_ops_frame2.write(frame2);
  data_buffer_ops_frame2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_ops_frame3.write(frame3);
  data_buffer_ops_frame3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_ops_frame4.write(frame4);
  data_buffer_ops_frame4.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_ops_outputs.write(outputs);
  data_buffer_ops_outputs.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set kernel arguments
  run_ops.set_arg(optical_flow_hw_ptr_frame0, data_buffer_ops_frame0);
  run_ops.set_arg(optical_flow_hw_ptr_frame1, data_buffer_ops_frame1);
  run_ops.set_arg(optical_flow_hw_ptr_frame2, data_buffer_ops_frame2);
  run_ops.set_arg(optical_flow_hw_ptr_frame3, data_buffer_ops_frame3);
  run_ops.set_arg(optical_flow_hw_ptr_frame4, data_buffer_ops_frame4);
  run_ops.set_arg(optical_flow_hw_ptr_outputs, data_buffer_ops_outputs);


  //run kernel
  run_ops.start();
  run_ops.wait();

  // read data from buffer objects
  data_buffer_ops_outputs.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_ops_outputs.read(outputs);

  // 20 iterations
  int iterations = 20;
  std::cout << "Running " << iterations << " iterations ..." << std::endl;

  double start_whole_time = omp_get_wtime();

  double start_iteration_time;
  double  optical_flow_hw_time = 0;


  for (int iter = 0; iter < iterations; iter++)
  {
    start_iteration_time = omp_get_wtime();

    data_buffer_ops_frame0.write(frame0);
    data_buffer_ops_frame0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_ops_frame1.write(frame1);
    data_buffer_ops_frame1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_ops_frame2.write(frame2);
    data_buffer_ops_frame2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_ops_frame3.write(frame3);
    data_buffer_ops_frame3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_ops_frame4.write(frame4);
    data_buffer_ops_frame4.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_ops_outputs.write(outputs);
    data_buffer_ops_outputs.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set kernel arguments
    run_ops.set_arg(optical_flow_hw_ptr_frame0, data_buffer_ops_frame0);
    run_ops.set_arg(optical_flow_hw_ptr_frame1, data_buffer_ops_frame1);
    run_ops.set_arg(optical_flow_hw_ptr_frame2, data_buffer_ops_frame2);
    run_ops.set_arg(optical_flow_hw_ptr_frame3, data_buffer_ops_frame3);
    run_ops.set_arg(optical_flow_hw_ptr_frame4, data_buffer_ops_frame4);
    run_ops.set_arg(optical_flow_hw_ptr_outputs, data_buffer_ops_outputs);

    //run kernel
    run_ops.start();
    run_ops.wait();

    // read data from buffer objects
    data_buffer_ops_outputs.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_ops_outputs.read(outputs);
    optical_flow_hw_time += omp_get_wtime() - start_iteration_time;
  }
  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations << " iterations" << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "optical_flow time: " << (optical_flow_hw_time / iterations) * 1000 << " ms" << endl;

}

int main(int argc, char** argv)
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running optical_flow benchmark C++ HLS" << std::endl;
  std::cout << "=======================================" << std::endl;

  // parse command line arguments
  std::string dataPath("");
  std::string outFile("");
  parse_sdsoc_command_line_args(argc, argv, dataPath, outFile);

  // create actual file names according to the datapath
  std::string frame_files[5];
  std::string reference_file;
  frame_files[0] = dataPath + "/frame1.ppm";
  frame_files[1] = dataPath + "/frame2.ppm";
  frame_files[2] = dataPath + "/frame3.ppm";
  frame_files[3] = dataPath + "/frame4.ppm";
  frame_files[4] = dataPath + "/frame5.ppm";
  reference_file = dataPath + "/ref.flo";

  // read in images and convert to grayscale
  printf("Reading input files ... \n");

  CByteImage imgs[5];
  for (int i = 0; i < 5; i++)
  {
    CByteImage tmpImg;
    ReadImage(tmpImg, frame_files[i].c_str());
    imgs[i] = ConvertToGray(tmpImg);
  }

  // read in reference flow file
  printf("Reading reference output flow... \n");

  CFloatImage refFlow;
  ReadFlowFile(refFlow, reference_file.c_str());

  // sw version host code
  static pixel_t frames[5][MAX_HEIGHT][MAX_WIDTH];
  static velocity_t outputs[MAX_HEIGHT][MAX_WIDTH];
  static pixel_t outputs_hw[MAX_HEIGHT][2 * MAX_WIDTH];
  // use native C datatype arrays
  for (int f = 0; f < 5; f++)
    for (int i = 0; i < MAX_HEIGHT; i++)
      for (int j = 0; j < MAX_WIDTH; j++) {
        frames[f][i][j] = imgs[f].Pixel(j, i, 0) / 255.0f;
      }

  // run
  optical_flow_hw(frames[0], frames[1], frames[2], frames[3], frames[4], outputs_hw);
  inverse_transform_velocity(outputs_hw, outputs);

/*
  // check results
  printf("Checking results:\n");

  write_results(outputs, refFlow, outFile);
*/

  return EXIT_SUCCESS;

}
