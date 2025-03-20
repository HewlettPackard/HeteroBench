# Compiling HeteroBench from Source

This document provides detailed instructions for setting up, configuring, and compiling HeteroBench from source code. This approach is recommended for users who need to run on the AMD or Intel GPUs, AMD FPGAs, customize the benchmarks, or run on specialized hardware configurations.

In this guidance, we assume users already setup the hardware environment correctly, for example, the nvidia toolkit, AMD rocm, and intel oneapi.

## Environment Setup

Before compiling HeteroBench, you'll need to configure your environment according to the specific hardware accelerators you plan to use.

### Basic Requirements

- **Operating System**: Linux (Ubuntu 20.04 or newer recommended)
- **Required Packages**: Python 3.8+, CMake 3.16+, GCC/G++ 9.0+
- **Python Libraries**: NumPy (used primarily for data structures and as a reference for validation)

### Hardware Environments

HeteroBench supports the following hardware platforms, each requiring specific development environments:

- **CPU**: No additional setup required beyond the basic development tools
- **NVIDIA GPUs**: Requires CUDA Toolkit and NVIDIA HPC SDK
  - [NVIDIA CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)
  - [NVIDIA HPC SDK Documentation](https://developer.nvidia.com/hpc-sdk-downloads)
- **AMD GPUs**: Requires ROCm Platform
  - [AMD ROCm Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)
- **Intel GPUs**: Requires oneAPI Base Toolkit
  - [Intel oneAPI Installation Guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- **AMD FPGAs**: Requires Vitis Development Environment
  - [AMD Vitis Installation Guide](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html)

### Install basic dependencies (Ubuntu example, skip if already satisfied):

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev python3-pip
pip3 install numpy
```

### Compiler Paths Configuration

Below are some **examples** of compiler paths, please adjust them according to the system details.

#### Set Path to Clang and OpenMP if installed from LLVM

```bash
export LLVM_INSTALL_PATH=/path/to/llvm-project/llvm-install
export PATH=$LLVM_INSTALL_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_INSTALL_PATH/llvm/lib:$LD_LIBRARY_PATH
```

### GPU-Specific Environment Setup

#### NVIDIA GPU

```bash
export NVHPC=/opt/nvidia/hpc_sdk
export PATH=$NVHPC/Linux_x86_64/24.5/compilers/bin:$PATH
export MANPATH=$NVHPC/Linux_x86_64/24.5/compilers/man:$MANPATH
export LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/24.5/compilers/lib:$LD_LIBRARY_PATH
```

#### AMD GPU

```bash
export ROCM_PATH=/opt/rocm-6.1.1
export PATH=$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
```

#### Intel GPU

```bash
source /opt/intel/oneapi/setvars.sh
```

### FPGA Setup (AMD)

For FPGA support, install Xilinx Vitis and set up the environment:

```bash
source /path/to/xilinx/vitis/settings64.sh
```

## Custom Configuration

HeteroBench uses configuration files to control benchmark parameters and environment settings. Users are free to modify any default settings.

### Environment Configuration

Edit the environment settings in `./config_json/env_config.json`, for example:

- **Xilinx Libraries**: Set the paths to Xilinx XRT and HLS
- **Profiling Tools**: Configure profiling tools for performance analysis
  - For NVIDIA GPU: `ncu --set full`
  - For AMD GPU: `rocprof --timestamp on`
  - For Intel GPU: `vtune -collect gpu-offload -knob collect-programming-api=true --`

### Benchmark Configuration

Edit benchmark settings in `./config_json/proj_config.json`:

- **Input/Output Paths**: Set custom input and output paths
- **Data sizes**: Set the custom data and parameter values. 
  - **Make sure to regenerate the input data to overwrite the data with old sizes**
- **Iteration Count**: Adjust the number of iterations for each kernel
- **GPU Brand**: Choose between `nvidia`, `amd`, and `intel`
- **GPU Architecture**: Specify the architecture code for your GPU
- **FPGA Board**: Set the path to your FPGA board files

> Note: each benchmark will run 1 warm-up test intuitively and then run 1 iteration by default. Users can increase the iteration times, then the benchmark will run multiple times of tests, and then get the average time consumption.

## Adding Support for New GPU Architectures

HeteroBench can be extended to support new GPU architectures using one of the following methods:

### Method 1: Update proj_config.json

Edit the `gpu_arch` field in `config_json/proj_config.json`.

### Method 2: Update heterobench.py

Modify the `gpu_arch_map` dictionary in the `heterobench.py` file to include your GPU model in the architecture detection logic.

## Usage Instructions

The main script `heterobench.py` manages all benchmarks and provides a unified interface:

```bash
python heterobench.py <benchmark> <action> <backend> <options>
```

### Parameters

- **benchmark**: Benchmark name (full name or abbreviation)
  - Example: `canny_edge_detection` or `ced`

- **action**: Operation to perform
  - `build`: Compile the benchmark
  - `run`: Execute the benchmark
  - `clean`: Remove compiled outputs for CPU/GPU
  - `clean_fpga`: Remove FPGA-related compiled files
  - `clean_all`: Remove all generated files

- **backend**: Target hardware
  - `python`: Python implementation
  - `numba`: Numba-accelerated Python implementation
  - `cpu`: C++ code on CPU
  - `gpu_omp`: OpenMP C++ code on GPU
  - `gpu_acc`: OpenACC C++ code on GPU (NVIDIA brand only)
  - `gpu_cuda`: CUDA code on GPU (NVIDIA brand only)
  - `fpga`: HLS C++ code on FPGA

- **options**: Additional settings
  - `parallel`: Use parallel implementations (default)
  - `serial`: Run serial implementation on CPU
  - `fpga_compile`: Run FPGA synthesis
  - `fpga_link`: Run FPGA implementation
  - `fpga_all`: Run both FPGA compile and link

### Example Commands

Build and run the Canny Edge Detection benchmark using Python:
```bash
python heterobench.py ced build python
python heterobench.py ced run python
```

Build and run the Sobel Filter benchmark on CPU with OpenMP:
```bash
python heterobench.py sobel_filter build cpu
python heterobench.py sobel_filter run cpu
```

Build and run the CNN benchmark on FPGA:
```bash
python heterobench.py cnn build fpga fpga_all # This is for hardware implementation
python heterobench.py cnn build fpga # This is for the software build
python heterobench.py cnn run fpga
```

Build and run the 3 Matrix Multiplication benchmark on CPU in serial mode:
```bash
python heterobench.py 3mm build cpu serial
python heterobench.py 3mm run cpu serial
```

## Running on Systems without GPUs

HeteroBench can run on CPU-only systems by using the appropriate backend options:

```bash
# Build and run serial CPU benchmarks
python heterobench.py all build cpu serial
python heterobench.py all run cpu serial

# Build and run OpenMP CPU benchmarks
python heterobench.py all build cpu
python heterobench.py all run cpu
```

**Note**: Running Python-based benchmarks on CPUs may take over 10 hours, and Numba-accelerated versions may take over 3 hours.

## Profiling and Results

After running a benchmark, performance results are available in two forms:

1. **Terminal Output**: Runtime information for each kernel is displayed
2. **Log Files**: Detailed profiling results are saved in the `logs/` directory

The logs contain kernel-level timing data, enabling analysis of performance bottlenecks and optimization opportunities.

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   - Verify compiler paths are correctly set
   - Ensure all dependencies are installed
   - Check GPU/FPGA drivers and SDKs are properly configured

2. **Runtime Errors**
   - Confirm hardware is properly detected
   - Verify input files exist at the specified paths
   - Check for sufficient memory and storage

3. **Performance Issues**
   - Try increasing iteration count for more stable results
   - Check for system resource contention
   - Verify hardware-specific optimizations are enabled
