# HeteroBench: Multi-kernel Benchmarks for Heterogeneous Systems

## Overview
HeteroBench is a comprehensive benchmark suite designed to evaluate heterogeneous systems with various accelerators, including CPUs, GPUs (NVIDIA, AMD, Intel), and FPGAs (AMD). It features multi-kernel applications spanning domains like image processing, machine learning, numerical computation, and physical simulation. HeteroBench aims to assist users in assessing performance, optimizing hardware usage, and facilitating decision-making in HPC environments.

## Key Features
- **Broad Hardware Compatibility**: Supports CPUs, GPUs from multiple brands, and FPGAs.
- **Multi-language Support**: Benchmarks available in Python, Numba-accelerated Python, and various C++ versions (serial, OpenMP, OpenACC, CUDA, and Vitis HLS).
- **Customizable Kernel Placement**: Enables kernel-level optimization for different hardware backends by users.
- **Fair Comparisons**: Standardized algorithms across platforms for unbiased benchmarking.
- **User-friendly Design**: Simplified setup with configuration files and a top-level Python script for project management.

## Directory Structure

At the top level, you will find the following files and directories:

- **LICENSE.md**: Open source license information for the repository.
- **README.md**: This file, providing an overview description.
- **Dockerfile**: Docker file used to create the container.
- **run.sh**: A single-point run file for building and running all scripts.
- **Documentation/**: Detail documents and instructions.
  - **[Description.md](./Documentation/Description.md)**: A detail description and quick start guide of this project.
  - **[Notes.md](./Documentation/Notes.md)**: Some instruction notes to add new GPUs, run without GPUs, or change FPGAs.
  - **[Docker_Setup.md](./Documentation/Docker_Setup.md)**: Information on setting up Docker for running applications on NVIDIA GPUs.
  - **[Source_Compilation.md](./Documentation/Source_Compilation.md)**: Instructions for compiling and running HeteroBench from source code.
- **HeteroBench/**: The main directory containing all benchmark codes:
  - **heterobench.py**: The main Python script for managing and executing all benchmarks.
  - **benchmarks/**: Contains 11 subdirectories, one for each benchmark. Each includes Python and C++ implementations along with Makefiles.
  - **config_json/**: Contains configuration files specifying system environment settings and benchmark parameters.
  - **logs/**: Generated after running benchmarks, contains log files and terminal outputs showing time consumption profiling results.

## Contributing

We welcome contributions to HeteroBench! If you'd like to add new benchmarks, support for new hardware, or improve existing code, please submit a pull request or open an issue.

## Cite Us

If you find our project helpful, please consider cite our [ICPE paper](https://doi.org/10.1145/3676151.3719366):

```
```

## Support and Contact

If you have any questions or need assistance with HeteroBench, feel free to contact:

- Alok Mishra: alok.mishra@hpe.com
- Hongzheng Tian: hongzhet@uci.edu, or hongzheng.tian@hpe.com
