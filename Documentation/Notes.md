
# Additional Notes
## Adding New GPU Architectures

HeteroBench supports adding new GPU architectures. Here are two methods to add a new architecture:

### Method 1: Update proj_config.json File

Edit the file `config_json/proj_config.json` to update the GPU architecture (`gpu_arch`) for each benchmark:

```bash
sed -i 's/"gpu_arch": ""/"gpu_arch": "YOUR_ARCHITECTURE"/g' HeteroBench/config_json/proj_config.json
```

Replace `YOUR_ARCHITECTURE` with your GPU architecture identifier (e.g., `sm_80` for NVIDIA RTX A16).

### Method 2: Update heterobench.py

The `heterobench.py` file contains a mapping dictionary `gpu_arch_map` of GPU models to their architecture codes. You can add your GPU model to it to ensure automatic architecture detection.

## Running Without a GPU

HeteroBench can be run on systems without a GPU by using only CPU implementations:

1. To build and run C++ benchmarks on a CPU only, modify the Dockerfile:
   ```bash
   sed -i '69,71 {s/^/# /}' Dockerfile
   echo "" >> Dockerfile
   echo "RUN python3 heterobench.py all build cpu serial" >> Dockerfile
   ```
   This will comment out the GPU related build, and add a CPU serial build in addition.

2. Update run.sh to run CPU benchmarks:
   ```bash
   sed -i '40,42 {s/^/# /}' run.sh
   echo "" >> run.sh
   echo "run cpu serial" >> run.sh
   ```

3. For Docker execution on systems without a GPU, remove the `--gpus all` parameter:
   ```bash
   sed -i '31 {s/--gpus all -it/-it/g}' run.sh
   ```

## Running on an FPGA

Users can follow the steps in [Source_Compilation.md](Source_Compilation.md) to run benchmarks on an FPGA. We use AMD Alveo U280 by default, specifically the `xilinx_u280_gen3x16_xdma_1_202211_1` platform. If you need to use a different FPGA, please update the `"fpga_board"` field of each benchmark in `config_json/proj_config.json` to your specific platform version. You can do this by running:

```bash
sed -i 's/"fpga_board": "xilinx_u280_gen3x16_xdma_1_202211_1"/"fpga_board": "YOUR_FPGA_PLATFORM"/g' HeteroBench/config_json/proj_config.json
```

**Additionally**, it's important to note that if your FPGA board doesn't have **HBM** (High-bandwidth memory), you'll need to comment out the HBM-related lines in the `xclbin_overlay.cfg` file located in each benchmark's `fpga_impl` directory. You can do this by running the following command:

```bash
find HeteroBench/benchmarks -name "xclbin_overlay.cfg" -exec sed -i 's/.*HBM.*/#&/' {} \;
```

This command will search for all `xclbin_overlay.cfg` files in the benchmarks directory and remove any lines containing "HBM".