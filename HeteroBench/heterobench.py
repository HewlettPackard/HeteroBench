"""
(C) Copyright [2024] Hewlett Packard Enterprise Development LP

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the Software),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import json
import subprocess
import argparse
import os
import logging
from datetime import datetime
import re

def get_nvidia_arch():
    nvidia_arch_info = '''
    reference source <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>
    
    ============= NVIDIA GPU Architecture =============
    sm_20: GeForce 400, 500, 600, GT-630
    sm_30: GeForce 700, GT-730
    sm_35: Tesla K40
    sm_37: Tesla K80
    sm_52: Quadro M6000, GeForce 900, GTX-970, GTX-980, GTX Titan X
    sm_60: Tesla P100
    sm_61: GTX 1030, GTX 1050, GTX 1060, GTX 1070, GTX 1080, Tesla P4, Tesla P40, Tesla P6
    sm_70: Tesla V100, Titan V
    sm_72: Jetson AGX Xavier, Jetson Xavier NX
    sm_75: GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, Titan RTX, Quadro RTX 4000, Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000, Tesla T4
    sm_80: A100, A100X, A16
    sm_86: RTX 3050, RTX 3060, RTX 3070m RTX 3080, RTX 3090, RTX A2000, RTX A3000, RTX A4000, RTX A5000, RTX A6000
    sm_87: RTX 4080, RTX 4090, RTX 6000, Tesla L40
    sm_90: H100, H200
    sm_100: B100, B200, RTX 5080, RTX 5090
    '''

    gpu_arch_map = {
        'geforce 400': 'sm_20', 'geforce 500': 'sm_20', 'geforce 600': 'sm_20', 'gt-630': 'sm_20',
        'geforce 700': 'sm_30', 'gt-730': 'sm_30',
        'tesla k40': 'sm_35',
        'tesla k80': 'sm_37',
        'quadro m6000': 'sm_52', 'geforce 900': 'sm_52', 'gtx-970': 'sm_52', 'gtx-980': 'sm_52', 'gtx titan x': 'sm_52',
        'tesla p100': 'sm_60',
        'gtx 1030': 'sm_61', 'gtx 1050': 'sm_61', 'gtx 1060': 'sm_61', 'gtx 1070': 'sm_61', 'gtx 1080': 'sm_61',
        'tesla p4': 'sm_61', 'tesla p40': 'sm_61', 'tesla p6': 'sm_61',
        'tesla v100': 'sm_70', 'titan v': 'sm_70',
        'jetson agx xavier': 'sm_72', 'jetson xavier nx': 'sm_72',
        'gtx 1660': 'sm_75', 'rtx 2060': 'sm_75', 'rtx 2070': 'sm_75', 'rtx 2080': 'sm_75', 'titan rtx': 'sm_75',
        'quadro rtx': 'sm_75', 'tesla t4': 'sm_75',
        'a100': 'sm_80', 'a100x': 'sm_80', 'a16': 'sm_80',
        'rtx 3050': 'sm_86', 'rtx 3060': 'sm_86', 'rtx 3070': 'sm_86', 'rtx 3080': 'sm_86', 'rtx 3090': 'sm_86',
        'rtx a2000': 'sm_86', 'rtx a3000': 'sm_86', 'rtx a4000': 'sm_86', 'rtx a5000': 'sm_86', 'rtx a6000': 'sm_86',
        'rtx 4080': 'sm_87', 'rtx 4090': 'sm_87', 'rtx 6000': 'sm_87', 'tesla l40': 'sm_87',
        'h100': 'sm_90', 'h200': 'sm_90',
        'b100': 'sm_100', 'b200': 'sm_100', 'rtx 5080': 'sm_100', 'rtx 5090': 'sm_100'
    }

    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True)
        gpu_name = result.stdout.strip().lower()

        for model, arch in gpu_arch_map.items():
            if re.search(rf'\b{re.escape(model)}\b', gpu_name):
                return arch

        print(f"Unknown NVIDIA GPU architecture for {gpu_name}")
        print("Please directly give the GPU architecture in \"gpu_arch\" in proj_config.json:")
        print(nvidia_arch_info)
        return None

    except Exception as e:
        print(f"Error getting NVIDIA GPU architecture: {e}")
        return None
    
def get_amd_arch():
    try:
        result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True)
        output_lines = result.stdout.splitlines()

        gpu_name_line = None
        for line in output_lines:
            if 'card series' in line.lower() or 'gfx version' in line.lower():
                gpu_name_line = line.strip().lower()
                break
        
        if not gpu_name_line:
            print("No relevant GPU information found in rocm-smi output.")
            return None
        
        if 'mi100' in gpu_name_line or 'mi210' in gpu_name_line:
            return 'gfx90a'
        elif 'aldebaran' in gpu_name_line:
            return 'gfx90a'
        elif 'mi50' in gpu_name_line or 'mi60' in gpu_name_line:
            return 'gfx908'
        else:
            print(f"Unknown AMD GPU architecture for {gpu_name_line}")
            return None
    except Exception as e:
        print(f"Error getting AMD GPU architecture: {e}")
        return None
    
def get_intel_arch():
    return 'spir64'

def get_gpu_arch(gpu_brand):
    if gpu_brand == 'nvidia':
        return get_nvidia_arch()
    elif gpu_brand == 'amd':
        return get_amd_arch()
    elif gpu_brand == 'intel':
        return get_intel_arch()
    else:
        print(f"Unsupported GPU brand: {gpu_brand}")
        return None

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def add_prefix_to_sources(sources, prefix):
    return [f'{prefix}/{source}' for source in sources]

def setup_logging(log_file):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, 'w')  # 'w' to overwrite the log file
                                  ,logging.StreamHandler()  # uncomment this line to print logs to console
                                  ])

def run_makefile(benchmark, backend, env, action, options, log_file=None):

    cwd = os.getcwd()
    if backend['python'] or backend['numba']:
        # preprocess the path $(COVA_BENCH_PATH) into $COVA_BENCH_PATH
        input_path = os.path.expandvars(benchmark['parameters']['input'].replace('$(', '$').replace(')', ''))
        output_path = os.path.expandvars(benchmark['parameters']['output'].replace('$(', '$').replace(')', ''))
        # if not exists and the output path is not "", create the output directory
        if not os.path.exists(output_path) and output_path:
            os.makedirs(output_path)
        if benchmark['name'] == 'canny_edge_detection':
            print(f'current benchmark: {benchmark["name"]}')
            low_threshold = str(benchmark['parameters']['low_threshold'])
            high_threshold = str(benchmark['parameters']['high_threshold'])
            iterations = str(benchmark['parameters']['iteration_times'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", \
                                    f'{cwd}/{input_path}', f'{cwd}/{output_path}', low_threshold, high_threshold, iterations], stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", \
                                    f'{cwd}/{input_path}', f'{cwd}/{output_path}', low_threshold, high_threshold, iterations], stdout=f, stderr=f, check=True)
        elif benchmark['name'] == 'optical_flow':
            print(f'current benchmark: {benchmark["name"]}')
            max_height = str(benchmark['parameters']['max_height'])
            max_width = str(benchmark['parameters']['max_width'])
            iterations = str(benchmark['parameters']['iteration_times'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", \
                                    f'{cwd}/{input_path}', f'{cwd}/{output_path}', max_height, max_width, iterations], stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", \
                                    f'{cwd}/{input_path}', f'{cwd}/{output_path}', max_height, max_width, iterations], stdout=f, stderr=f, check=True)
        elif benchmark['name'] == 'sobel_filter':
            print(f'current benchmark: {benchmark["name"]}')
            iterations = str(benchmark['parameters']['iteration_times'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", \
                                    f'{cwd}/{input_path}', f'{cwd}/{output_path}', iterations], stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", \
                                    f'{cwd}/{input_path}', f'{cwd}/{output_path}', iterations], stdout=f, stderr=f, check=True)
        elif benchmark['name'] == 'convolutional_neural_network':
            print(f'current benchmark: {benchmark["name"]}')
            iterations = str(benchmark['parameters']['iteration_times'])
            conv2d_stride = str(benchmark['parameters']['conv2d_stride'])
            conv2d_padding = str(benchmark['parameters']['conv2d_padding'])
            conv2d_bias = str(benchmark['parameters']['conv2d_bias'])
            pooling_size = str(benchmark['parameters']['pooling_size'])
            pooling_stride = str(benchmark['parameters']['pooling_stride'])
            input_size_h = str(benchmark['parameters']['input_size_h'])
            input_size_w = str(benchmark['parameters']['input_size_w'])
            conv_kernel_size_h = str(benchmark['parameters']['conv_kernel_size_h'])
            conv_kernel_size_w = str(benchmark['parameters']['conv_kernel_size_w'])
            full_connect_layer_size_w = str(benchmark['parameters']['full_connect_layer_size_w'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", \
                                    f'{cwd}/{input_path}', f'{cwd}/{output_path}', iterations, conv2d_stride, \
                                    conv2d_padding, conv2d_bias, pooling_size, pooling_stride, input_size_h, \
                                    input_size_w, conv_kernel_size_h, conv_kernel_size_w, full_connect_layer_size_w], \
                                    stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", \
                                    f'{cwd}/{input_path}', f'{cwd}/{output_path}', iterations, conv2d_stride, \
                                    conv2d_padding, conv2d_bias, pooling_size, pooling_stride, input_size_h, \
                                    input_size_w, conv_kernel_size_h, conv_kernel_size_w, full_connect_layer_size_w], \
                                    stdout=f, stderr=f, check=True)
        elif benchmark['name'] == 'multilayer_perceptron':
            print(f'current benchmark: {benchmark["name"]}')
            iterations = str(benchmark['parameters']['iteration_times'])
            layer0_h1 = str(benchmark['parameters']['layer0_h1'])
            layer0_w1 = str(benchmark['parameters']['layer0_w1'])
            layer0_w2 = str(benchmark['parameters']['layer0_w2'])
            layer1_h1 = str(benchmark['parameters']['layer1_h1'])
            layer1_w1 = str(benchmark['parameters']['layer1_w1'])
            layer1_w2 = str(benchmark['parameters']['layer1_w2'])
            layer2_h1 = str(benchmark['parameters']['layer2_h1'])
            layer2_w1 = str(benchmark['parameters']['layer2_w1'])
            layer2_w2 = str(benchmark['parameters']['layer2_w2'])
            layer3_h1 = str(benchmark['parameters']['layer3_h1'])
            layer3_w1 = str(benchmark['parameters']['layer3_w1'])
            layer3_w2 = str(benchmark['parameters']['layer3_w2'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", \
                                    f'{cwd}/{input_path}', f'{cwd}/{output_path}', iterations, layer0_h1, \
                                    layer0_w1, layer0_w2, layer1_h1, layer1_w1, layer1_w2, layer2_h1, \
                                    layer2_w1, layer2_w2, layer3_h1, layer3_w1, layer3_w2], stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", \
                                    f'{cwd}/{input_path}', f'{cwd}/{output_path}', iterations, layer0_h1, \
                                    layer0_w1, layer0_w2, layer1_h1, layer1_w1, layer1_w2, layer2_h1, \
                                    layer2_w1, layer2_w2, layer3_h1, layer3_w1, layer3_w2], stdout=f, stderr=f, check=True)
        elif benchmark['name'] == 'one_head_attention':
            print(f'current benchmark: {benchmark["name"]}')
            input_path = str(benchmark['parameters']['input'])
            output_path = str(benchmark['parameters']['output'])
            iterations = str(benchmark['parameters']['iteration_times'])
            batch_size = str(benchmark['parameters']['batch_size'])
            N = str(benchmark['parameters']['N'])
            d = str(benchmark['parameters']['d'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", input_path, \
                                    output_path, iterations, batch_size, N, d], stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", input_path, \
                                    output_path, iterations, batch_size, N, d], stdout=f, stderr=f, check=True)
        elif benchmark['name'] == 'spam_filter':
            print(f'current benchmark: {benchmark["name"]}')
            iterations = str(benchmark['parameters']['iteration_times'])
            num_features = str(benchmark['parameters']['num_features'])
            num_samples = str(benchmark['parameters']['num_samples'])
            num_training = str(benchmark['parameters']['num_training'])
            num_testing = str(benchmark['parameters']['num_testing'])
            step_size = str(benchmark['parameters']['step_size'])
            num_epochs = str(benchmark['parameters']['num_epochs'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", \
                                    f'{cwd}/{input_path}', iterations, num_features, \
                                    num_samples, num_training, num_testing, step_size, num_epochs], \
                                    stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", \
                                    f'{cwd}/{input_path}', iterations, num_features, \
                                    num_samples, num_training, num_testing, step_size, num_epochs], \
                                    stdout=f, stderr=f, check=True)
        elif benchmark['name'] == '3_matrix_multiplication':
            print(f'current benchmark: {benchmark["name"]}')
            iterations = str(benchmark['parameters']['iteration_times'])
            ni = str(benchmark['parameters']['ni'])
            nj = str(benchmark['parameters']['nj'])
            nk = str(benchmark['parameters']['nk'])
            nl = str(benchmark['parameters']['nl'])
            nm = str(benchmark['parameters']['nm'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", \
                                    iterations, ni, nj, nk, nl, nm], stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", \
                                    iterations, ni, nj, nk, nl, nm], stdout=f, stderr=f, check=True)
        elif benchmark['name'] == 'alternating_direction_implicit':
            print(f'current benchmark: {benchmark["name"]}')
            iterations = str(benchmark['parameters']['iteration_times'])
            tsteps = str(benchmark['parameters']['tsteps'])
            n = str(benchmark['parameters']['n'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", \
                                    iterations, tsteps, n], stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", \
                                    iterations, tsteps, n], stdout=f, stderr=f, check=True)
        elif benchmark['name'] == 'digit_recog':
            print(f'current benchmark: {benchmark["name"]}')
            input_path = str(benchmark['parameters']['input'])
            iterations = str(benchmark['parameters']['iteration_times'])
            num_training = str(benchmark['parameters']['num_training'])
            class_size = str(benchmark['parameters']['class_size'])
            digit_width = str(benchmark['parameters']['digit_width'])
            k_const = str(benchmark['parameters']['k_const'])
            para_factor = str(benchmark['parameters']['para_factor'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", input_path, \
                                    iterations, num_training, class_size, digit_width, k_const, para_factor], \
                                    stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", input_path, \
                                    iterations, num_training, class_size, digit_width, k_const, para_factor], \
                                    stdout=f, stderr=f, check=True)
        elif benchmark['name'] == 'parallelize_particle':
            print(f'current benchmark: {benchmark["name"]}')
            iterations = str(benchmark['parameters']['iteration_times'])
            nparticles = str(benchmark['parameters']['nparticles'])
            savefreq = str(benchmark['parameters']['savefreq'])
            with open(log_file, 'a') as f:
                if backend['python']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/main.py", \
                                    iterations, nparticles, savefreq], stdout=f, stderr=f, check=True)
                elif backend['numba']:
                    subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python_numba/main.py", \
                                    iterations, nparticles, savefreq], stdout=f, stderr=f, check=True)
        else:
            print(f"Unknown benchmark '{benchmark['name']}'.")
        return
    
    if backend['cpu']:
        # add ./cpu_impl/ for cpu_sources 
        cpu_sources_with_prefix = add_prefix_to_sources(benchmark['krnl_sources'], 'cpu_impl')
    if backend['gpu_omp']:
        # add ./gpu_impl/ for gpu_sources 
        gpu_sources_with_prefix = add_prefix_to_sources(benchmark['krnl_sources'], 'gpu_impl')
    if backend['gpu_cuda']:
        # add ./cuda_impl/ for gpu_sources 
        gpu_sources_with_prefix = add_prefix_to_sources(benchmark['krnl_sources'], 'cuda_impl')
        # change the .cpp files to .cu files
        gpu_sources_with_prefix = [source.replace('.cpp', '.cu') for source in gpu_sources_with_prefix]
    if backend['gpu_acc']:
        # add ./acc_impl/ for gpu_sources
        gpu_sources_with_prefix = add_prefix_to_sources(benchmark['krnl_sources'], 'acc_impl')
    if backend['fpga']:
        # add ./fpga_impl/ for fpga_sources 
        fpga_sources_with_prefix = add_prefix_to_sources(benchmark['krnl_sources'], 'fpga_impl')

    if options == 'serial':
        proj_path = f"./benchmarks/{benchmark['name']}/homobackend_cpu/Cpp"
    elif backend['hetero']:
        proj_path = f"./benchmarks/{benchmark['name']}/heterobackend/cpu_gpu"
        print(f"Currently, running heterobackend for {benchmark['name']} is not supported.")
        print(f'Please directly run the Makefile in {proj_path} to build the benchmark.')
        raise NotImplementedError
    elif backend['cpu']:
        proj_path = f"./benchmarks/{benchmark['name']}/homobackend_cpu/OpenMP"
    elif backend['gpu_omp']:
        proj_path = f"./benchmarks/{benchmark['name']}/homobackend_gpu/OpenMP"
    elif backend['gpu_cuda']:
        proj_path = f"./benchmarks/{benchmark['name']}/homobackend_gpu/CUDA"
    elif backend['gpu_acc']:
        proj_path = f"./benchmarks/{benchmark['name']}/homobackend_gpu/OpenACC"
    elif backend['fpga']:
        proj_path = f"./benchmarks/{benchmark['name']}/homobackend_fpga/test_1"
    else:
        pass #TODO: handle other heterobackend cases

    from jinja2 import FileSystemLoader, Environment
    template_loader = FileSystemLoader(searchpath = proj_path)
    template_env = Environment(loader = template_loader)
    print(f'current processing path: {proj_path}')
    makefile_template = template_env.get_template('Makefile.jinja')

    make_cmd = [
        "make",
        "-C", proj_path,

        # general environment variables
        f"compiler={env['compiler']}",
        f"cxxflags={env['cxxflags']}",
        f"ldflags={env['ldflags']}",
        f"openmp_libs={env['openmp_libs']}",
        f"lomptarget={env['lomptarget']}",
        f"profiling_tool={env['profiling_tool']}",

        # opencv environment variables if needed
        f"opencv_libs={env['opencv_libs']}",

        # benchmark specific environment variables
        # f"sources={benchmark['source']}",
        f"target={benchmark['abbreviation']}",
        f"iteration_times={benchmark['parameters']['iteration_times']}"
    ]
    if backend['gpu_cuda']:
        make_cmd.append(f"cuda_compiler={env['cuda_compiler']}")

    make_template_vars = {
        "compiler": env['compiler'],
        "cxxflags": env['cxxflags'],
        "ldflags": env['ldflags'],
        "openmp_libs": env['openmp_libs'],
        "lomptarget": env['lomptarget'],
        "profiling_tool": env['profiling_tool'],
        "opencv_libs": env['opencv_libs'],
        "target": benchmark['abbreviation'],
        "iteration_times": benchmark['parameters']['iteration_times']
    }
    if backend['gpu_cuda']:
        make_template_vars["cuda_compiler"] = env['cuda_compiler']

    # if there are multiple sources in the benchmark['source'], 
    # or the source is a list of sources, then we need to handle it
    if benchmark['source'] and isinstance(benchmark['source'], list):
        make_cmd.append(f"sources={' '.join(benchmark['source'])}")
        make_template_vars["sources"] = ' '.join(benchmark['source'])
    elif benchmark['source']:
        make_cmd.append(f"sources={benchmark['source']}")
        make_template_vars["sources"] = benchmark['source']

    output_path = f"{cwd}/{benchmark['parameters']['output']}"
    # # if the output_path exists, remove it; only for debug purpose
    # if os.path.exists(output_path):
    #     subprocess.run(["rm", "-rf", output_path], check=True)
    dir_path = os.path.dirname(output_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if benchmark['name'] == 'canny_edge_detection':
        make_cmd.append(f"input={cwd}/{benchmark['parameters']['input']}")
        make_cmd.append(f"output={output_path}")
        make_cmd.append(f"low_threshold={benchmark['parameters']['low_threshold']}")
        make_cmd.append(f"high_threshold={benchmark['parameters']['high_threshold']}")
        
        make_template_vars["input"] = f"{cwd}/{benchmark['parameters']['input']}"
        make_template_vars["output"] = output_path
        make_template_vars["low_threshold"] = benchmark['parameters']['low_threshold']
        make_template_vars["high_threshold"] = benchmark['parameters']['high_threshold']

    elif benchmark['name'] == 'gpu_test':
        pass
    elif benchmark['name'] == 'optical_flow':
        make_cmd.append(f"input={cwd}/{benchmark['parameters']['input']}")
        make_cmd.append(f"output={output_path}")
        make_cmd.append(f"max_height={benchmark['parameters']['max_height']}")
        make_cmd.append(f"max_width={benchmark['parameters']['max_width']}")

        make_template_vars["input"] = f"{cwd}/{benchmark['parameters']['input']}"
        make_template_vars["output"] = output_path
        make_template_vars["max_height"] = benchmark['parameters']['max_height']
        make_template_vars["max_width"] = benchmark['parameters']['max_width']

    elif benchmark['name'] == 'sobel_filter':
        make_cmd.append(f"input={cwd}/{benchmark['parameters']['input']}")
        make_cmd.append(f"output={output_path}")

        make_template_vars["input"] = f"{cwd}/{benchmark['parameters']['input']}"
        make_template_vars["output"] = output_path

    elif benchmark['name'] == 'convolutional_neural_network':
        # for the fpga version of this benchmark, we need to remove the pad_input.cpp 
        # from the sources because it is included inside the conv2d.cpp
        if backend['fpga']:
            fpga_sources_with_prefix.remove('fpga_impl/pad_input.cpp')

        # Check if the input_path exists and all required files are inside
        input_path = os.path.expandvars(benchmark['parameters']['input'].replace('$(', '$').replace(')', ''))
        required_files = ['b_fc.bin', 'input_image.bin', 'W_conv.bin', 'W_fc.bin']
        if (not os.path.exists(input_path)) or not all(os.path.exists(os.path.join(input_path, file)) for file in required_files):
            if not os.path.exists(input_path):
                os.makedirs(input_path)
            print(f"Generating input data for {benchmark['name']}...")
            iterations = str(benchmark['parameters']['iteration_times'])
            conv2d_stride = str(benchmark['parameters']['conv2d_stride'])
            conv2d_padding = str(benchmark['parameters']['conv2d_padding'])
            conv2d_bias = str(benchmark['parameters']['conv2d_bias'])
            pooling_size = str(benchmark['parameters']['pooling_size'])
            pooling_stride = str(benchmark['parameters']['pooling_stride'])
            input_size_h = str(benchmark['parameters']['input_size_h'])
            input_size_w = str(benchmark['parameters']['input_size_w'])
            conv_kernel_size_h = str(benchmark['parameters']['conv_kernel_size_h'])
            conv_kernel_size_w = str(benchmark['parameters']['conv_kernel_size_w'])
            full_connect_layer_size_w = str(benchmark['parameters']['full_connect_layer_size_w'])
            subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/tool.py", \
                            f'{cwd}/{input_path}', f'{cwd}/{output_path}', iterations, conv2d_stride, \
                            conv2d_padding, conv2d_bias, pooling_size, pooling_stride, input_size_h, \
                            input_size_w, conv_kernel_size_h, conv_kernel_size_w, full_connect_layer_size_w])

        make_cmd.append(f"input={cwd}/{benchmark['parameters']['input']}")
        make_cmd.append(f"output={output_path}")
        make_cmd.append(f"conv2d_stride={benchmark['parameters']['conv2d_stride']}")
        make_cmd.append(f"conv2d_padding={benchmark['parameters']['conv2d_padding']}")
        make_cmd.append(f"conv2d_bias={benchmark['parameters']['conv2d_bias']}")
        make_cmd.append(f"pooling_size={benchmark['parameters']['pooling_size']}")
        make_cmd.append(f"pooling_stride={benchmark['parameters']['pooling_stride']}")
        make_cmd.append(f"input_size_h={benchmark['parameters']['input_size_h']}")
        make_cmd.append(f"input_size_w={benchmark['parameters']['input_size_w']}")
        make_cmd.append(f"conv_kernel_size_h={benchmark['parameters']['conv_kernel_size_h']}")
        make_cmd.append(f"conv_kernel_size_w={benchmark['parameters']['conv_kernel_size_w']}")
        make_cmd.append(f"full_connect_layer_size_w={benchmark['parameters']['full_connect_layer_size_w']}")
    
        make_template_vars["input"] = f"{cwd}/{benchmark['parameters']['input']}"
        make_template_vars["output"] = output_path
        make_template_vars["conv2d_stride"] = benchmark['parameters']['conv2d_stride']
        make_template_vars["conv2d_padding"] = benchmark['parameters']['conv2d_padding']
        make_template_vars["conv2d_bias"] = benchmark['parameters']['conv2d_bias']
        make_template_vars["pooling_size"] = benchmark['parameters']['pooling_size']
        make_template_vars["pooling_stride"] = benchmark['parameters']['pooling_stride']
        make_template_vars["input_size_h"] = benchmark['parameters']['input_size_h']
        make_template_vars["input_size_w"] = benchmark['parameters']['input_size_w']
        make_template_vars["conv_kernel_size_h"] = benchmark['parameters']['conv_kernel_size_h']
        make_template_vars["conv_kernel_size_w"] = benchmark['parameters']['conv_kernel_size_w']
        make_template_vars["full_connect_layer_size_w"] = benchmark['parameters']['full_connect_layer_size_w']

    elif benchmark['name'] == 'multilayer_perceptron':
        # for the fpga version of this benchmark, we need to change the dot_add.cpp
        # to dot_add0.cpp, dot_add1.cpp, dot_add2.cpp, and dot_add3.cpp
        if backend['fpga']:
            fpga_sources_with_prefix.remove('fpga_impl/dot_add.cpp')
            fpga_sources_with_prefix.append('fpga_impl/dot_add0.cpp')
            fpga_sources_with_prefix.append('fpga_impl/dot_add1.cpp')
            fpga_sources_with_prefix.append('fpga_impl/dot_add2.cpp')
            fpga_sources_with_prefix.append('fpga_impl/dot_add3.cpp')

        # Check if the input_path exists and all required files are inside
        input_path = os.path.expandvars(benchmark['parameters']['input'].replace('$(', '$').replace(')', ''))
        required_files = ['a0.bin', 'b0.bin', 'b1.bin', 'b2.bin', 'b3.bin', 'w0.bin', 'w1.bin', 'w2.bin', 'w3.bin']
        if (not os.path.exists(input_path)) or not all(os.path.exists(os.path.join(input_path, file)) for file in required_files):
            if not os.path.exists(input_path):
                os.makedirs(input_path)
            print(f"Generating input data for {benchmark['name']}...")
            iterations = str(benchmark['parameters']['iteration_times'])
            layer0_h1 = str(benchmark['parameters']['layer0_h1'])
            layer0_w1 = str(benchmark['parameters']['layer0_w1'])
            layer0_w2 = str(benchmark['parameters']['layer0_w2'])
            layer1_h1 = str(benchmark['parameters']['layer1_h1'])
            layer1_w1 = str(benchmark['parameters']['layer1_w1'])
            layer1_w2 = str(benchmark['parameters']['layer1_w2'])
            layer2_h1 = str(benchmark['parameters']['layer2_h1'])
            layer2_w1 = str(benchmark['parameters']['layer2_w1'])
            layer2_w2 = str(benchmark['parameters']['layer2_w2'])
            layer3_h1 = str(benchmark['parameters']['layer3_h1'])
            layer3_w1 = str(benchmark['parameters']['layer3_w1'])
            layer3_w2 = str(benchmark['parameters']['layer3_w2'])
            subprocess.run([f'{env["python_command"]}', f"benchmarks/{benchmark['name']}/python/tool.py", \
                            f'{cwd}/{input_path}', f'{cwd}/{output_path}', iterations, layer0_h1, \
                            layer0_w1, layer0_w2, layer1_h1, layer1_w1, layer1_w2, layer2_h1, \
                            layer2_w1, layer2_w2, layer3_h1, layer3_w1, layer3_w2])
        
        make_cmd.append(f"input={cwd}/{benchmark['parameters']['input']}")
        make_cmd.append(f"output={output_path}")
        make_cmd.append(f"layer0_h1={benchmark['parameters']['layer0_h1']}")
        make_cmd.append(f"layer0_w1={benchmark['parameters']['layer0_w1']}")
        make_cmd.append(f"layer0_w2={benchmark['parameters']['layer0_w2']}")
        make_cmd.append(f"layer1_h1={benchmark['parameters']['layer1_h1']}")
        make_cmd.append(f"layer1_w1={benchmark['parameters']['layer1_w1']}")
        make_cmd.append(f"layer1_w2={benchmark['parameters']['layer1_w2']}")
        make_cmd.append(f"layer2_h1={benchmark['parameters']['layer2_h1']}")
        make_cmd.append(f"layer2_w1={benchmark['parameters']['layer2_w1']}")
        make_cmd.append(f"layer2_w2={benchmark['parameters']['layer2_w2']}")
        make_cmd.append(f"layer3_h1={benchmark['parameters']['layer3_h1']}")
        make_cmd.append(f"layer3_w1={benchmark['parameters']['layer3_w1']}")
        make_cmd.append(f"layer3_w2={benchmark['parameters']['layer3_w2']}")

        make_template_vars["input"] = f"{cwd}/{benchmark['parameters']['input']}"
        make_template_vars["output"] = output_path
        make_template_vars["layer0_h1"] = benchmark['parameters']['layer0_h1']
        make_template_vars["layer0_w1"] = benchmark['parameters']['layer0_w1']
        make_template_vars["layer0_w2"] = benchmark['parameters']['layer0_w2']
        make_template_vars["layer1_h1"] = benchmark['parameters']['layer1_h1']
        make_template_vars["layer1_w1"] = benchmark['parameters']['layer1_w1']
        make_template_vars["layer1_w2"] = benchmark['parameters']['layer1_w2']
        make_template_vars["layer2_h1"] = benchmark['parameters']['layer2_h1']
        make_template_vars["layer2_w1"] = benchmark['parameters']['layer2_w1']
        make_template_vars["layer2_w2"] = benchmark['parameters']['layer2_w2']
        make_template_vars["layer3_h1"] = benchmark['parameters']['layer3_h1']
        make_template_vars["layer3_w1"] = benchmark['parameters']['layer3_w1']
        make_template_vars["layer3_w2"] = benchmark['parameters']['layer3_w2']

    elif benchmark['name'] == 'one_head_attention':
        # for the fpga version of this benchmark, we need to change the matmul.cpp
        # to matmul0.cpp, matmul1.cpp
        if backend['fpga']:
            fpga_sources_with_prefix.remove('fpga_impl/matmul.cpp')
            fpga_sources_with_prefix.append('fpga_impl/matmul0.cpp')
            fpga_sources_with_prefix.append('fpga_impl/matmul1.cpp')
        
        make_cmd.append(f"input={cwd}/{benchmark['parameters']['input']}")
        make_cmd.append(f"output={output_path}")
        make_cmd.append(f"batch_size={benchmark['parameters']['batch_size']}")
        make_cmd.append(f"N={benchmark['parameters']['N']}")
        make_cmd.append(f"d={benchmark['parameters']['d']}")

        make_template_vars["input"] = f"{cwd}/{benchmark['parameters']['input']}"
        make_template_vars["output"] = output_path
        make_template_vars["batch_size"] = benchmark['parameters']['batch_size']
        make_template_vars["N"] = benchmark['parameters']['N']
        make_template_vars["d"] = benchmark['parameters']['d']

    elif benchmark['name'] == 'spam_filter':
        make_cmd.append(f"input={cwd}/{benchmark['parameters']['input']}")
        make_cmd.append(f"num_features={benchmark['parameters']['num_features']}")
        make_cmd.append(f"num_samples={benchmark['parameters']['num_samples']}")
        make_cmd.append(f"num_training={benchmark['parameters']['num_training']}")
        make_cmd.append(f"num_testing={benchmark['parameters']['num_testing']}")
        make_cmd.append(f"step_size={benchmark['parameters']['step_size']}")
        make_cmd.append(f"num_epochs={benchmark['parameters']['num_epochs']}")

        make_template_vars["input"] = f"{cwd}/{benchmark['parameters']['input']}"
        make_template_vars["num_features"] = benchmark['parameters']['num_features']
        make_template_vars["num_samples"] = benchmark['parameters']['num_samples']
        make_template_vars["num_training"] = benchmark['parameters']['num_training']
        make_template_vars["num_testing"] = benchmark['parameters']['num_testing']
        make_template_vars["step_size"] = benchmark['parameters']['step_size']
        make_template_vars["num_epochs"] = benchmark['parameters']['num_epochs']

    elif benchmark['name'] == '3_matrix_multiplication':
        # for the fpga version of this benchmark, we need to change the kernel_3mm_0.cpp
        # kernel_3mm_1.cpp, kernel_3mm_2.cpp to kernel_3mm.cpp
        if backend['fpga']:
            fpga_sources_with_prefix.remove('fpga_impl/kernel_3mm_0.cpp')
            fpga_sources_with_prefix.remove('fpga_impl/kernel_3mm_1.cpp')
            fpga_sources_with_prefix.remove('fpga_impl/kernel_3mm_2.cpp')
            fpga_sources_with_prefix.append('fpga_impl/kernel_3mm.cpp')
        
        make_cmd.append(f"ni={benchmark['parameters']['ni']}")
        make_cmd.append(f"nj={benchmark['parameters']['nj']}")
        make_cmd.append(f"nk={benchmark['parameters']['nk']}")
        make_cmd.append(f"nl={benchmark['parameters']['nl']}")
        make_cmd.append(f"nm={benchmark['parameters']['nm']}")

        make_template_vars["ni"] = benchmark['parameters']['ni']
        make_template_vars["nj"] = benchmark['parameters']['nj']
        make_template_vars["nk"] = benchmark['parameters']['nk']
        make_template_vars["nl"] = benchmark['parameters']['nl']
        make_template_vars["nm"] = benchmark['parameters']['nm']

    elif benchmark['name'] == 'alternating_direction_implicit':
        make_cmd.append(f"tsteps={benchmark['parameters']['tsteps']}")
        make_cmd.append(f"n={benchmark['parameters']['n']}")

        make_template_vars["tsteps"] = benchmark['parameters']['tsteps']
        make_template_vars["n"] = benchmark['parameters']['n']
        
    elif benchmark['name'] == 'digit_recog':
        # for the fpga version of this benchmark, we need to delete all the .cpp files
        # and replace with DigitRec_hw.cpp
        if backend['fpga']:
            fpga_sources_with_prefix = ['fpga_impl/DigitRec_hw.cpp']

        make_cmd.append(f"num_training={benchmark['parameters']['num_training']}")
        make_cmd.append(f"class_size={benchmark['parameters']['class_size']}")
        make_cmd.append(f"digit_width={benchmark['parameters']['digit_width']}")
        make_cmd.append(f"k_const={benchmark['parameters']['k_const']}")
        make_cmd.append(f"para_factor={benchmark['parameters']['para_factor']}")

        make_template_vars["num_training"] = benchmark['parameters']['num_training']
        make_template_vars["class_size"] = benchmark['parameters']['class_size']
        make_template_vars["digit_width"] = benchmark['parameters']['digit_width']
        make_template_vars["k_const"] = benchmark['parameters']['k_const']
        make_template_vars["para_factor"] = benchmark['parameters']['para_factor']

    elif benchmark['name'] == 'parallelize_particle':
        # for the fpga version of this benchmark, we need to delete all the .cpp files
        # and replace with top_kernel.cpp
        if backend['fpga']:
            fpga_sources_with_prefix = ['fpga_impl/top_kernel.cpp']
            
        make_cmd.append(f"nparticles={benchmark['parameters']['nparticles']}")
        make_cmd.append(f"savefreq={benchmark['parameters']['savefreq']}")

        make_template_vars["nparticles"] = benchmark['parameters']['nparticles']
        make_template_vars["savefreq"] = benchmark['parameters']['savefreq']

    else:
        pass

    # gpu environment variables if needed
    if backend['gpu_omp']:
        make_cmd.append(f"openmp_offload_libs={env['openmp_offload_libs']}")
        
        make_template_vars["openmp_offload_libs"] = env['openmp_offload_libs']

    # fpga environment variables if needed
    if backend['fpga']:
        make_cmd.append(f"xilinx_xrt={env['xilinx_xrt']}")
        make_cmd.append(f"xilinx_hls={env['xilinx_hls']}")
        make_cmd.append(f"fpga_board={benchmark['fpga_board']}")
        make_cmd.append(f"kernel_frequency={benchmark['kernel_frequency']}")

        make_template_vars["xilinx_xrt"] = env['xilinx_xrt']
        make_template_vars["xilinx_hls"] = env['xilinx_hls']
        make_template_vars["fpga_board"] = benchmark['fpga_board']
        make_template_vars["kernel_frequency"] = benchmark['kernel_frequency']

    # benchmark code resources
    if backend['cpu']:
        make_cmd.append(f"cpu_sources={' '.join(cpu_sources_with_prefix)}")
        make_template_vars["cpu_sources"] = ' '.join(cpu_sources_with_prefix)
    if backend['gpu_omp']:
        make_cmd.append(f"gpu_sources={' '.join(gpu_sources_with_prefix)}")
        make_template_vars["gpu_sources"] = ' '.join(gpu_sources_with_prefix)
    if backend['gpu_cuda']:
        make_cmd.append(f"gpu_sources={' '.join(gpu_sources_with_prefix)}")
        make_template_vars["gpu_sources"] = ' '.join(gpu_sources_with_prefix)
    if backend['gpu_acc']:
        make_cmd.append(f"gpu_sources={' '.join(gpu_sources_with_prefix)}")
        make_template_vars["gpu_sources"] = ' '.join(gpu_sources_with_prefix)
    if backend['fpga']:
        make_cmd.append(f"fpga_sources={' '.join(fpga_sources_with_prefix)}")
        make_template_vars["fpga_sources"] = ' '.join(fpga_sources_with_prefix)

    # write the makefile
    with open(f"{proj_path}/Makefile", 'w') as f:
        f.write(makefile_template.render(make_template_vars))

    if action == "build":
        if options == 'fpga_compile':
            # with open(log_file, 'a') as f:
            #     subprocess.run(make_cmd + ["fpga_compile"], stdout=f, stderr=f, check=True)
            subprocess.run(make_cmd + ["fpga_compile"], check=True)
        elif options == 'fpga_link':
            # with open(log_file, 'a') as f:
            #     subprocess.run(make_cmd + ["fpga_link"], stdout=f, stderr=f, check=True)
            subprocess.run(make_cmd + ["fpga_link"], check=True)
        elif options == 'fpga_all':
            # with open(log_file, 'a') as f:
            #     subprocess.run(make_cmd + ["fpga_all"], stdout=f, stderr=f, check=True)
            subprocess.run(make_cmd + ["fpga_all"], check=True)
        else:
            # with open(log_file, 'a') as f:
            #     subprocess.run(make_cmd + ["all"], stdout=f, stderr=f, check=True)
            subprocess.run(make_cmd + ["all"], check=True)
    elif action == "clean":
        # with open(log_file, 'a') as f:
            # subprocess.run(make_cmd + ["clean"], stdout=f, stderr=f, check=True)
        subprocess.run(make_cmd + ["clean"], check=True)
    elif action == "clean_all":
        # with open(log_file, 'a') as f:
        #     subprocess.run(make_cmd + ["clean_all"], stdout=f, stderr=f, check=True)
        subprocess.run(make_cmd + ["clean_all"], check=True)
    elif action == "clean_fpga":
        # with open(log_file, 'a') as f:
        #     subprocess.run(make_cmd + ["clean_fpga"], stdout=f, stderr=f, check=True)
        subprocess.run(make_cmd + ["clean_fpga"], check=True)
    elif action == "run":
        with open(log_file, 'a') as f:
            subprocess.run(make_cmd + ["run"], stdout=f, stderr=f, check=True)  
    else:
        pass

def run_heterobench(args, proj_config, env_config, log_file):
    # get the benchmark configuration
    # benchmark = next((b for b in proj_config["benchmarks"] if b["name"] == args.benchmark), None)
    # find benchmark in the "name" or in the "abbreviation" field
    benchmark = next((b for b in proj_config["benchmarks"] if b["name"] == args.benchmark or b["abbreviation"] == args.benchmark), None)

    if not benchmark:
        print(f"Benchmark '{args.benchmark}' not found.")
        return

    # get the environment configuration
    # has_cpu_impl = benchmark['cpu_sources'] != []
    # has_gpu_impl = benchmark['gpu_sources'] != []
    # has_fpga_impl = benchmark['fpga_sources'] != []

    has_cpu_impl = args.backend == 'cpu'
    has_gpu_omp_impl = args.backend == 'gpu_omp' or args.backend == 'hetero'
    has_gpu_cuda_impl = args.backend == 'gpu_cuda'
    has_gpu_acc_impl = args.backend == 'gpu_acc'
    has_fpga_impl = args.backend == 'fpga' or args.backend == 'hetero'
    python_impl = args.backend == 'python'
    numba_impl = args.backend == 'numba'
    hetero_impl = args.backend == 'hetero'

    backend = {
        "cpu": has_cpu_impl,
        "gpu_omp": has_gpu_omp_impl,
        "gpu_cuda": has_gpu_cuda_impl,
        "gpu_acc": has_gpu_acc_impl,
        "fpga": has_fpga_impl,
        "python": python_impl,
        "numba": numba_impl,
        "hetero": hetero_impl
    }

    has_gpu_impl = has_gpu_omp_impl or has_gpu_cuda_impl or has_gpu_acc_impl
    has_acc_impl =  has_fpga_impl or has_gpu_omp_impl or has_gpu_cuda_impl or has_gpu_acc_impl

    if args.options == 'serial' and has_acc_impl:
        raise ValueError(f"The benchmark has " + ('GPU' if has_gpu_impl else 'FPGA') + \
                         " implementation. Please DO NOT use 'serial' option.")
    
    if args.options == 'fpga_compile' and not has_fpga_impl:
        raise ValueError(f"The benchmark does not have FPGA implementation. Please DO NOT use 'fpga_compile' option.")
    
    if args.options == 'fpga_link' and not has_fpga_impl:
        raise ValueError(f"The benchmark does not have FPGA implementation. Please DO NOT use 'fpga_link' option.")
    
    if args.options == 'fpga_all' and not has_fpga_impl:
        raise ValueError(f"The benchmark does not have FPGA implementation. Please DO NOT use 'fpga_all' option.")

    gpu_arch = None
    
    if has_gpu_impl:
        gpu_brand = benchmark["gpu_brand"]
        gpu_arch = benchmark["gpu_arch"]
        if has_gpu_cuda_impl and gpu_brand != "nvidia":
            raise ValueError(f"The benchmark has GPU CUDA implementation but the GPU brand is not NVIDIA.")
        # if gpu_arch is not specified, get it automatically
        if gpu_arch == "":
            gpu_arch = get_gpu_arch(gpu_brand)

        if gpu_brand == "nvidia":
            if has_gpu_omp_impl:
                choose_env = env_config["environments"]["nvidia_nvc++"]
            elif has_gpu_cuda_impl:
                choose_env = env_config["environments"]["nvidia_cuda"]
            elif has_gpu_acc_impl:
                choose_env = env_config["environments"]["nvidia_acc"]
            else:
                choose_env = env_config["environments"]["nvidia_clang++"]
        elif gpu_brand == "amd":
            choose_env = env_config["environments"]["amd"]
        elif gpu_brand == "intel":
            choose_env = env_config["environments"]["intel"]
            
        # for intel gpu, we don't need -march parameter
        if gpu_brand == "intel":
            openmp_offload_libs = f'{choose_env["openmp_offload_libs"]}'
            lomptarget = "-lomptarget"
        elif gpu_brand == "nvidia" and has_gpu_omp_impl:
            openmp_offload_libs = f'{choose_env["openmp_offload_libs"]}'
            lomptarget = ""
        else:
            openmp_offload_libs = f'{choose_env["openmp_offload_libs"]} -march={gpu_arch}'
            lomptarget = "-lomptarget"
        
    elif has_fpga_impl:
        choose_env = env_config["environments"]["fpga"]
        openmp_offload_libs = f'{choose_env["openmp_offload_libs"]}'
        lomptarget = "-lomptarget"
    else:
        choose_env = env_config["environments"]["standard_c++"]
        openmp_offload_libs = f'{choose_env["openmp_offload_libs"]}'
        lomptarget = "-lomptarget"

    if has_gpu_cuda_impl:
        cuda_compiler = choose_env["cuda_compiler"]
    else:
        cuda_compiler = ""

    env = {
        "python_command": env_config["environments"]["python_command"],
        "compiler": choose_env["compiler"],
        "cuda_compiler": cuda_compiler,
        "cxxflags": choose_env["cxxflags"],
        "ldflags": choose_env["ldflags"],
        "openmp_libs": choose_env["openmp_libs"],
        "openmp_offload_libs": openmp_offload_libs,
        "lomptarget": lomptarget,
        "profiling_tool": choose_env["profiling_tool"],
        "ignore_warnings": choose_env["ignore_warnings"],
        "opencv_libs": env_config["environments"]["opencv"]["opencv_libs"],
        "xilinx_xrt": env_config["environments"]["xilinx"]["xilinx_xrt"],
        "xilinx_hls": env_config["environments"]["xilinx"]["xilinx_hls"]
    }

    if choose_env['ignore_warnings']:
        env['cxxflags'] += " -w"

    run_makefile(benchmark, backend, env, args.action, args.options, log_file)

def main():
    # command line arguments list
    parser = argparse.ArgumentParser(description='Run benchmarks with specified actions.')
    parser.add_argument('benchmark', type=str, help='The name of the benchmark to run.')
    parser.add_argument('action', choices=['build', 'run', 'clean', 'clean_all', 'clean_fpga'], \
                        help='The action to perform on the benchmark.')
    parser.add_argument('backend', choices=['cpu', 'gpu_omp', 'gpu_cuda', 'gpu_acc', 'fpga', 'python', 'numba', 'hetero'], \
                            help='The backend to build the benchmark.')
    parser.add_argument('options', nargs='?', default='parallel', \
                        choices=['parallel', 'serial', 'fpga_compile', 'fpga_link', 'fpga_all'], \
                        help='The options to build the benchmark.')

    args = parser.parse_args()

    running_benchmark = args.benchmark

    # setup logging if the action is executing the benchmark
    if args.action == "run":
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists('logs'):
            os.makedirs('logs')
        # log_file = f'logs_{args.benchmark}_{args.action}_{args.options}.log'
        log_file = f'logs/{args.benchmark}_{args.action}{f"_{args.options}" if args.action == "run" else ""}_{current_time}.log'
        # hw_type = f'' + ("_cpu" if has_cpu_impl else "") + ("_gpu" if has_gpu_impl else "") + ("_fpga" if has_fpga_impl else "")
        # log_file = f'logs{hw_type}_{args.benchmark}_{args.action}_{args.options}.log'
        setup_logging(log_file)
    else:
        log_file = None

    # load config files
    proj_config = load_config('config_json/proj_config.json')
    env_config = load_config('config_json/env_config.json')

    if args.benchmark == 'all':
        for benchmark in proj_config["benchmarks"]:
            args.benchmark = benchmark["name"]
            run_heterobench(args, proj_config, env_config, log_file)
    else:
        run_heterobench(args, proj_config, env_config, log_file)

    print(f'Finished {args.action} benchmark: {running_benchmark}')
    if log_file != None:
        print(f'Check the result in the {log_file}')
        # print everything in the log file
        with open(log_file, 'r') as f:
            print(f.read())

if __name__ == "__main__":
    main()