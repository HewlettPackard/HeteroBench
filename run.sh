# (C) Copyright [2024] Hewlett Packard Enterprise Development LP

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the Software),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

HAS_GPU=false
container_name="heterobench-container"
temp_name="heterobench-temp-container"

if [ "$HAS_GPU" = true ] ; then
    docker build -t --build-arg GPU_ENV_VAR=NVIDIA $container_name .
else
    docker build -t $container_name .
fi

docker create --name $temp_name ${container_name}
docker cp $temp_name:/workspace/HeteroBench/logs .
docker rm $temp_name

# if [ ! -d logs ]; then
#     mkdir logs
# fi
#
# run() {
#     filename=logs/$1_$(date +"%Y%m%d_%I%M%S").log
#     echo "Output for the next run is in the file - ${filename}."
#     echo "Running all benchmarks for $1 ..."
#     time docker run --gpus all -it heterobench-container python3 /workspace/HeteroBench/heterobench.py all run $1 > ${filename}
#     echo "Done."
#     echo
# }
#
# # RUN different benchmarks
# # run python
# # run numba
# run cpu
# run gpu_omp
# run gpu_acc
# run gpu_cuda
