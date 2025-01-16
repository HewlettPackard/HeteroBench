CXX := clang++
CXXFLAGS := -Wall -Wextra -pedantic -std=c++11 -fopenmp -w
LDFLAGS := -fopenmp
OPENMP_LIBS := -lgomp

# input path
INPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/convolutional_neural_network/input

# output path
OUTPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/convolutional_neural_network/output

# iteration_times
ITERATIONS := 1
# conv2d_stride
CONV2D_STRIDE := 1
# conv2d_padding
CONV2D_PADDING := 1
# conv2d_bias
CONV2D_BIAS := 0.1
# pooling_size
POOLING_SIZE := 2
# pooling_stride
POOLING_STRIDE := 2
# input_size_h
INPUT_SIZE_H := 1024
# input_size_w
INPUT_SIZE_W := 2048
# conv_kernel_size_h
CONV_KERNEL_SIZE_H := 3
# conv_kernel_size_w
CONV_KERNEL_SIZE_W := 3
# full_connect_layer_size_w
FULL_CONNECT_LAYER_SIZE_W := 2048

# Project name
TARGET_EXEC := cnn_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS) \
	-DCONV2D_STRIDE=$(CONV2D_STRIDE) \
	-DCONV2D_PADDING=$(CONV2D_PADDING) \
	-DCONV2D_BIAS=$(CONV2D_BIAS) \
	-DPOOLING_SIZE=$(POOLING_SIZE) \
	-DPOOLING_STRIDE=$(POOLING_STRIDE) \
	-DINPUT_SIZE_H=$(INPUT_SIZE_H) \
	-DINPUT_SIZE_W=$(INPUT_SIZE_W) \
	-DCONV_KERNEL_SIZE_H=$(CONV_KERNEL_SIZE_H) \
	-DCONV_KERNEL_SIZE_W=$(CONV_KERNEL_SIZE_W) \
	-DFULL_CONNECT_LAYER_SIZE_W=$(FULL_CONNECT_LAYER_SIZE_W)

# CPU Source files
CPU_SOURCES := cpu_impl/conv2d.cpp cpu_impl/relu.cpp cpu_impl/max_pooling.cpp cpu_impl/pad_input.cpp cpu_impl/dot_add.cpp cpu_impl/softmax.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

CPU_OBJECTS := $(CPU_SOURCES:.cpp=.o)

INCLUDE_DIRS := cpu_impl/include
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(CPU_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(OPENMP_LIBS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile other CPU source files
cpu_impl/%.o: cpu_impl/%.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# Run
run:
	./$(TARGET_EXEC) $(INPUT_PATH) $(OUTPUT_PATH)

# Clean up
clean:
	rm -f $(TARGET_EXEC) $(OBJECTS) $(CPU_OBJECTS)

.PHONY: all clean run help

help:
	@echo "Usage: make [all|clean|run|help]"
	@echo "  all:        Build the project"
	@echo "  clean:      Clean up the project"
	@echo "  run:        Run the project"
	@echo "  help:       Show this help message"