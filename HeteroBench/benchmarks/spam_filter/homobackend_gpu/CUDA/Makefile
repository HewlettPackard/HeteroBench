CXX := nvc++
NVCC := nvcc
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := -cuda

# input path
INPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/spam_filter/data

# iterations
ITERATIONS := 1

# num of features
NUM_FEATURES := 10000

# num of samples
NUM_SAMPLES := 5000

# num of training
NUM_TRAINING := 4500

# num of testing
NUM_TESTING := 500

# step size
STEP_SIZE := 60000

# num of epochs
NUM_EPOCHS := 5

# Project name
TARGET_EXEC := spf_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS) -DNUM_FEATURES=$(NUM_FEATURES) -DNUM_SAMPLES=$(NUM_SAMPLES) -DNUM_TRAINING=$(NUM_TRAINING) -DNUM_TESTING=$(NUM_TESTING) -DSTEP_SIZE=$(STEP_SIZE) -DNUM_EPOCHS=$(NUM_EPOCHS)

# GPU Source files
GPU_SOURCES := cuda_impl/dotProduct.cu cuda_impl/Sigmoid.cu cuda_impl/computeGradient.cu cuda_impl/updateParameter.cu

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cu=.o)

# Include directories
INCLUDE_DIRS := cuda_impl/include $(IMAGELIB_DIR)
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
	$(CXX)  $(LDFLAGS) -o $@ $^

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

cuda_impl/Sigmoid.o: cuda_impl/Sigmoid.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile GPU source file
cuda_impl/%.o: cuda_impl/%.cu
	$(NVCC) -lineinfo  $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# Run
run:
	./$(TARGET_EXEC) -p $(INPUT_PATH)

# Clean up
clean:
	rm -f $(TARGET_EXEC) $(OBJECTS) $(GPU_OBJECTS)

.PHONY: all clean run help

help:
	@echo "Usage: make [all|clean|run|help]"
	@echo "  all:        Build the project"
	@echo "  clean:      Clean up the project"
	@echo "  run:        Run the project"
	@echo "  help:       Show this help message"