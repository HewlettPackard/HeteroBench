CXX := nvc++
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := 
OPENMP_OFFLOAD_LIBS := -mp=gpu

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
GPU_SOURCES := gpu_impl/dotProduct.cpp gpu_impl/Sigmoid.cpp gpu_impl/computeGradient.cpp gpu_impl/updateParameter.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := gpu_impl/include $(IMAGELIB_DIR)
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Link OpenMP target offload library
OPENMP_OFFLOAD_LIB := 

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
	$(CXX) $(OPENMP_OFFLOAD_LIBS) $(LDFLAGS) -o $@ $^ $(OPENMP_OFFLOAD_LIB)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile other GPU source files
gpu_impl/%.o: gpu_impl/%.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(OPENMP_OFFLOAD_LIBS) $(INCLUDE_PARAMS) -c $< -o $@

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