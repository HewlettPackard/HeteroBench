CXX := nvc++
NVCC := nvcc
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := -cuda

# num of test
NUM_TEST := 2000

NUM_TRAINING := 18000
CLASS_SIZE := 1800
DIGIT_WIDTH := 4
K_CONST := 3
PARA_FACTOR := 40

# Project name
TARGET_EXEC := dgr_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DNUM_TEST=$(NUM_TEST) -DNUM_TRAINING=$(NUM_TRAINING) -DCLASS_SIZE=$(CLASS_SIZE) -DDIGIT_WIDTH=$(DIGIT_WIDTH) -DK_CONST=$(K_CONST) -DPARA_FACTOR=$(PARA_FACTOR)

# GPU Source files
GPU_SOURCES := cuda_impl/popcount.cu cuda_impl/update_knn.cu cuda_impl/knn_vote.cu

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cu=.o)

# Include directories
INCLUDE_DIRS := cuda_impl/include
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
	$(CXX)  $(LDFLAGS) -o $@ $^

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile GPU source file
cuda_impl/%.o: cuda_impl/%.cu
	$(NVCC) -lineinfo $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# Run
run:
	./$(TARGET_EXEC)

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