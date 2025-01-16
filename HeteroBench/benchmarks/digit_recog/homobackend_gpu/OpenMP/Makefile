CXX := nvc++
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := 
OPENMP_OFFLOAD_LIBS := -mp=gpu

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
GPU_SOURCES := gpu_impl/popcount.cpp gpu_impl/update_knn.cpp gpu_impl/knn_vote.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := gpu_impl/include
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