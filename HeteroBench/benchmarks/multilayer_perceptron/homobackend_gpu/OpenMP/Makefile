CXX := nvc++
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := 
OPENMP_OFFLOAD_LIBS := -mp=gpu

# input path
INPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/multilayer_perceptron/input

# output path
OUTPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/multilayer_perceptron/output

# iteration_times
ITERATIONS := 1
# layer0_h1
LAYER0_H1 := 3072
# layer0_w1
LAYER0_W1 := 2048
# layer0_w2
LAYER0_W2 := 4096
# layer1_h1
LAYER1_H1 := 3072
# layer1_w1
LAYER1_W1 := 4096
# layer1_w2
LAYER1_W2 := 4096
# layer2_h1
LAYER2_H1 := 3072
# layer2_w1
LAYER2_W1 := 4096
# layer2_w2
LAYER2_W2 := 4096
# layer3_h1
LAYER3_H1 := 3072
# layer3_w1
LAYER3_W1 := 4096
# layer3_w2
LAYER3_W2 := 1024

# Project name
TARGET_EXEC := mlp_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS) \
	-DL0_H1=$(LAYER0_H1) \
	-DL0_W1=$(LAYER0_W1) \
	-DL0_W2=$(LAYER0_W2) \
	-DL1_H1=$(LAYER1_H1) \
	-DL1_W1=$(LAYER1_W1) \
	-DL1_W2=$(LAYER1_W2) \
	-DL2_H1=$(LAYER2_H1) \
	-DL2_W1=$(LAYER2_W1) \
	-DL2_W2=$(LAYER2_W2) \
	-DL3_H1=$(LAYER3_H1) \
	-DL3_W1=$(LAYER3_W1) \
	-DL3_W2=$(LAYER3_W2)

GPU_SOURCES := gpu_impl/dot_add.cpp gpu_impl/sigmoid.cpp gpu_impl/softmax.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

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
	./$(TARGET_EXEC) $(INPUT_PATH) $(OUTPUT_PATH)

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