CXX := nvc++
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := -acc

# input path
INPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/one_head_attention/input

# output path
OUTPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/one_head_attention/output

# iteration_times
ITERATIONS := 1
# batch_size
BATCH_SIZE := 8
# N
N := 1024
# d
D := 128

# Project name
TARGET_EXEC := oha_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS) \
	-DBATCH_SIZE=$(BATCH_SIZE) \
	-DN=$(N) \
	-DD=$(D)

# GPU Source files
GPU_SOURCES := acc_impl/transpose.cpp acc_impl/softmax.cpp acc_impl/matmul.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

INCLUDE_DIRS := acc_impl/include
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile GPU source file
acc_impl/%.o: acc_impl/%.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(LDFLAGS) $(INCLUDE_PARAMS)  -c $< -o $@

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