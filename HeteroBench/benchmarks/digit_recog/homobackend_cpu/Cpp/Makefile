CXX := clang++
CXXFLAGS := -Wall -Wextra -pedantic -std=c++11 -fopenmp -w
LDFLAGS := -fopenmp
OPENMP_LIBS := -lgomp

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

# CPU Source files
CPU_SOURCES := cpu_impl/popcount.cpp cpu_impl/update_knn.cpp cpu_impl/knn_vote.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

CPU_OBJECTS := $(CPU_SOURCES:.cpp=.o)

# Include directories
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
	./$(TARGET_EXEC)

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