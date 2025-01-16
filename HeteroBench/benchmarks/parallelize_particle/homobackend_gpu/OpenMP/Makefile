CXX := nvc++
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := 
OPENMP_OFFLOAD_LIBS := -mp=gpu

# nsteps
NSTEPS := 1

# num of particles
NPARTICLES := 1000000

# num of save frequency
SAVEFREQ := 10

# Project name
TARGET_EXEC := ppc_sw

# Source files
SOURCES := main.cpp common.cpp

# MACRO DEFINITION
MACRO := -DNSTEPS=$(NSTEPS) -DNPARTICLES=$(NPARTICLES) -DSAVEFREQ=$(SAVEFREQ)

# GPU source files
GPU_SOURCES := gpu_impl/compute_forces.cpp gpu_impl/grid.cpp gpu_impl/move_particles.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

# Include directories, also include input_data.h under current directory
INCLUDE_DIRS := gpu_impl/include .
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Link OpenMP target offload library
OPENMP_OFFLOAD_LIB := 

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
	$(CXX) $(OPENMP_OFFLOAD_LIBS) $(LDFLAGS) -o $@ $^ $(OPENMP_OFFLOAD_LIB)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

common.o: common.cpp
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