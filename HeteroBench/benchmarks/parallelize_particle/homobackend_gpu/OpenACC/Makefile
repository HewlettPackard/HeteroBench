CXX := nvc++
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := -acc

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
GPU_SOURCES := acc_impl/compute_forces.cpp acc_impl/grid.cpp acc_impl/move_particles.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

# Include directories, also include input_data.h under current directory
INCLUDE_DIRS := acc_impl/include .
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

common.o: common.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile other GPU source files
acc_impl/%.o: acc_impl/%.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(LDFLAGS) $(INCLUDE_PARAMS) -c $< -o $@

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