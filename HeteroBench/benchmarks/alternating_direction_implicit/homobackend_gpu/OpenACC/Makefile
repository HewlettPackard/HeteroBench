CXX := nvc++
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := -acc

# iterations
ITERATIONS := 1

# TSTEPS, N
TSTEPS := 50
N := 1024

# Project name
TARGET_EXEC := krnl_adi_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS) -DTSTEPS=$(TSTEPS) -DN=$(N)

# GPU source files
GPU_SOURCES := acc_impl/init_array.cpp acc_impl/kernel_adi.cpp

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

# To compile other GPU source files
acc_impl/%.o: acc_impl/%.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(LDFLAGS) $(INCLUDE_PARAMS)  -c $< -o $@

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