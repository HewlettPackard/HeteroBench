/*
 * (C) Copyright [2024] Hewlett Packard Enterprise Development LP
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the Software),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
 
#include "cpu_impl.h"
#include "common.h"
#include "omp.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <vector>

#define DEBUG 0
using namespace std;

linkedlist_static grid_static[gridsize2];
particle_t particles_dyn[NPARTICLES];
particle_t particles[NPARTICLES];
bool compare_particles(const particle_t& p1, const particle_t& p2) {
	const double epsilon = 1e-10; // Tolerance for floating-point comparison
	return std::fabs(p1.x - p2.x) < epsilon &&
		std::fabs(p1.y - p2.y) < epsilon &&
		std::fabs(p1.vx - p2.vx) < epsilon &&
		std::fabs(p1.vy - p2.vy) < epsilon &&
		std::fabs(p1.ax - p2.ax) < epsilon &&
		std::fabs(p1.ay - p2.ay) < epsilon;
}

bool compare_particle_arrays(particle_t* arr1, particle_t* arr2, int size) {
	for (int i = 0; i < size; ++i) {
		if (!compare_particles(arr1[i], arr2[i])) {
			return false;
		}
	}
	return true;
}

void parallelize_particle_simulation(int n, int nsteps, int savefreq, FILE* fsave)
{
	double size = SIZE;


	// Create a grid for optimizing the interactions
	int gridSize = GRID_SIZE; // TODO: Rounding errors?

	grid_t grid;
	grid_init(grid, gridSize);

	//Upper part no need to be reimplemented
	for (int i = 0; i < n; ++i)
	{
		grid_add(grid, &particles_dyn[i]);
	}

	for (int i = 0; i < grid.size * grid.size; i++) {
		if (grid.node_counts[i] > 1)
			printf("Grid coordinate %d: %d nodes\n", i, grid.node_counts[i]);
	}
	std::cout << "Running 1 warm up iteration ..." << std::endl;
	// 1 warm up iteration
	compute_forces(particles_dyn, n, grid);
	move_particles(particles_dyn, n, grid);


	std::cout << "Done" << std::endl;
	std::cout << "Running " << nsteps << " iterations ..." << std::endl;
	// Simulate a number of time steps
	double start_whole_time = omp_get_wtime();

	double start_iteration_time;
	double start_compute_forces_time = 0;
	double start_move_particles_time = 0;

	for (int step = 0; step < nsteps; step++)
	{
		// Compute forces
		start_iteration_time = omp_get_wtime();
		compute_forces(particles_dyn, n, grid);
		start_compute_forces_time += omp_get_wtime() - start_iteration_time;

		// Move particles
		start_iteration_time = omp_get_wtime();
		move_particles(particles_dyn, n, grid);
		start_move_particles_time += omp_get_wtime() - start_iteration_time;

		if (fsave && (step % savefreq) == 0)
			save(fsave, n, particles);
	}
	std::cout << "Done" << std::endl;

	// double end_whole_time = omp_get_wtime() - start_whole_time;
	// std::cout << "Total " << nsteps << " number of steps" << endl;
	// std::cout << "Single iteration time: " << (end_whole_time / nsteps) * 1000 << " ms" << endl;
	// std::cout << "Compute forces time: " << (start_compute_forces_time / nsteps) * 1000 << " ms" << endl;
	// std::cout << "Move particles time: " << (start_move_particles_time / nsteps) * 1000 << " ms" << endl;

	grid_clear(grid);


	if (fsave)
	{
		fclose(fsave);
	}
}
void parallelize_particle_simulation_static(int n, int nsteps, int savefreq, FILE* fsave)
{
	std::cout << "Running 1 warm up iteration ..." << std::endl;
	for (int i = 0; i < n; ++i)
	{
		grid_add_static(grid_static, &particles[i]);
	}

	// 1 warm up iteration
	compute_forces_static(particles, n, grid_static);
	move_particles_static(particles, n, grid_static);

	std::cout << "Done" << std::endl;
	std::cout << "Running " << nsteps << " iterations ..." << std::endl;
	// Simulate a number of time steps
	double start_whole_time = omp_get_wtime();

	double start_iteration_time;
	double start_compute_forces_time = 0;
	double start_move_particles_time = 0;

	for (int step = 0; step < nsteps; step++)
	{
		// Compute forces
		start_iteration_time = omp_get_wtime();
		compute_forces_static(particles, n, grid_static);
		start_compute_forces_time += omp_get_wtime() - start_iteration_time;

		// Move particles
		start_iteration_time = omp_get_wtime();
		move_particles_static(particles, n, grid_static);
		start_move_particles_time += omp_get_wtime() - start_iteration_time;


		if (fsave && (step % savefreq) == 0)
			save(fsave, n, particles);
	}
	double end_whole_time = omp_get_wtime() - start_whole_time;
	std::cout << "Done" << std::endl;
	std::cout << "Total " << nsteps << " number of steps" << endl;
	std::cout << "Single iteration time: " << (end_whole_time / nsteps) * 1000 << " ms" << endl;
	std::cout << "Compute forces time: " << (start_compute_forces_time / nsteps) * 1000 << " ms" << endl;
	std::cout << "Move particles time: " << (start_move_particles_time / nsteps) * 1000 << " ms" << endl;


	if (fsave)
	{
		fclose(fsave);
	}
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{
	std::cout << "=======================================" << std::endl;
	std::cout << "Running parallelize_particle benchmark C++ Serial" << std::endl;
	std::cout << "=======================================" << std::endl;

	char *savename = read_string( argc, argv, "-o", NULL );
	
	FILE *fsave = savename ? fopen( savename, "w" ) : NULL ;

	int n = NPARTICLES;
	int nsteps = NSTEPS;
	int savefreq = SAVEFREQ;

	double size = set_size(n);
	init_particles(n, particles_dyn, particles);
	parallelize_particle_simulation(n, nsteps, savefreq, fsave);
	parallelize_particle_simulation_static(n, nsteps, savefreq, fsave);

	bool arrays_equal = compare_particle_arrays(particles_dyn, particles, NPARTICLES);

	if (arrays_equal) {
		std::cout << "The particle arrays are equal." << std::endl;
	}
	else {
		std::cout << "The particle arrays are not equal." << std::endl;
	}
	return 0;
}
