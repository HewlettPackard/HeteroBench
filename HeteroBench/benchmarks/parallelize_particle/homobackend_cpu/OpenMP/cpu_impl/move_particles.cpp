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

using namespace std;

void move_particles_static(particle_t* particles, int n, linkedlist_static grid[gridsize2])
{
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		int gc = grid_coord_flat(gridsize, particles[i].x, particles[i].y);

		grid_move(particles[i], gridsize);

		// Re-add the particle if it has changed grid position
		if (gc != grid_coord_flat(gridsize, particles[i].x, particles[i].y))
		{
			// if (! grid_remove(grid, &particles[i], gc))
			// {
			// 	fprintf(stdout, "Error: Failed to remove particle '%p'. Code must be faulty. Blame source writer.\n", &particles[i]);
			// 	exit(3);
			// }
			grid_add_static(grid, &particles[i]);
		}
	}
}