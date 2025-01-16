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

inline int Min(int a, int b) { return a < b ? a : b; }
inline int Max(int a, int b) { return a > b ? a : b; }

//
//  interact two particles
//
void apply_force_static(particle_t& particle, particle_t& neighbor)
{
	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
	double r2 = dx * dx + dy * dy;
	if (r2 > cutoff * cutoff)
		return;
	r2 = fmax(r2, min_r * min_r);
	double r = sqrt(r2);

	//
	//  very simple short-range repulsive force
	//
	double coef = (1 - cutoff / r) / r2 / mass;
	particle.ax += coef * dx;
	particle.ay += coef * dy;
}

void compute_forces_static(particle_t* particles, int n, linkedlist_static grid_static[gridsize2])
{
	for (int i = 0; i < n; i++)
	{
		// Reset acceleration
		particles[i].ax = particles[i].ay = 0;

		// Use the grid to traverse neighbours
		int gx = grid_coord(particles[i].x);
		int gy = grid_coord(particles[i].y);

		for (int x = Max(gx - 1, 0); x <= Min(gx + 1, gridsize - 1); x++)
		{
			for (int y = Max(gy - 1, 0); y <= Min(gy + 1, gridsize - 1); y++)
			{
				linkedlist_static* curr = &grid_static[x * gridsize + y];
				int t = curr->index_now;
				while (t != 0)
				{
					apply_force_static(particles[i], (curr->value[t]));
					t--;
				}
			}
		}
	}
}