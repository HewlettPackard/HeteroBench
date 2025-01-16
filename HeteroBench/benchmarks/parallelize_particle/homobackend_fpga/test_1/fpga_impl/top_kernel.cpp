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
 
#include "include/fpga_impl.h"
using namespace std;

inline int Min(int a, int b) { return a < b ? a : b; }
inline int Max(int a, int b) { return a > b ? a : b; }

void apply_force(particle_t& particle, particle_t& neighbor)
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

int grid_coord(double c)
{
    return (int)floor(c / cutoff);
}

int grid_coord_flat(int size, double x, double y)
{
    return grid_coord(x) * size + grid_coord(y);
}


void grid_add_static(linkedlist_static grad_static[gridsize2], particle_t* p)
{
#pragma HLS INLINE
    int gridCoord = grid_coord_flat(gridsize, p->x, p->y);

    grad_static[gridCoord].value[grad_static[gridCoord].index_now] = *p;
    grad_static[gridCoord].index_now++;

}

//
//  integrate the ODE
//
void grid_move(particle_t& p, double size)
{
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    //
    //  bounce from walls
    //
    while (p.x < 0 || p.x > size)
    {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size)
    {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
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
#pragma HLS PIPELINE II=1
                linkedlist_static* curr = &grid_static[x * gridsize + y];
                int t = curr->index_now;
                while (t != 0)
                {
                    apply_force(particles[i], (curr->value[t]));
                    t--;
                }
            }
        }
    }
}

void move_particles_static(particle_t* particles, int n, linkedlist_static grid[gridsize2])
{
    for (int i = 0; i < n; i++)
    {
#pragma HLS PIPELINE II=1
        int gc = grid_coord_flat(gridsize, particles[i].x, particles[i].y);

        grid_move(particles[i], gridsize);

        // Re-add the particle if it has changed grid position
        if (gc != grid_coord_flat(gridsize, particles[i].x, particles[i].y))
        {
            grid_add_static(grid, &particles[i]);
        }
    }
}


void top_kernel(particle_t particles[NPARTICLES], linkedlist_static grid_static[gridsize2]) {

#pragma HLS INTERFACE m_axi bundle=particle_in port=particles max_read_burst_length=16 num_read_outstanding=64
#pragma HLS INTERFACE s_axilite bundle=control port=particles
#pragma HLS INTERFACE m_axi bundle=grid_static_out port=grid_static max_write_burst_length=16 num_write_outstanding=64

#pragma HLS INTERFACE s_axilite bundle=control port=grid_static
#pragma HLS INTERFACE s_axilite bundle=control port=return

    for (int step = 0; step < NSTEPS; step++) {
        for (int i = 0; i < NPARTICLES; ++i) {
#pragma HLS PIPELINE II=1
            grid_add_static(grid_static, &particles[i]);
        }

        compute_forces_static(particles, NPARTICLES, grid_static);
        move_particles_static(particles, NPARTICLES, grid_static);
    }

}
