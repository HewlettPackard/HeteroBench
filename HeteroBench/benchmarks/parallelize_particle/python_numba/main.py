"""
(C) Copyright [2024] Hewlett Packard Enterprise Development LP

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the Software),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import math
import numpy as np
import random
import time
import sys
from numba import jit

if len(sys.argv) == 4:
    nsteps = int(sys.argv[1])
    nparticles = int(sys.argv[2])
    savefreq = int(sys.argv[3])
else:
    print("Did not provide command line arguments correctly. Using default values")
    print("Usage: python main.py <nsteps> <nparticles> <savefreq>")
    nsteps = 20
    nparticles = 1000000
    savefreq = 10

# Tuned constants
DENSITY = 0.0005
MASS = 0.01
CUTOFF = 0.01
MIN_R = CUTOFF / 100
DT = 0.0005

SIZE = math.sqrt(DENSITY * nparticles)
GRID_SIZE = int((SIZE / CUTOFF) + 1)
GRIDSIZE = GRID_SIZE
GRIDSIZE2 = GRID_SIZE * GRID_SIZE
MAX_LINKEDLIST_SIZE = 10

# Particle data structure using NumPy arrays
dtype = np.dtype([('x', np.float64), ('y', np.float64), ('vx', np.float64), ('vy', np.float64), ('ax', np.float64), ('ay', np.float64)])
particles_dyn = np.zeros(nparticles, dtype=dtype)
particles_static = np.zeros(nparticles, dtype=dtype)

# Grid structure using NumPy arrays
grid_static = np.zeros((GRIDSIZE2, MAX_LINKEDLIST_SIZE), dtype=dtype)
grid_static_count = np.zeros(GRIDSIZE2, dtype=np.int32)

@jit(nopython=True, cache=True)
def grid_coord(c):
    return int(math.floor(c / CUTOFF))

@jit(nopython=True, cache=True)
def grid_coord_flat(size, x, y):
    return grid_coord(x) * size + grid_coord(y)

@jit(nopython=True, cache=True)
def grid_add_static(grid_static, grid_static_count, particle):
    grid_coord = grid_coord_flat(GRIDSIZE, particle['x'], particle['y'])
    count = grid_static_count[grid_coord]
    grid_static[grid_coord, count] = particle
    grid_static_count[grid_coord] += 1

@jit(nopython=True, cache=True)
def grid_move(particle, size):
    particle['vx'] += particle['ax'] * DT
    particle['vy'] += particle['ay'] * DT
    particle['x'] += particle['vx'] * DT
    particle['y'] += particle['vy'] * DT

    while particle['x'] < 0 or particle['x'] > size:
        particle['x'] = -particle['x'] if particle['x'] < 0 else 2 * size - particle['x']
        particle['vx'] = -particle['vx']

    while particle['y'] < 0 or particle['y'] > size:
        particle['y'] = -particle['y'] if particle['y'] < 0 else 2 * size - particle['y']
        particle['vy'] = -particle['vy']

@jit(nopython=True, cache=True)
def apply_force(particle, neighbor):
    dx = neighbor['x'] - particle['x']
    dy = neighbor['y'] - particle['y']
    r2 = dx * dx + dy * dy
    if r2 > CUTOFF * CUTOFF:
        return
    r2 = max(r2, MIN_R * MIN_R)
    r = math.sqrt(r2)
    coef = (1 - CUTOFF / r) / r2 / MASS
    particle['ax'] += coef * dx
    particle['ay'] += coef * dy

@jit(nopython=True, cache=True)
def compute_forces_static(particles, grid_static, grid_static_count):
    for i in range(particles.shape[0]):
        particle = particles[i]
        particle['ax'] = particle['ay'] = 0
        gx = grid_coord(particle['x'])
        gy = grid_coord(particle['y'])
        for x in range(max(gx - 1, 0), min(gx + 1, GRIDSIZE - 1) + 1):
            for y in range(max(gy - 1, 0), min(gy + 1, GRIDSIZE - 1) + 1):
                grid_index = x * GRIDSIZE + y
                t = grid_static_count[grid_index]
                while t > 0:
                    apply_force(particle, grid_static[grid_index, t-1])
                    t -= 1

@jit(nopython=True, cache=True)
def move_particles_static(particles, grid_static, grid_static_count):
    new_grid_static = np.zeros_like(grid_static)
    new_grid_static_count = np.zeros_like(grid_static_count)
    for i in range(particles.shape[0]):
        particle = particles[i]
        old_gc = grid_coord_flat(GRIDSIZE, particle['x'], particle['y'])
        grid_move(particle, GRIDSIZE)
        new_gc = grid_coord_flat(GRIDSIZE, particle['x'], particle['y'])
        if old_gc != new_gc:
            grid_add_static(new_grid_static, new_grid_static_count, particle)
    return new_grid_static, new_grid_static_count

@jit(nopython=True, cache=True)
def compare_particles(p1, p2):
    epsilon = 1e-3
    return (abs(p1['x'] - p2['x']) < epsilon and abs(p1['y'] - p2['y']) < epsilon and
            abs(p1['vx'] - p2['vx']) < epsilon and abs(p1['vy'] - p2['vy']) < epsilon and
            abs(p1['ax'] - p2['ax']) < epsilon and abs(p1['ay'] < epsilon))

def init_particles(n, particles0, particles1):
    random.seed(time.time())
    sx = math.ceil(math.sqrt(n))
    sy = (n + sx - 1) // sx

    shuffle = list(range(n))
    random.shuffle(shuffle)

    for i in range(n):
        k = shuffle[i]
        particles0[i]['x'] = SIZE * (1 + (k % sx)) / (1 + sx)
        particles0[i]['y'] = SIZE * (1 + (k // sx)) / (1 + sy)
        particles0[i]['vx'] = random.random() * 2 - 1
        particles0[i]['vy'] = random.random() * 2 - 1
        particles1[i]['x'] = particles0[i]['x']
        particles1[i]['y'] = particles0[i]['y']
        particles1[i]['vx'] = particles0[i]['vx']
        particles1[i]['vy'] = particles0[i]['vy']

def parallelize_particle_simulation_static(n, nsteps, savefreq):
    global particles_static, grid_static, grid_static_count

    grid_static.fill(0)
    grid_static_count.fill(0)

    for i in range(n):
        grid_add_static(grid_static, grid_static_count, particles_static[i])

    print("Running 1 warm-up iteration ...")
    compute_forces_static(particles_static, grid_static, grid_static_count)
    grid_static, grid_static_count = move_particles_static(particles_static, grid_static, grid_static_count)
    print("Done")

    start_compute_forces_time = 0
    start_move_particles_time = 0

    print("Running " + str(nsteps) + " iteration ...")
    start_whole_time = time.time()
    for step in range(nsteps):
        start_iteration_time = time.time()
        compute_forces_static(particles_static, grid_static, grid_static_count)
        start_compute_forces_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        grid_static, grid_static_count = move_particles_static(particles_static, grid_static, grid_static_count)
        start_move_particles_time += time.time() - start_iteration_time
    end_whole_time = time.time() - start_whole_time
    print("Done")
    print("Static simulation: Total time for {} steps: {:.4f} seconds".format(nsteps, end_whole_time))
    print("Single iteration time: {:.4f} ms".format((end_whole_time / nsteps) * 1000))
    print("Compute forces time: {:.4f} ms".format((start_compute_forces_time / nsteps) * 1000))
    print("Move particles time: {:.4f} ms".format((start_move_particles_time / nsteps) * 1000))

def main():
    print("=======================================")
    print("Running parallelize_particle benchmark Python Numba")
    print("=======================================")
    global particles_dyn, particles_static

    init_particles(nparticles, particles_dyn, particles_static)

    parallelize_particle_simulation_static(nparticles, nsteps, savefreq)

if __name__ == "__main__":
    main()
