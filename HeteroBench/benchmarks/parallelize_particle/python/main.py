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

# Particle data structure
class Particle:
    def __init__(self, x=0, y=0, vx=0, vy=0, ax=0, ay=0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay

# Global variables for particles
particles_dyn = [Particle() for _ in range(nparticles)]
particles_static = [Particle() for _ in range(nparticles)]

class LinkedListNode:
    def __init__(self, particle=None, next=None):
        self.particle = particle
        self.next = next

class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = [None] * (size * size)
        self.node_counts = np.zeros(size * size, dtype=int)

class LinkedListStatic:
    def __init__(self):
        self.value = [None] * MAX_LINKEDLIST_SIZE
        self.index_now = 0

def grid_coord(c):
    return int(math.floor(c / CUTOFF))

def grid_coord_flat(size, x, y):
    return grid_coord(x) * size + grid_coord(y)

def grid_init(grid, size):
    grid.size = size
    grid.grid = [None] * (size * size)
    grid.node_counts = np.zeros(size * size, dtype=int)

def grid_add(grid, particle):
    grid_coord = grid_coord_flat(grid.size, particle.x, particle.y)
    new_node = LinkedListNode(particle, grid.grid[grid_coord])
    grid.grid[grid_coord] = new_node

def grid_add_static(grid_static, particle):
    grid_coord = grid_coord_flat(GRIDSIZE, particle.x, particle.y)
    grid_static[grid_coord].value[grid_static[grid_coord].index_now] = particle
    grid_static[grid_coord].index_now += 1

def grid_move(particle, size):
    particle.vx += particle.ax * DT
    particle.vy += particle.ay * DT
    particle.x += particle.vx * DT
    particle.y += particle.vy * DT

    while particle.x < 0 or particle.x > size:
        particle.x = -particle.x if particle.x < 0 else 2 * size - particle.x
        particle.vx = -particle.vx

    while particle.y < 0 or particle.y > size:
        particle.y = -particle.y if particle.y < 0 else 2 * size - particle.y
        particle.vy = -particle.vy


def apply_force(particle, neighbor):
    dx = neighbor.x - particle.x
    dy = neighbor.y - particle.y
    r2 = dx * dx + dy * dy
    if r2 > CUTOFF * CUTOFF:
        return
    r2 = max(r2, MIN_R * MIN_R)
    r = math.sqrt(r2)
    coef = (1 - CUTOFF / r) / r2 / MASS
    particle.ax += coef * dx
    particle.ay += coef * dy

def compute_forces(particles, grid):
    for particle in particles:
        particle.ax = particle.ay = 0
        gx = grid_coord(particle.x)
        gy = grid_coord(particle.y)
        for x in range(max(gx - 1, 0), min(gx + 1, grid.size - 1) + 1):
            for y in range(max(gy - 1, 0), min(gy + 1, grid.size - 1) + 1):
                node = grid.grid[x * grid.size + y]
                while node is not None:
                    apply_force(particle, node.particle)
                    node = node.next

def compute_forces_static(particles, grid_static):
    for particle in particles:
        particle.ax = particle.ay = 0
        gx = grid_coord(particle.x)
        gy = grid_coord(particle.y)
        for x in range(max(gx - 1, 0), min(gx + 1, GRIDSIZE - 1) + 1):
            for y in range(max(gy - 1, 0), min(gy + 1, GRIDSIZE - 1) + 1):
                cell = grid_static[x * GRIDSIZE + y]
                t = cell.index_now
                while t > 0:
                    apply_force(particle, cell.value[t-1])
                    t -= 1

def move_particles(particles, grid):
    for particle in particles:
        gc = grid_coord_flat(grid.size, particle.x, particle.y)
        grid_move(particle, grid.size)
        new_gc = grid_coord_flat(grid.size, particle.x, particle.y)
        if gc != new_gc:

            grid_add(grid, particle)

def move_particles_static(particles, grid_static):
    for particle in particles:
        gc = grid_coord_flat(GRIDSIZE, particle.x, particle.y)
        grid_move(particle, GRIDSIZE)
        new_gc = grid_coord_flat(GRIDSIZE, particle.x, particle.y)
        if gc != new_gc:
            grid_add_static(grid_static, particle)

def compare_particles(p1, p2):
    epsilon = 1e-3
    return (abs(p1.x - p2.x) < epsilon and abs(p1.y - p2.y) < epsilon and
            abs(p1.vx - p2.vx) < epsilon and abs(p1.vy - p2.vy) < epsilon and
            abs(p1.ax - p2.ax) < epsilon and abs(p1.ay - p2.ay) < epsilon)

def compare_particle_arrays(arr1, arr2, size):
    for i in range(size):
        if not compare_particles(arr1[i], arr2[i]):  
            print(i)
            return False
    return True

def init_particles(n, particles0, particles1):
    random.seed(time.time())
    sx = math.ceil(math.sqrt(n))
    sy = (n + sx - 1) // sx

    shuffle = list(range(n))
    random.shuffle(shuffle)

    for i in range(n):
        k = shuffle[i]
        particles0[i].x = SIZE * (1 + (k % sx)) / (1 + sx)
        particles0[i].y = SIZE * (1 + (k // sx)) / (1 + sy)
        particles0[i].vx = random.random() * 2 - 1
        particles0[i].vy = random.random() * 2 - 1
        particles1[i].x = particles0[i].x 
        particles1[i].y = particles0[i].y
        particles1[i].vx =  particles0[i].vx
        particles1[i].vy =particles0[i].vy

def parallelize_particle_simulation(n, nsteps, savefreq):
    global particles_dyn
    size = SIZE
    print(f"{size} particle size")

    gridSize = GRID_SIZE
    print(f"{gridSize} grid size")
    grid = Grid(gridSize)
    grid_init(grid, gridSize)

    for i in range(n):
        grid_add(grid, particles_dyn[i])

    print("Running 1 warm-up iteration ...")
    compute_forces(particles_dyn, grid)
    move_particles(particles_dyn, grid)
    print("Done")

    start_compute_forces_time = 0
    start_move_particles_time = 0
    
    print("Running " + str(nsteps) +" iteration ...")
    
    start_whole_time = time.time()
    for step in range(nsteps):
        start_iteration_time = time.time()
        compute_forces(particles_dyn, grid)
        start_compute_forces_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        move_particles(particles_dyn, grid)
        start_move_particles_time += time.time() - start_iteration_time
    end_whole_time = time.time() - start_whole_time
    print("Done")
    # print("Dynamic simulation: Total time for {} steps: {:.4f} seconds".format(nsteps, end_whole_time))
    # print("Single iteration time: {:.4f} ms".format((end_whole_time / nsteps) * 1000))
    # print("Compute forces time: {:.4f} ms".format((start_compute_forces_time / nsteps) * 1000))
    # print("Move particles time: {:.4f} ms".format((start_move_particles_time / nsteps) * 1000))

def parallelize_particle_simulation_static(n, nsteps, savefreq):
    global particles_static
    size = SIZE
    grid_static = [LinkedListStatic() for _ in range(GRIDSIZE2)]

    for i in range(n):
        grid_add_static(grid_static, particles_static[i])

    print("Running 1 warm-up iteration ...")
    compute_forces_static(particles_static, grid_static)
    move_particles_static(particles_static, grid_static)
    print("Done")

    start_compute_forces_time = 0
    start_move_particles_time = 0

    print("Running " + str(nsteps) +" iteration ...")
    start_whole_time = time.time()
    for step in range(nsteps):
        start_iteration_time = time.time()
        compute_forces_static(particles_static, grid_static)
        start_compute_forces_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        move_particles_static(particles_static, grid_static)
        start_move_particles_time += time.time() - start_iteration_time
    end_whole_time = time.time() - start_whole_time
    print("Done")
    print("Static simulation: Total time for {} steps: {:.4f} seconds".format(nsteps, end_whole_time))
    print("Single iteration time: {:.4f} ms".format((end_whole_time / nsteps) * 1000))
    print("Compute forces time: {:.4f} ms".format((start_compute_forces_time / nsteps) * 1000))
    print("Move particles time: {:.4f} ms".format((start_move_particles_time / nsteps) * 1000))

def main():
    print("=======================================")
    print("Running parallel_particle benchmark Python")
    print("=======================================")
    global particles_dyn, particles_static

    init_particles(nparticles, particles_dyn, particles_static)


    parallelize_particle_simulation(nparticles, nsteps, savefreq)
    parallelize_particle_simulation_static(nparticles, nsteps, savefreq)

    arrays_equal = compare_particle_arrays(particles_dyn, particles_static, nparticles)
    if arrays_equal:
        print("The particle arrays are equal.")
    else:
        print("The particle arrays are not equal.")

if __name__ == "__main__":
    main()
