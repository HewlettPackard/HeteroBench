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
 
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>

#include "cpu_impl.h"

//
// Calculate the grid coordinate from a real coordinate
//
int grid_coord(double c)
{
    return (int)floor(c / cutoff);
}

int grid_coord_flat(int size, double x, double y)
{
    return grid_coord(x) * size + grid_coord(y);
}

//
// initialize grid and fill it with particles
// 
void grid_init(grid_t& grid, int size)
{
    grid.size = size;

    // Initialize grid
    grid.grid = (linkedlist**)malloc(sizeof(linkedlist*) * size * size);

    if (grid.grid == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for the grid!\n");
        exit(1);
    }

    memset(grid.grid, 0, sizeof(linkedlist*) * size * size);
    grid.node_counts = (int*)malloc(grid.size * grid.size * sizeof(int));
    memset(grid.grid, 0, sizeof(int*) * size * size);
}

//
// adds a particle pointer to the grid
//
void grid_add(grid_t& grid, particle_t* p)
{
    int gridCoord = grid_coord_flat(grid.size, p->x, p->y);

    linkedlist_t* newElement = (linkedlist_t*)malloc(sizeof(linkedlist));
    newElement->value = p;

    // Beginning of critical section
    newElement->next = grid.grid[gridCoord];

    grid.grid[gridCoord] = newElement;

}
void grid_add_static(linkedlist_static grad_static[gridsize2], particle_t* p)
{
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

//
// Removes a particle from a grid
//
bool grid_remove(grid_t& grid, particle_t* p, int gridCoord)
{
    if (gridCoord == -1)
        gridCoord = grid_coord_flat(grid.size, p->x, p->y);

    // No elements?
    if (grid.grid[gridCoord] == 0)
    {
        return false;
    }

    // Beginning of critical section

    linkedlist_t** nodePointer = &(grid.grid[gridCoord]);
    linkedlist_t* current = grid.grid[gridCoord];

    while (current && (current->value != p))
    {
        nodePointer = &(current->next);
        current = current->next;
    }

    if (current)
    {
        *nodePointer = current->next;
        free(current);
    }

    // End of critical section

    return !!current;
}

//
// clears a grid from values and deallocates any memory from the heap.
//
void grid_clear(grid_t& grid)
{
    for (int i = 0; i < grid.size * grid.size; ++i)
    {
        linkedlist_t* curr = grid.grid[i];
        while (curr != 0)
        {
            linkedlist_t* tmp = curr->next;
            free(curr);
            curr = tmp;
        }
    }
    free(grid.grid);
}

int grid_size(grid_t& grid)
{
    int count = 0;
    for (int i = 0; i < grid.size * grid.size; ++i)
    {
        linkedlist_t* curr = grid.grid[i];
        while (curr != 0)
        {
            count++;
            curr = curr->next;
        }
    }

    return count;
}