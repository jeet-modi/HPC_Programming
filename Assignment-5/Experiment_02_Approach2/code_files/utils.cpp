#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <omp.h>

#include "utils.h"

using namespace std;

void interpolation(double *grid_buf, Points *pt_data)
{
    memset(grid_buf, 0, (size_t)GRID_X * GRID_Y * sizeof(double));

    for (int idx = 0; idx < NUM_Points; idx++)
    {
        double posx = pt_data[idx].x;
        double posy = pt_data[idx].y;

        int cell_x = (int)(posx / dx);
        int cell_y = (int)(posy / dy);

        if (cell_x >= NX)
            cell_x = NX - 1;
        if (cell_y >= NY)
            cell_y = NY - 1;
        if (cell_x < 0)
            cell_x = 0;
        if (cell_y < 0)
            cell_y = 0;

        double frac_x = (posx - (double)cell_x * dx) / dx;
        double frac_y = (posy - (double)cell_y * dy) / dy;

        grid_buf[cell_y * GRID_X + cell_x] += (1.0 - frac_x) * (1.0 - frac_y);
        grid_buf[cell_y * GRID_X + cell_x + 1] += frac_x * (1.0 - frac_y);
        grid_buf[(cell_y + 1) * GRID_X + cell_x] += (1.0 - frac_x) * frac_y;
        grid_buf[(cell_y + 1) * GRID_X + cell_x + 1] += frac_x * frac_y;
    }
}


unsigned int hlp1(unsigned int &rng_state)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

double grand(unsigned int &rng_state)
{
    return hlp1(rng_state) * (1.0 / 4294967296.0);
}


void mover_immediate_serial(Points *pt_data, double stepX, double stepY)
{
    const double inv_rand = 1.0 / RAND_MAX;
    for (int idx = 0; idx < NUM_Points; idx++)
    {
        double shift_x = (rand() * inv_rand * 2.0 - 1.0) * stepX;
        double shift_y = (rand() * inv_rand * 2.0 - 1.0) * stepY;

        pt_data[idx].x += shift_x;
        pt_data[idx].y += shift_y;

        if (pt_data[idx].x < 0.0 || pt_data[idx].x > 1.0 || pt_data[idx].y < 0.0 || pt_data[idx].y > 1.0)
        {
            pt_data[idx].x = rand() * inv_rand;
            pt_data[idx].y = rand() * inv_rand;
        }
    }
}


void mover_immediate_parallel(Points *pt_data, double stepX, double stepY)
{
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned int rng_state = (unsigned int)(thread_id + 1) * 1234567891u;

#pragma omp for schedule(static)
        for (int idx = 0; idx < NUM_Points; idx++)
        {
            double shift_x = (grand(rng_state) * 2.0 - 1.0) * stepX;
            double shift_y = (grand(rng_state) * 2.0 - 1.0) * stepY;

            pt_data[idx].x += shift_x;
            pt_data[idx].y += shift_y;

            if (pt_data[idx].x < 0.0 || pt_data[idx].x > 1.0 ||
                pt_data[idx].y < 0.0 || pt_data[idx].y > 1.0)
            {
                pt_data[idx].x = grand(rng_state);
                pt_data[idx].y = grand(rng_state);
            }
        }
    }
}


void mover_simple_parallel(Points *pt_data, double stepX, double stepY)
{
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned int rng_state = (unsigned int)(thread_id + 1) * 362436069u;

#pragma omp for schedule(static)
        for (int idx = 0; idx < NUM_Points; idx++)
        {
            while (1)
            {
                double shift_x = (grand(rng_state) * 2.0 - 1.0) * stepX;
                double shift_y = (grand(rng_state) * 2.0 - 1.0) * stepY;

                double next_x = pt_data[idx].x + shift_x;
                double next_y = pt_data[idx].y + shift_y;

                if (next_x >= 0.0 && next_x <= 1.0 &&
                    next_y >= 0.0 && next_y <= 1.0)
                {
                    pt_data[idx].x = next_x;
                    pt_data[idx].y = next_y;
                    break;
                }
            }
        }
    }
}


void save_mesh(double *grid_buf)
{
    FILE *file_ptr = fopen("Mesh.out", "w");
    if (!file_ptr)
    {
        fprintf(stderr, "Error creating Mesh.out\n");
        exit(1);
    }
    for (int row = 0; row < GRID_Y; row++)
    {
        for (int col = 0; col < GRID_X; col++)
        {
            fprintf(file_ptr, "%lf ", grid_buf[row * GRID_X + col]);
        }
        fprintf(file_ptr, "\n");
    }
    fclose(file_ptr);
}