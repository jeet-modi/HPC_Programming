#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

void interpolation(double *mesh_value, Points *points) {
    // Clear the global mesh initially
    int total_cells = GRID_X * GRID_Y;
    memset(mesh_value, 0, total_cells * sizeof(double));

    double scale_x = (double)NX;
    double scale_y = (double)NY;
    double area_cell = dx * dy;

    #pragma omp parallel
    {
        // Thread-local privatization to prevent race conditions without slow atomics
        double *thread_mesh = (double*)calloc(total_cells, sizeof(double));

        #pragma omp for schedule(static)
        for (int idx = 0; idx < NUM_Points; idx++) {
            double pos_x = points[idx].x;
            double pos_y = points[idx].y;

            int grid_i = (int)(pos_x * scale_x);
            int grid_j = (int)(pos_y * scale_y);

            grid_i = (grid_i >= NX) ? NX - 1 : ((grid_i < 0) ? 0 : grid_i);
            grid_j = (grid_j >= NY) ? NY - 1 : ((grid_j < 0) ? 0 : grid_j);

            double frac_x = (pos_x * scale_x) - grid_i;
            double frac_y = (pos_y * scale_y) - grid_j;

            double one_minus_x = 1.0 - frac_x;
            double one_minus_y = 1.0 - frac_y;

            // Area-scaled weights based strictly on the assignment formula
            double weight_00 = one_minus_x * one_minus_y * area_cell;
            double weight_10 = frac_x * one_minus_y * area_cell;
            double weight_01 = one_minus_x * frac_y * area_cell;
            double weight_11 = frac_x * frac_y * area_cell;

            int index_base = grid_j * GRID_X + grid_i;

            // Update local memory (no locks needed)
            thread_mesh[index_base] += weight_00;
            thread_mesh[index_base + 1] += weight_10;
            thread_mesh[index_base + GRID_X] += weight_01;
            thread_mesh[index_base + GRID_X + 1] += weight_11;
        }

        // Reduction phase: safely merge local meshes back to the global mesh
        #pragma omp critical
        {
            for (int k = 0; k < total_cells; k++) {
                mesh_value[k] += thread_mesh[k];
            }
        }
        
        free(thread_mesh);
    }
}

void save_mesh(double *mesh_value) {
    FILE *file_ptr = fopen("Mesh_parallel.out", "w"); // Updated name
    if (!file_ptr) {
        printf("Error creating Mesh_parallel.out\n");
        exit(1);
    }
    for (int row = 0; row < GRID_Y; row++) {
        for (int col = 0; col < GRID_X; col++) {
            fprintf(file_ptr, "%lf ", mesh_value[row * GRID_X + col]);
        }
        fprintf(file_ptr, "\n");
    }
    fclose(file_ptr);
}