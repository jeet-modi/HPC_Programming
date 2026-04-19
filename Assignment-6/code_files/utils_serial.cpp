#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

void interpolation(double *mesh_value, Points *points) {
    double scale_x = (double)NX; 
    double scale_y = (double)NY;
    double area_cell = dx * dy;

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

        double weight_00 = one_minus_x * one_minus_y * area_cell;
        double weight_10 = frac_x * one_minus_y * area_cell;
        double weight_01 = one_minus_x * frac_y * area_cell;
        double weight_11 = frac_x * frac_y * area_cell;

        int index_base = grid_j * GRID_X + grid_i;

        mesh_value[index_base] += weight_00;
        mesh_value[index_base + 1] += weight_10;
        mesh_value[index_base + GRID_X] += weight_01;
        mesh_value[index_base + GRID_X + 1] += weight_11;
    }
}

void save_mesh(double *mesh_value) {
    FILE *file_ptr = fopen("Mesh_serial.out", "w"); // Updated name
    if (!file_ptr) {
        printf("Error creating Mesh_serial.out\n");
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