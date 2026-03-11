#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
long NUM_Points;
int Maxiter;
double dx, dy;

int main()
{
    int Nx_values[3] = {250,500,1000};
    int Ny_values[3] = {100,200,400};

    NUM_Points = 100000000;
    Maxiter = 10;

    FILE *fp = fopen("exp2_results.csv","w");
    fprintf(fp,"ProblemIndex,InterpolationTime\n");

    for(int g=0; g<3; g++)
    {
        NX = Nx_values[g];
        NY = Ny_values[g];

        GRID_X = NX+1;
        GRID_Y = NY+1;

        dx = 1.0/NX;
        dy = 1.0/NY;

        double *mesh_value = (double*)calloc(GRID_X*GRID_Y,sizeof(double));
        Points *points = (Points*)malloc(NUM_Points*sizeof(Points));

        initializepoints(points);

        double total = 0;

        for(int iter=0;iter<Maxiter;iter++)
        {
            double start = omp_get_wtime();
            interpolation(mesh_value,points);
            double end = omp_get_wtime();

            total += end-start;
        }

        fprintf(fp,"%d,%.3f\n",g+1,total);

        free(mesh_value);
        free(points);
    }

    fclose(fp);

    return 0;
}