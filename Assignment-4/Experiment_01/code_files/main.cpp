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
    long points_range[5] = {100,10000,1000000,100000000,1000000000};

    Maxiter = 10;

    FILE *fp = fopen("exp1_results.csv","w");
    fprintf(fp,"Nx,Ny,Particles,TotalInterpolationTime\n");

    for(int g=0; g<3; g++)
    {
        NX = Nx_values[g];
        NY = Ny_values[g];

        GRID_X = NX + 1;
        GRID_Y = NY + 1;

        dx = 1.0/NX;
        dy = 1.0/NY;

        for(int p=0;p<5;p++)
        {
            NUM_Points = points_range[p];

            double *mesh_value = (double*)calloc(GRID_X*GRID_Y,sizeof(double));
            Points *points = (Points*)malloc(NUM_Points*sizeof(Points));

            double total = 0.0;

            for(int iter=0;iter<Maxiter;iter++)
            {
                initializepoints(points);

                double start = omp_get_wtime();
                interpolation(mesh_value,points);
                double end = omp_get_wtime();

                total += (end-start);
            }

            fprintf(fp,"%d,%d,%ld,%.3f\n",NX,NY,NUM_Points,total);

            free(mesh_value);
            free(points);
        }
    }

    fclose(fp);

    return 0;
}