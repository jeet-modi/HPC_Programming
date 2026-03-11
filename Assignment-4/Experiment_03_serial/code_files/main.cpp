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
    NX = 1000;
    NY = 400;
    NUM_Points = 14000000;
    Maxiter = 10;

    GRID_X = NX+1;
    GRID_Y = NY+1;

    dx = 1.0/NX;
    dy = 1.0/NY;

    double *mesh_value = (double*)calloc(GRID_X*GRID_Y,sizeof(double));
    Points *points = (Points*)malloc(NUM_Points*sizeof(Points));

    initializepoints(points);

    FILE *fp = fopen("exp3_serial.csv","w");
    fprintf(fp,"Iter,InterpTime,MoverTime,Total\n");

    for(int i=0;i<Maxiter;i++)
    {
        double t1 = omp_get_wtime();
        interpolation(mesh_value,points);
        double t2 = omp_get_wtime();

        mover_serial(points,dx,dy);
        double t3 = omp_get_wtime();

        fprintf(fp,"%d,%.3f,%.3f,%.3f\n",
                i+1,
                t2-t1,
                t3-t2,
                t3-t1);
    }

    fclose(fp);

    return 0;
}