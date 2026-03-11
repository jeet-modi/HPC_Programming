#ifndef UTILS_H
#define UTILS_H

#include "init.h"

void interpolation(double *mesh_value, Points *points);
void mover_serial(Points *points,double dx,double dy);
void mover_parallel(Points *points,double dx,double dy);

#endif