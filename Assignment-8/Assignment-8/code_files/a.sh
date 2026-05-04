#!/bin/bash

MPIEXEC=/opt/homebrew/bin/mpirun
INPUT=input.bin
CSV=timings_e.csv

echo "mpi,omp,interp,reduce,broadcast,norm,mover,denorm,total" > $CSV

pairs=(
"1 2" "2 1"
"1 4" "2 2" "4 1"
"1 8" "2 4" "4 2" "8 1"
"1 16" "2 8" "4 4" "8 2" "16 1"
"1 32" "2 16" "4 8" "8 4" "16 2" "32 1"
"1 64" "2 32" "4 16" "8 8" "16 4" "32 2" "64 1"
)

for p in "${pairs[@]}"
do
    MPI=$(echo $p | awk '{print $1}')
    OMP=$(echo $p | awk '{print $2}')

    export OMP_NUM_THREADS=$OMP

    OUT=$($MPIEXEC --oversubscribe -np $MPI ./mpi $INPUT 10)

    INTERP=$(echo "$OUT" | grep "Interpolation" | awk '{print $NF}')
    REDUCE=$(echo "$OUT" | grep "MPI Reduce" | awk '{print $NF}')
    BROADCAST=$(echo "$OUT" | grep "MPI Broadcast" | awk '{print $NF}')
    NORM=$(echo "$OUT" | grep "Normalization" | awk '{print $NF}')
    MOVER=$(echo "$OUT" | grep "Mover" | awk '{print $NF}')
   
    DENORM=$(echo "$OUT" | grep "Denormalization" | awk '{print $NF}')
    TOTAL=$(echo "$OUT" | grep "Total Time" | awk '{print $NF}')
    
    echo "$MPI,$OMP,$INTERP,$REDUCE,$BROADCAST,$NORM,$MOVER,$DENORM,$TOTAL" >> $CSV
done

echo "Done: $CSV"
