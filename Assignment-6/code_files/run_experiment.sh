#!/bin/bash

CONFIGS=("input_A.bin" "input_B.bin" "input_C.bin" "input_D.bin" "input_E.bin")
THREADS=(2 4 6 8 10 12 14 16)

echo "Config,Mode,Threads,Time" > results.csv

echo "========================================"
echo "    STARTING PARALLEL EXPERIMENTS       "
echo "========================================"

for config in "${CONFIGS[@]}"; do
    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t
        
        T1=$(./pic_parallel $config)
        T2=$(./pic_parallel $config)
        T3=$(./pic_parallel $config)
        
        AVG=$(echo "scale=6; ($T1 + $T2 + $T3) / 3" | bc)
        echo "Parallel | $config | Threads: $t | Time: $AVG s"
        
        echo "$config,Parallel,$t,$AVG" >> results.csv
    done
done

echo ""
echo "========================================"
echo "      STARTING SERIAL EXPERIMENTS       "
echo "========================================"

for config in "${CONFIGS[@]}"; do
    T1=$(./pic_serial $config)
    T2=$(./pic_serial $config)
    T3=$(./pic_serial $config)
    
    AVG=$(echo "scale=6; ($T1 + $T2 + $T3) / 3" | bc)
    echo "Serial   | $config | Threads: 1 | Time: $AVG s"
    
    echo "$config,Serial,1,$AVG" >> results.csv
done

echo ""
echo "All experiments complete! Data saved to results.csv."
