#!/usr/bin/env bash
# PHASE 3 -- OpenMP.  Save with:
#   bash scripts/run_omp.sh >> outputs/run_omp.txt 2>&1
#   CC=gcc bash scripts/run_omp.sh >> outputs/run_omp_gcc.txt 2>&1
#   N=10000 BLK=128 bash scripts/run_omp.sh >> outputs/run_omp_big.txt 2>&1
set -e
cd "$(dirname "$0")/../omp"
CC=${CC:-icx}; STD=-std=c11
if [ "$CC" = "gcc" ]; then OMPF=-fopenmp; XHOST=${XHOST:--march=native};
else                        OMPF=-qopenmp; XHOST=${XHOST:--xHost}; fi
N=${N:-5000}; BLK=${BLK:-32}; MAXT=$(nproc)

echo "# Compiler: $($CC --version 2>/dev/null | head -1)"
$CC -O3 $XHOST $OMPF $STD matmul_omp.c -o matmul_omp
echo "# built matmul_omp | N=$N default block=$BLK | cores available=$MAXT"
echo

echo "## 1) Thread scaling  (N=$N, block=$BLK, schedule=static)  T=1,2,4,8,16,24"
for T in 1 2 4 8 16 24; do
    [ "$T" -le "$MAXT" ] || { echo "skip T=$T (>$MAXT cores)"; continue; }
    OMP_NUM_THREADS=$T OMP_SCHEDULE=static ./matmul_omp $N $BLK 3
done
echo

T24=$(( MAXT < 24 ? MAXT : 24 ))
echo "## 2) Block-size sweep  (N=$N, T=$T24, schedule=static)  block=0(no-block),32,64,128,256"
for B in 0 32 64 128 256; do
    OMP_NUM_THREADS=$T24 OMP_SCHEDULE=static ./matmul_omp $N $B 3
done
echo

echo "## 3) Schedule sweep  (N=$N, T=$T24, block=$BLK)  static / dynamic / guided"
for S in static dynamic guided; do
    OMP_NUM_THREADS=$T24 OMP_SCHEDULE=$S ./matmul_omp $N $BLK 3
done
echo

echo "## Confirm winner bigger, e.g.:"
echo "##   OMP_NUM_THREADS=$T24 OMP_SCHEDULE=<best> ./omp/matmul_omp 10000 <best_block> 5   (>> outputs/omp_confirm.txt 2>&1)"
