#!/usr/bin/env bash
# PHASE 4 -- MPI.  Save with:
#   bash scripts/run_mpi.sh >> outputs/run_mpi.txt 2>&1
#   N=10000 BLK=64 bash scripts/run_mpi.sh >> outputs/run_mpi_big.txt 2>&1
# Uses mpiicx if present (Intel), else mpicc (OpenMPI/GCC).
set -e
cd "$(dirname "$0")/../mpi"

if command -v mpiicx >/dev/null 2>&1; then MPICC=mpiicx; XHOST=${XHOST:--xHost};
else                                       MPICC=mpicc;  XHOST=${XHOST:--march=native}; fi
STD=-std=c11
N=${N:-5000}; BLK=${BLK:-64}; MAXP=$(nproc)

echo "# MPI wrapper: $MPICC ($($MPICC --version 2>/dev/null | head -1))"
$MPICC -O3 $XHOST $STD          matmul_mpi.c -o matmul_mpi
$MPICC -O3 $XHOST $STD -DP2P    matmul_mpi.c -o matmul_mpi_p2p
echo "# built matmul_mpi (collective) and matmul_mpi_p2p | N=$N block=$BLK | cores=$MAXP"
echo

echo "## 1) Process scaling (strong)  N=$N block=$BLK  P=1,2,4,8,16,24  [collectives]"
for P in 1 2 4 8 16 24; do
    [ "$P" -le "$MAXP" ] || { echo "skip P=$P (>$MAXP cores)"; continue; }
    mpirun -np $P ./matmul_mpi $N $BLK 3
done
echo

P8=$(( MAXP < 8 ? MAXP : 8 ))
echo "## 2) Collectives vs manual point-to-point  (N=$N, P=$P8)"
mpirun -np $P8 ./matmul_mpi     $N $BLK 3
mpirun -np $P8 ./matmul_mpi_p2p $N $BLK 3
echo

echo "## 3) Weak scaling (per-process work ~constant: n = 4000 * P^(1/3))  P=1,2,4,8"
for PAIR in "1:4000" "2:5040" "4:6350" "8:8000"; do
    P=${PAIR%:*}; NW=${PAIR#*:}
    [ "$P" -le "$MAXP" ] || { echo "skip P=$P"; continue; }
    mpirun -np $P ./matmul_mpi $NW $BLK 3
done
echo

echo "## Confirm winner bigger, e.g.:"
echo "##   mpirun -np $P8 ./mpi/matmul_mpi 10000 <best_block> 3   (>> outputs/mpi_confirm.txt 2>&1)"
