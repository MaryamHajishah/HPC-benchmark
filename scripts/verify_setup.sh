#!/usr/bin/env bash
# verify_setup.sh -- READ-ONLY check that MPI is configured right before the big runs.
# Builds throwaway binaries in /tmp; does NOT modify any repo file or binary.
# Run from the repo root:   bash scripts/verify_setup.sh
[ -d mpi ] && [ -d scripts ] || { echo "!! run from the repo root (where mpi/ and scripts/ live)"; exit 1; }

echo "==================================================================="
echo " HPC-benchmark setup check   host=$(hostname)   $(date +%H:%M:%S)"
echo "==================================================================="

echo; echo "[1] repo state"
git log --oneline -1 2>/dev/null; git status --short 2>/dev/null | head

echo; echo "[2] compilers (want Intel oneAPI 2023.x for BOTH)"
icx    --version 2>/dev/null | head -1 || echo "  icx NOT found -- source /opt/intel/oneapi/setvars.sh"
mpiicx --version 2>/dev/null | head -1 || echo "  mpiicx NOT found"

echo; echo "[3] restrict fix in the MPI kernel?"
if grep -q 'restrict' mpi/matmul_mpi.c; then
  echo "  PRESENT:"; grep -n 'restrict' mpi/matmul_mpi.c | head
else
  echo "  >>> NO 'restrict' in mpi/matmul_mpi.c -- kernel NOT fixed on this PC"
fi

echo; echo "[4] MPI script: wrapper, flags, block default"
grep -nE 'MPICC=|XHOST=|BLK=\$\{BLK' scripts/run_mpi.sh

echo; echo "[5] does the inner multiply VECTORIZE? (throwaway build in /tmp)"
mpiicx -O3 -xHost -std=c11 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
       mpi/matmul_mpi.c -o /tmp/mmpi_chk 2>&1 | grep -iE 'vectoriz' | head
[ -x /tmp/mmpi_chk ] || echo "  >>> MPI build FAILED"

echo; echo "[6] SPEED PROOF @ n=5000 block=32  (want SEQ and MPI P=1 BOTH ~30 GF)"
icx -O3 -xHost -std=c11 seq/matmul_seq.c -o /tmp/seq_chk 2>/dev/null
printf "  SEQ    : "; /tmp/seq_chk 5000 32 1
printf "  MPI P=1: "; mpirun -np 1 /tmp/mmpi_chk 5000 32 1

echo; echo "==================================================================="
echo " Done -- paste this whole output back."
echo "==================================================================="
