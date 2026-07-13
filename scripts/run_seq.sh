#!/usr/bin/env bash
# PHASE 2 -- sequential baseline.  Save with:
#   bash scripts/run_seq.sh >> outputs/run_seq.txt 2>&1
#   CC=gcc XHOST=-march=native bash scripts/run_seq.sh >> outputs/run_seq_gcc.txt 2>&1
set -e
cd "$(dirname "$0")/../seq"
CC=${CC:-icx}
if [ "$CC" = "gcc" ]; then XHOST=${XHOST:--march=native}; else XHOST=${XHOST:--xHost}; fi
STD=-std=c11
N1=${N1:-2000}   # loop-order size
N2=${N2:-5000}   # flag-sweep + block-sweep size (matches the report tables)

echo "# Compiler: $($CC --version 2>/dev/null | head -1)"
echo

echo "## Loop order  (n=$N1): naive i-j-k vs cache-friendly i-k-j vs blocked"
$CC -O3 $XHOST -DNAIVE $STD matmul_seq.c -o matmul_naive
$CC -O3 $XHOST         $STD matmul_seq.c -o matmul_seq
printf "i-j-k naive    "; ./matmul_naive "$N1" 0  1
printf "i-k-j          "; ./matmul_seq   "$N1" 0  1
printf "i-k-j blocked  "; ./matmul_seq   "$N1" 64 1
echo

echo "## Flag sweep  (n=$N2, i-k-j, no blocking)"
for OPT in "-O0" "-O1" "-O2" "-O3" "-O3 $XHOST"; do
    $CC $OPT $STD matmul_seq.c -o matmul_seq
    printf "%-16s " "$OPT"; ./matmul_seq "$N2" 0 1
done
echo

echo "## Block-size sweep  (n=$N2, -O3 $XHOST)"
$CC -O3 $XHOST $STD matmul_seq.c -o matmul_seq
for BS in 0 8 16 32 64; do ./matmul_seq "$N2" "$BS" 1; done
echo

echo "## Confirm winner bigger, e.g.:  ./seq/matmul_seq 10000 <best_block> 5   (>> outputs/seq_confirm.txt 2>&1)"
