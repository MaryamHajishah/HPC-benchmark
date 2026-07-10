#!/usr/bin/env bash
# PHASE 1 -- machine specs.  Save with:  bash scripts/specs.sh >> outputs/specs.txt 2>&1
echo "===================== CPU (lscpu) ====================="
lscpu
echo
echo "===================== Memory (free -h) ================"
free -h
echo
echo "===================== Cache / topology ================"
lstopo-no-graphics 2>/dev/null || hwloc-ls 2>/dev/null || echo "(lstopo/hwloc not installed -- lscpu cache lines above are enough)"
echo
echo "===================== Compilers ======================="
icx  --version 2>/dev/null | head -1 || echo "icx  not found"
icc  --version 2>/dev/null | head -1 || echo "icc  not found (deprecated, ok)"
gcc  --version 2>/dev/null | head -1 || echo "gcc  not found"
mpirun --version 2>/dev/null | head -1 || echo "mpirun not found"
