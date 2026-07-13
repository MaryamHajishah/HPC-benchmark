#!/usr/bin/env bash
# LAST CHANCE — run AFTER final_run.sh finishes. Fully unattended.
#     bash scripts/last_chance.sh
# What the bisect revealed: block=32 is slow (~15 GF/s) for EVERY kernel form at
# n=2000, and block=0 is ~25 everywhere; only the real seq binary at n>=5000
# reaches 32 with block=32. So: (1) redo the bisect at n=5000 where it can
# discriminate, (2) sweep block sizes on the REAL MPI binary (flat and VLA),
# (3) rebuild with the empirically best (variant, block) and run the full MPI
# sweeps with it. Commits+pushes at every milestone; Ctrl-C is safe.
cd "$(dirname "$0")/.."
mkdir -p outputs
exec > >(tee -a outputs/last_chance_master.log) 2>&1
say(){ echo; echo "##### [$(date +%H:%M:%S)] $*"; }
commit_push(){
    git add outputs >/dev/null 2>&1 || true
    git diff --cached --quiet 2>/dev/null || git commit -m "last chance: $1" >/dev/null 2>&1 || true
    git push >/dev/null 2>&1 && echo ">> pushed ($1)" || echo ">> push failed (commit is local)"
}
trap 'say "INTERRUPTED — securing"; commit_push "partial"; exit 130' INT TERM
command -v icx >/dev/null 2>&1 || source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
if command -v mpiicx >/dev/null 2>&1; then MPICC=mpiicx; MPIXHOST=-xCORE-AVX2;
else MPICC=mpicc; MPIXHOST=-march=native; fi
gf(){ echo "$1" | grep -o '[0-9.]\+ GFLOP' | head -1 | cut -d' ' -f1; }

############################################################
say "1/4  Bisect at n=5000 (where seq is known-fast at b32) + the n=2000 theory check  (~3 min)"
############################################################
icx -O3 -xHost -std=c11 scripts/kernel_bisect.c -o scripts/kernel_bisect
{ echo "## bisect n=5000 block=32 (real discriminating size)"
  ./scripts/kernel_bisect 5000 32 1
  echo "## theory check: the REAL seq binary at n=2000 block=32 (never measured before)"
  icx -O3 -xHost -std=c11 seq/matmul_seq.c -o seq/matmul_seq
  ./seq/matmul_seq 2000 32 1
  ./seq/matmul_seq 2000 16 1
  echo "## vectorization reports (writable path this time): seq kernel vs bisect kernels"
  mkdir -p outputs/optrpt && cd outputs/optrpt
  icx -O3 -xHost -std=c11 -qopt-report=3 -c ../../seq/matmul_seq.c      -o seq.o 2>&1 | head -2
  icx -O3 -xHost -std=c11 -qopt-report=3 -c ../../scripts/kernel_bisect.c -o kb.o 2>&1 | head -2
  grep -m 40 -i "vector\|remainder\|interchang\|unroll\|fma" ./*.yaml ./*.optrpt 2>/dev/null || echo "(no report files found: $(ls))"
  cd ../..
} 2>&1 | tee -a outputs/kernel_bisect_icx.txt

############################################################
say "2/4  Block-size sweep on the REAL MPI binary, flat and VLA variants  (~3 min)"
############################################################
$MPICC -O3 $MPIXHOST -std=c11              mpi/matmul_mpi.c -o mpi/mm_flat
$MPICC -O3 $MPIXHOST -std=c11 -DKERNEL_VLA mpi/matmul_mpi.c -o mpi/mm_vla
BESTGF=0; BESTV=""; BESTB=0
{ echo "## MPI-binary block sweep, n=5000, direct run"
  for V in flat vla; do for B in 0 16 24 32; do
      OUT=$(./mpi/mm_$V 5000 $B 1); G=$(gf "$OUT")
      echo "$V b=$B : $OUT"
      if python3 -c "exit(0 if float('${G:-0}') > float('$BESTGF') else 1)"; then
          BESTGF=$G; BESTV=$V; BESTB=$B; fi
  done; done
  echo ">> WINNER: variant=$BESTV block=$BESTB at $BESTGF GFLOP/s (seq best is 32.2 b32 / 24.9 b0)"
} 2>&1 | tee -a outputs/kernel_bisect_icx.txt
commit_push "bisect n=5000 + MPI block sweep (winner: $BESTV b=$BESTB @ $BESTGF)"

KV=""; [ "$BESTV" = vla ] && KV=-DKERNEL_VLA
$MPICC -O3 $MPIXHOST -std=c11 $KV       mpi/matmul_mpi.c -o mpi/matmul_mpi
$MPICC -O3 $MPIXHOST -std=c11 $KV -DP2P mpi/matmul_mpi.c -o mpi/matmul_mpi_p2p

############################################################
say "3/4  MPI n=10000 sweep with variant=$BESTV block=$BESTB  (~9-14 min)"
############################################################
{ echo "# MPI wrapper: $MPICC | flags: -O3 $MPIXHOST $KV | N=10000 block=$BESTB (empirically re-tuned; b32 pathological in this binary)"
  echo "## Process scaling (strong)  N=10000 block=$BESTB  [collectives]"
  for P in 1 2 4 8 16 24; do mpirun -np $P ./mpi/matmul_mpi 10000 $BESTB 3; done
  echo; echo "## Collectives vs point-to-point  (N=10000, P=8)"
  mpirun -np 8 ./mpi/matmul_mpi     10000 $BESTB 3
  mpirun -np 8 ./mpi/matmul_mpi_p2p 10000 $BESTB 3
  echo; echo "## Weak scaling (n = 4000 * P^(1/3))"
  for PAIR in "1:4000" "2:5040" "4:6350" "8:8000"; do
      mpirun -np ${PAIR%:*} ./mpi/matmul_mpi ${PAIR#*:} $BESTB 3; done
} 2>&1 | tee outputs/run_mpi_10000_FIXED.txt
commit_push "MPI n=10000 with retuned block=$BESTB"

############################################################
say "4/4  MPI n=15000 sweep (long — Ctrl-C safe, every finished line counts)"
############################################################
{ echo "# MPI wrapper: $MPICC | flags: -O3 $MPIXHOST $KV | N=15000 block=$BESTB (retuned)"
  echo "# NOTE: reps=2 (not 3) for this sweep only — lab-closure time constraint; min statistic unchanged"
  echo "## Process scaling (strong)  N=15000 block=$BESTB  [collectives, P<=16]"
  for P in 1 2 4 8 16; do mpirun -np $P ./mpi/matmul_mpi 15000 $BESTB 2; done
} 2>&1 | tee outputs/run_mpi_15000_FIXED.txt
rm -f mpi/mm_flat mpi/mm_vla
commit_push "MPI n=15000 with retuned block=$BESTB — done"
say "ALL DONE (winner was $BESTV block=$BESTB @ $BESTGF GF/s). Screenshot with:"
echo "   cat outputs/kernel_bisect_icx.txt"
echo "   cat outputs/run_mpi_10000_FIXED.txt"
echo "   cat outputs/run_mpi_15000_FIXED.txt"
