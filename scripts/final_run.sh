#!/usr/bin/env bash
# FINAL LAB RUN — fully unattended, no prompts. From the repo root:
#     bash scripts/final_run.sh
#
# Order (most critical first; longest job last so Ctrl-C late loses least):
#   1. kernel bisect with icx  ->  outputs/kernel_bisect_icx.txt          ~3 min
#   2. A/B the real MPI binary (flat vs -DKERNEL_VLA), keep the winner    ~2 min
#   3. IF fixed: MPI n=10000 sweep      -> outputs/run_mpi_10000_FIXED.txt ~12 min
#      IF NOT fixed: diagnostics bundle only (no pointless re-runs)        ~3 min
#   4. clean sequential n=15000 baseline -> outputs/seq_15000_clean.txt   ~11 min
#   5. Advisor roofline (guarded, 15-min timeout) -> outputs/roofline.html ~8 min
#      [ mid-run git commit+push here — everything above is secured ]
#   6. IF fixed: MPI n=15000 sweep      -> outputs/run_mpi_15000_FIXED.txt ~22 min
#   7. final git commit+push
#
# Ctrl-C at any point is SAFE: a trap commits+pushes whatever exists.
# Screenshots: everything is also in outputs/final_run_master.log — at the end,
# `cat` each outputs/ file and screenshot the terminal (list printed at the end).

cd "$(dirname "$0")/.."
mkdir -p outputs
exec > >(tee -a outputs/final_run_master.log) 2>&1

say(){ echo; echo "##### [$(date +%H:%M:%S)] $*"; }
commit_push(){
    git add outputs >/dev/null 2>&1 || true
    git diff --cached --quiet 2>/dev/null || git commit -m "final lab run: $1" >/dev/null 2>&1 || true
    git push >/dev/null 2>&1 && echo ">> pushed to GitHub ($1)" || echo ">> push failed (commit is local — push manually later)"
}
trap 'say "INTERRUPTED — securing results"; commit_push "partial (interrupted)"; exit 130' INT TERM

command -v icx >/dev/null 2>&1 || source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
if command -v mpiicx >/dev/null 2>&1; then MPICC=mpiicx; MPIXHOST=-xCORE-AVX2;
else MPICC=mpicc; MPIXHOST=-march=native; fi

if ! grep -q KERNEL_VLA mpi/matmul_mpi.c; then
    echo "FATAL: mpi/matmul_mpi.c has no KERNEL_VLA variant — run 'git pull' first."; exit 1
fi

############################################################
say "1/6  Kernel bisect with icx (no MPI): which source form is slow?"
############################################################
icx -O3 -xHost -std=c11 scripts/kernel_bisect.c -o scripts/kernel_bisect
{ echo "# icx -O3 -xHost | $(icx --version 2>/dev/null | head -1)"
  ./scripts/kernel_bisect 2000 32 3
  ./scripts/kernel_bisect 2000 0 3
} 2>&1 | tee outputs/kernel_bisect_icx.txt

############################################################
say "2/6  A/B the REAL MPI binary: flat vs -DKERNEL_VLA (direct run, n=5000)"
############################################################
$MPICC -O3 $MPIXHOST -std=c11              mpi/matmul_mpi.c -o mpi/mm_flat
$MPICC -O3 $MPIXHOST -std=c11 -DKERNEL_VLA mpi/matmul_mpi.c -o mpi/mm_vla
OUT_FLAT=$(./mpi/mm_flat 5000 32 1); echo "flat : $OUT_FLAT" | tee -a outputs/kernel_bisect_icx.txt
OUT_VLA=$( ./mpi/mm_vla  5000 32 1); echo "vla  : $OUT_VLA"  | tee -a outputs/kernel_bisect_icx.txt
GF_FLAT=$(echo "$OUT_FLAT" | grep -o '[0-9.]\+ GFLOP' | head -1 | cut -d' ' -f1)
GF_VLA=$( echo "$OUT_VLA"  | grep -o '[0-9.]\+ GFLOP' | head -1 | cut -d' ' -f1)
KV=""; python3 -c "exit(0 if float('${GF_VLA:-0}') > float('${GF_FLAT:-0}')*1.15 else 1)" && KV=-DKERNEL_VLA
GF_BEST=$(python3 -c "print(max(float('${GF_FLAT:-0}'),float('${GF_VLA:-0}')))")
FIXED=no; python3 -c "exit(0 if $GF_BEST >= 24.0 else 1)" && FIXED=yes
echo ">> winner: ${KV:-flat} (flat=$GF_FLAT vla=$GF_VLA GFLOP/s) — kernel fixed: $FIXED" | tee -a outputs/kernel_bisect_icx.txt
$MPICC -O3 $MPIXHOST -std=c11 $KV       mpi/matmul_mpi.c -o mpi/matmul_mpi
$MPICC -O3 $MPIXHOST -std=c11 $KV -DP2P mpi/matmul_mpi.c -o mpi/matmul_mpi_p2p
echo ">> verify under mpirun -np 1:"
mpirun -np 1 ./mpi/matmul_mpi 5000 32 1 | tee -a outputs/kernel_bisect_icx.txt

############################################################
if [ "$FIXED" = yes ]; then
say "3/6  MPI n=10000 sweep with the fixed kernel (~12 min)"
############################################################
{ echo "# MPI wrapper: $MPICC | flags: -O3 $MPIXHOST $KV | N=10000 block=32 | KERNEL FIXED (P1=$GF_BEST GF/s direct)"
  echo "## Process scaling (strong)  N=10000 block=32  [collectives]"
  for P in 1 2 4 8 16 24; do mpirun -np $P ./mpi/matmul_mpi 10000 32 3; done
  echo; echo "## Collectives vs point-to-point  (N=10000, P=8)"
  mpirun -np 8 ./mpi/matmul_mpi     10000 32 3
  mpirun -np 8 ./mpi/matmul_mpi_p2p 10000 32 3
  echo; echo "## Weak scaling (n = 4000 * P^(1/3))"
  for PAIR in "1:4000" "2:5040" "4:6350" "8:8000"; do
      mpirun -np ${PAIR%:*} ./mpi/matmul_mpi ${PAIR#*:} 32 3; done
} 2>&1 | tee outputs/run_mpi_10000_FIXED.txt
else
say "3/6  Kernel NOT fixed — skipping MPI re-runs (today's pre-fix data stands); collecting diagnostics instead"
############################################################
{ echo "## diagnostics for the unresolved half-speed kernel"
  echo "### wrapper invocation:"
  mpiicx -show 2>&1 | head -3;  mpiicx -compile_info 2>&1 | head -3
  echo "### bisect built with the MPI wrapper (tests the link-time hypothesis):"
  $MPICC -O3 $MPIXHOST -std=c11 scripts/kernel_bisect.c -o scripts/kb_mpi 2>&1
  ./scripts/kb_mpi 2000 32 3
  echo "### vectorization report of the flat kernel:"
  ( cd mpi && icx -O3 -xHost -std=c11 -qopt-report=3 -c matmul_mpi.c -o /dev/null 2>&1
    grep -m 20 -i "vector\|remainder\|unroll" *.optrpt *.yaml 2>/dev/null )
} 2>&1 | tee -a outputs/kernel_bisect_icx.txt
fi

############################################################
say "4/6  Clean sequential n=15000 baseline (last one was perturbed: 237 min vs 258 med)"
############################################################
icx -O3 -xHost -std=c11 seq/matmul_seq.c -o seq/matmul_seq
{ echo "== clean sequential baseline n=15000, block=32, reps=3 =="
  ./seq/matmul_seq 15000 32 3
} 2>&1 | tee outputs/seq_15000_clean.txt

############################################################
say "5/6  Advisor roofline (guarded; skipped automatically if advisor is missing/broken)"
############################################################
if command -v advisor >/dev/null 2>&1; then
    icx -O3 -xHost -g -std=c11 seq/matmul_seq.c -o seq/matmul_seq
    rm -rf ./adv
    timeout 300 advisor --collect=survey            --project-dir=./adv -- ./seq/matmul_seq 5000 32 1 \
 && timeout 600 advisor --collect=tripcounts --flop --project-dir=./adv -- ./seq/matmul_seq 5000 32 1 \
 && timeout 120 advisor --report=roofline --project-dir=./adv --report-output=outputs/roofline.html \
 && echo ">> roofline OK -> outputs/roofline.html (open at home, hover the big dot, screenshot)" \
 || echo ">> advisor failed or timed out — roofline stays a report blank, not a problem"
else echo ">> advisor not found — skipping roofline"; fi
commit_push "bisect+A/B, mpi10000(if fixed), clean seq15000, roofline"

############################################################
if [ "$FIXED" = yes ]; then
say "6/6  MPI n=15000 sweep (~22 min, the long one — Ctrl-C here is safe, partial lines still count)"
############################################################
{ echo "# MPI wrapper: $MPICC | flags: -O3 $MPIXHOST $KV | N=15000 block=32 | KERNEL FIXED"
  echo "## Process scaling (strong)  N=15000 block=32  [collectives, P<=16: B is 1.8 GB/rank]"
  for P in 1 2 4 8 16; do mpirun -np $P ./mpi/matmul_mpi 15000 32 3; done
} 2>&1 | tee outputs/run_mpi_15000_FIXED.txt
else say "6/6  skipped (kernel not fixed)"; fi

rm -f mpi/mm_flat mpi/mm_vla scripts/kb_mpi
commit_push "final"
say "ALL DONE — kernel fixed: $FIXED"
echo "For the professor's screenshots, cat + screenshot each of these:"
echo "   cat outputs/kernel_bisect_icx.txt"
[ "$FIXED" = yes ] && echo "   cat outputs/run_mpi_10000_FIXED.txt" && echo "   cat outputs/run_mpi_15000_FIXED.txt"
echo "   cat outputs/seq_15000_clean.txt"
echo "(roofline: open outputs/roofline.html in a browser — can be done at home)"
