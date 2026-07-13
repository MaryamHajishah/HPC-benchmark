#!/usr/bin/env bash
# FULL PC2 TEST SESSION — everything except CUDA (CUDA runs on Colab).
#
#   bash scripts/preflight_pc2.sh          # run this FIRST — must end in "All checks passed"
#   bash scripts/run_all_pc2.sh            # interactive: shows output live + pauses for screenshots
#
#   NOPAUSE=1 bash scripts/run_all_pc2.sh          # no screenshot pauses
#   SMOKE=1 NOPAUSE=1 bash scripts/run_all_pc2.sh  # tiny sizes — logic test only, ~1 min
#
# Every block prints to the TERMINAL (screenshot it when prompted) AND is saved
# under outputs/pc2_<host>_<date>/.  At the end, results are also copied to the
# canonical filenames that scripts/make_plots.py reads.
#
# Expected wall time on the lab i9-12900K (excluding pauses):
#   specs 1' · loop-order -O0 1' · run_seq 7' · run_omp 2' · confirm-10000 14'
#   · confirm-15000 47' · MPI diagnosis 4' · MPI-10000 10-20' · MPI-15000 25-50'
#   ≈ 2 h – 2 h 45'.  Run blocks strictly one at a time (this script does).
set -uo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
trap 'echo; echo "*** ABORTED in block: ${BLOCK:-setup} (line $LINENO). Files so far are in ${OUT:-outputs}. ***"' ERR
set -e

SMOKE=${SMOKE:-0}
NOPAUSE=${NOPAUSE:-0}

# ---------------- sizes ------------------------------------------------------
if [ "$SMOKE" = 1 ]; then
    N_LOOP=400; N_MED=500; N_BIG1=600; N_BIG2=800; REPS=1
    PLIST="1 2 4"; PLIST15="1 2 4"; WEAK_PAIRS="1:300 2:380 4:480"
else
    N_LOOP=2000; N_MED=5000; N_BIG1=10000; N_BIG2=15000; REPS=3
    PLIST="1 2 4 8 16 24"; PLIST15="1 2 4 8 16"; WEAK_PAIRS="1:4000 2:5040 4:6350 8:8000"
fi
MAXP=$(nproc)

# ---------------- toolchain (same choices as the Makefile) --------------------
if ! command -v icx >/dev/null 2>&1 && [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi
if command -v icx >/dev/null 2>&1; then CC=icx; XHOST=-xHost; OMPF=-qopenmp;
else CC=gcc; XHOST=-march=native; OMPF=-fopenmp; fi
if command -v mpiicx >/dev/null 2>&1; then MPICC=mpiicx; MPIXHOST=-xHost;
else MPICC=mpicc; MPIXHOST=-march=native; fi
MPIRUN="mpirun"
MPI_FLAVOR="$(mpirun --version 2>/dev/null | head -1)"
grep -qi "open mpi" <<<"$MPI_FLAVOR" && MPIRUN="mpirun --oversubscribe"

OUT="outputs/pc2_$(hostname)_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"

gf_of() { grep -oE '[0-9]+(\.[0-9]+)? GFLOP/s' <<<"$1" | head -1 | cut -d' ' -f1; }
pause() {
    [ "$NOPAUSE" = 1 ] && return 0
    [ -t 0 ] || return 0
    echo; read -r -p ">>> 📸  SCREENSHOT the block above, then press Enter to continue... " _ || true
    echo
}
banner() {
    BLOCK="$1"
    echo; echo "############################################################"
    echo "###  $1   (expected: $2)"
    echo "###  $(date '+%H:%M:%S')  ->  $OUT"
    echo "############################################################"
}

echo "run_all_pc2.sh — host=$(hostname) date=$(date) CC=$CC MPICC=$MPICC SMOKE=$SMOKE"
echo "MPI runtime: ${MPI_FLAVOR:-unknown}"
echo "Output dir : $OUT"
[ "$SMOKE" = 1 ] && echo "*** SMOKE MODE: tiny sizes, results are NOT report data ***"

# ---------------- build everything upfront ------------------------------------
BLOCK="build"
$CC -O3 $XHOST        -std=c11 seq/matmul_seq.c -o seq/matmul_seq
$CC -O3 $XHOST $OMPF  -std=c11 omp/matmul_omp.c -o omp/matmul_omp
$MPICC -O3 $MPIXHOST  -std=c11 mpi/matmul_mpi.c -o mpi/matmul_mpi
$MPICC -O3 $MPIXHOST  -std=c11 -DP2P mpi/matmul_mpi.c -o mpi/matmul_mpi_p2p
echo "build OK ($CC $XHOST | $MPICC $MPIXHOST)"

# ================ 1/8 SPECS ====================================================
banner "1/8  MACHINE SPECS" "1 min"
bash scripts/specs.sh 2>&1 | tee "$OUT/specs.txt"
pause

# ================ 2/8 LOOP ORDER AT -O0 ========================================
banner "2/8  LOOP ORDER at -O0 (raw cache effect, n=$N_LOOP)" "1 min"
{
    $CC -O0 -DNAIVE -std=c11 seq/matmul_seq.c -o seq/matmul_naive_O0
    $CC -O0         -std=c11 seq/matmul_seq.c -o seq/matmul_seq_O0
    printf "i-j-k -O0  "; ./seq/matmul_naive_O0 "$N_LOOP" 0 1
    printf "i-k-j -O0  "; ./seq/matmul_seq_O0   "$N_LOOP" 0 1
} 2>&1 | tee "$OUT/looporder_O0.txt"
pause

# ================ 3/8 SEQUENTIAL SWEEPS ========================================
banner "3/8  SEQUENTIAL: loop order / flag sweep / block sweep" "7 min (the -O0 run at n=$N_MED alone is ~3 min)"
N1=$N_LOOP N2=$N_MED CC=$CC XHOST=$XHOST bash scripts/run_seq.sh 2>&1 | tee "$OUT/run_seq.txt"
pause

# ================ 4/8 OPENMP SWEEPS ============================================
banner "4/8  OPENMP: thread / block / schedule sweeps (n=$N_MED)" "2 min"
N=$N_MED BLK=32 CC=$CC XHOST=$XHOST bash scripts/run_omp.sh 2>&1 | tee "$OUT/run_omp.txt"
pause

# ================ 5/8 CONFIRM n=N_BIG1 =========================================
banner "5/8  CONFIRM n=$N_BIG1: sequential baseline + OpenMP strong scaling" "14 min"
{
    echo "== sequential baseline (>=60s) =="
    ./seq/matmul_seq "$N_BIG1" 0  "$REPS"
    ./seq/matmul_seq "$N_BIG1" 32 "$REPS"
    echo "== OpenMP strong scaling =="
    for T in 1 2 4 8 16 24; do
        [ "$T" -le "$MAXP" ] || { echo "skip T=$T (>$MAXP cores)"; continue; }
        OMP_NUM_THREADS=$T OMP_SCHEDULE=static ./omp/matmul_omp "$N_BIG1" 32 "$REPS"
    done
} 2>&1 | tee "$OUT/confirm_${N_BIG1}_b32.txt"
pause

# ================ 6/8 CONFIRM n=N_BIG2 =========================================
banner "6/8  CONFIRM n=$N_BIG2: sequential baseline + OpenMP strong scaling" "47 min — the long one, let it run"
{
    echo "== sequential baseline n=$N_BIG2 =="
    ./seq/matmul_seq "$N_BIG2" 0  "$REPS"
    ./seq/matmul_seq "$N_BIG2" 32 "$REPS"
    echo "== OpenMP strong scaling n=$N_BIG2 =="
    for T in 1 2 4 8 16 24; do
        [ "$T" -le "$MAXP" ] || { echo "skip T=$T (>$MAXP cores)"; continue; }
        OMP_NUM_THREADS=$T OMP_SCHEDULE=static ./omp/matmul_omp "$N_BIG2" 32 "$REPS"
    done
} 2>&1 | tee "$OUT/confirm_${N_BIG2}_b32.txt"
pause

# ================ 7/8 MPI DIAGNOSIS (§6.5) =====================================
banner "7/8  MPI DIAGNOSIS — is the kernel at half speed, and why?" "4 min"
{
    echo "# MPI wrapper : $MPICC ($($MPICC --version 2>/dev/null | head -1))"
    echo "# MPI runtime : $MPI_FLAVOR"
    echo
    echo "## Reference: sequential kernel, n=$N_MED block=32"
    SEQ_OUT=$(./seq/matmul_seq "$N_MED" 32 1); echo "seq       : $SEQ_OUT"
    SEQ_GF=$(gf_of "$SEQ_OUT")

    echo
    echo "## Test 1 — MPI binary run DIRECTLY (no mpirun): isolates the BUILD"
    DIR_OUT=$(./mpi/matmul_mpi "$N_MED" 32 1); echo "direct    : $DIR_OUT"
    DIR_GF=$(gf_of "$DIR_OUT")

    if awk -v a="$DIR_GF" -v b="$SEQ_GF" 'BEGIN{exit !(a < 0.7*b)}'; then
        echo ">> direct run is <70% of sequential ($DIR_GF vs $SEQ_GF) => BUILD problem (Hypothesis A)."
        if [ "$MPICC" = mpiicx ]; then
            echo ">> what the wrapper really invokes:"; $MPICC -show || true
            echo ">> rebuilding with explicit -xCORE-AVX2 ..."
            MPIXHOST="-xCORE-AVX2"
            $MPICC -O3 $MPIXHOST -std=c11 mpi/matmul_mpi.c -o mpi/matmul_mpi
            $MPICC -O3 $MPIXHOST -std=c11 -DP2P mpi/matmul_mpi.c -o mpi/matmul_mpi_p2p
            DIR_OUT=$(./mpi/matmul_mpi "$N_MED" 32 1); echo "rebuilt   : $DIR_OUT"
            DIR_GF=$(gf_of "$DIR_OUT")
            awk -v a="$DIR_GF" -v b="$SEQ_GF" 'BEGIN{exit !(a >= 0.85*b)}' \
                && echo ">> HYPOTHESIS A CONFIRMED: -xCORE-AVX2 fixed the build ($DIR_GF GFLOP/s). All runs below use it." \
                || echo ">> rebuild did NOT fix it ($DIR_GF GFLOP/s) — investigate manually (see LAB-CHECKLIST-MPI.md) before trusting the runs below."
        else
            echo ">> (non-Intel wrapper — check compile flags manually)"
        fi
    else
        echo ">> direct run matches sequential ($DIR_GF vs $SEQ_GF GFLOP/s) => the BINARY is fine."
    fi

    echo
    echo "## Test 2 — same binary under mpirun -np 1: isolates the LAUNCHER (pinning)"
    PLAIN_OUT=$($MPIRUN -np 1 ./mpi/matmul_mpi "$N_MED" 32 1); echo "mpirun    : $PLAIN_OUT"
    PLAIN_GF=$(gf_of "$PLAIN_OUT")
    if grep -qi "open mpi" <<<"$MPI_FLAVOR"; then
        PIN_OUT=$($MPIRUN --bind-to core -np 1 ./mpi/matmul_mpi "$N_MED" 32 1)
    else
        PIN_OUT=$(I_MPI_PIN_PROCESSOR_LIST=0 $MPIRUN -np 1 ./mpi/matmul_mpi "$N_MED" 32 1)
    fi
    echo "pinned    : $PIN_OUT"
    PIN_GF=$(gf_of "$PIN_OUT")

    PIN_ENV=0
    if awk -v p="$PLAIN_GF" -v d="$DIR_GF" 'BEGIN{exit !(p < 0.8*d)}' \
       && awk -v q="$PIN_GF" -v d="$DIR_GF" 'BEGIN{exit !(q >= 0.9*d)}'; then
        echo ">> HYPOTHESIS B CONFIRMED: mpirun pins the rank onto a slow core ($PLAIN_GF plain vs $PIN_GF pinned)."
        PIN_ENV=1
    else
        echo ">> launcher looks fine (plain $PLAIN_GF vs pinned $PIN_GF vs direct $DIR_GF GFLOP/s)."
    fi

    # Persist the decisions to a state file: this block runs in a subshell
    # (because of the | tee), so plain exports would NOT reach the runs below.
    {
        echo "MPIXHOST=\"$MPIXHOST\""
        if [ "$PIN_ENV" = 1 ] && [ "$MAXP" -eq 24 ] && ! grep -qi "open mpi" <<<"$MPI_FLAVOR"; then
            # P-cores (one rank per physical P-core) first, then HT siblings, then E-cores.
            echo "export I_MPI_PIN_PROCESSOR_LIST=\"0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15,16,17,18,19,20,21,22,23\""
        elif [ "$PIN_ENV" = 1 ] && grep -qi "open mpi" <<<"$MPI_FLAVOR"; then
            echo "export OMPI_MCA_hwloc_base_binding_policy=core"
        fi
    } > "$OUT/.mpi_env"
    [ "$PIN_ENV" = 1 ] && echo ">> pinning WILL BE APPLIED to all MPI runs below (see .mpi_env)."
    echo
    echo "## Verdict summary:  seq=$SEQ_GF  direct=$DIR_GF  mpirun=$PLAIN_GF  pinned=$PIN_GF  (GFLOP/s)"
} 2>&1 | tee "$OUT/mpi_diagnosis.txt"
# Apply the diagnosis decisions in THIS shell (flags + optional pinning).
# shellcheck disable=SC1090,SC1091
. "$OUT/.mpi_env"
echo "applied: MPIXHOST=$MPIXHOST  pin=${I_MPI_PIN_PROCESSOR_LIST:-${OMPI_MCA_hwloc_base_binding_policy:-none}}"
pause

# ================ 8/8 MPI FULL RUNS ============================================
banner "8/8a MPI n=$N_BIG1: strong scaling + collectives-vs-p2p + weak scaling" "10-20 min"
{
    echo "# MPI wrapper: $MPICC | flags: -O3 $MPIXHOST | N=$N_BIG1 block=32 | cores=$MAXP"
    echo "# pinning: I_MPI_PIN_PROCESSOR_LIST=${I_MPI_PIN_PROCESSOR_LIST:-<unset>} OMPI_bind=${OMPI_MCA_hwloc_base_binding_policy:-<unset>}"
    echo
    echo "## 1) Process scaling (strong)  N=$N_BIG1 block=32  [collectives]"
    for P in $PLIST; do
        [ "$P" -le "$MAXP" ] || { echo "skip P=$P (>$MAXP cores)"; continue; }
        $MPIRUN -np "$P" ./mpi/matmul_mpi "$N_BIG1" 32 "$REPS"
    done
    echo
    P8=$(( MAXP < 8 ? MAXP : 8 ))
    echo "## 2) Collectives vs point-to-point  (N=$N_BIG1, P=$P8)"
    $MPIRUN -np "$P8" ./mpi/matmul_mpi     "$N_BIG1" 32 "$REPS"
    $MPIRUN -np "$P8" ./mpi/matmul_mpi_p2p "$N_BIG1" 32 "$REPS"
    echo
    echo "## 3) Weak scaling (per-process work ~constant: n = n1 * P^(1/3))"
    for PAIR in $WEAK_PAIRS; do
        P=${PAIR%:*}; NW=${PAIR#*:}
        [ "$P" -le "$MAXP" ] || { echo "skip P=$P"; continue; }
        $MPIRUN -np "$P" ./mpi/matmul_mpi "$NW" 32 "$REPS"
    done
} 2>&1 | tee "$OUT/run_mpi_${N_BIG1}_FIXED.txt"
pause

banner "8/8b MPI n=$N_BIG2: strong scaling (P capped by memory)" "25-50 min"
{
    echo "# MPI wrapper: $MPICC | flags: -O3 $MPIXHOST | N=$N_BIG2 block=32"
    echo "## Process scaling (strong)  N=$N_BIG2 block=32  [collectives]"
    AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    for P in $PLIST15; do
        [ "$P" -le "$MAXP" ] || { echo "skip P=$P (>$MAXP cores)"; continue; }
        # memory: rank0 holds A+B+C (3B); every other rank holds B + its A/C stripes (B + 2B/P)
        NEED_KB=$(awk -v n="$N_BIG2" -v p="$P" 'BEGIN{B=n*n*8/1024; printf "%d", 3*B + (p-1)*(B + 2*B/p) + 2097152}')
        if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
            echo "skip P=$P (needs ~$((NEED_KB/1024/1024)) GB, only $((AVAIL_KB/1024/1024)) GB available)"
            continue
        fi
        $MPIRUN -np "$P" ./mpi/matmul_mpi "$N_BIG2" 32 "$REPS"
    done
} 2>&1 | tee "$OUT/run_mpi_${N_BIG2}_FIXED.txt"
pause

# ================ canonical copies + summary ===================================
BLOCK="finish"
if [ "$SMOKE" != 1 ]; then
    cp "$OUT/specs.txt"                    outputs/specs.txt
    cp "$OUT/looporder_O0.txt"             outputs/looporder_O0.txt
    cp "$OUT/run_seq.txt"                  outputs/run_seq.txt
    cp "$OUT/run_omp.txt"                  outputs/run_omp.txt
    cp "$OUT/confirm_${N_BIG1}_b32.txt"    outputs/confirm_10000_b32.txt
    cp "$OUT/confirm_${N_BIG2}_b32.txt"    outputs/confirm_15000_b32.txt
    cp "$OUT/run_mpi_${N_BIG1}_FIXED.txt"  outputs/run_mpi_10000_FIXED.txt
    cp "$OUT/run_mpi_${N_BIG2}_FIXED.txt"  outputs/run_mpi_15000_FIXED.txt
    echo "Canonical copies written to outputs/ (make_plots.py will now use the FIXED MPI data)."
fi

echo
echo "################ ALL DONE — $(date '+%H:%M:%S') ################"
echo "Raw session files:  $OUT/"
ls -l "$OUT"
echo
echo "Still to do BY HAND on this PC:"
echo "  1. Intel Advisor roofline (screenshot the GUI):"
echo "       advixe-cl -collect roofline -project-dir ./adv -- ./seq/matmul_seq 5000 32 3"
echo "       advixe-gui ./adv &"
echo "  2. git add outputs && git commit -m 'PC2 full re-run' && git push"
echo "  3. back home:  python3 scripts/make_plots.py   (figures update automatically)"
