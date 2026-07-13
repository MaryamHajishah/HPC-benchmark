#!/usr/bin/env bash
# PREFLIGHT for the PC2 full re-run.
#   bash scripts/preflight_pc2.sh
# Verifies toolchain, hardware, memory and disk, then compiles every binary
# (seq / omp / mpi) and runs a tiny correctness test on each (max_err must be 0).
# Exit code 0 = safe to run scripts/run_all_pc2.sh.   Takes ~1 minute.
set -u
export LC_ALL=C
cd "$(dirname "$0")/.."

PASS=0; WARN=0; FAIL=0
ok()   { printf '[ OK ] %s\n' "$1"; PASS=$((PASS+1)); }
warn() { printf '[WARN] %s\n' "$1"; WARN=$((WARN+1)); }
fail() { printf '[FAIL] %s\n' "$1"; FAIL=$((FAIL+1)); }
gf_of() { grep -oE '[0-9]+(\.[0-9]+)? GFLOP/s' <<<"$1" | head -1 | cut -d' ' -f1; }
err_ok() { grep -qE 'max_err=0(\.0+)?( |$)' <<<"$1"; }   # exact zero only

echo "================ HPC-benchmark preflight — $(hostname), $(date) ================"
echo

# ---------- 1. Intel oneAPI environment -------------------------------------
if ! command -v icx >/dev/null 2>&1 && [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

# ---------- 2. Compilers ------------------------------------------------------
if command -v icx >/dev/null 2>&1; then
    CC=icx; XHOST=-xHost; OMPF=-qopenmp
    ok "icx found: $(icx --version 2>/dev/null | head -1)"
else
    CC=gcc; XHOST=-march=native; OMPF=-fopenmp
    warn "icx NOT found — falling back to gcc ($(gcc --version | head -1)). Lab runs must use icx: run 'source /opt/intel/oneapi/setvars.sh --force' first."
fi
if command -v mpiicx >/dev/null 2>&1; then MPICC=mpiicx; MPIXHOST=-xHost; ok "mpiicx found (Intel MPI wrapper)";
elif command -v mpicc >/dev/null 2>&1; then MPICC=mpicc; MPIXHOST=-march=native; warn "mpiicx not found — using mpicc ($(mpicc --version 2>/dev/null | head -1))";
else MPICC=""; fail "no MPI compiler (mpiicx/mpicc) found"; fi
if command -v mpirun >/dev/null 2>&1; then ok "mpirun found: $(mpirun --version 2>/dev/null | head -1)";
else fail "mpirun not found"; fi

# ---------- 3. Hardware -------------------------------------------------------
MODEL=$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)
NCPU=$(nproc)
if grep -q "12900K" <<<"$MODEL"; then ok "CPU is the expected lab chip: $MODEL"; else warn "CPU is '$MODEL' — NOT an i9-12900K; timings will not match PC1/the report"; fi
if [ "$NCPU" -eq 24 ]; then ok "24 logical CPUs (8P+8E, HT on)"; else warn "nproc=$NCPU (expected 24) — thread/process sweeps will be capped at $NCPU"; fi

AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
# MPI n=15000, P=16 needs ~36 GB (rank0: A+B+C = 5.4 GB, 15 ranks × (B + stripes) ≈ 2 GB each)
if   [ "$AVAIL_GB" -ge 40 ]; then ok "MemAvailable = ${AVAIL_GB} GB (enough for MPI n=15000 up to P=16)"
elif [ "$AVAIL_GB" -ge 22 ]; then warn "MemAvailable = ${AVAIL_GB} GB — MPI n=15000 will be capped at P=8 (P=16 needs ~36 GB). run_all_pc2.sh caps automatically."
else warn "MemAvailable = ${AVAIL_GB} GB — n=15000 MPI runs will be heavily capped; close other programs"; fi

DISK_KB=$(df -Pk . | awk 'NR==2{print $4}')
if [ "$DISK_KB" -ge 2097152 ]; then ok "Disk free: $((DISK_KB/1024/1024)) GB"; else fail "Less than 2 GB free disk here"; fi

LOAD=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$LOAD" 'BEGIN{exit !(l < 1.0)}'; then ok "System idle (load $LOAD)"; else warn "Load average is $LOAD — someone/something else is running; timings will be polluted. Close it or pick another PC."; fi

# ---------- 4. Repo + write access --------------------------------------------
for f in seq/matmul_seq.c omp/matmul_omp.c mpi/matmul_mpi.c scripts/run_seq.sh scripts/run_omp.sh; do
    [ -f "$f" ] && ok "found $f" || fail "missing $f — clone/pull the full repo first"
done
mkdir -p outputs && touch outputs/.write_test 2>/dev/null && rm -f outputs/.write_test \
    && ok "outputs/ is writable" || fail "cannot write to outputs/"

# ---------- 5. Compile + tiny correctness test (oracle must be 0) --------------
echo
echo "--- compile + correctness smoke tests (n=400) ---"
if $CC -O3 $XHOST -std=c11 seq/matmul_seq.c -o seq/matmul_seq 2>/tmp/pf_err; then
    OUTP=$(./seq/matmul_seq 400 32 1)
    echo "  seq : $OUTP"
    err_ok "$OUTP" && ok "sequential compiles and max_err=0" || fail "sequential kernel WRONG RESULT"
else fail "sequential does not compile: $(head -1 /tmp/pf_err)"; fi

if $CC -O3 $XHOST $OMPF -std=c11 omp/matmul_omp.c -o omp/matmul_omp 2>/tmp/pf_err; then
    OUTP=$(OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./omp/matmul_omp 400 32 1)
    echo "  omp : $OUTP"
    err_ok "$OUTP" && ok "OpenMP compiles and max_err=0 (4 threads)" || fail "OpenMP kernel WRONG RESULT"
else fail "OpenMP does not compile: $(head -1 /tmp/pf_err)"; fi

if [ -n "$MPICC" ]; then
    if $MPICC -O3 $MPIXHOST -std=c11 mpi/matmul_mpi.c -o mpi/matmul_mpi 2>/tmp/pf_err \
       && $MPICC -O3 $MPIXHOST -std=c11 -DP2P mpi/matmul_mpi.c -o mpi/matmul_mpi_p2p 2>>/tmp/pf_err; then
        MPIRUN="mpirun"
        mpirun --version 2>/dev/null | head -1 | grep -qi "open mpi" && MPIRUN="mpirun --oversubscribe"
        if OUTP=$($MPIRUN -np 2 ./mpi/matmul_mpi 400 32 1 2>&1); then
            echo "  mpi : $OUTP"
            err_ok "$OUTP" && ok "MPI (collective+p2p) compiles and max_err=0 (2 procs)" || fail "MPI kernel WRONG RESULT"
        else fail "mpirun -np 2 failed: $(head -1 <<<"$OUTP")"; fi
    else fail "MPI does not compile: $(head -1 /tmp/pf_err)"; fi
fi

# ---------- 6. Vectorization sanity -------------------------------------------
echo
echo "--- vectorization sanity (seq, n=1500, no blocking) ---"
OUTP=$(./seq/matmul_seq 1500 0 1); echo "  $OUTP"
GF=$(gf_of "$OUTP")
if [ "$CC" = icx ]; then THRESH=15; else THRESH=5; fi
if awk -v g="$GF" -v t="$THRESH" 'BEGIN{exit !(g >= t)}'; then
    ok "sequential kernel at $GF GFLOP/s (>= $THRESH ⇒ vectorization active)"
else
    fail "only $GF GFLOP/s — vectorization looks OFF; check that $XHOST was accepted (grep the -qopt-report)"
fi

# ---------- 7. Optional tools ---------------------------------------------------
command -v advixe-cl >/dev/null 2>&1 || command -v advisor >/dev/null 2>&1 \
    && ok "Intel Advisor found (roofline, §4.5)" \
    || warn "Intel Advisor not found — the roofline screenshot (§4.5) can't be taken on this PC"
command -v git >/dev/null 2>&1 && ok "git found (commit outputs at the end)" || warn "git not found — copy outputs/ home manually"

# ---------- summary --------------------------------------------------------------
echo
echo "================ SUMMARY: $PASS ok, $WARN warnings, $FAIL failures ================"
if [ "$FAIL" -gt 0 ]; then
    echo "DO NOT run run_all_pc2.sh until every [FAIL] above is resolved."
    exit 1
fi
echo "All checks passed. Run:   bash scripts/run_all_pc2.sh"
echo "(expected total: ~2–3 h; the script pauses after each block so you can screenshot)"
