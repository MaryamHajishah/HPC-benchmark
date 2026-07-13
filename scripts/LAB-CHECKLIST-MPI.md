# Lab checklist — fix the MPI half-speed kernel, then re-run everything

**The problem:** the MPI kernel at P=1 reaches only ~14–15 GFLOP/s, while the identical
sequential kernel reaches ~32 GFLOP/s. 15 ≈ half of 32, and there are exactly two things
that can halve this kernel on our machine. Test both, in this order.

**Take screenshots of every terminal command + output** (the professor wants them),
and redirect everything into `outputs/` as shown.

---

## 0. Setup (once per session)

```bash
cd ~/HPC-benchmark        # or wherever the repo is
source /opt/intel/oneapi/setvars.sh --force
make mpi                  # builds mpi/matmul_mpi and mpi/matmul_mpi_p2p
```

## 1. Hypothesis A — the binary itself is slow (build/flags problem)

Run the MPI binary **directly, without mpirun** (an MPI program with no launcher runs
as a single process — same math, no runtime involved):

```bash
./mpi/matmul_mpi 5000 32 3 | tee -a ../outputs/mpi_diag.txt
```

* **~30 GFLOP/s → the binary is FINE.** The build was never the problem; it's the
  launcher (go to step 2).
* **~15 GFLOP/s → it IS the build.** Check what the wrapper really invokes and force AVX2:

```bash
mpiicx -show | tee -a outputs/mpi_diag.txt        # shows the real compiler + flags
cd mpi
mpiicx -O3 -xCORE-AVX2 -qopt-report=3 -std=c11 matmul_mpi.c -o matmul_mpi
grep -i "vector" matmul_mpi.optrpt | head          # kernel loop must say VECTORIZED
./matmul_mpi 5000 32 3                             # expect ~30 GFLOP/s now
```

## 2. Hypothesis B — mpirun pins rank 0 onto an E-core

Our i9-12900K is hybrid: 8 fast P-cores (logical CPUs **0–15**, hyper-threaded pairs)
plus 8 slow E-cores (logical CPUs **16–23**). An E-core runs this kernel at roughly
half P-core speed — which also fits "15 vs 32" exactly. Intel MPI pins processes by
default, and it does not know P-cores from E-cores.

See where ranks actually land:

```bash
I_MPI_DEBUG=4 mpirun -np 1 ./mpi/matmul_mpi 5000 32 3 2>&1 | tee -a outputs/mpi_diag.txt
# look for the "Rank ... Pin cpu" table in the output
```

Force ranks onto P-cores (one rank per physical P-core) and re-time:

```bash
I_MPI_PIN_PROCESSOR_LIST=0,2,4,6,8,10,12,14 mpirun -np 1 ./mpi/matmul_mpi 5000 32 3
```

(If the lab uses OpenMPI instead of Intel MPI: `mpirun --bind-to core --report-bindings -np 1 ...`)

## 3. Confirm the fix

Whichever hypothesis was right, one P=1 run must now show **~30 GFLOP/s**:

```bash
mkdir -p outputs
mpirun -np 1 ./mpi/matmul_mpi 5000 32 3 | tee -a outputs/mpi_fixed_confirm.txt
```

Write down (for the report, section 7.5): which hypothesis was correct, the exact
command/flag/env-var that fixed it, and the before/after GFLOP/s.

## 4. Re-run the full MPI experiment set (with the fix applied)

If the fix was a **pin list**, export it first so every run below uses it:

```bash
export I_MPI_PIN_PROCESSOR_LIST=0,2,4,6,8,10,12,14   # only if Hypothesis B
```

If the fix was a **flag**, edit `scripts/run_mpi.sh` line `XHOST=${XHOST:--xHost}`
to `-xCORE-AVX2` (or just prepend `XHOST=-xCORE-AVX2` to each command).

```bash
# strong scaling + collectives-vs-p2p + weak scaling, n=10000  (~30–40 min)
N=10000 BLK=32 bash scripts/run_mpi.sh >> outputs/run_mpi_10000_FIXED.txt 2>&1

# strong scaling at n=15000  (~60–75 min — start it and let it run)
N=15000 BLK=32 bash scripts/run_mpi.sh >> outputs/run_mpi_15000_FIXED.txt 2>&1
```

Note: with only 8 P-cores pinned, the P=16 and P=24 points will spill onto E-cores /
hyperthreads — that is fine and is itself a result we explain in the report.

## 5. Intel Advisor roofline (one run, one screenshot) — report section 4.5

```bash
cd seq && icx -O3 -xHost -g -std=c11 matmul_seq.c -o matmul_seq && cd ..
advixe-cl -collect roofline -project-dir ./adv -- ./seq/matmul_seq 5000 32 3
advixe-gui ./adv &      # open GUI, Survey & Roofline view → screenshot the roofline
```

Screenshot with the kernel dot visible; note its arithmetic-intensity (x) and GFLOP/s (y).
If `advixe-cl` is missing try `advisor --collect=roofline ...` (newer oneAPI name).

## 6. Bring home

* `outputs/mpi_diag.txt`, `outputs/mpi_fixed_confirm.txt`
* `outputs/run_mpi_10000_FIXED.txt`, `outputs/run_mpi_15000_FIXED.txt`
* the Advisor roofline screenshot (PNG)
* terminal screenshots of each block above

Then: `git add outputs && git commit -m "MPI fixed runs + roofline" && git push`
(or just send me the files) and I fill sections 7.2–7.5 and 4.5 of the report.
