#!/usr/bin/env python3
"""Generate every figure of the HPC report from the raw files in outputs/.

Usage:  python3 scripts/make_plots.py          (from the repo root)
Output: report/figures/*.png  (300 dpi, ready to insert in the report)

Re-run this after adding outputs/run_mpi_10000_FIXED.txt and
outputs/run_mpi_15000_FIXED.txt — the MPI figures switch to the fixed data
automatically (and drop the "PRE-FIX" watermark).
"""
import os, re, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
FIG  = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

# The single sequential baselines (block=32, -O3 -xHost) reused everywhere.
TSEQ = {5000: 7.732878, 10000: 61.610451, 15000: 208.155467}

C = {"seq": "#4C72B0", "omp": "#DD8452", "mpi": "#55A868", "cuda": "#C44E52",
     "ideal": "#888888", "fp64": "#8172B3"}

def read(fname):
    p = os.path.join(OUT, fname)
    return open(p).read() if os.path.exists(p) else ""

RE_TMIN  = re.compile(r"t_min=([\d.]+) s")
RE_GF    = re.compile(r"([\d.]+) GFLOP/s")
RE_OMP   = re.compile(r"omp n=(\d+) block=(\d+) threads=(\d+) sched=(\w+).*?t_min=([\d.]+) s.*?([\d.]+) GFLOP/s")
RE_MPI   = re.compile(r"mpi n=(\d+) block=(\d+) procs=(\d+) mode=(\w+).*?t_tot=([\d.]+) s \(comm=([\d.]+) comp=([\d.]+), comm%=([\d.]+)\) \| ([\d.]+) GFLOP/s")
RE_CUDA  = re.compile(r"cuda (fp\d+)\s+(\S+)\s+n=(\d+) \| kernel=([\d.]+) ms \| ([\d.]+) GFLOP/s \| H2D=([\d.]+) ms")

def bar(ax, labels, values, color, unit="GFLOP/s", fmt="{:.1f}"):
    b = ax.bar(labels, values, color=color, width=0.6)
    ax.bar_label(b, fmt=fmt, padding=2, fontsize=9)
    ax.set_ylabel(unit); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=300); plt.close(fig)
    print("wrote", name)

# ---------------------------------------------------------------- Fig 4.1 loop order
def fig_looporder():
    txt = read("looporder_O0.txt")
    gf = {m.group(1): float(m.group(2)) for m in
          re.finditer(r"(i-\w-\w) -O0.*?([\d.]+) GFLOP/s", txt)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4))
    bar(a1, ["i-j-k", "i-k-j"], [gf.get("i-j-k", 0), gf.get("i-k-j", 0)], C["seq"], fmt="{:.2f}")
    a1.set_title("-O0: cache effect visible (2.0×)", fontsize=10)
    bar(a2, ["i-j-k", "i-k-j"], [25.29, 25.29], C["seq"], fmt="{:.2f}")
    a2.set_title("-O3 -xHost: compiler interchange\n→ identical", fontsize=10)
    fig.suptitle("Fig 4.1 — Loop order, n=2000 (GFLOP/s)", fontsize=11)
    save(fig, "fig4_1_looporder.png")

# ---------------------------------------------------------------- Fig 4.2 flags
def fig_flags():
    txt = read("run_seq.txt")
    sec = txt.split("## Flag sweep")[1].split("##")[0]
    labels, vals = [], []
    for line in sec.strip().splitlines():
        m = re.match(r"(-\S+(?: -\S+)?)\s+n=5000.*?([\d.]+) GFLOP/s", line)
        if m: labels.append(m.group(1)); vals.append(float(m.group(2)))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bar(ax, labels, vals, C["seq"], fmt="{:.2f}")
    ax.set_title("Fig 4.2 — Compiler flags, sequential i-k-j, n=5000")
    save(fig, "fig4_2_flags.png")

# ---------------------------------------------------------------- Fig 4.3 block sweep
def fig_blocks_seq():
    txt = read("run_seq.txt")
    sec = txt.split("## Block-size sweep")[1].split("##")[0]
    labels, vals = [], []
    for line in sec.strip().splitlines():
        m = re.match(r"n=5000 block=(\d+).*?([\d.]+) GFLOP/s", line)
        if m:
            labels.append("none" if m.group(1) == "0" else m.group(1))
            vals.append(float(m.group(2)))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bar(ax, labels, vals, C["seq"], fmt="{:.2f}")
    ax.set_xlabel("block size"); ax.set_title("Fig 4.3 — Cache-block sweep, sequential, n=5000, -O3 -xHost")
    save(fig, "fig4_3_blocks.png")

# ---------------------------------------------------------------- OMP helpers
def omp_scaling(fname, n):
    rows = [m for m in RE_OMP.finditer(read(fname)) if int(m.group(1)) == n]
    seen = {}
    for m in rows:
        t = int(m.group(3))
        if m.group(4) == "static" and t not in seen:
            seen[t] = float(m.group(5))
    return dict(sorted(seen.items()))

# ---------------------------------------------------------------- Fig 5.1/5.2 OMP scaling
def fig_omp_scaling():
    d10 = omp_scaling("confirm_10000_b32.txt", 10000)
    d15 = omp_scaling("confirm_15000_b32.txt", 15000)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for d, n, c, mk in [(d10, 10000, C["omp"], "o"), (d15, 15000, C["mpi"], "s")]:
        T = list(d); S = [TSEQ[n] / d[t] for t in T]
        a1.plot(T, S, mk + "-", color=c, label=f"n={n}")
        a2.plot(T, [s / t * 100 for s, t in zip(S, T)], mk + "-", color=c, label=f"n={n}")
    tmax = max(d10)
    a1.plot([1, 8], [1, 8], "--", color=C["ideal"], label="ideal (≤8 P-cores)")
    a1.set_xlabel("threads"); a1.set_ylabel("speed-up  S = T_seq_best / T_p")
    a1.set_title("strong scaling"); a1.legend(); a1.grid(alpha=.3)
    a1.set_xticks(list(d10))
    a2.axhline(100, ls="--", color=C["ideal"]); a2.set_xlabel("threads")
    a2.set_ylabel("efficiency  E = S/p  (%)"); a2.set_title("efficiency")
    a2.legend(); a2.grid(alpha=.3); a2.set_xticks(list(d10))
    fig.suptitle("Fig 5.1 — OpenMP thread scaling (block=32, static)", fontsize=11)
    save(fig, "fig5_1_omp_scaling.png")

# ---------------------------------------------------------------- Fig 5.3 OMP block / 5.4 schedule
def fig_omp_sweeps():
    txt = read("run_omp.txt")
    blk = txt.split("## 2)")[1].split("##")[0]
    sch = txt.split("## 3)")[1].split("##")[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    labels, vals = [], []
    for m in RE_OMP.finditer(blk):
        labels.append("none" if m.group(2) == "0" else m.group(2)); vals.append(float(m.group(6)))
    bar(a1, labels, vals, C["omp"], fmt="{:.0f}")
    a1.set_xlabel("block size"); a1.set_title("block sweep (n=5000, T=24)")
    labels, vals = [], []
    for m in RE_OMP.finditer(sch):
        labels.append(m.group(4)); vals.append(float(m.group(6)))
    bar(a2, labels, vals, C["omp"], fmt="{:.0f}")
    a2.set_title("schedule sweep (n=5000, T=24, b=64)")
    fig.suptitle("Fig 5.2 — OpenMP block-size and schedule sweeps (GFLOP/s)", fontsize=11)
    save(fig, "fig5_2_omp_sweeps.png")

# ---------------------------------------------------------------- MPI
def mpi_scaling(fname, n):
    rows = {}
    for m in RE_MPI.finditer(read(fname)):
        if int(m.group(1)) == n and m.group(4) == "collective":
            p = int(m.group(3))
            if p not in rows:
                rows[p] = (float(m.group(5)), float(m.group(8)))  # t_tot, comm%
    return dict(sorted(rows.items()))

def kernel_really_fixed(fname, n):
    """A *_FIXED file counts as fixed only if its P=1 run matches the sequential
    kernel speed (the 2026-07-13 session wrote pre-fix data into FIXED files)."""
    d = mpi_scaling(fname, n)
    return 1 in d and TSEQ[n] / d[1][0] > 0.8

def fig_mpi_scaling():
    fixed10, fixed15 = "run_mpi_10000_FIXED.txt", "run_mpi_15000_FIXED.txt"
    use_fixed = kernel_really_fixed(fixed10, 10000)
    src10 = fixed10 if use_fixed else "run_mpi_10000_v2.txt"
    src15 = fixed15 if kernel_really_fixed(fixed15, 15000) else "confirm_15000_b32.txt"
    d10, d15 = mpi_scaling(src10, 10000), mpi_scaling(src15, 15000)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for d, n, c, mk in [(d10, 10000, C["mpi"], "o"), (d15, 15000, C["fp64"], "s")]:
        if not d: continue
        P = list(d); S = [TSEQ[n] / d[p][0] for p in P]
        a1.plot(P, S, mk + "-", color=c, label=f"n={n}")
        a2.plot(P, [d[p][1] for p in P], mk + "-", color=c, label=f"n={n}")
    a1.plot([1, 8], [1, 8], "--", color=C["ideal"], label="ideal (≤8)")
    a1.set_xlabel("processes"); a1.set_ylabel("speed-up vs T_seq_best")
    a1.set_title("strong scaling"); a1.legend(); a1.grid(alpha=.3)
    a2.set_xlabel("processes"); a2.set_ylabel("communication share of t_tot (%)")
    a2.set_title("communication share"); a2.legend(); a2.grid(alpha=.3)
    tag = "" if use_fixed else "  [PRE-FIX kernel — see §7.5]"
    fig.suptitle(f"Fig 6.1 — MPI process scaling (row-block, collectives, block=32){tag}",
                 fontsize=10, color="black" if use_fixed else "#B03030")
    save(fig, "fig6_1_mpi_scaling.png")

# ---------------------------------------------------------------- CUDA
def cuda_rows(fname, prec):
    rows = {}
    for m in RE_CUDA.finditer(read(fname)):
        if m.group(1) == prec:
            rows[m.group(2)] = (float(m.group(5)), float(m.group(4)), float(m.group(6)))
    return rows  # name -> (GF/s, kernel ms, H2D ms)

ORDER = ["naive-b8", "naive-b16", "naive-b32", "tiled-16", "tiled-32",
         "reg-4x4", "reg-8x8", "cublas"]

def fig_cuda_progression():
    r5  = cuda_rows("cuda_5000_reg (1).txt", "fp32")
    r10 = cuda_rows("cuda_10000_reg (1).txt", "fp32")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = range(len(ORDER)); w = 0.38
    b1 = ax.bar([i - w/2 for i in x], [r5.get(k, (0,))[0] for k in ORDER], w,
                color=C["cuda"], label="n=5000")
    b2 = ax.bar([i + w/2 for i in x], [r10.get(k, (0,))[0] for k in ORDER], w,
                color=C["seq"], label="n=10000")
    ax.bar_label(b1, fmt="{:.0f}", fontsize=8, rotation=90, padding=2)
    ax.bar_label(b2, fmt="{:.0f}", fontsize=8, rotation=90, padding=2)
    ax.set_xticks(list(x)); ax.set_xticklabels(ORDER, rotation=20)
    ax.set_ylabel("GFLOP/s"); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
    ax.legend(); ax.set_ylim(0, 5200)
    ax.set_title("Fig 7.1 — CUDA FP32 progression on the Tesla T4: naive → tiled → register-blocked → cuBLAS")
    save(fig, "fig7_1_cuda_progression.png")

def fig_cuda_fp64():
    r32 = cuda_rows("cuda_10000_reg (1).txt", "fp32")
    r64 = cuda_rows("cuda_10000_reg (1).txt", "fp64")
    keys = ["naive-b32", "tiled-32", "reg-8x8", "cublas"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    x = range(len(keys)); w = 0.38
    b1 = a1.bar([i - w/2 for i in x], [r32[k][0] for k in keys], w, color=C["cuda"], label="FP32")
    b2 = a1.bar([i + w/2 for i in x], [r64[k][0] for k in keys], w, color=C["fp64"], label="FP64")
    a1.bar_label(b1, fmt="{:.0f}", fontsize=8); a1.bar_label(b2, fmt="{:.0f}", fontsize=8)
    a1.set_xticks(list(x)); a1.set_xticklabels(keys, rotation=15)
    a1.set_ylabel("GFLOP/s"); a1.legend(); a1.grid(axis="y", alpha=.3); a1.set_axisbelow(True)
    a1.set_title("FP32 vs FP64, n=10000 (T4: FP64 peak = 1/32 FP32)")
    # transfer share for best kernels, n=10000 FP32
    for k, c in [("reg-8x8", C["cuda"]), ("cublas", C["seq"])]:
        gf, ker, h2d = r32[k]
        a2.bar(k, ker, color=c, label=f"{k} kernel")
        a2.bar(k, h2d, bottom=ker, color="#BBBBBB")
    a2.set_ylabel("time (ms)"); a2.grid(axis="y", alpha=.3); a2.set_axisbelow(True)
    a2.set_title("kernel vs H2D copy (grey), n=10000 FP32")
    fig.suptitle("Fig 7.2 — Precision gap and transfer cost", fontsize=11)
    save(fig, "fig7_2_cuda_fp64_transfer.png")

# ---------------------------------------------------------------- Fig 8.1 grand comparison
def fig_grand():
    labels = ["seq -O0\n(naive)", "seq best\n(AVX2+b32)", "OpenMP\n8T", "MPI 16P\n[pre-fix]",
              "CUDA reg-8×8\nFP32", "cuBLAS\nFP32"]
    vals = [1.44, 32.46, 231.41, 125.81, 3021.27, 4272.51]
    if kernel_really_fixed("run_mpi_10000_FIXED.txt", 10000):
        d = mpi_scaling("run_mpi_10000_FIXED.txt", 10000)
        if d:
            best_p = min(d, key=lambda p: d[p][0])
            vals[3] = 2e12 * 1e-9 / d[best_p][0] * (10000**3) / 1e12  # 2n^3/t/1e9
            vals[3] = 2 * 10000**3 / d[best_p][0] / 1e9
            labels[3] = f"MPI {best_p}P\n(fixed)"
    fig, ax = plt.subplots(figsize=(8, 4))
    b = ax.bar(labels, vals, color=[C["seq"], C["seq"], C["omp"], C["mpi"], C["cuda"], "#777777"])
    ax.bar_label(b, fmt="{:.0f}", fontsize=9)
    ax.set_yscale("log"); ax.set_ylabel("GFLOP/s (log scale)")
    ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
    ax.set_title("Fig 8.1 — The whole project in one chart (n=10000 unless noted, doubles on CPU)")
    save(fig, "fig8_1_grand_comparison.png")

if __name__ == "__main__":
    fig_looporder(); fig_flags(); fig_blocks_seq()
    fig_omp_scaling(); fig_omp_sweeps()
    fig_mpi_scaling()
    fig_cuda_progression(); fig_cuda_fp64()
    fig_grand()
    print("\nAll figures in report/figures/")
