#!/usr/bin/env python3
"""Compare old pseudo-random stride vs new curand-style uniform random LBA generation.

Requires: numpy, matplotlib
    pip install numpy matplotlib

Usage:
    python3 tools/compare_random.py
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Drive parameters ---
DRIVE_SIZE = 8 * 10**12             # 8 TB in bytes
LBA_SIZE   = 512                    # bytes
LBA_MAX    = DRIVE_SIZE // LBA_SIZE # 15,625,000,000

# --- Workload parameters ---
N_BLOCKS       = 8                  # 4 KB I/O at 512 B LBAs
TOTAL_THREADS  = 2048
IOS_PER_THREAD = 500
TOTAL_IOS      = TOTAL_THREADS * IOS_PER_THREAD

print(f"lba_max        = {LBA_MAX:,}")
print(f"total_threads  = {TOTAL_THREADS}")
print(f"ios_per_thread = {IOS_PER_THREAD}")
print(f"total I/Os     = {TOTAL_IOS:,}")

# ------------------------------------------------------------------
#  Old logic: fixed tid-derived odd stride (from gpu_worker.cu before)
#
#  lba_step = ((tid * 1315423911) | 1) * n_blocks
#  slba += lba_step  (mod lba_max)
# ------------------------------------------------------------------
def generate_old(total_threads, ios_per_thread, n_blocks, lba_max):
    lbas = np.empty(total_threads * ios_per_thread, dtype=np.uint64)
    idx = 0
    for tid in range(total_threads):
        slba = np.uint64(tid * n_blocks)
        lba_step = np.uint64(((tid * 1315423911) | 1) * n_blocks)

        if lba_max and slba + n_blocks >= lba_max:
            slba = np.uint64(slba - (slba // lba_max) * lba_max)
        if lba_max and lba_step >= lba_max:
            lba_step = np.uint64(lba_step - (lba_step // lba_max) * lba_max)
        if lba_step == 0:
            lba_step = np.uint64(n_blocks if n_blocks else 1)

        for _ in range(ios_per_thread):
            lbas[idx] = slba
            idx += 1
            slba = np.uint64(slba + lba_step)
            if lba_max and slba >= lba_max:
                slba = np.uint64(slba - lba_max)
    return lbas

# ------------------------------------------------------------------
#  New logic: uniform random (simulates two 32-bit curand -> 64-bit)
#
#  r = (curand() << 32) | curand()
#  slba = r % (lba_max - n_blocks)
# ------------------------------------------------------------------
def generate_new(total_ios, n_blocks, lba_max):
    bound = lba_max - n_blocks
    hi = np.random.randint(0, 2**32, size=total_ios, dtype=np.uint64)
    lo = np.random.randint(0, 2**32, size=total_ios, dtype=np.uint64)
    r = (hi << np.uint64(32)) | lo
    return r % np.uint64(bound)

print("\nGenerating old (stride-based) LBAs...")
old_lbas = generate_old(TOTAL_THREADS, IOS_PER_THREAD, N_BLOCKS, LBA_MAX)

print("Generating new (curand-style) LBAs...")
new_lbas = generate_new(TOTAL_IOS, N_BLOCKS, LBA_MAX)

# ------------------------------------------------------------------
#  Histogram
# ------------------------------------------------------------------
BINS = 256

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].hist(old_lbas, bins=BINS, color="tomato", edgecolor="black",
             linewidth=0.3, alpha=0.85)
axes[0].set_title("Old: pseudo-random stride  (tid * 1315423911 | 1) * n_blocks",
                  fontsize=13)
axes[0].set_ylabel("Count")
axes[0].axhline(TOTAL_IOS / BINS, color="black", ls="--", lw=1,
                label=f"ideal uniform = {TOTAL_IOS // BINS}")
axes[0].legend()

axes[1].hist(new_lbas, bins=BINS, color="steelblue", edgecolor="black",
             linewidth=0.3, alpha=0.85)
axes[1].set_title("New: curand() uniform random  (two 32-bit -> 64-bit) % bound",
                  fontsize=13)
axes[1].set_ylabel("Count")
axes[1].set_xlabel(f"LBA  (drive: 8 TB, 512 B sectors, lba_max = {LBA_MAX:,})")
axes[1].axhline(TOTAL_IOS / BINS, color="black", ls="--", lw=1,
                label=f"ideal uniform = {TOTAL_IOS // BINS}")
axes[1].legend()

plt.tight_layout()
plt.savefig("tools/compare_random.png", dpi=150)
print(f"\nSaved  -> tools/compare_random.png")
plt.show()
