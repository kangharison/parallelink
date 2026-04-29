#!/usr/bin/env python3
"""Plot histograms from binary LBA dumps produced by compare_random.cu.

Requires: numpy, matplotlib
    pip install numpy matplotlib

Usage:
    # 1. Build and run the CUDA program first:
    #    nvcc -o compare_random tools/compare_random.cu -lcurand -O2
    #    ./compare_random
    #
    # 2. Then plot:
    #    python3 tools/compare_random.py
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt

LBA_MAX = 8 * 10**12 // 512  # must match compare_random.cu

def load_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 8
    return np.array(struct.unpack(f"<{n}Q", data), dtype=np.uint64)

try:
    old_lbas = load_bin("old_lbas.bin")
    new_lbas = load_bin("new_lbas.bin")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run compare_random (CUDA binary) first to generate the .bin files.")
    sys.exit(1)

total = len(old_lbas)
print(f"Loaded {total:,} samples from each file")

BINS = 256

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].hist(old_lbas, bins=BINS, color="tomato", edgecolor="black",
             linewidth=0.3, alpha=0.85)
axes[0].set_title("Old: pseudo-random stride  (tid * 1315423911 | 1) * n_blocks",
                  fontsize=13)
axes[0].set_ylabel("Count")
axes[0].axhline(total / BINS, color="black", ls="--", lw=1,
                label=f"ideal uniform = {total // BINS}")
axes[0].legend()

axes[1].hist(new_lbas, bins=BINS, color="steelblue", edgecolor="black",
             linewidth=0.3, alpha=0.85)
axes[1].set_title("New: curand() XORWOW  (two 32-bit -> 64-bit) % bound",
                  fontsize=13)
axes[1].set_ylabel("Count")
axes[1].set_xlabel(f"LBA  (drive: 8 TB, 512 B sectors, lba_max = {LBA_MAX:,})")
axes[1].axhline(total / BINS, color="black", ls="--", lw=1,
                label=f"ideal uniform = {total // BINS}")
axes[1].legend()

plt.tight_layout()
plt.savefig("compare_random.png", dpi=150)
print(f"Saved -> compare_random.png")
plt.show()
