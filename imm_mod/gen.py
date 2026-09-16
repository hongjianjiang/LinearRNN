#!/usr/bin/env python3
# gen.py
"""
IMM-Mod (Prime + Invertible) dataset generator (STEPWISE TARGETS)
-----------------------------------------------------------------

Task:
  Iterated 3x3 matrix multiplication over Z_m (mod prime m),
  matrices sampled from {-1,0,1}^{3x3}, but REJECT until invertible mod m.

Labels (stepwise):
  v_t = (P_t)[qk] mod m,  where P_t = M1 @ ... @ Mt (mod m)

Format:
  src: "T|m|qk|mat1|...|matT"
  tgt: "v1|v2|...|vT"

Splits (length generalization):
  train/val_bin0/test_bin0: T in [1, T_train]
  test_bin1:                T in [T_train+1, T_train+T_gap]
  test_bin2:                T in [T_train+T_gap+1, T_train+2*T_gap]

Notes:
- m is fixed by --m (default 29). Prefer prime m (29, 31, 37, ...).
- full-rank is enforced by default (can disable with --no_full_rank).
"""

from __future__ import annotations

import os
import argparse
import numpy as np
from tqdm import tqdm


# -------------------------
# math helpers
# -------------------------
def det3_int(M: np.ndarray) -> int:
    # exact integer det for 3x3
    a, b, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]
    g, h, i = M[2, 0], M[2, 1], M[2, 2]
    return int(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))


def is_invertible_mod_prime(M: np.ndarray, m: int) -> bool:
    # For prime m: invertible iff det != 0 mod m
    return (det3_int(M) % m) != 0


def matmul3_mod(A: np.ndarray, B: np.ndarray, m: int) -> np.ndarray:
    return ((A % m) @ (B % m)) % m


def flatten_rowmajor(M: np.ndarray) -> str:
    return ",".join(str(int(x)) for x in M.reshape(-1))


# -------------------------
# sampling
# -------------------------
def sample_matrix_pm1(rng: np.random.Generator) -> np.ndarray:
    vals = np.array([-1, 0, 1], dtype=np.int64)
    return rng.choice(vals, size=(3, 3)).astype(np.int64)


def sample_matrix_pm1_invertible(rng: np.random.Generator, m: int, max_tries: int = 100_000) -> np.ndarray:
    # rejection sample until invertible mod prime m
    for _ in range(max_tries):
        M = sample_matrix_pm1(rng)
        if is_invertible_mod_prime(M, m):
            return M
    raise RuntimeError(f"Failed to sample invertible pm1 matrix after {max_tries} tries (m={m}).")


def gen_one_stepwise(
    rng: np.random.Generator,
    T: int,
    m: int,
    qk: int,
    full_rank: bool,
) -> tuple[str, str]:
    """
    Returns:
      src: "T|m|qk|mat1|...|matT"
      tgt: "v1|v2|...|vT" where v_t = (P_t)[qk] mod m
    """
    P = np.eye(3, dtype=np.int64) % m
    mats: list[np.ndarray] = []
    vs: list[str] = []

    for _t in range(T):
        Mt = sample_matrix_pm1_invertible(rng, m) if full_rank else sample_matrix_pm1(rng)
        mats.append(Mt)
        P = matmul3_mod(P, Mt, m)
        vt = int(P.reshape(-1)[qk] % m)
        vs.append(str(vt))

    src = f"{T}|{m}|{qk}|" + "|".join(flatten_rowmajor(M) for M in mats)
    tgt = "|".join(vs)
    return src, tgt


def write_split(
    out_dir: str,
    split: str,
    n_samples: int,
    T_lo: int,
    T_hi: int,
    rng: np.random.Generator,
    m: int,
    qk: int,
    full_rank: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    src_path = os.path.join(out_dir, f"{split}_src.txt")
    tgt_path = os.path.join(out_dir, f"{split}_tgt.txt")

    with open(src_path, "w", encoding="utf-8") as fsrc, open(tgt_path, "w", encoding="utf-8") as ftgt:
        pbar = tqdm(total=n_samples, desc=f"{split}_src.txt", dynamic_ncols=True)
        for _ in range(n_samples):
            T = int(rng.integers(T_lo, T_hi + 1))
            src, tgt = gen_one_stepwise(rng, T=T, m=m, qk=qk, full_rank=full_rank)
            fsrc.write(src + "\n")
            ftgt.write(tgt + "\n")
            pbar.update(1)
        pbar.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)

    # Length generalization
    ap.add_argument("--T_train", type=int, default=50)
    ap.add_argument("--T_gap", type=int, default=100)

    # Sizes
    ap.add_argument("--n_train", type=int, default=70_000)
    ap.add_argument("--n_val", type=int, default=20_000)
    ap.add_argument("--n_test", type=int, default=10_000)

    # Modulus / query
    ap.add_argument("--m", type=int, default=29, help="prefer a prime modulus (default 29)")
    ap.add_argument("--qk", type=int, default=0, help="query index in [0..8]; 0 means (0,0)")

    # Sampling
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_full_rank", action="store_true", help="disable invertibility rejection sampling")

    args = ap.parse_args()
    if not (0 <= args.qk <= 8):
        raise ValueError(f"--qk must be in [0..8], got {args.qk}")
    if args.m < 2:
        raise ValueError(f"--m must be >=2, got {args.m}")

    rng = np.random.default_rng(args.seed)
    full_rank = not args.no_full_rank

    T_train = args.T_train
    T_gap = args.T_gap

    splits = [
        ("train",     args.n_train, 1, T_train),
        ("val_bin0",  args.n_val,   1, T_train),
        ("test_bin0", args.n_test,  1, T_train),
        ("test_bin1", args.n_test,  T_train + 1,         T_train + T_gap),
        ("test_bin2", args.n_test,  T_train + T_gap + 1, T_train + 2 * T_gap),
    ]

    for name, n, lo, hi in splits:
        write_split(
            out_dir=args.out_dir,
            split=name,
            n_samples=n,
            T_lo=lo,
            T_hi=hi,
            rng=rng,
            m=args.m,
            qk=args.qk,
            full_rank=full_rank,
        )

    print(f"[OK] wrote dataset to {args.out_dir}")
    print("  src: T|m|qk|mat1|...|matT")
    print("  tgt: v1|...|vT, where v_t=(P_t)[qk] mod m")
    print(f"  m={args.m} qk={args.qk} full_rank={full_rank}")
    print(f"  T_train={args.T_train} T_gap={args.T_gap} => bin2 max T={args.T_train + 2*args.T_gap}")


if __name__ == "__main__":
    main()
