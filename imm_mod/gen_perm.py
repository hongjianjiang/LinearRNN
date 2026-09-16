#!/usr/bin/env python3
# gen_perm.py
"""
Iterated 3x3 PERMUTATION Matrix Product (S_3 word-tracking) dataset generator.

Motivation: the mod-m iterated matrix mult task (gen.py) requires tracking an
exact residue in Z_m through T compounding multiplications -- any small
representational imprecision compounds (a "mixing" map), and empirically
neither GRU nor DeltaNet sustain tracking past ~4-20 steps regardless of
supervision density (see eval_by_t.py findings). The paper's own theory
(Theorem 7 vs 5/6) draws a sharp line between PD (permutation-diagonal)
LRNNs -- NC^1-complete -- and DPLR LRNNs -- PNC^1-complete, strictly harder.
Restricting the matrix family to permutation matrices moves this task onto
the easy (NC^1) side of that line: S_3 has exactly 6 elements, closed under
multiplication, so the state space is small, discrete, and non-diverging --
no numeric drift is possible. This is the same family of finite-state
word-problem tasks (the S_5 word problem is the canonical NC^1-complete
problem, cited directly in the paper) that RNNs are documented to learn and
length-generalize on.

Task:
  Iterated 3x3 permutation matrix multiplication.
  M_t is the permutation matrix of a uniformly random element of S_3.
  P_t = M_1 @ M_2 @ ... @ M_t (ordinary matrix product -- permutation
  matrices are closed under multiplication, so P_t is always itself a
  permutation matrix, entries always in {0,1}, no modulus needed).

Labels (stepwise):
  v_t = index of P_t among the 6 elements of S_3 (canonical enumeration
  order below) -- i.e. "which group element is the running product right
  now", not a single matrix entry. This directly supervises the full state
  instead of one lossy bit of it.

Format (unchanged from gen.py, so train_rnn_relu.py needs zero changes):
  src: "T|m|qk|mat1|...|matT"   (m=6 fixed; qk is unused, kept as 0 for
                                  format compatibility)
  tgt: "v1|v2|...|vT"           (v_t in [0,5])

Splits (length generalization): identical convention to gen.py --
  train/val_bin0/test_bin0: T in [1, T_train]
  test_bin1:                T in [T_train+1, T_train+T_gap]
  test_bin2:                T in [T_train+T_gap+1, T_train+2*T_gap]
"""

from __future__ import annotations

import os
import argparse
from itertools import permutations

import numpy as np
from tqdm import tqdm

# Canonical enumeration of S_3 (6 elements). PERMS[i] = (pi(0), pi(1), pi(2)).
PERMS = list(permutations(range(3)))
assert len(PERMS) == 6
PERM_TO_IDX = {p: i for i, p in enumerate(PERMS)}


def perm_matrix(p) -> np.ndarray:
    M = np.zeros((3, 3), dtype=np.int64)
    for row, col in enumerate(p):
        M[row, col] = 1
    return M


PERM_MATS = [perm_matrix(p) for p in PERMS]


def matrix_to_perm_idx(M: np.ndarray) -> int:
    # M is guaranteed to be a 0/1 permutation matrix (product of permutation
    # matrices); decode which S_3 element it is.
    p = tuple(int(np.argmax(M[row])) for row in range(3))
    return PERM_TO_IDX[p]


def flatten_rowmajor(M: np.ndarray) -> str:
    return ",".join(str(int(x)) for x in M.reshape(-1))


def gen_one_stepwise(rng: np.random.Generator, T: int) -> tuple[str, str]:
    P = np.eye(3, dtype=np.int64)  # identity permutation matrix
    mats: list[np.ndarray] = []
    vs: list[str] = []

    for _t in range(T):
        gi = int(rng.integers(0, 6))
        Mt = PERM_MATS[gi]
        mats.append(Mt)
        P = P @ Mt  # still a 0/1 permutation matrix, exactly (no residue error possible)
        vs.append(str(matrix_to_perm_idx(P)))

    src = f"{T}|6|0|" + "|".join(flatten_rowmajor(M) for M in mats)
    tgt = "|".join(vs)
    return src, tgt


def write_split(out_dir: str, split: str, n_samples: int, T_lo: int, T_hi: int, rng: np.random.Generator) -> None:
    os.makedirs(out_dir, exist_ok=True)
    src_path = os.path.join(out_dir, f"{split}_src.txt")
    tgt_path = os.path.join(out_dir, f"{split}_tgt.txt")

    with open(src_path, "w", encoding="utf-8") as fsrc, open(tgt_path, "w", encoding="utf-8") as ftgt:
        pbar = tqdm(total=n_samples, desc=f"{split}_src.txt", dynamic_ncols=True)
        for _ in range(n_samples):
            T = int(rng.integers(T_lo, T_hi + 1))
            src, tgt = gen_one_stepwise(rng, T=T)
            fsrc.write(src + "\n")
            ftgt.write(tgt + "\n")
            pbar.update(1)
        pbar.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--T_train", type=int, default=100)
    ap.add_argument("--T_gap", type=int, default=100)
    ap.add_argument("--n_train", type=int, default=70_000)
    ap.add_argument("--n_val", type=int, default=20_000)
    ap.add_argument("--n_test", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    T_train, T_gap = args.T_train, args.T_gap

    splits = [
        ("train",     args.n_train, 1, T_train),
        ("val_bin0",  args.n_val,   1, T_train),
        ("test_bin0", args.n_test,  1, T_train),
        ("test_bin1", args.n_test,  T_train + 1,         T_train + T_gap),
        ("test_bin2", args.n_test,  T_train + T_gap + 1, T_train + 2 * T_gap),
    ]

    for name, n, lo, hi in splits:
        write_split(args.out_dir, name, n, lo, hi, rng)

    print(f"[OK] wrote dataset to {args.out_dir}")
    print("  src: T|m|qk|mat1|...|matT  (m=6 fixed, qk unused)")
    print("  tgt: v1|...|vT, where v_t = index of P_t in S_3 (canonical order)")
    print(f"  T_train={T_train} T_gap={T_gap} => bin2 max T={T_train + 2 * T_gap}")


if __name__ == "__main__":
    main()
