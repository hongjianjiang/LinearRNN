#!/usr/bin/env python3
"""
PERMUTATION-matrix 3x3 iterated product, FINAL binary label. NumPy version.

Same motivation as imm_mod/gen_perm.py: the original NO-MOD task (gen.py)
samples matrices from {-1,0,1} with unbounded integer growth -- a "mixing"
map where a small representational error compounds catastrophically, and
where honest (non-leaking) RNN/DeltaNet/RWKV-7 all collapse to chance by
T~15-20, per eval_by_T.py findings. This is imm_unbound's harder cousin:
sparse supervision (a single label at the end, not dense per-step) on top
of the same mixing-map difficulty.

Restricting the matrix family to 3x3 PERMUTATION matrices (elements of
S_3, the paper's own PD/NC^1 side of the PD-vs-DPLR line) removes the
mixing/unbounded-growth problem entirely: permutation matrices are closed
under multiplication (P_T is always itself a 0/1 permutation matrix, no
integer blowup, nothing to approximate), while the label semantics and
file format are otherwise identical to gen.py -- so train_rnn.py /
train_deltanet.py / train_rwkv7.py need zero changes.

Format (unchanged from gen.py):
  src: "T|a1,...,a9|...|a1,...,a9"   (each block a flattened 3x3 0/1 matrix)
  tgt: "0" or "1"

Label:
  y = 1 iff (P_T)[0,0] == 0,  P_T = M_1 @ M_2 @ ... @ M_T (permutation matrices)
"""

from __future__ import annotations
import os
import argparse
import random
from typing import List, Tuple

import numpy as np

PERM_LIST = [
    (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
]


def eye3() -> np.ndarray:
    return np.eye(3, dtype=np.int64)


def perm_matrix(p) -> np.ndarray:
    M = np.zeros((3, 3), dtype=np.int64)
    for row, col in enumerate(p):
        M[row, col] = 1
    return M


PERM_MATS = [perm_matrix(p) for p in PERM_LIST]


def sample_perm_matrix(rng: random.Random) -> np.ndarray:
    return PERM_MATS[rng.randrange(6)]


def label_for_seq(mats: List[np.ndarray]) -> int:
    P = eye3()
    for M in mats:
        P = P @ M  # stays a 0/1 permutation matrix throughout -- no overflow, no mixing
    return 1 if P[0, 0] == 0 else 0


def serialize_src(T: int, mats: List[np.ndarray]) -> str:
    parts = [str(T)]
    for M in mats:
        flat = M.reshape(-1)
        parts.append(",".join(str(int(x)) for x in flat))
    return "|".join(parts)


def gen_split_examples(
    rng: random.Random,
    T_min: int,
    T_max: int,
    n_per_len: int,
    balanced: bool,
    max_tries: int,
) -> Tuple[List[str], List[str]]:

    src_lines: List[str] = []
    tgt_lines: List[str] = []

    for T in range(T_min, T_max + 1):
        need = n_per_len

        if not balanced:
            for _ in range(need):
                mats = [sample_perm_matrix(rng) for _ in range(T)]
                y = label_for_seq(mats)
                src_lines.append(serialize_src(T, mats))
                tgt_lines.append(str(y))

        else:
            need1 = need // 2
            need0 = need - need1
            got0 = got1 = 0
            tries = 0

            while (got0 < need0) or (got1 < need1):
                tries += 1
                if max_tries > 0 and tries > max_tries:
                    rem = (need0 - got0) + (need1 - got1)
                    for _ in range(rem):
                        mats = [sample_perm_matrix(rng) for _ in range(T)]
                        y = label_for_seq(mats)
                        src_lines.append(serialize_src(T, mats))
                        tgt_lines.append(str(y))
                    break

                mats = [sample_perm_matrix(rng) for _ in range(T)]
                y = label_for_seq(mats)

                if y == 0 and got0 < need0:
                    got0 += 1
                    src_lines.append(serialize_src(T, mats))
                    tgt_lines.append("0")

                elif y == 1 and got1 < need1:
                    got1 += 1
                    src_lines.append(serialize_src(T, mats))
                    tgt_lines.append("1")

    return src_lines, tgt_lines


def write_split(out_dir: str, split: str, src_lines: List[str], tgt_lines: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{split}_src.txt"), "w") as f:
        f.write("\n".join(src_lines) + "\n")
    with open(os.path.join(out_dir, f"{split}_tgt.txt"), "w") as f:
        f.write("\n".join(tgt_lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T_train", type=int, default=100)
    ap.add_argument("--T_gap", type=int, default=100)
    ap.add_argument("--n_train_per_len", type=int, default=200)
    ap.add_argument("--n_val_per_len", type=int, default=50)
    ap.add_argument("--n_test_per_len", type=int, default=50)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--max_tries", type=int, default=200000)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    T0_min, T0_max = 1, args.T_train
    T1_min, T1_max = args.T_train + 1, args.T_train + args.T_gap
    T2_min, T2_max = args.T_train + args.T_gap + 1, args.T_train + 2 * args.T_gap

    print(f"[out] {args.out_dir}")

    splits = [
        ("train",     T0_min, T0_max, args.n_train_per_len),
        ("val_bin0",  T0_min, T0_max, args.n_val_per_len),
        ("test_bin0", T0_min, T0_max, args.n_test_per_len),
        ("test_bin1", T1_min, T1_max, args.n_test_per_len),
        ("test_bin2", T2_min, T2_max, args.n_test_per_len),
    ]

    for name, Tmin, Tmax, n in splits:
        print(f"[gen] {name} T=[{Tmin},{Tmax}]")
        src, tgt = gen_split_examples(rng, Tmin, Tmax, n, args.balanced, args.max_tries)
        write_split(args.out_dir, name, src, tgt)


if __name__ == "__main__":
    main()
