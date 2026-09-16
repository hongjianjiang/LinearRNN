#!/usr/bin/env python3
import argparse, random
import numpy as np

def parse_src_line(line: str):
    parts = line.strip().split("|")
    T = int(parts[0])
    mats = []
    for s in parts[1:]:
        a = np.fromstring(s, sep=",", dtype=np.int64)
        assert a.size == 9
        mats.append(a.reshape(3,3))
    assert len(mats) == T
    return T, np.stack(mats, axis=0)  # (T,3,3)

def label_cap(mats_T33: np.ndarray, cap: int) -> int:
    r = np.array([1,0,0], dtype=np.int64)
    for t in range(mats_T33.shape[0]):
        r = r @ mats_T33[t]
        r = np.clip(r, -cap, cap)
    return int(r[0] == 0)

def label_int64(mats_T33: np.ndarray) -> int:
    r = np.array([1,0,0], dtype=np.int64)
    mats = mats_T33.astype(np.int64, copy=False)
    for t in range(mats.shape[0]):
        r = r @ mats[t]   # will overflow as int64
    return int(r[0] == 0)

def label_mod(mats_T33: np.ndarray, primes: list[int]) -> int:
    mats = mats_T33.astype(np.int64, copy=False)
    ok = True
    for p in primes:
        r = np.array([1,0,0], dtype=np.int64)
        for t in range(mats.shape[0]):
            r = (r @ mats[t]) % p
        ok = ok and (r[0] % p == 0)
        if not ok:
            return 0
    return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--label_mode", choices=["cap","int64","mod"], required=True)
    ap.add_argument("--cap", type=int, default=1000000)
    ap.add_argument("--primes", type=str, default="1000000007,1000000009")
    ap.add_argument("--num_check", type=int, default=2000,
                    help="How many random examples to check.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    primes = [int(x) for x in args.primes.split(",") if x.strip()]

    # read all lines (fast enough for spot-check)
    with open(args.src, "r") as fs:
        src_lines = [ln for ln in fs if ln.strip()]
    with open(args.tgt, "r") as ft:
        tgt_lines = [ln for ln in ft if ln.strip()]

    if len(src_lines) != len(tgt_lines):
        raise RuntimeError(f"Line count mismatch: src={len(src_lines)} tgt={len(tgt_lines)}")

    n = len(src_lines)
    k = min(args.num_check, n)
    idxs = rng.sample(range(n), k)

    bad = 0
    for j, i in enumerate(idxs, 1):
        T, mats = parse_src_line(src_lines[i])
        y_file = int(tgt_lines[i].strip())
        if args.label_mode == "cap":
            y = label_cap(mats, args.cap)
        elif args.label_mode == "int64":
            y = label_int64(mats)
        else:
            y = label_mod(mats, primes)

        if y != y_file:
            bad += 1
            if bad <= 10:
                print(f"[mismatch] i={i} T={T} file={y_file} recomputed={y}")
        if j % 200 == 0:
            print(f"checked {j}/{k} mismatches={bad}")

    print(f"DONE checked={k} mismatches={bad} ({100*bad/k:.2f}%)")

if __name__ == "__main__":
    main()
