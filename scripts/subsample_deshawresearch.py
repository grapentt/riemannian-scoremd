"""
Reproducible subsampling of DE Shaw Cα trajectories.

Reads raw .h5 files from data/deshawresearch/, applies a stride-based
subsample, Kabsch-aligns to the first training frame, then saves
train/test splits as .npy files under data/processed/.

Output files:
    data/processed/{protein}_train.npy   shape (N_train, n, 3), float32, Å
    data/processed/{protein}_test.npy    shape (N_test,  n, 3), float32, Å

The first training frame is also saved as the reference frame:
    data/processed/{protein}_ref.npy     shape (n, 3), float32, Å

Usage:
    python scripts/subsample_deshawresearch.py [--protein chignolin|bba|all]
                                               [--stride 10]
                                               [--test-frac 0.10]
                                               [--seed 42]
                                               [--dry-run]

Default strides produce:
    chignolin:  50 000 frames / stride 5  → 10 000 train + 1111 test
    bba:        70 000 frames / stride 1  → 63 000 train + 7000 test

Re-running with the same --seed and --stride always produces identical splits.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Kabsch alignment (pure numpy, no JAX dependency)
# ---------------------------------------------------------------------------

def kabsch_align(mobile: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Align mobile (n, 3) to ref (n, 3) by Kabsch rotation (no translation —
    both should already be centred).  Returns aligned mobile (n, 3).
    """
    H = mobile.T @ ref                    # (3, 3)
    U, _, Vh = np.linalg.svd(H)
    # Correct for reflection
    d = np.linalg.det(Vh.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vh.T @ D @ U.T                    # (3, 3)
    return mobile @ R.T


def kabsch_align_batch(coords: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Align batch (T, n, 3) to ref (n, 3).  Returns aligned batch (T, n, 3).
    Both coords and ref should be centred.
    """
    aligned = np.empty_like(coords)
    for i in range(coords.shape[0]):
        aligned[i] = kabsch_align(coords[i], ref)
    return aligned


# ---------------------------------------------------------------------------
# Protein configurations
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent

PROTEIN_CONFIGS = {
    "chignolin": {
        "h5_files": [_ROOT / "data" / "deshawresearch" / "chignolin-0_ca.h5"],
        "n_atoms": 10,
        "default_stride": 5,       # 50k → 10k frames
    },
    "bba": {
        "h5_files": [
            _ROOT / "data" / "deshawresearch" / "bba-0_ca.h5",
            _ROOT / "data" / "deshawresearch" / "bba-1_ca.h5",
        ],
        "n_atoms": 28,
        "default_stride": 1,       # 70k → 70k frames (keep all)
    },
}


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_protein(
    protein_name: str,
    stride: int | None,
    test_frac: float,
    seed: int,
    dry_run: bool,
    out_dir: Path,
):
    import mdtraj as md

    cfg = PROTEIN_CONFIGS[protein_name]
    h5_files = cfg["h5_files"]
    n_atoms = cfg["n_atoms"]
    effective_stride = stride if stride is not None else cfg["default_stride"]

    # ---- Load ----
    missing = [f for f in h5_files if not f.exists()]
    if missing:
        print(f"  [ERROR] Missing files: {missing}")
        return False

    print(f"  Loading {len(h5_files)} file(s)...")
    traj = md.load([str(f) for f in h5_files])
    T_raw = traj.n_frames
    assert traj.n_atoms == n_atoms, f"Expected {n_atoms} atoms, got {traj.n_atoms}"
    print(f"  Loaded {T_raw} frames, {n_atoms} Cα atoms")

    # ---- Convert nm → Å ----
    coords = traj.xyz * 10.0    # (T, n, 3), Å, float32

    # ---- Stride ----
    coords = coords[::effective_stride]
    T_sub = coords.shape[0]
    print(f"  After stride={effective_stride}: {T_sub} frames")

    # ---- Centre each frame ----
    coords -= coords.mean(axis=1, keepdims=True)    # (T, n, 3)

    # ---- Reproducible shuffle ----
    rng = np.random.default_rng(seed)
    idx = rng.permutation(T_sub)
    coords = coords[idx]

    # ---- Train / test split ----
    n_test = max(1, int(round(T_sub * test_frac)))
    n_train = T_sub - n_test
    train = coords[:n_train]    # (N_train, n, 3)
    test  = coords[n_train:]    # (N_test,  n, 3)
    print(f"  Split: {n_train} train, {n_test} test (test_frac={test_frac})")

    # ---- Kabsch align to first training frame ----
    ref = train[0].copy()       # (n, 3), already centred
    print(f"  Kabsch-aligning to first training frame...")
    train = kabsch_align_batch(train, ref)
    test  = kabsch_align_batch(test, ref)

    # ---- Save ----
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{protein_name}_train.npy"
    test_path  = out_dir / f"{protein_name}_test.npy"
    ref_path   = out_dir / f"{protein_name}_ref.npy"

    if dry_run:
        print(f"  [dry-run] Would save:")
        print(f"    {train_path}  shape={train.shape}")
        print(f"    {test_path}   shape={test.shape}")
        print(f"    {ref_path}    shape={ref.shape}")
    else:
        np.save(train_path, train.astype(np.float32))
        np.save(test_path,  test.astype(np.float32))
        np.save(ref_path,   ref.astype(np.float32))
        print(f"  Saved:")
        print(f"    {train_path}  {train.shape}")
        print(f"    {test_path}   {test.shape}")
        print(f"    {ref_path}    {ref.shape}")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Subsample DE Shaw trajectories → data/processed/*.npy"
    )
    parser.add_argument(
        "--protein", default="all",
        choices=list(PROTEIN_CONFIGS.keys()) + ["all"],
        help="Which protein to process (default: all)",
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help="Stride for subsampling (default: protein-specific: chignolin=5, bba=1)",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.10,
        help="Fraction of frames held out for test set (default: 0.10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffle (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be saved without writing files",
    )

    args = parser.parse_args()

    out_dir = _ROOT / "data" / "processed"
    proteins = list(PROTEIN_CONFIGS.keys()) if args.protein == "all" else [args.protein]

    print(f"\n{'='*60}")
    print("DE Shaw trajectory subsampling")
    print(f"  seed={args.seed}  test_frac={args.test_frac}  dry_run={args.dry_run}")
    print(f"{'='*60}")

    all_ok = True
    for protein_name in proteins:
        print(f"\n[{protein_name}]")
        ok = process_protein(
            protein_name=protein_name,
            stride=args.stride,
            test_frac=args.test_frac,
            seed=args.seed,
            dry_run=args.dry_run,
            out_dir=out_dir,
        )
        all_ok = all_ok and ok

    print(f"\n{'='*60}")
    if all_ok:
        print("Done.")
    else:
        print("Errors encountered — check output above.")
    print(f"{'='*60}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
