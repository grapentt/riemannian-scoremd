"""
Offline pre-computation of noised training data (Phase A caching).

Usage:
    python scripts/precompute_noised_data.py \\
        --protein bba \\
        --n-repeats 5 \\
        --output data/precomputed/

This script pre-runs Phase A (marginal_prob + score_target) for every training
frame at randomly sampled diffusion times, saving (x_t, s_true, t) tensors to
disk. Training then runs Phase B only (pure JIT, ~0.7 ms/step on CPU), which is
~400× faster than the online prepare_batch approach.

Strategy:
  For each "noise repeat" r in 1..n_repeats:
    - Sample t_i ~ Uniform[t_min, t_max] for each frame i
    - Compute x_t_i = marginal_prob(x_i, t_i) and s_true_i = score_target(...)
    - Save to data/precomputed/{protein}_noised_r{r}.npz

Training uses these pre-computed files by shuffling across noise repeats per epoch.
n_repeats=5 gives 5 independent noised views of the dataset (different t samples),
providing enough stochasticity for training while being fast at inference time.

n_repeats=10 is recommended for a proper training run. Each repeat takes:
  BBA (n=28, 63k frames): 63000 × 8ms = 504s ≈ 8.4 min per repeat
  Chignolin (n=10, 9k frames): 9000 × ~2ms = 18s per repeat
"""

import sys
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from manifold.pointcloud_jax import ShapeManifold
from diffusion.manifold_sde import ManifoldVP
from training.score_loss import prepare_batch


PROTEINS = {
    "chignolin": {"n": 10, "d": 3, "alpha": 1.0},
    "bba":       {"n": 28, "d": 3, "alpha": 1.0},
}


def precompute_repeat(
    manifold, sde, train_data, t_min, t_max, rng, repeat_idx, out_path, B=128
):
    """
    Compute (x_t, s_true, t) for all frames in train_data.
    Processes in batches of B to show progress.
    Saves to out_path as a compressed .npz file.
    """
    N, n, d = train_data.shape
    rng, t_key = jax.random.split(rng)
    t_all = jax.random.uniform(t_key, (N,), minval=t_min, maxval=t_max)

    x_t_all = np.zeros((N, n, d), dtype=np.float32)
    s_true_all = np.zeros((N, n, d), dtype=np.float32)

    n_batches = (N + B - 1) // B
    t0 = time.time()

    for b in range(n_batches):
        start = b * B
        end = min(start + B, N)
        x0_batch = jnp.array(train_data[start:end])
        t_batch = t_all[start:end]

        rng, noise_key = jax.random.split(rng)
        x_t_batch, s_true_batch = prepare_batch(
            manifold, sde, x0_batch, t_batch, noise_key
        )

        x_t_all[start:end] = np.array(x_t_batch)
        s_true_all[start:end] = np.array(s_true_batch)

        if (b + 1) % 20 == 0 or b == n_batches - 1:
            elapsed = time.time() - t0
            rate = (end) / elapsed
            eta = (N - end) / rate if rate > 0 else 0
            print(f"  repeat {repeat_idx:2d}  [{end:6d}/{N}]  "
                  f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s  "
                  f"({rate:.0f} frames/s)")

    np.savez_compressed(
        out_path,
        x_t=x_t_all,
        s_true=s_true_all,
        t=np.array(t_all),
    )
    elapsed = time.time() - t0
    print(f"  → saved {out_path}  ({elapsed:.1f}s)")
    return rng


def main():
    parser = argparse.ArgumentParser(description="Pre-compute noised training data.")
    parser.add_argument("--protein", choices=list(PROTEINS.keys()), default="bba")
    parser.add_argument("--n-repeats", type=int, default=5,
                        help="Number of independent noise draws per dataset")
    parser.add_argument("--output", type=str,
                        default=str(_ROOT / "data" / "precomputed"),
                        help="Output directory")
    parser.add_argument("--t-min", type=float, default=0.01)
    parser.add_argument("--t-max", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for prepare_batch calls (affects only progress reporting)")
    parser.add_argument("--start-repeat", type=int, default=0,
                        help="Resume from this repeat index (0-based)")
    args = parser.parse_args()

    out_dir = Path(args.output) / args.protein
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    data_path = _ROOT / "data" / "processed" / f"{args.protein}_train.npy"
    ref_path  = _ROOT / "data" / "processed" / f"{args.protein}_ref.npy"
    train_data = np.load(data_path)
    x_ref = np.load(ref_path)
    N, n, d = train_data.shape
    print(f"Protein: {args.protein}  N={N} frames, n={n} Cα, d={d}")
    print(f"Output dir: {out_dir}")
    print(f"n_repeats: {args.n_repeats}, t_min={args.t_min}, t_max={args.t_max}\n")

    info = PROTEINS[args.protein]
    manifold = ShapeManifold(dim=d, numpoints=n, alpha=1.0, base=x_ref)
    sde = ManifoldVP(manifold)

    # ---- Warm up s_exp JIT ----
    print("Warming up s_exp JIT cache...")
    rng = jax.random.PRNGKey(args.seed)
    rng, t_key, n_key = jax.random.split(rng, 3)
    x0_warmup = jnp.array(train_data[:4])
    t_warmup = jax.random.uniform(t_key, (4,), minval=args.t_min, maxval=args.t_max)
    t0 = time.time()
    prepare_batch(manifold, sde, x0_warmup, t_warmup, n_key)
    print(f"  JIT warm-up: {time.time()-t0:.2f}s\n")

    # ---- Estimate time ----
    print("Timing one batch to estimate total time...")
    rng, t_key, n_key = jax.random.split(rng, 3)
    x0_b = jnp.array(train_data[:args.batch_size])
    t_b = jax.random.uniform(t_key, (args.batch_size,), minval=args.t_min, maxval=args.t_max)
    t0 = time.time()
    prepare_batch(manifold, sde, x0_b, t_b, n_key)
    dt_per_sample = (time.time() - t0) / args.batch_size
    total_est = dt_per_sample * N * args.n_repeats
    print(f"  {dt_per_sample*1000:.2f} ms/sample → {N} frames × {args.n_repeats} repeats = {total_est/60:.1f} min total\n")

    # ---- Pre-compute repeats ----
    for r in range(args.start_repeat, args.n_repeats):
        out_path = out_dir / f"noised_r{r:02d}.npz"
        if out_path.exists():
            print(f"  repeat {r:2d}: already exists at {out_path}, skipping")
            continue
        print(f"Computing repeat {r}/{args.n_repeats-1}...")
        rng = precompute_repeat(
            manifold, sde, train_data,
            args.t_min, args.t_max, rng, r,
            out_path, B=args.batch_size
        )

    print(f"\nDone. All {args.n_repeats} repeats saved to {out_dir}/")

    # ---- Print training estimate with precomputed data ----
    # Phase B only: ~0.7ms/step at B=32
    steps_per_epoch = N // 32  # or use a batch size
    phase_b_ms = 0.7  # ms per step (from benchmark)
    print(f"\n[Estimated training time with precomputed data (Phase B only)]")
    print(f"  Steps/epoch: {steps_per_epoch} (B=32)")
    print(f"  ~{phase_b_ms:.1f} ms/step → {steps_per_epoch*phase_b_ms/1000:.1f}s/epoch")
    print(f"  1000 epochs: {steps_per_epoch*phase_b_ms*1000/3600:.1f}h")


if __name__ == "__main__":
    main()
