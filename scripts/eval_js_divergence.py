"""
Evaluate a trained TangentScoreModel via reverse-SDE sampling and JS divergence
on pairwise Cα-Cα distances.

This is the canonical sample-quality metric used by ScoreMD (Plainer 2025) and
Xu et al. 2026: generate N conformations via reverse diffusion, compute all
pairwise Cα distances per frame, compare their marginal distributions to those
of the held-out test set via JS divergence, report mean and per-pair.

Usage:
    python scripts/eval_js_divergence.py \
        --ckpt checkpoints/bba_local_run1/ckpt_final.npz \
        --protein bba \
        --n-samples 500 \
        --n-steps 500 \
        --out checkpoints/bba_local_run1/eval_js.npz

Output npz keys:
    js_per_pair   (n_pairs,)  JS divergence for each atom pair
    mean_js       scalar      mean over all pairs
    median_js     scalar
    samples       (N, n, d)   generated conformations
    ts            (n_steps+1,)
"""

import argparse, sys, time
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from manifold.pointcloud_jax import ShapeManifold
from diffusion.manifold_sde import ManifoldVP
from diffusion.manifold_solvers import ManifoldEulerMaruyama
from models.tangent_mlp import TangentScoreModel
from training.train_manifold import load_checkpoint


# ---------------------------------------------------------------------------
# JS divergence helpers
# ---------------------------------------------------------------------------

def js_divergence_1d(p_vals, q_vals, n_bins=50):
    """JS divergence between two 1-D empirical distributions."""
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max())
    if hi == lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins)
    q_hist, _ = np.histogram(q_vals, bins=bins)
    p_hist = p_hist.astype(float) + 1e-8
    q_hist = q_hist.astype(float) + 1e-8
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    m = 0.5 * (p_hist + q_hist)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(p_hist, m) + 0.5 * kl(q_hist, m)


def pairwise_distances(x):
    """
    Compute all upper-triangle pairwise Cα distances.
    x: (N, n, d) → returns (N, n*(n-1)//2)
    """
    N, n, d = x.shape
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = np.linalg.norm(x[:, i] - x[:, j], axis=-1)  # (N,)
            dists.append(d_ij)
    return np.stack(dists, axis=1)  # (N, n_pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--protein", default="bba", choices=["bba", "chignolin"])
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--t-start", type=float, default=0.99)
    parser.add_argument("--t-end",   type=float, default=0.01)
    parser.add_argument("--n-bins",  type=int, default=50)
    parser.add_argument("--input-scale", type=float, default=6.26)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # ---- Config per protein ----
    cfg = {
        "bba":       dict(n=28, d=3, alpha=1.0,
                          test_path="data/processed/bba_test.npy"),
        "chignolin": dict(n=10, d=3, alpha=1.0,
                          test_path="data/processed/chignolin_test.npy"),
    }[args.protein]

    out_path = args.out or str(Path(args.ckpt).parent / "eval_js.npz")

    print(f"Protein:    {args.protein}  (n={cfg['n']}, d={cfg['d']})")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Samples:    {args.n_samples}  steps={args.n_steps}")
    print(f"Output:     {out_path}")
    print()

    # ---- Setup ----
    test_data = np.load(cfg['test_path'])   # (N_test, n, d)
    print(f"Test set:   {test_data.shape}")

    # Use mean of test set as base point (required by s_exp for alignment)
    base_point = test_data.mean(axis=0)    # (n, d)
    manifold = ShapeManifold(dim=cfg['d'], numpoints=cfg['n'], alpha=cfg['alpha'],
                             base=base_point)
    sde      = ManifoldVP(manifold)
    model    = TangentScoreModel(hidden_dims=(256,)*4, input_scale=args.input_scale)

    # ---- Load checkpoint ----
    example_x = test_data[:1]
    params = load_checkpoint(args.ckpt, model, manifold, sde, example_x)
    print(f"Checkpoint loaded.")

    # ---- Score function ----
    # Solver calls score_fn(x, t) where x:(N,1,n,d), t:scalar.
    # Model expects (x_flat:(B,nd), t_col:(B,1)).
    def score_fn(x, t):
        N_batch = x.shape[0]
        x_flat = x.reshape(N_batch, n * d)
        t_col  = jnp.full((N_batch, 1), float(t))
        s_flat = model.apply(params, x_flat, t_col)   # (N, nd)
        return s_flat.reshape(N_batch, 1, n, d)

    # ---- Reverse SDE sampling ----
    solver = ManifoldEulerMaruyama(sde=sde, manifold=manifold,
                                   mode='reverse', score_fn=score_fn)

    n, d = cfg['n'], cfg['d']
    rng = jax.random.PRNGKey(args.seed)

    ts = np.linspace(args.t_start, args.t_end, args.n_steps + 1)
    dt = float(ts[1] - ts[0])   # negative (reverse time)

    # Initial noise ~ N(0, sigma(T)^2 * I), shape (N_samples, 1, n, d)
    rng, init_key = jax.random.split(rng)
    sigma_T = float(sde.sigma(args.t_start))
    x = jax.random.normal(init_key, (args.n_samples, 1, n, d)) * sigma_T

    print(f"Sampling {args.n_samples} conformations via reverse SDE "
          f"({args.n_steps} steps, dt={dt:.4f})...")
    t0 = time.time()

    for i, t_val in enumerate(ts[:-1]):
        rng, step_key = jax.random.split(rng)
        x = solver.step(x, float(t_val), dt, step_key)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step {i+1}/{args.n_steps}  ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"Sampling done in {elapsed:.1f}s")

    samples = np.array(x[:, 0])   # (N_samples, n, d)
    print(f"Samples shape: {samples.shape}")
    print(f"Samples stats: mean={samples.mean():.3f}  std={samples.std():.3f}  "
          f"min={samples.min():.3f}  max={samples.max():.3f}")

    # ---- Pairwise distances ----
    n_pairs = n * (n - 1) // 2
    print(f"\nComputing pairwise distances ({n_pairs} pairs)...")

    ref_dists = pairwise_distances(test_data)      # (N_test, n_pairs)
    gen_dists = pairwise_distances(samples)        # (N_samples, n_pairs)

    # ---- JS divergence per pair ----
    js_per_pair = np.zeros(n_pairs)
    for k in range(n_pairs):
        js_per_pair[k] = js_divergence_1d(
            ref_dists[:, k], gen_dists[:, k], n_bins=args.n_bins
        )

    mean_js   = float(js_per_pair.mean())
    median_js = float(np.median(js_per_pair))
    max_js    = float(js_per_pair.max())

    print(f"\n{'='*50}")
    print(f"JS divergence (pairwise Cα distances, {n_pairs} pairs):")
    print(f"  Mean:   {mean_js:.4f}")
    print(f"  Median: {median_js:.4f}")
    print(f"  Max:    {max_js:.4f}")
    print(f"  p25:    {np.percentile(js_per_pair, 25):.4f}")
    print(f"  p75:    {np.percentile(js_per_pair, 75):.4f}")
    print(f"{'='*50}")

    # Interpretation guide
    print()
    print("Reference thresholds (from ScoreMD/Xu 2026 literature):")
    print("  < 0.05  → good (well-calibrated distribution)")
    print("  0.05–0.15 → moderate (plausible but shifted)")
    print("  > 0.15  → poor (distribution mismatch)")

    # ---- Save ----
    np.savez(out_path,
             js_per_pair=js_per_pair,
             mean_js=np.array(mean_js),
             median_js=np.array(median_js),
             max_js=np.array(max_js),
             samples=samples,
             ts=ts,
             n_samples=np.array(args.n_samples),
             n_steps=np.array(args.n_steps))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
