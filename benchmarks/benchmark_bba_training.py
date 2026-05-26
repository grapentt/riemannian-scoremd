"""
Quick BBA training speed benchmark (Phase 3.6 pre-flight).

Measures wall time for:
  - Phase A: prepare_batch (s_exp + score_target) for B=32 samples at n=28
  - Phase B: JIT-compiled train_step (network forward + grad + EMA update)
  - Total step time and estimated training time for 1000 epochs

Usage:
    python benchmarks/benchmark_bba_training.py
"""

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from manifold.pointcloud_jax import ShapeManifold
from diffusion.manifold_sde import ManifoldVP
from models.tangent_mlp import TangentScoreModel
from training.score_loss import prepare_batch, riemannian_dsm_loss_from_noised
from training.train_manifold import make_train_step

import optax

def main():
    # ---- Load BBA data ----
    data_path = _ROOT / "data" / "processed" / "bba_train.npy"
    ref_path  = _ROOT / "data" / "processed" / "bba_ref.npy"

    print(f"Loading BBA training data from {data_path}")
    train_data = np.load(data_path)
    x_ref = np.load(ref_path)
    N, n, d = train_data.shape
    print(f"  {N} frames, n={n} Cα, d={d}  →  nd={n*d} dims")

    # ---- Build manifold + SDE ----
    manifold = ShapeManifold(dim=d, numpoints=n, alpha=1.0, base=x_ref)
    sde = ManifoldVP(manifold)

    # ---- Model: 256×4 (Phase 3.6 size) ----
    model = TangentScoreModel(hidden_dims=(256, 256, 256, 256))
    nd = n * d
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, nd)), jnp.zeros((1, 1)))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {n_params:,}")

    # ---- Optimizer (same as training) ----
    schedule = optax.cosine_decay_schedule(init_value=3e-4, decay_steps=10000, alpha=1e-5/3e-4)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)
    ema_params = params

    train_step = make_train_step(model, manifold, sde, optimizer, likelihood_weighting=True)

    B = 32
    rng = jax.random.PRNGKey(42)

    # Sample a mini-batch
    x0_batch = jnp.array(train_data[:B])
    rng, t_key, noise_key = jax.random.split(rng, 3)
    t_batch = jax.random.uniform(t_key, (B,), minval=0.01, maxval=0.99)

    # ---- Warm-up Phase A (1 call to warm JIT caches in s_exp) ----
    print("\n[Warm-up Phase A — s_exp JIT compilation]")
    t0 = time.time()
    x_t, s_true = prepare_batch(manifold, sde, x0_batch, t_batch, noise_key)
    dt_warmup = time.time() - t0
    print(f"  First prepare_batch (includes JIT compile): {dt_warmup:.2f}s")

    # ---- Warm-up Phase B (JIT compile train_step) ----
    print("[Warm-up Phase B — train_step JIT compilation]")
    t0 = time.time()
    params, ema_params, opt_state, loss = train_step(
        params, ema_params, opt_state, x_t, s_true, t_batch
    )
    loss.block_until_ready()
    dt_b_warmup = time.time() - t0
    print(f"  First train_step (includes JIT compile): {dt_b_warmup:.2f}s  loss={float(loss):.4f}")

    # ---- Benchmark N_REPS steps (post warm-up) ----
    N_REPS = 10
    print(f"\n[Benchmark: {N_REPS} steps post warm-up, B={B}]")

    times_a, times_b = [], []

    for i in range(N_REPS):
        rng, t_key, noise_key = jax.random.split(rng, 3)
        x0_batch = jnp.array(train_data[i*B:(i+1)*B])
        t_batch = jax.random.uniform(t_key, (B,), minval=0.01, maxval=0.99)

        t0 = time.time()
        x_t, s_true = prepare_batch(manifold, sde, x0_batch, t_batch, noise_key)
        dt_a = time.time() - t0
        times_a.append(dt_a)

        t0 = time.time()
        params, ema_params, opt_state, loss = train_step(
            params, ema_params, opt_state, x_t, s_true, t_batch
        )
        loss.block_until_ready()
        dt_b = time.time() - t0
        times_b.append(dt_b)

    mean_a = np.mean(times_a)
    mean_b = np.mean(times_b)
    mean_total = mean_a + mean_b

    print(f"  Phase A (prepare_batch):   {mean_a*1000:.1f} ms/step  (min={min(times_a)*1000:.1f}, max={max(times_a)*1000:.1f})")
    print(f"  Phase B (train_step JIT):  {mean_b*1000:.1f} ms/step  (min={min(times_b)*1000:.1f}, max={max(times_b)*1000:.1f})")
    print(f"  Total:                     {mean_total*1000:.1f} ms/step")

    # ---- Estimate full training time ----
    # BBA: 63k frames, B=32 → ~1969 steps/epoch
    steps_per_epoch = max(1, N // B)
    secs_per_epoch = mean_total * steps_per_epoch
    print(f"\n[Projected training time for BBA, 63k frames, B={B}]")
    print(f"  Steps/epoch:       {steps_per_epoch}")
    print(f"  Time/epoch:        {secs_per_epoch:.1f}s ({secs_per_epoch/60:.1f} min)")
    print(f"  100 epochs:        {100*secs_per_epoch/60:.0f} min ({100*secs_per_epoch/3600:.1f} h)")
    print(f"  1000 epochs:       {1000*secs_per_epoch/60:.0f} min ({1000*secs_per_epoch/3600:.1f} h)")
    print(f"  [Phase A dominates — {100*mean_a/mean_total:.0f}% of step time]")

    # ---- Suggest batch size / strategy ----
    print(f"\n[Phase A breakdown per sample]")
    print(f"  prepare_batch per sample: {mean_a*1000/B:.2f} ms  (n={n} Cα)")
    print(f"  [Compare: chignolin n=10 was ~{24.4:.1f} ms total per step at B=32]")
    # BBA n=28: s_exp is O(n²d) per iter, so roughly (28/214)² × faster than AK at n=214
    # but actually s_exp cost scales with n² distance matrix, so n=28 is ~(28²/10²) = 7.84× slower than chignolin
    # Per-sample breakdown
    print(f"  Estimated: s_exp+s_log ∝ n² → n=28 should be ~{(28/10)**2:.1f}× slower than chignolin per sample")


if __name__ == "__main__":
    main()
