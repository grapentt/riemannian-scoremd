"""
Phase 3 smoke test (Task 3.5): verify that the TangentScoreModel trains on
chignolin (n=10, 9k frames) and that the DSM loss decreases.

Gates:
  1. Model initializes and forward pass runs without error
  2. Loss is finite at epoch 0
  3. Final loss < initial loss (training makes progress)

Usage:
    pytest tests/test_score_model.py -v
    pytest tests/test_score_model.py -v --runslow   # include slow tests
    python tests/test_score_model.py           # verbose standalone run
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from manifold.pointcloud_jax import ShapeManifold
from diffusion.manifold_sde import ManifoldVP
from models.tangent_mlp import TangentScoreModel, PotentialTangentScoreModel
from training.score_loss import riemannian_dsm_loss
from training.train_manifold import train

# ---------------------------------------------------------------------------
# Synthetic data helper (fast, no file I/O)
# ---------------------------------------------------------------------------

def make_synthetic_data(n=10, d=3, N=200, seed=0):
    """
    Generate N synthetic protein-like conformations (Cα positions).
    Uses a simple random walk backbone to get realistic bond distances.
    """
    rng = np.random.default_rng(seed)
    coords = np.zeros((N, n, d), dtype=np.float32)
    for i in range(N):
        # Random walk with step size 3.8 Å (typical Cα-Cα bond)
        pos = np.zeros((n, d))
        for j in range(1, n):
            direction = rng.standard_normal(d)
            direction /= np.linalg.norm(direction)
            pos[j] = pos[j-1] + 3.8 * direction
        pos -= pos.mean(axis=0)   # centre
        coords[i] = pos.astype(np.float32)
    return coords


def make_manifold(x_ref, n=10, d=3):
    """Build ShapeManifold from reference frame."""
    base = np.array(x_ref, dtype=np.float32)
    base -= base.mean(axis=0)
    return ShapeManifold(dim=d, numpoints=n, alpha=1.0, base=base)


# ---------------------------------------------------------------------------
# Test 1: Forward pass (model init + single prediction)
# ---------------------------------------------------------------------------

def test_forward_pass():
    """Model initializes and produces finite output of the right shape."""
    n, d = 10, 3
    B = 4
    nd = n * d

    model = TangentScoreModel(hidden_dims=(64, 64))
    x_flat = jnp.ones((B, nd))
    t_col = 0.5 * jnp.ones((B, 1))

    params = model.init(jax.random.PRNGKey(0), x_flat, t_col)
    out = model.apply(params, x_flat, t_col)

    assert out.shape == (B, nd), f"Expected ({B}, {nd}), got {out.shape}"
    assert bool(jnp.all(jnp.isfinite(out))), "Non-finite output"

    print(f"  ✓ forward pass: shape={out.shape}, finite=True  PASS")
    return True


# ---------------------------------------------------------------------------
# Test 2: Loss finite on single batch
# ---------------------------------------------------------------------------

def test_loss_finite():
    """DSM loss is finite for a small batch of synthetic frames."""
    n, d = 10, 3
    B = 8
    x0 = make_synthetic_data(n=n, N=B)  # (B, n, d)
    x_ref = x0[0]
    manifold = make_manifold(x_ref, n=n, d=d)
    sde = ManifoldVP(manifold)

    model = TangentScoreModel(hidden_dims=(64, 64))
    nd = n * d
    dummy_x = jnp.zeros((1, nd))
    dummy_t = jnp.zeros((1, 1))
    params = model.init(jax.random.PRNGKey(0), dummy_x, dummy_t)

    score_fn = lambda x_flat, t_col: model.apply(params, x_flat, t_col)
    t_batch = 0.5 * jnp.ones(B)
    rng = jax.random.PRNGKey(1)

    loss = riemannian_dsm_loss(
        score_fn=score_fn,
        manifold=manifold,
        sde=sde,
        x0=jnp.array(x0),
        t=t_batch,
        rng=rng,
    )
    loss_val = float(loss)
    ok = np.isfinite(loss_val)
    status = "PASS" if ok else "FAIL"
    print(f"  {'✓' if ok else '✗'} loss finite: loss={loss_val:.4f}  {status}")
    return ok


# ---------------------------------------------------------------------------
# Test 3: Loss decreases over 200 epochs (synthetic data)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_loss_decreases():
    """
    Train for 200 epochs on chignolin data (or 200 synthetic frames as fallback).
    Gate: final loss < initial loss (any improvement).
    """
    n, d = 10, 3

    # Use real data if available — synthetic random-walk proteins have very
    # different geometry and make the loss explode at small t (alpha/sigma >> 1)
    data_path = _ROOT / "data" / "processed" / "chignolin_train.npy"
    if data_path.exists():
        x0 = np.load(data_path)[:500]   # 500 frames, fast on CPU
        label = "real chignolin (500 frames)"
    else:
        x0 = make_synthetic_data(n=n, N=200)
        label = "synthetic (200 frames)"

    x_ref = x0[0]
    manifold = make_manifold(x_ref, n=x0.shape[1], d=x0.shape[2])
    sde = ManifoldVP(manifold)

    model = TangentScoreModel(hidden_dims=(64, 64, 64))
    print(f"  [{label}]")

    state, history = train(
        model=model,
        manifold=manifold,
        sde=sde,
        train_data=x0,
        n_epochs=200,
        batch_size=32,
        learning_rate=3e-4,
        log_every=50,
        seed=42,
    )

    first_loss = history[0][1]
    last_loss = history[-1][1]
    ok = last_loss < first_loss and not np.isnan(last_loss)
    status = "PASS" if ok else "FAIL"
    print(f"  {'✓' if ok else '✗'} loss decreases: "
          f"initial={first_loss:.4f} → final={last_loss:.4f}  {status}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: PotentialTangentScoreModel forward pass
# ---------------------------------------------------------------------------

def test_potential_model_forward():
    """Conservative model initializes and produces finite output."""
    n, d = 10, 3
    B = 4
    nd = n * d

    model = PotentialTangentScoreModel(hidden_dims=(64, 64))
    x_flat = jnp.ones((B, nd)) * 0.1
    t_col = 0.5 * jnp.ones((B, 1))

    params = model.init(jax.random.PRNGKey(0), x_flat, t_col)
    out = model.apply(params, x_flat, t_col)

    assert out.shape == (B, nd), f"Expected ({B}, {nd}), got {out.shape}"
    assert bool(jnp.all(jnp.isfinite(out))), "Non-finite output"

    print(f"  ✓ PotentialTangentScoreModel forward: shape={out.shape}  PASS")
    return True


# ---------------------------------------------------------------------------
# pytest wrappers
# ---------------------------------------------------------------------------

def test_score_forward():
    assert test_forward_pass()

def test_score_loss_finite():
    assert test_loss_finite()

@pytest.mark.slow
def test_score_loss_decreases():
    assert test_loss_decreases()

def test_potential_score_forward():
    assert test_potential_model_forward()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-real-data", action="store_true",
                        help="Use real chignolin data from data/processed/ if available")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("Phase 3 smoke test: TangentScoreModel + Riemannian DSM loss")
    print("=" * 65 + "\n")

    results = [
        ("forward pass (TangentScoreModel)",         test_forward_pass()),
        ("loss finite on single batch",               test_loss_finite()),
        ("loss decreases (200 epochs, synthetic)",    test_loss_decreases()),
        ("forward pass (PotentialTangentScoreModel)", test_potential_model_forward()),
    ]

    if args.use_real_data:
        data_path = _ROOT / "data" / "processed" / "chignolin_train.npy"
        if data_path.exists():
            print(f"\n--- Real chignolin data ({data_path}) ---")
            x_train = np.load(data_path)[:500]   # 500 frames for quick test
            x_ref = x_train[0]
            n, d = x_train.shape[1], x_train.shape[2]
            manifold = make_manifold(x_ref, n=n, d=d)
            sde = ManifoldVP(manifold)
            model = TangentScoreModel(hidden_dims=(128, 128, 128))
            state, history = train(
                model=model, manifold=manifold, sde=sde,
                train_data=x_train, n_epochs=200, batch_size=32,
                learning_rate=1e-3, log_every=50, seed=0,
            )
            first, last = history[0][1], history[-1][1]
            ok = last < first
            results.append(("real chignolin loss decreases (200 epochs)", ok))
        else:
            print(f"\n  [SKIP] real data not found at {data_path}")

    passed = sum(r for _, r in results)
    total = len(results)
    print(f"\n{'='*65}")
    print(f"SUMMARY: {passed}/{total} passed")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    if passed == total:
        print("\nALL PHASE 3 SMOKE TESTS PASSED")
    print("=" * 65)
    sys.exit(0 if passed == total else 1)
