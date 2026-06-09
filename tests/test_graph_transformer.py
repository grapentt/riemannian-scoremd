"""
Tests for GraphTransformerScoreModel.

Gates:
  1. Forward pass (both modes): correct shape, finite output
  2. Loss finite on a single batch (both modes)
  3. Loss decreases over 200 epochs (non-conservative, fast)
  4. Both modes produce same-shape output; conservative ≠ non-conservative at init

Usage:
    pytest riemannian-scoremd/tests/test_graph_transformer.py -v
    python riemannian-scoremd/tests/test_graph_transformer.py
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from manifold.pointcloud_jax import ShapeManifold
from diffusion.manifold_sde import ManifoldVP
from models.graph_transformer_jax import GraphTransformerScoreModel
from training.score_loss import riemannian_dsm_loss_from_noised


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# BBA dimensions
N_ATOMS = 28
D = 3
ND = N_ATOMS * D


def make_synthetic_bba(N: int = 64, seed: int = 0):
    """Generate N synthetic BBA-like conformations (Cα positions, centred)."""
    rng = np.random.default_rng(seed)
    coords = np.zeros((N, N_ATOMS, D), dtype=np.float32)
    for i in range(N):
        pos = np.zeros((N_ATOMS, D))
        for j in range(1, N_ATOMS):
            direction = rng.standard_normal(D)
            direction /= np.linalg.norm(direction)
            pos[j] = pos[j - 1] + 3.8 * direction
        pos -= pos.mean(axis=0)
        coords[i] = pos.astype(np.float32)
    return coords


def make_manifold_and_sde(x0):
    """Build ShapeManifold + ManifoldVP from reference frame."""
    base = x0[0].copy()
    base -= base.mean(axis=0)
    manifold = ShapeManifold(dim=D, numpoints=N_ATOMS, alpha=1.0, base=base)
    sde = ManifoldVP(manifold)
    return manifold, sde


def make_noised_batch(manifold, sde, x0, B: int = 8, seed: int = 1):
    """Generate a batch of (x_t, s_true, t) from synthetic data using online noising."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x0), B)
    x_batch = jnp.array(x0[idx])        # (B, n, d)
    t_batch = jnp.array(rng.uniform(0.1, 0.5, B).astype(np.float32))

    # Compute noised samples and score targets online
    jax_rng = jax.random.PRNGKey(seed)
    x_t_list, s_true_list = [], []
    for b in range(B):
        xb = x_batch[b:b+1, None, :, :]    # (1, 1, n, d)
        tb = float(t_batch[b])
        rng_b, jax_rng = jax.random.split(jax_rng)
        xt_b, _, sigma_b = sde.marginal_prob(xb, jnp.array([[tb]]), rng_b)
        # horizontal score target via prelog + project_G
        from training.score_loss import score_target
        st_b = score_target(manifold, xt_b, xb, jnp.array([[tb]]), sde)
        x_t_list.append(xt_b[0, 0])
        s_true_list.append(st_b[0, 0])

    x_t = jnp.stack(x_t_list)      # (B, n, d)
    s_true = jnp.stack(s_true_list) # (B, n, d)
    return x_t, s_true, t_batch


# ---------------------------------------------------------------------------
# Test 1: Forward pass — both conservative and non-conservative modes
# ---------------------------------------------------------------------------

def test_forward_pass():
    """GraphTransformerScoreModel: correct output shape and finite values (both modes)."""
    B = 4
    x_flat = jnp.ones((B, ND))
    t_col  = 0.5 * jnp.ones((B, 1))

    for conservative in [False, True]:
        model = GraphTransformerScoreModel(n=N_ATOMS, d=D, hidden_dim=32, num_layers=1,
                                           num_heads=4, dim_head=8,
                                           conservative=conservative)
        params = model.init(jax.random.PRNGKey(0), x_flat, t_col)
        out = model.apply(params, x_flat, t_col)

        assert out.shape == (B, ND), f"conservative={conservative}: expected ({B},{ND}), got {out.shape}"
        assert bool(jnp.all(jnp.isfinite(out))), f"conservative={conservative}: non-finite output"
        print(f"  ✓ forward pass conservative={conservative}: shape={out.shape}  PASS")

    return True


# ---------------------------------------------------------------------------
# Test 2: Loss finite — both modes
# ---------------------------------------------------------------------------

def test_loss_finite():
    """DSM loss is finite for a small batch with both model modes."""
    x0 = make_synthetic_bba(N=32, seed=0)
    manifold, sde = make_manifold_and_sde(x0)

    B = 8
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(x0), B)
    x_t    = jnp.array(x0[idx])
    t      = 0.3 * jnp.ones(B)
    s_true = jnp.zeros_like(x_t)

    all_ok = True
    for conservative in [False, True]:
        model = GraphTransformerScoreModel(n=N_ATOMS, d=D, hidden_dim=32, num_layers=1,
                                           num_heads=4, dim_head=8,
                                           conservative=conservative)
        params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, ND)), jnp.zeros((1, 1)))
        score_fn = lambda x_flat, t_col: model.apply(params, x_flat, t_col)

        loss = riemannian_dsm_loss_from_noised(
            score_fn=score_fn, manifold=manifold, sde=sde,
            x_t=x_t, s_true=s_true, t=t,
            likelihood_weighting=False, normalize_targets=False,
        )
        loss_val = float(loss)
        ok = np.isfinite(loss_val)
        all_ok = all_ok and ok
        print(f"  {'✓' if ok else '✗'} loss finite conservative={conservative}: "
              f"loss={loss_val:.4f}  {'PASS' if ok else 'FAIL'}")

    return all_ok


# ---------------------------------------------------------------------------
# Test 3: Loss decreases over 200 epochs (non-conservative — fast)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_loss_decreases():
    """Train 200 epochs on 64 synthetic BBA frames (non-conservative) — loss must decrease."""
    x0 = make_synthetic_bba(N=64, seed=42)
    manifold, sde = make_manifold_and_sde(x0)

    # Non-conservative: fast, ~0.3ms/step
    model = GraphTransformerScoreModel(n=N_ATOMS, d=D, hidden_dim=32, num_layers=1,
                                       num_heads=4, dim_head=8, conservative=False)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, ND)), jnp.zeros((1, 1)))

    # Use precomputed-style batch: x_t ≈ x0, s_true = zeros, t=0.3
    # This is a smoke test for training mechanics (loss finiteness + gradient flow)
    B = 32
    rng_np = np.random.default_rng(0)
    idx = rng_np.integers(0, len(x0), B)
    x_t   = jnp.array(x0[idx])
    s_true = jnp.zeros_like(x_t)
    t_arr  = 0.3 * jnp.ones(B)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-3),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        def loss_fn(p):
            sf = lambda xf, tc: model.apply(p, xf, tc)
            return riemannian_dsm_loss_from_noised(
                score_fn=sf, manifold=manifold, sde=sde,
                x_t=x_t, s_true=s_true, t=t_arr,
                likelihood_weighting=False, normalize_targets=False,
            )
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    history = []
    for ep in range(200):
        params, opt_state, loss = train_step(params, opt_state)
        if ep % 50 == 0 or ep == 199:
            history.append((ep, float(loss)))
            print(f"    epoch {ep:3d}: loss={float(loss):.4f}")

    first_loss = history[0][1]
    last_loss  = history[-1][1]
    ok = last_loss < first_loss and not np.isnan(last_loss)
    status = "PASS" if ok else "FAIL"
    print(f"  {'✓' if ok else '✗'} loss decreases: "
          f"initial={first_loss:.4f} → final={last_loss:.4f}  {status}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: Conservative and non-conservative modes produce different outputs
# ---------------------------------------------------------------------------

def test_modes_differ():
    """Conservative and non-conservative modes must produce different outputs at init."""
    B = 4
    x_flat = jnp.ones((B, ND))
    t_col  = 0.5 * jnp.ones((B, 1))
    rng    = jax.random.PRNGKey(0)

    model_nc = GraphTransformerScoreModel(n=N_ATOMS, d=D, hidden_dim=32, num_layers=1,
                                          num_heads=4, dim_head=8, conservative=False)
    model_c  = GraphTransformerScoreModel(n=N_ATOMS, d=D, hidden_dim=32, num_layers=1,
                                          num_heads=4, dim_head=8, conservative=True)

    params_nc = model_nc.init(rng, x_flat, t_col)
    params_c  = model_c.init(rng, x_flat, t_col)

    out_nc = model_nc.apply(params_nc, x_flat, t_col)
    out_c  = model_c.apply(params_c,  x_flat, t_col)

    # Both finite
    assert bool(jnp.all(jnp.isfinite(out_nc))), "non-conservative: non-finite output"
    assert bool(jnp.all(jnp.isfinite(out_c))),  "conservative: non-finite output"

    # Outputs differ (different parameterization → different function at init)
    max_diff = float(jnp.abs(out_nc - out_c).max())
    ok = max_diff > 1e-6
    print(f"  {'✓' if ok else '✗'} modes differ: max_diff={max_diff:.4f}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Forward pass (both modes)")
    print("=" * 60)
    r1 = test_forward_pass()

    print("\n" + "=" * 60)
    print("TEST 2: Loss finite (both modes)")
    print("=" * 60)
    r2 = test_loss_finite()

    print("\n" + "=" * 60)
    print("TEST 3: Loss decreases 200 epochs (non-conservative)")
    print("=" * 60)
    r3 = test_loss_decreases()

    print("\n" + "=" * 60)
    print("TEST 4: Conservative ≠ non-conservative")
    print("=" * 60)
    r4 = test_modes_differ()

    passed = sum([r1, r2, r3, r4])
    print(f"\n{'='*60}")
    print(f"Results: {passed}/4 passed")
    if passed < 4:
        sys.exit(1)
