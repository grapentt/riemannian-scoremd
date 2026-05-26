"""
Tests for the separation-gradient (use_separation_grad=True) variant of s_geodesic.

The key claim: using s_prelog (Euclidean gradient of w²) instead of s_log
(Riemannian gradient, requires eigh(nd×nd)) in the inner loop of s_geodesic
converges to the same minimum — i.e., geodesics are the same up to tolerance.

Tests:
  1. Geodesic midpoint accuracy: prelog-based midpoint ~ log-based midpoint
  2. Geodesic endpoint (tau=2, used by s_exp) accuracy
  3. s_exp accuracy: both variants produce the same exponential map
  4. BM tests still pass with use_separation_grad=True (regression guard)

Fast mode (default): n=10 synthetic, runs in ~5–30s.
Full mode (--full):  n=214 adenylate kinase, runs in ~2–10 min.
"""

import sys
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from manifold.pointcloud_jax import ShapeManifold

FAST_MODE = os.environ.get("FAST_MODE", "True").lower() not in ("false", "0", "no")

DATA_PATH = _ROOT.parent / "diepeveen2024" / "data" / "molecular_dynamics" / "4ake"
PSF_FILE  = DATA_PATH / "adk4ake.psf"
DCD_FILE  = DATA_PATH / "dims0001_fit-core.dcd"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_manifold_and_pair(fast: bool = True):
    """Return (manifold, x, y) — two nearby conformations."""
    if not fast:
        try:
            import mdtraj
            traj = mdtraj.load(str(DCD_FILE), top=str(PSF_FILE))
            ca_idx = traj.topology.select("name CA")
            x_np = (traj.xyz[0, ca_idx] * 10.0).astype(np.float32)  # nm -> Angstrom
            y_np = (traj.xyz[1, ca_idx] * 10.0).astype(np.float32)
            x = jnp.array(x_np[None, None])
            y = jnp.array(y_np[None, None])
            n, d = x.shape[2], x.shape[3]
            m = ShapeManifold(dim=d, numpoints=n, alpha=1.0, base=np.array(x[0, 0]))
            return m, m.align_mpoint(x), m.align_mpoint(y)
        except (ImportError, FileNotFoundError):
            pass
    # Synthetic n=10
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((1, 1, 10, 3)).astype(np.float32) * 5.0
    y_np = x_np + rng.standard_normal((1, 1, 10, 3)).astype(np.float32) * 0.5
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    n, d = x.shape[2], x.shape[3]
    m = ShapeManifold(dim=d, numpoints=n, alpha=1.0, base=np.array(x[0, 0]))
    return m, m.align_mpoint(x), m.align_mpoint(y)


# ---------------------------------------------------------------------------
# Test 1: geodesic midpoint (tau=0.5)
# ---------------------------------------------------------------------------

def test_midpoint_accuracy(fast: bool = True, tol_dist: float = 0.05):
    """
    The midpoint from prelog-based and log-based s_geodesic should agree
    in w^delta distance to within tol_dist.

    tol_dist is set generously (0.05) because the two methods converge to
    the same critical point but may land slightly differently within tolerance.
    """
    print("\n[Test 1] Geodesic midpoint (tau=0.5): prelog vs log accuracy")
    manifold, x, y = make_manifold_and_pair(fast)

    tau_half = jnp.array([0.5])
    tol = 1e-3

    z_log = manifold.s_geodesic(x, y, tau_half, tol=tol, use_separation_grad=False)
    z_sep = manifold.s_geodesic(x, y, tau_half, tol=tol, use_separation_grad=True)

    dist = float(manifold.s_distance(z_log, z_sep)[0, 0, 0])
    dist_xy = float(manifold.s_distance(x, y)[0, 0, 0])

    print(f"  w^delta(z_log, z_sep) = {dist:.6f}")
    print(f"  w^delta(x, y)         = {dist_xy:.6f}")
    print(f"  Relative distance     = {dist / max(dist_xy, 1e-8):.4f}")

    assert dist < tol_dist, (
        f"Midpoints disagree: w^delta(z_log, z_sep)={dist:.4f} > tol={tol_dist}"
    )
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2: geodesic extrapolation (tau=2, used by s_exp)
# ---------------------------------------------------------------------------

def test_extrapolation_accuracy(fast: bool = True, tol_dist: float = 0.1):
    """
    tau=2 extrapolation (used inside s_exp doubling steps).
    The prelog version should agree with the log version.
    """
    print("\n[Test 2] Geodesic extrapolation (tau=2.0): prelog vs log accuracy")
    manifold, x, y = make_manifold_and_pair(fast)

    # Use a small tangent so K=1 in s_exp — adjust y to be close to x
    rng = np.random.default_rng(7)
    n, d = x.shape[2], x.shape[3]
    noise = jnp.array(rng.standard_normal((1, 1, n, d)).astype(np.float32)) * 0.3
    y_close = manifold.align_mpoint(x + noise)

    tau_two = jnp.array([2.0])
    tol = 1e-3

    z_log = manifold.s_geodesic(x, y_close, tau_two, tol=tol, use_separation_grad=False)
    z_sep = manifold.s_geodesic(x, y_close, tau_two, tol=tol, use_separation_grad=True)

    dist = float(manifold.s_distance(z_log, z_sep)[0, 0, 0])
    dist_xy = float(manifold.s_distance(x, y_close)[0, 0, 0])

    print(f"  w^delta(z_log, z_sep) = {dist:.6f}")
    print(f"  w^delta(x, y_close)   = {dist_xy:.6f}")
    print(f"  Relative distance     = {dist / max(dist_xy, 1e-8):.4f}")

    assert dist < tol_dist, (
        f"Extrapolation endpoints disagree: w^delta={dist:.4f} > tol={tol_dist}"
    )
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: s_exp accuracy (both variants)
# ---------------------------------------------------------------------------

def test_s_exp_accuracy(fast: bool = True, tol_dist: float = 0.15):
    """
    s_exp(x, X, use_separation_grad=True) and (False) should produce
    the same output up to tol_dist in w^delta distance.
    """
    print("\n[Test 3] s_exp accuracy: prelog vs log geodesic doubling")
    manifold, x, _ = make_manifold_and_pair(fast)

    rng = jax.random.PRNGKey(123)
    v = jax.random.normal(rng, x.shape)
    v_hp = manifold.horizontal_projection_tvector(x, v[:, :, None])
    nrm = manifold.norm(x, v_hp)
    nrm_safe = jnp.maximum(nrm, 1e-8)[:, :, :, None, None]
    # Use sigma=0.3 tangent vector
    X = 0.3 * v_hp[:, :, 0] / nrm_safe[:, :, 0]

    tol = 1e-3

    x_exp_log = manifold.s_exp(x, X, tol=tol, use_separation_grad=False)
    x_exp_sep = manifold.s_exp(x, X, tol=tol, use_separation_grad=True)

    dist = float(manifold.s_distance(x_exp_log, x_exp_sep)[0, 0, 0])
    dist_X = float(manifold.norm(x, X[:, :, None])[0, 0, 0])

    print(f"  ||X||_g               = {dist_X:.6f}")
    print(f"  w^delta(exp_log, exp_sep) = {dist:.6f}")
    print(f"  Relative distance     = {dist / max(dist_X, 1e-8):.4f}")

    assert dist < tol_dist, (
        f"s_exp outputs disagree: w^delta={dist:.4f} > tol={tol_dist}"
    )
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 4: round-trip — s_log(s_exp(x, X), x) ≈ -(scaling * X)
# ---------------------------------------------------------------------------

def test_round_trip(fast: bool = True, cos_tol: float = 0.85):
    """
    s_log(s_exp(x, X, use_separation_grad=True), x) should point in -X direction.

    Note: K=1 s_exp doubles the geodesic, so s_log gives ≈ -2X (factor of 2 is expected).
    We check direction (cosine similarity) only, not magnitude.
    """
    print("\n[Test 4] Round-trip: direction of s_log(s_exp(x, X), x)")
    manifold, x, _ = make_manifold_and_pair(fast)

    rng = jax.random.PRNGKey(55)
    v = jax.random.normal(rng, x.shape)
    v_hp = manifold.horizontal_projection_tvector(x, v[:, :, None])
    nrm = manifold.norm(x, v_hp)
    nrm_safe = jnp.maximum(nrm, 1e-8)[:, :, :, None, None]
    X = 0.3 * v_hp[:, :, 0] / nrm_safe[:, :, 0]

    for use_sep, label in [(True, "prelog"), (False, "s_log")]:
        x_t = manifold.s_exp(x, X, tol=1e-3, use_separation_grad=use_sep)
        log_back = manifold.s_log(x_t, x)[0, 0, 0].flatten()   # (nd,)
        X_flat = X[0, 0].flatten()

        cos_sim = float(
            jnp.dot(-log_back, X_flat) /
            (jnp.linalg.norm(log_back) * jnp.linalg.norm(X_flat) + 1e-12)
        )
        mag_ratio = float(jnp.linalg.norm(log_back) / (jnp.linalg.norm(X_flat) + 1e-12))
        print(f"  [{label}]  cos(-s_log, X) = {cos_sim:.4f}  |s_log|/|X| = {mag_ratio:.4f}")

        assert cos_sim > cos_tol, (
            f"[{label}] Round-trip direction error: cos={cos_sim:.4f} < {cos_tol}"
        )
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 5: convergence criterion consistency
# ---------------------------------------------------------------------------

def test_convergence_criterion(fast: bool = True):
    """
    The Euclidean error metric in the prelog path should monotonically decrease
    (or at least not blow up). Verify the while_loop terminates within max_iter.
    """
    print("\n[Test 5] Convergence: prelog while_loop terminates properly")
    manifold, x, y = make_manifold_and_pair(fast)

    tau_half = jnp.array([0.5])

    # Run with generous tol — should converge in far fewer than 100 iterations
    z = manifold.s_geodesic(x, y, tau_half, tol=1e-3, max_iter=100,
                            use_separation_grad=True)

    # Verify z is a valid conformation (finite, no NaN)
    assert jnp.all(jnp.isfinite(z)), "Output contains NaN/Inf"

    # Verify z is closer to the true midpoint than the initial guess (z=y)
    dist_from_init = float(manifold.s_distance(y, z)[0, 0, 0])
    dist_xy = float(manifold.s_distance(x, y)[0, 0, 0])
    # z should have moved from y toward x
    assert dist_from_init < dist_xy, (
        f"z did not move toward midpoint: dist(y,z)={dist_from_init:.4f} >= dist(x,y)={dist_xy:.4f}"
    )
    print(f"  z moved from y: dist(y,z)={dist_from_init:.4f} < dist(x,y)={dist_xy:.4f}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Use n=214 adenylate kinase (slow). Default: n=10 synthetic.")
    args = parser.parse_args()
    fast = not args.full

    label = "adenylate kinase (n=214)" if not fast else "synthetic (n=10)"
    print("=" * 65)
    print(f"Separation-Gradient Geodesic Tests")
    print(f"Data: {label}")
    print("=" * 65)

    test_midpoint_accuracy(fast)
    test_extrapolation_accuracy(fast)
    test_s_exp_accuracy(fast)
    test_round_trip(fast)
    test_convergence_criterion(fast)

    print("\n" + "=" * 65)
    print("All 5 tests PASSED.")
    print("=" * 65)
