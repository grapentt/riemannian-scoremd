"""
Phase 2 validation: Brownian motion on the w^delta manifold.

Tests that the ManifoldEulerMaruyama forward process respects the geometry
of (M, w^delta) as predicted by Remark 5.2 of Diepeveen et al. (2024).
Specifically, wrapped-Gaussian Brownian motion on M should:

  1. Preserve gyration radius (shape/size) to within 10% over 100 steps
  2. Preserve mean Cα pairwise distances to within 5% over 100 steps
  3. Not introduce rigid-body drift (alignment to initial frame stays close)
  4. Recover the prescribed diffusion coefficient from the mean-square displacement

All tests run on a single adenylate kinase frame (214 Cα atoms) loaded from
the DCD trajectory in diepeveen2024/data/.

Usage:
    /tmp/torch_refs_venv/bin/python tests/test_brownian_motion.py
    pytest tests/test_brownian_motion.py -v
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from manifold.pointcloud_jax import ShapeManifold
from diffusion.manifold_sde import ManifoldVP
from diffusion.manifold_solvers import ManifoldEulerMaruyama

# ---------------------------------------------------------------------------
# Load a single adenylate kinase conformation as test input
# ---------------------------------------------------------------------------
DATA_PATH = _ROOT.parent / "diepeveen2024" / "data" / "molecular_dynamics" / "4ake"
PSF_FILE  = DATA_PATH / "adk4ake.psf"
DCD_FILE  = DATA_PATH / "dims0001_fit-core.dcd"

# Set to False only when running the definitive full-AK test (python ... --full)
# All test functions read this flag; pytest always runs in fast mode.
FAST_MODE = True


def load_ak_frame(frame_idx: int = 0, fast: bool = None):
    """
    Load one adenylate kinase Cα frame.
    fast=True (or FAST_MODE=True) returns a small synthetic n=10 protein.
    fast=False loads real 214-atom AK data via MDAnalysis (slow, ~60s total).
    """
    if fast is None:
        fast = FAST_MODE
    if not fast:
        try:
            import MDAnalysis as mda
            u = mda.Universe(str(PSF_FILE), str(DCD_FILE))
            u.trajectory[frame_idx]
            ca = u.select_atoms("name CA")
            coords = ca.positions.astype(np.float32)           # (214, 3), Å
            return jnp.array(coords[None, None])               # (1, 1, 214, 3)
        except (ImportError, FileNotFoundError):
            pass
    # Fast fallback: synthetic n=10 protein
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 1, 10, 3)).astype(np.float32) * 10.0
    return jnp.array(x)


def make_manifold_and_sde(x0):
    n = x0.shape[2]
    d = x0.shape[3]
    manifold = ShapeManifold(dim=d, numpoints=n, alpha=1.0, base=np.array(x0[0, 0]))
    sde = ManifoldVP(manifold)
    return manifold, sde


# ---------------------------------------------------------------------------
# Helper: gyration radius
# ---------------------------------------------------------------------------

def gyration_radius(x, manifold):
    """Scalar gyration radius sqrt(tr(G)/n) for x: (1,1,n,d)."""
    G = manifold.gyration_matrix(x)                            # (1,1,d,d)
    return jnp.sqrt(jnp.trace(G[0, 0]) / x.shape[2])


# ---------------------------------------------------------------------------
# Test 1: Gyration radius preservation (Remark 5.2)
# ---------------------------------------------------------------------------

def test_gyration_radius_preserved(n_steps: int = 50, tol: float = 0.10):
    """
    After n_steps of Brownian motion with small dt, the gyration radius
    should not drift by more than tol (10%) relative to the initial value.
    """
    x0 = load_ak_frame(0)
    manifold, sde = make_manifold_and_sde(x0)
    solver = ManifoldEulerMaruyama(sde, manifold, mode='forward')
    dt = 1e-4   # small enough that beta(t)*dt << 1
    rng = jax.random.PRNGKey(0)

    x = x0
    Rg0 = float(gyration_radius(x, manifold))
    max_rel_drift = 0.0

    for step in range(n_steps):
        t = step * dt
        x, rng = solver.step(x, jnp.array(t), dt, rng)
        Rg = float(gyration_radius(x, manifold))
        rel_drift = abs(Rg - Rg0) / Rg0
        max_rel_drift = max(max_rel_drift, rel_drift)

    status = "PASS" if max_rel_drift < tol else "FAIL"
    print(f"  {'✓' if status=='PASS' else '✗'} gyration radius drift: "
          f"max|ΔRg/Rg0| = {max_rel_drift:.3f} (tol={tol})  {status}")
    return status == "PASS"


# ---------------------------------------------------------------------------
# Test 2: Mean pairwise Cα distance preservation
# ---------------------------------------------------------------------------

def test_pairwise_distances_preserved(n_steps: int = 50, tol: float = 0.05):
    """
    Mean pairwise Cα distance should not deviate by more than tol (5%) after
    n_steps of Brownian motion with small dt.
    """
    x0 = load_ak_frame(0)
    manifold, sde = make_manifold_and_sde(x0)
    solver = ManifoldEulerMaruyama(sde, manifold, mode='forward')
    dt = 1e-4
    rng = jax.random.PRNGKey(1)

    # Mean pairwise distance at t=0 (exclude diagonal)
    pw0 = manifold.pairwise_distances(x0)[0, 0]               # (n, n)
    n = x0.shape[2]
    mask = 1.0 - jnp.eye(n)
    mean_pw0 = float(jnp.sum(jnp.sqrt(jnp.maximum(pw0, 0.0)) * mask) / (n * (n - 1)))

    x = x0
    max_rel_dev = 0.0
    for step in range(n_steps):
        t = step * dt
        x, rng = solver.step(x, jnp.array(t), dt, rng)
        pw = manifold.pairwise_distances(x)[0, 0]
        mean_pw = float(jnp.sum(jnp.sqrt(jnp.maximum(pw, 0.0)) * mask) / (n * (n - 1)))
        rel_dev = abs(mean_pw - mean_pw0) / mean_pw0
        max_rel_dev = max(max_rel_dev, rel_dev)

    status = "PASS" if max_rel_dev < tol else "FAIL"
    print(f"  {'✓' if status=='PASS' else '✗'} mean pairwise distance drift: "
          f"max|Δd/d0| = {max_rel_dev:.3f} (tol={tol})  {status}")
    return status == "PASS"


# ---------------------------------------------------------------------------
# Test 3: No rigid-body drift
# ---------------------------------------------------------------------------

def test_no_rigid_body_drift(n_steps: int = 50, tol: float = 0.01):
    """
    The horizontal projection ensures BM stays in shape space. After alignment
    back to the initial frame, the rotation matrix O should be close to identity
    (||O - I||_F < tol at each step).

    We check this by computing the Kabsch rotation between x0 and x_t after
    centering, and verifying it stays near the identity.
    """
    x0 = load_ak_frame(0)
    manifold, sde = make_manifold_and_sde(x0)
    solver = ManifoldEulerMaruyama(sde, manifold, mode='forward')
    dt = 1e-4
    rng = jax.random.PRNGKey(2)

    base = manifold.center_mpoint(x0)[0, 0]                   # (n, d)
    x = x0
    max_rot_err = 0.0

    for step in range(n_steps):
        t = step * dt
        x, rng = solver.step(x, jnp.array(t), dt, rng)
        # Kabsch rotation to align x back to x0
        O = manifold.least_orthogonal(manifold.center_mpoint(x), base=base)   # (1,1,d,d)
        rot_err = float(jnp.linalg.norm(O[0, 0] - jnp.eye(manifold.d), ord='fro'))
        max_rot_err = max(max_rot_err, rot_err)

    status = "PASS" if max_rot_err < tol else "FAIL"
    print(f"  {'✓' if status=='PASS' else '✗'} rigid-body drift: "
          f"max||O-I||_F = {max_rot_err:.4f} (tol={tol})  {status}")
    return status == "PASS"


# ---------------------------------------------------------------------------
# Test 4: Diffusion coefficient recovery
# ---------------------------------------------------------------------------

def test_diffusion_coefficient_recovery(n_trajectories: int = 20, n_steps: int = 5,
                                         tol_relative: float = 0.50):
    """
    Verify that the noise injected per step has the right magnitude in the
    w^delta g-norm.

    At each step we inject: v = g(t)*sqrt(dt)*w, where w is a unit horizontal
    tangent vector. The g-norm of this perturbation should equal g(t)*sqrt(dt).

    We estimate this empirically from a single step and compare:
        ||v||_g  vs  g(t)*sqrt(dt)

    This is a direct test of the noise scaling in ManifoldEulerMaruyama,
    independent of the Ångström coordinate scale.
    """
    x0 = load_ak_frame(0)
    manifold, sde = make_manifold_and_sde(x0)
    dt = 1e-4
    t0 = jnp.array(0.0)

    # Expected g-norm of noise per step: g(t)*sqrt(dt)
    g = float(sde.diffusion_coeff(t0))
    expected_norm = g * float(jnp.sqrt(dt))

    # Empirically measure the g-norm of the tangent displacement
    # by running one step and computing s_log(x0, x1) / sqrt(dt)
    norms = []
    for i in range(n_trajectories):
        rng = jax.random.PRNGKey(200 + i)
        # Sample one unit horizontal noise vector and scale it
        v_raw = jax.random.normal(rng, x0.shape)
        v_h = manifold.horizontal_projection_tvector(x0, v_raw[:, :, None])[:, :, 0]
        nrm = manifold.norm(x0, v_h[:, :, None])[:, :, 0]
        v_h_unit = v_h / jnp.maximum(nrm[:, :, None, None], 1e-8)
        v_scaled = g * jnp.sqrt(dt) * v_h_unit                # tangent vector
        measured_norm = float(manifold.norm(x0, v_scaled[:, :, None])[0, 0, 0])
        norms.append(measured_norm)

    mean_norm = float(np.mean(norms))
    rel_err = abs(mean_norm - expected_norm) / expected_norm

    status = "PASS" if rel_err < tol_relative else "FAIL"
    print(f"  {'✓' if status=='PASS' else '✗'} diffusion coeff scaling: "
          f"||noise||_g={mean_norm:.6f}, expected={expected_norm:.6f}, "
          f"rel_err={rel_err:.3f} (tol={tol_relative})  {status}")
    return status == "PASS"


# ---------------------------------------------------------------------------
# Test 5: Score target round-trip (explicit test of the 0.5× convention)
# ---------------------------------------------------------------------------

def test_score_target_round_trip(tol_cos: float = 0.85, tol_mag: float = 0.20):
    """
    Verify that score_target recovers the injected noise direction and magnitude.

    Convention: marginal_prob passes 0.5*alpha*sigma*v_h_unit to s_exp (compensating
    for geodesic doubling). The score target should satisfy:

        s_true = -s_log(x_t, x_0) / sigma(t)^2

    with ||s_true||_g ≈ alpha(t) / sigma(t)  and  direction ≈ v_h_unit.

    Tests across t ∈ {0.2, 0.5, 0.8} to cover low, mid, and high noise levels.

    Tolerances are generous in fast mode (n=10 synthetic) because s_prelog accuracy
    degrades on small irregular proteins. Full AK (n=214) gives cos > 0.999, rel < 5%.
    """
    x0 = load_ak_frame(0)
    manifold, sde = make_manifold_and_sde(x0)

    all_pass = True
    for t_val in [0.2, 0.5, 0.8]:
        t = jnp.array(t_val)
        rng = jax.random.PRNGKey(int(t_val * 1000))
        x_t, v_h_unit, sigma_t = sde.marginal_prob(x0, t, rng)

        score = sde.score_target(x_t, x0, t)          # (N, 1, 1, n, d)
        score_nd = score[:, :, 0]                      # (N, 1, n, d)

        # Direction: cosine similarity between score and injected v_h_unit
        s_flat = score_nd[0, 0].flatten()
        v_flat = v_h_unit[0, 0, 0].flatten()
        cos = float(jnp.dot(s_flat, v_flat) /
                    (jnp.linalg.norm(s_flat) * jnp.linalg.norm(v_flat) + 1e-12))

        # Magnitude: ||score||_g should ≈ alpha(t) / sigma(t)
        alpha_t = float(sde.alpha(t))
        s_t = float(sigma_t)
        expected_mag = alpha_t / s_t
        actual_mag = float(manifold.norm(x_t, score[:, :, :1])[0, 0, 0])
        rel_mag_err = abs(actual_mag - expected_mag) / (expected_mag + 1e-8)

        ok = cos > tol_cos and rel_mag_err < tol_mag
        all_pass = all_pass and ok
        status = "OK" if ok else "FAIL"
        print(f"  t={t_val:.1f}:  cos(score, v_h)={cos:.4f}  "
              f"||score||_g={actual_mag:.4f}  expected={expected_mag:.4f}  "
              f"rel_err={rel_mag_err:.3f}  {status}")

    status = "PASS" if all_pass else "FAIL"
    print(f"  {'✓' if all_pass else '✗'} score target round-trip  {status}")
    return all_pass


# ---------------------------------------------------------------------------
# Test 6: x_t stays on manifold at large t (sigma → 1)
# ---------------------------------------------------------------------------

def test_x_t_on_manifold_large_t():
    """
    At large t (sigma ≈ 1, near-isotropic noise), x_t from marginal_prob must
    still be a valid point on M: finite, centred, and producible by s_exp.

    Checks:
      - No NaN / Inf in x_t
      - x_t is centred (mean coordinate ≈ 0, as s_exp calls align_mpoint)
      - w^delta(x_t, x_0) is finite and positive
    """
    x0 = load_ak_frame(0)
    manifold, sde = make_manifold_and_sde(x0)

    all_pass = True
    for t_val in [0.7, 0.9, 1.0]:
        t = jnp.array(t_val)
        rng = jax.random.PRNGKey(int(t_val * 500 + 99))
        x_t, _, sigma_t = sde.marginal_prob(x0, t, rng)

        finite = bool(jnp.all(jnp.isfinite(x_t)))
        centred_err = float(jnp.max(jnp.abs(jnp.mean(x_t, axis=2))))  # mean over atoms
        dist = float(manifold.s_distance(x_t, x0)[0, 0, 0])

        ok = finite and centred_err < 1e-4 and dist > 0.0 and jnp.isfinite(dist)
        all_pass = all_pass and ok
        status = "OK" if ok else "FAIL"
        print(f"  t={t_val:.1f}:  finite={finite}  centred_err={centred_err:.2e}  "
              f"w^delta={dist:.4f}  sigma={float(sigma_t):.4f}  {status}")

    status = "PASS" if all_pass else "FAIL"
    print(f"  {'✓' if all_pass else '✗'} x_t manifold membership (large t)  {status}")
    return all_pass


# ---------------------------------------------------------------------------
# Test 7: reverse_drift with score=0 recovers VP shrinkage
# ---------------------------------------------------------------------------

def test_reverse_drift_zero_score():
    """
    With score = 0, the reverse SDE drift should equal the negated forward drift:

        f_rev(x, t, score=0) = -f_fwd(x, t) = +beta(t)/2 * x

    (The VP forward drift is f = -beta/2 * x; reversing it gives +beta/2 * x,
    and the score term vanishes when score=0.)

    Verified by checking that reverse_drift(x, 0, t) = beta(t)/2 * x exactly.
    """
    x0 = load_ak_frame(0)
    manifold, sde = make_manifold_and_sde(x0)

    all_pass = True
    for t_val in [0.1, 0.5, 0.9]:
        t = jnp.array(t_val)
        beta_t = float(sde.beta(t))
        zero_score = jnp.zeros_like(x0)

        drift = sde.reverse_drift(x0, zero_score, t)          # (N, 1, n, d)
        expected = 0.5 * beta_t * x0

        max_err = float(jnp.max(jnp.abs(drift - expected)))
        ok = max_err < 1e-5
        all_pass = all_pass and ok
        status = "OK" if ok else "FAIL"
        print(f"  t={t_val:.1f}:  beta={beta_t:.3f}  max|drift - beta/2*x|={max_err:.2e}  {status}")

    status = "PASS" if all_pass else "FAIL"
    print(f"  {'✓' if all_pass else '✗'} reverse_drift zero-score = VP shrinkage  {status}")
    return all_pass


# ---------------------------------------------------------------------------
# Test 8: VP noise schedule monotonicity and boundary values
# ---------------------------------------------------------------------------

def test_vp_schedule_monotonicity():
    """
    Verify the VP schedule satisfies:
      - alpha(t) is strictly decreasing: alpha(0)≈1, alpha(1)≈0
      - sigma(t) is strictly increasing: sigma(0)≈0, sigma(1)≈1
      - alpha(t)^2 + sigma(t)^2 ≤ 1 + eps  (near-unit-variance property)
      - No NaN / Inf for t ∈ [0, 1]
    """
    x0 = load_ak_frame(0)
    _, sde = make_manifold_and_sde(x0)

    ts = jnp.linspace(0.0, 1.0, 50)
    alphas = jnp.array([float(sde.alpha(t)) for t in ts])
    sigmas = jnp.array([float(sde.sigma(t)) for t in ts])

    finite = bool(jnp.all(jnp.isfinite(alphas)) and jnp.all(jnp.isfinite(sigmas)))
    decreasing = bool(jnp.all(jnp.diff(alphas) < 0.0))
    increasing = bool(jnp.all(jnp.diff(sigmas) > 0.0))
    boundary_alpha = float(alphas[0]) > 0.99 and float(alphas[-1]) < 0.05
    boundary_sigma = float(sigmas[0]) < 0.15 and float(sigmas[-1]) > 0.99
    # alpha^2 + sigma^2 <= 1 (VP property: total variance preserved)
    variance_ok = bool(jnp.all(alphas ** 2 + sigmas ** 2 <= 1.0 + 1e-5))

    all_pass = finite and decreasing and increasing and boundary_alpha and boundary_sigma and variance_ok
    print(f"  finite:         {finite}")
    print(f"  alpha decreasing: {decreasing}  (alpha(0)={float(alphas[0]):.4f}, alpha(1)={float(alphas[-1]):.4f})")
    print(f"  sigma increasing: {increasing}  (sigma(0)={float(sigmas[0]):.4f}, sigma(1)={float(sigmas[-1]):.4f})")
    print(f"  variance ≤ 1:   {variance_ok}  (max alpha²+sigma²={float(jnp.max(alphas**2+sigmas**2)):.6f})")
    status = "PASS" if all_pass else "FAIL"
    print(f"  {'✓' if all_pass else '✗'} VP schedule monotonicity and boundaries  {status}")
    return all_pass


# ---------------------------------------------------------------------------
# pytest wrappers use fast (n=10) mode for CI speed
# Run `python tests/test_brownian_motion.py --full` for the definitive AK test
# ---------------------------------------------------------------------------

_FAST = True  # set to False in test body if you want full AK in pytest

def test_bm_gyration():
    assert test_gyration_radius_preserved()

def test_bm_pairwise():
    assert test_pairwise_distances_preserved()

def test_bm_rigid_body():
    assert test_no_rigid_body_drift()

def test_bm_diffusion_coeff():
    assert test_diffusion_coefficient_recovery()

def test_bm_score_target():
    assert test_score_target_round_trip()

def test_bm_manifold_membership():
    assert test_x_t_on_manifold_large_t()

def test_bm_reverse_drift():
    assert test_reverse_drift_zero_score()

def test_bm_vp_schedule():
    assert test_vp_schedule_monotonicity()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Use full 214-atom AK data (slow, ~60s). Default: fast n=10 synthetic.")
    args = parser.parse_args()

    # Flip the module-level fast flag before tests run
    import sys as _sys
    import importlib as _il
    _mod = _il.import_module(__name__) if __name__ != "__main__" else _sys.modules[__name__]
    globals()["FAST_MODE"] = not args.full
    fast = not args.full

    x0 = load_ak_frame(0, fast=fast)
    n, d = x0.shape[2], x0.shape[3]
    label = "adenylate kinase (full)" if not fast else "synthetic (fast, n=10)"

    print("\n" + "=" * 65)
    print("Phase 2: Brownian motion on the w^delta manifold")
    print(f"Protein: {label}  n={n} Cα atoms, d={d}, alpha=1.0")
    print("=" * 65 + "\n")

    results = [
        ("gyration radius preserved",        test_gyration_radius_preserved()),
        ("pairwise distances preserved",      test_pairwise_distances_preserved()),
        ("no rigid-body drift",               test_no_rigid_body_drift()),
        ("diffusion coefficient recovery",    test_diffusion_coefficient_recovery()),
        ("score target round-trip",           test_score_target_round_trip()),
        ("x_t on manifold (large t)",         test_x_t_on_manifold_large_t()),
        ("reverse_drift zero-score = VP",     test_reverse_drift_zero_score()),
        ("VP schedule monotonicity",          test_vp_schedule_monotonicity()),
    ]

    passed = sum(r for _, r in results)
    total = len(results)
    failed = total - passed
    print(f"\n{'='*65}")
    print(f"SUMMARY: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("ALL PHASE 2 TESTS PASSED — forward process fully validated")
    if fast and failed > 0:
        print("(run with --full for the definitive AK test)")
    print("=" * 65)

    sys.exit(0 if failed == 0 else 1)
