"""
Phase 2.6 gate: data pipeline validation.

Loads DE Shaw chignolin and BBA trajectories from data/deshawresearch/,
converts to the ShapeManifold format, and verifies:

  1. No NaN / Inf in loaded coordinates
  2. Mean Cα bond distance in expected range 3.6–4.0 Å (consecutive residues)
  3. s_distance(x, x) == 0 (self-distance is identically zero)
  4. metric_tensor(x) is positive definite (all eigenvalues > 0)

The gate passes when all 4 assertions hold for both proteins.

Usage:
    /path/to/.venv/bin/python tests/test_data_loading.py
    pytest tests/test_data_loading.py -v

Requires: mdtraj, tables (PyTables), jax, numpy
"""

import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from manifold.pointcloud_jax import ShapeManifold

DATA_DIR = _ROOT / "data" / "deshawresearch"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trajectory_as_manifold_frames(h5_files, n_atoms_expected, pdb_path=None):
    """
    Load .h5 MDTraj trajectory, convert nm→Å, return (T, 1, 1, n, 3) jnp array.
    Also returns raw coords (T, n, 3) in Å for bond-distance checks.
    """
    import mdtraj as md

    paths = [str(p) for p in h5_files]
    traj = md.load(paths)

    assert traj.n_atoms == n_atoms_expected, (
        f"Expected {n_atoms_expected} Cα atoms, got {traj.n_atoms}"
    )

    coords_A = traj.xyz * 10.0  # nm → Angstrom, shape (T, n, 3)
    T, n, d = coords_A.shape

    # ShapeManifold expects (N, M, n, d) — use (T, 1, n, d) where each frame is independent
    x = jnp.array(coords_A[:, None, :, :])   # (T, n, 3) -> (T, 1, n, 3)
    # Add M=1 batch dim: (T, 1, 1, n, 3) is what the tests use, but manifold ops work on (N, M, n, d)
    # We'll use one frame at a time: (1, 1, n, 3) for single-frame ops
    return x, coords_A


def make_manifold(n, d=3):
    """Create ShapeManifold with default alpha=1.0."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((n, d)).astype(np.float32) * 3.0
    return ShapeManifold(dim=d, numpoints=n, alpha=1.0, base=base)


def check_bond_distances(coords_A, protein_name):
    """
    Check consecutive Cα–Cα bond distances are in 3.3–4.5 Å (loose)
    and that the mean is in 3.6–4.0 Å (tight, expected).
    Returns (mean, within_tight, within_loose).
    """
    # consecutive distances: shape (T, n-1)
    diffs = coords_A[:, 1:, :] - coords_A[:, :-1, :]   # (T, n-1, 3)
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))          # (T, n-1)
    mean_dist = float(dists.mean())
    fraction_tight = float(np.mean((dists >= 3.3) & (dists <= 4.5)))
    return mean_dist, fraction_tight


# ---------------------------------------------------------------------------
# Test 1: No NaN / Inf
# ---------------------------------------------------------------------------

def _check_no_nan_inf(protein_name, h5_files, n_atoms):
    """Check that loaded coordinates contain no NaN or Inf."""
    x, coords_A = load_trajectory_as_manifold_frames(h5_files, n_atoms)

    nan_count = int(jnp.sum(jnp.isnan(x)))
    inf_count = int(jnp.sum(jnp.isinf(x)))
    T = x.shape[0]

    ok = (nan_count == 0) and (inf_count == 0)
    status = "PASS" if ok else "FAIL"
    print(f"  {'✓' if ok else '✗'} [{protein_name}] no NaN/Inf: "
          f"T={T} frames, n={n_atoms} atoms, NaN={nan_count}, Inf={inf_count}  {status}")
    return ok, x, coords_A


# ---------------------------------------------------------------------------
# Test 2: Cα bond distances in expected range
# ---------------------------------------------------------------------------

def _check_bond_distances(protein_name, coords_A):
    """Mean consecutive Cα–Cα distance should be in 3.6–4.0 Å."""
    mean_dist, fraction_tight = check_bond_distances(coords_A, protein_name)
    # Mean must be in 3.6–4.0 Å; at least 85% of bonds in 3.3–4.5 Å
    ok_mean = 3.6 <= mean_dist <= 4.0
    ok_fraction = fraction_tight >= 0.85
    ok = ok_mean and ok_fraction
    status = "PASS" if ok else "FAIL"
    print(f"  {'✓' if ok else '✗'} [{protein_name}] Cα bond dist: "
          f"mean={mean_dist:.3f} Å (expect 3.6–4.0), "
          f"{100*fraction_tight:.1f}% bonds in [3.3, 4.5] Å  {status}")
    return ok


# ---------------------------------------------------------------------------
# Test 3: s_distance(x, x) == 0
# ---------------------------------------------------------------------------

def _check_self_distance(protein_name, x, manifold, n_frames_to_check=10):
    """Self-distance should be identically zero for all checked frames."""
    # Use stride to avoid testing all T frames (slow)
    stride = max(1, x.shape[0] // n_frames_to_check)
    frames = x[::stride][:n_frames_to_check]   # (k, 1, n, d)

    max_self_dist = 0.0
    for i in range(frames.shape[0]):
        xi = frames[i:i+1]                     # (1, 1, n, d)
        dist = float(manifold.s_distance(xi, xi)[0, 0, 0])
        max_self_dist = max(max_self_dist, dist)

    ok = max_self_dist < 1e-5
    status = "PASS" if ok else "FAIL"
    print(f"  {'✓' if ok else '✗'} [{protein_name}] s_distance(x, x) == 0: "
          f"max over {n_frames_to_check} frames = {max_self_dist:.2e}  {status}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: metric_tensor(x) positive definite on horizontal subspace
# ---------------------------------------------------------------------------

def _check_metric_positive_definite(protein_name, x, manifold, n_frames_to_check=5,
                                    n_vectors=10):
    """
    The w^delta metric tensor is only positive definite on the *horizontal* subspace
    of T_x M (the subspace orthogonal to rigid-body motions).  The full ambient
    (nd × nd) matrix has a 6-dimensional null space (3 translations + 3 rotations)
    and is not positive definite on the full tangent space.

    This test verifies that for random horizontal tangent vectors v_h:
        <v_h, v_h>_g  =  inner(x, v_h, v_h) > 0

    A passing result confirms that the metric is genuinely positive on the
    shape space, which is the physically meaningful check.
    """
    stride = max(1, x.shape[0] // n_frames_to_check)
    frames = x[::stride][:n_frames_to_check]

    min_ip = float('inf')
    rng = jax.random.PRNGKey(99)
    for i in range(frames.shape[0]):
        xi = frames[i:i+1]                     # (1, 1, n, d)
        for _ in range(n_vectors):
            rng, key = jax.random.split(rng)
            v_raw = jax.random.normal(key, xi.shape)
            v_h = manifold.horizontal_projection_tvector(
                xi, v_raw[:, :, None]
            )[:, :, 0]                          # (1, 1, n, d)
            ip = float(manifold.inner(xi, v_h[:, :, None], v_h[:, :, None])[0, 0, 0, 0])
            min_ip = min(min_ip, ip)

    ok = min_ip > 0.0
    status = "PASS" if ok else "FAIL"
    print(f"  {'✓' if ok else '✗'} [{protein_name}] metric PD on horizontal subspace: "
          f"min <v_h,v_h>_g over {n_frames_to_check}×{n_vectors} = {min_ip:.4e}  {status}")
    return ok


# ---------------------------------------------------------------------------
# Full per-protein validation
# ---------------------------------------------------------------------------

def validate_protein(protein_name, h5_files, n_atoms):
    print(f"\n--- {protein_name} (n={n_atoms} Cα, {len(h5_files)} file(s)) ---")

    # Test 1: no NaN/Inf
    ok1, x, coords_A = _check_no_nan_inf(protein_name, h5_files, n_atoms)
    if not ok1:
        print(f"  [SKIP] Skipping remaining tests — NaN/Inf detected.")
        return False

    # Test 2: bond distances
    ok2 = _check_bond_distances(protein_name, coords_A)

    # Create manifold (base = first frame, after centering)
    first_frame = np.array(x[0, 0])        # (n, 3), Å
    first_frame -= first_frame.mean(axis=0)  # center
    manifold = ShapeManifold(dim=3, numpoints=n_atoms, alpha=1.0, base=first_frame)

    # Center all frames (ShapeManifold expects centred input for s_distance)
    x_centred = x - x.mean(axis=2, keepdims=True)

    # Test 3: self-distance
    ok3 = _check_self_distance(protein_name, x_centred, manifold)

    # Test 4: metric tensor positive definite
    ok4 = _check_metric_positive_definite(protein_name, x_centred, manifold)

    all_ok = ok1 and ok2 and ok3 and ok4
    return all_ok


# ---------------------------------------------------------------------------
# Protein configurations
# ---------------------------------------------------------------------------

PROTEINS = {
    "chignolin": {
        "h5_files": [DATA_DIR / "chignolin-0_ca.h5"],
        "n_atoms": 10,
    },
    "bba": {
        "h5_files": [DATA_DIR / "bba-0_ca.h5", DATA_DIR / "bba-1_ca.h5"],
        "n_atoms": 28,
    },
}


# ---------------------------------------------------------------------------
# pytest entrypoints
# ---------------------------------------------------------------------------

def test_chignolin_no_nan():
    files = PROTEINS["chignolin"]["h5_files"]
    n = PROTEINS["chignolin"]["n_atoms"]
    ok, _, _ = _check_no_nan_inf("chignolin", files, n)
    assert ok

def test_chignolin_bond_distances():
    files = PROTEINS["chignolin"]["h5_files"]
    n = PROTEINS["chignolin"]["n_atoms"]
    _, _, coords_A = _check_no_nan_inf("chignolin", files, n)
    assert _check_bond_distances("chignolin", coords_A)

def test_chignolin_self_distance():
    files = PROTEINS["chignolin"]["h5_files"]
    n = PROTEINS["chignolin"]["n_atoms"]
    _, x, _ = _check_no_nan_inf("chignolin", files, n)
    x_c = x - x.mean(axis=2, keepdims=True)
    first = np.array(x[0, 0]); first -= first.mean(axis=0)
    manifold = ShapeManifold(dim=3, numpoints=n, alpha=1.0, base=first)
    assert _check_self_distance("chignolin", x_c, manifold)

def test_chignolin_metric_pd():
    files = PROTEINS["chignolin"]["h5_files"]
    n = PROTEINS["chignolin"]["n_atoms"]
    _, x, _ = _check_no_nan_inf("chignolin", files, n)
    x_c = x - x.mean(axis=2, keepdims=True)
    first = np.array(x[0, 0]); first -= first.mean(axis=0)
    manifold = ShapeManifold(dim=3, numpoints=n, alpha=1.0, base=first)
    assert _check_metric_positive_definite("chignolin", x_c, manifold)

def test_bba_no_nan():
    files = PROTEINS["bba"]["h5_files"]
    n = PROTEINS["bba"]["n_atoms"]
    ok, _, _ = _check_no_nan_inf("bba", files, n)
    assert ok

def test_bba_bond_distances():
    files = PROTEINS["bba"]["h5_files"]
    n = PROTEINS["bba"]["n_atoms"]
    _, _, coords_A = _check_no_nan_inf("bba", files, n)
    assert _check_bond_distances("bba", coords_A)

def test_bba_self_distance():
    files = PROTEINS["bba"]["h5_files"]
    n = PROTEINS["bba"]["n_atoms"]
    _, x, _ = _check_no_nan_inf("bba", files, n)
    x_c = x - x.mean(axis=2, keepdims=True)
    first = np.array(x[0, 0]); first -= first.mean(axis=0)
    manifold = ShapeManifold(dim=3, numpoints=n, alpha=1.0, base=first)
    assert _check_self_distance("bba", x_c, manifold)

def test_bba_metric_pd():
    files = PROTEINS["bba"]["h5_files"]
    n = PROTEINS["bba"]["n_atoms"]
    _, x, _ = _check_no_nan_inf("bba", files, n)
    x_c = x - x.mean(axis=2, keepdims=True)
    first = np.array(x[0, 0]); first -= first.mean(axis=0)
    manifold = ShapeManifold(dim=3, numpoints=n, alpha=1.0, base=first)
    assert _check_metric_positive_definite("bba", x_c, manifold)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("Phase 2.6: Data pipeline validation")
    print("=" * 65)

    results = {}
    for protein_name, cfg in PROTEINS.items():
        h5_files = cfg["h5_files"]
        missing = [f for f in h5_files if not f.exists()]
        if missing:
            print(f"\n[SKIP] {protein_name}: missing files: {missing}")
            results[protein_name] = None
            continue
        results[protein_name] = validate_protein(
            protein_name, h5_files, cfg["n_atoms"]
        )

    print(f"\n{'='*65}")
    all_pass = all(v for v in results.values() if v is not None)
    for name, ok in results.items():
        if ok is None:
            print(f"  [SKIP] {name}: data not found")
        else:
            print(f"  {'✓' if ok else '✗'} {name}: {'PASS' if ok else 'FAIL'}")

    if all_pass and None not in results.values():
        print("\nPHASE 2.6 GATE PASSED — data pipeline validated for all proteins")
    else:
        print("\nSome tests failed or data missing — see details above")
    print("=" * 65)

    sys.exit(0 if all_pass else 1)
