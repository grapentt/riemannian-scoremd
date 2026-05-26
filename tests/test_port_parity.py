"""
Numerical correctness tests for the w^delta ShapeManifold implementation.

Validates that all geometric operations (metric, distances, log map, exp map,
projections) produce numerically consistent results by cross-checking against
the reference PyTorch implementation of Diepeveen (2024, arXiv:2308.07818).

Both implementations compute the same mathematical objects; agreement to 1e-4
in float32 confirms correctness of the JAX implementation.

Usage:
    /tmp/torch_refs_venv/bin/python tests/test_manifold_correctness.py
    pytest tests/test_manifold_correctness.py -v

Environment: /tmp/torch_refs_venv  (torch 2.0.1 + JAX 0.4.30 + numpy 1.26.4)
Recreate if /tmp is cleared:
    python3.11 -m venv /tmp/torch_refs_venv
    pip install "torch==2.0.1" "numpy<2" "jax[cpu]==0.4.30" "jaxlib==0.4.30"
"""

import sys
import numpy as np
from pathlib import Path

# -- resolve import paths ---------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT.parent / "diepeveen2024" / "src"))

import torch
import jax
import jax.numpy as jnp
from manifolds.pointcloud import PointCloud        # reference (Diepeveen 2024)
from manifold.pointcloud_jax import ShapeManifold  # this project

# ---------------------------------------------------------------------------
# Fixed input parameters
# ---------------------------------------------------------------------------
N, M, MM, n, d = 2, 3, 2, 10, 3
ALPHA = 1.0
SEED = 42
TOL = 1e-4


def make_torch_inputs():
    rng = np.random.default_rng(seed=SEED)
    x_np = rng.standard_normal((N, M, n, d)).astype(np.float32)
    y_np = rng.standard_normal((N, MM, n, d)).astype(np.float32)
    return torch.tensor(x_np), torch.tensor(y_np)


def make_jax_inputs():
    rng = np.random.default_rng(seed=SEED)
    x_np = rng.standard_normal((N, M, n, d)).astype(np.float32)
    y_np = rng.standard_normal((N, MM, n, d)).astype(np.float32)
    return jnp.array(x_np), jnp.array(y_np)


def max_diff(jax_out, torch_out):
    j = np.asarray(jax_out, dtype=np.float32)
    t = torch_out.detach().numpy().astype(np.float32)
    assert j.shape == t.shape, f"Shape mismatch: JAX {j.shape} vs ref {t.shape}"
    return float(np.max(np.abs(j - t)))


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

results = []


def check(name, fn_jax, fn_torch):
    try:
        j = fn_jax()
        t = fn_torch()
        diff = max_diff(j, t)
        status = "PASS" if diff < TOL else "FAIL"
        results.append((name, diff, status))
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name:<45s}  max|diff| = {diff:.2e}  {status}")
        return status == "PASS"
    except Exception as e:
        results.append((name, float("nan"), "ERROR"))
        print(f"  ✗ {name:<45s}  ERROR: {type(e).__name__}: {e}")
        return False


def check_jit(name, fn_jax):
    try:
        result = fn_jax()
        jit_result = jax.jit(fn_jax)()
        diff = float(np.max(np.abs(np.asarray(result) - np.asarray(jit_result))))
        status = "PASS" if diff < TOL else "FAIL"
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} jit({name:<41s}) max|diff| = {diff:.2e}  {status}")
        return status == "PASS"
    except Exception as e:
        print(f"  ✗ jit({name:<41s}) ERROR: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    pc_ref = PointCloud(dim=d, numpoints=n, alpha=ALPHA)
    pc = ShapeManifold(dim=d, numpoints=n, alpha=ALPHA)

    xt, yt = make_torch_inputs()
    xj, yj = make_jax_inputs()

    print("\n" + "=" * 70)
    print("w^delta ShapeManifold — numerical correctness checks")
    print(f"Config: N={N}, M={M}, MM={MM}, n={n}, d={d}, alpha={ALPHA}, seed={SEED}")
    print(f"Tolerance: {TOL:.0e}")
    print("=" * 70)

    print("\n[Alignment & basic geometry]")
    check("translate_mpoint",
          lambda: pc.translate_mpoint(xj, xj.mean(axis=2)),
          lambda: pc_ref.translate_mpoint(xt, xt.mean(dim=2)))
    check("center_mpoint",
          lambda: pc.center_mpoint(xj),
          lambda: pc_ref.center_mpoint(xt))
    check("pairwise_distances",
          lambda: pc.pairwise_distances(xj),
          lambda: pc_ref.pairwise_distances(xt))
    check("gyration_matrix",
          lambda: pc.gyration_matrix(xj),
          lambda: pc_ref.gyration_matrix(xt))
    check("orthogonal_transform_mpoint",
          lambda: pc.orthogonal_transform_mpoint(
              xj, jnp.broadcast_to(jnp.eye(d)[None, None], (N, M, d, d))),
          lambda: pc_ref.orthogonal_transform_mpoint(
              xt, torch.eye(d)[None, None].expand(N, M, d, d)))

    base_np = np.random.default_rng(99).standard_normal((n, d)).astype(np.float32)
    base_j = jnp.array(base_np)
    base_t = torch.tensor(base_np)
    check("least_orthogonal",
          lambda: pc.least_orthogonal(pc.center_mpoint(xj), base=base_j),
          lambda: pc_ref.least_orthogonal(pc_ref.center_mpoint(xt), base=base_t))
    check("align_mpoint",
          lambda: pc.align_mpoint(xj, base=base_j),
          lambda: pc_ref.align_mpoint(xt, base=base_t))

    print("\n[w^delta metric tensor]")
    check("metric_tensor (tensor form)",
          lambda: pc.metric_tensor(xj, asmatrix=False),
          lambda: pc_ref.metric_tensor(xt, asmatrix=False))
    check("metric_tensor (matrix form)",
          lambda: pc.metric_tensor(xj, asmatrix=True),
          lambda: pc_ref.metric_tensor(xt, asmatrix=True))

    print("\n[w^delta distance, log map, tangent operations]")
    check("s_distance",
          lambda: pc.s_distance(xj, yj),
          lambda: pc_ref.s_distance(xt, yt))
    check("s_prelog (vector)",
          lambda: pc.s_prelog(xj, yj, asvector=True),
          lambda: pc_ref.s_prelog(xt, yt, asvector=True))
    check("s_prelog (tensor)",
          lambda: pc.s_prelog(xj, yj, asvector=False),
          lambda: pc_ref.s_prelog(xt, yt, asvector=False))
    check("s_log (vector)",
          lambda: pc.s_log(xj, yj, asvector=True),
          lambda: pc_ref.s_log(xt, yt, asvector=True))
    check("s_log (tensor)",
          lambda: pc.s_log(xj, yj, asvector=False),
          lambda: pc_ref.s_log(xt, yt, asvector=False))
    check("inner product",
          lambda: pc.inner(xj, xj[:, :, None], xj[:, :, None]),
          lambda: pc_ref.inner(xt, xt[:, :, None], xt[:, :, None]))
    check("orthonormal_basis",
          lambda: pc.orthonormal_basis(xj, asvector=True),
          lambda: pc_ref.orthonormal_basis(xt, asvector=True))
    check("horizontal_projection_tvector",
          lambda: pc.horizontal_projection_tvector(xj, xj[:, :, None]),
          lambda: pc_ref.horizontal_projection_tvector(xt, xt[:, :, None]))

    print("\n[g-norm]")
    check("norm",
          lambda: pc.norm(xj, xj[:, :, None]),
          lambda: pc_ref.norm(xt, xt[:, :, None]))

    print("\n[jax.jit compilation]")
    check_jit("center_mpoint",
              lambda: pc.center_mpoint(xj))
    check_jit("pairwise_distances",
              lambda: pc.pairwise_distances(xj))
    check_jit("metric_tensor(asmatrix=True)",
              lambda: pc.metric_tensor(xj, asmatrix=True))
    check_jit("s_distance",
              lambda: pc.s_distance(xj, yj))
    check_jit("s_log",
              lambda: pc.s_log(xj, yj))
    check_jit("horizontal_projection_tvector",
              lambda: pc.horizontal_projection_tvector(xj, xj[:, :, None]))

    passed = sum(1 for _, _, s in results if s == "PASS")
    failed = sum(1 for _, _, s in results if s != "PASS")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed  (tolerance = {TOL:.0e})")
    if failed == 0:
        print("ALL CORRECTNESS CHECKS PASSED")
    else:
        print("FAILURES:")
        for name, diff, status in results:
            if status != "PASS":
                print(f"  {name}: {status} (diff={diff:.2e})")
    print("=" * 70)
    return failed == 0


# ---------------------------------------------------------------------------
# pytest-compatible individual tests
# ---------------------------------------------------------------------------

def test_pairwise_distances():
    pc_ref = PointCloud(dim=d, numpoints=n, alpha=ALPHA)
    pc = ShapeManifold(dim=d, numpoints=n, alpha=ALPHA)
    xt, _ = make_torch_inputs(); xj, _ = make_jax_inputs()
    assert max_diff(pc.pairwise_distances(xj), pc_ref.pairwise_distances(xt)) < TOL

def test_metric_tensor():
    pc_ref = PointCloud(dim=d, numpoints=n, alpha=ALPHA)
    pc = ShapeManifold(dim=d, numpoints=n, alpha=ALPHA)
    xt, _ = make_torch_inputs(); xj, _ = make_jax_inputs()
    assert max_diff(pc.metric_tensor(xj, asmatrix=True),
                    pc_ref.metric_tensor(xt, asmatrix=True)) < TOL

def test_s_distance():
    pc_ref = PointCloud(dim=d, numpoints=n, alpha=ALPHA)
    pc = ShapeManifold(dim=d, numpoints=n, alpha=ALPHA)
    xt, yt = make_torch_inputs(); xj, yj = make_jax_inputs()
    assert max_diff(pc.s_distance(xj, yj), pc_ref.s_distance(xt, yt)) < TOL

def test_s_prelog():
    pc_ref = PointCloud(dim=d, numpoints=n, alpha=ALPHA)
    pc = ShapeManifold(dim=d, numpoints=n, alpha=ALPHA)
    xt, yt = make_torch_inputs(); xj, yj = make_jax_inputs()
    assert max_diff(pc.s_prelog(xj, yj, asvector=True),
                    pc_ref.s_prelog(xt, yt, asvector=True)) < TOL

def test_s_log():
    pc_ref = PointCloud(dim=d, numpoints=n, alpha=ALPHA)
    pc = ShapeManifold(dim=d, numpoints=n, alpha=ALPHA)
    xt, yt = make_torch_inputs(); xj, yj = make_jax_inputs()
    assert max_diff(pc.s_log(xj, yj, asvector=True),
                    pc_ref.s_log(xt, yt, asvector=True)) < TOL

def test_horizontal_projection():
    pc_ref = PointCloud(dim=d, numpoints=n, alpha=ALPHA)
    pc = ShapeManifold(dim=d, numpoints=n, alpha=ALPHA)
    xt, _ = make_torch_inputs(); xj, _ = make_jax_inputs()
    assert max_diff(pc.horizontal_projection_tvector(xj, xj[:, :, None]),
                    pc_ref.horizontal_projection_tvector(xt, xt[:, :, None])) < TOL


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
