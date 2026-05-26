"""
Benchmark and accuracy verification for s_exp optimisations.

Measures:
  1. Timing: s_geodesic prelog (separation grad) vs s_log (Riemannian grad)
  2. s_exp timing comparison: prelog vs s_log, with iteration count reporting
  3. Geodesic accuracy: w^delta endpoint error, midpoint error as a function of tolerance
  4. Score target accuracy: approx (ambient noising) vs exact s_exp

Usage:
    # Fast mode (n=10 synthetic, ~1–5 min):
    python benchmarks/benchmark_s_exp.py

    # Full mode (n=214 adenylate kinase, target: ≤50ms with prelog):
    python benchmarks/benchmark_s_exp.py --full

The fast mode gives meaningful relative comparisons; the full mode gives the
absolute numbers needed to assess whether Phase 2.5 hits the ≤50ms target.
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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_PATH = _ROOT.parent / "diepeveen2024" / "data" / "molecular_dynamics" / "4ake"
PSF_FILE  = DATA_PATH / "adk4ake.psf"
DCD_FILE  = DATA_PATH / "dims0001_fit-core.dcd"


def load_frames(fast: bool = True, n_frames: int = 2):
    """Return (x, y): two conformations from AK or synthetic, shape (1,1,n,3)."""
    if not fast:
        try:
            import mdtraj
            traj = mdtraj.load(str(DCD_FILE), top=str(PSF_FILE))
            ca_idx = traj.topology.select("name CA")
            frames = traj.xyz[:n_frames, ca_idx] * 10.0  # nm -> Angstrom
            x = jnp.array(frames[0][None, None].astype(np.float32))
            y = jnp.array(frames[1][None, None].astype(np.float32))
            return x, y
        except (ImportError, FileNotFoundError) as e:
            print(f"  [warn] mdtraj/DCD not available ({e}), falling back to n=10 synthetic")
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((1, 1, 10, 3)).astype(np.float32) * 5.0)
    y = jnp.array(rng.standard_normal((1, 1, 10, 3)).astype(np.float32) * 5.0)
    return x, y


def load_frame(fast: bool = True):
    """Return single frame (1,1,n,3)."""
    x, _ = load_frames(fast)
    return x


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def measure_time(fn, n_repeats: int = 3):
    """Return (median_s, std_s) over n_repeats calls (after 1 warmup)."""
    fn()  # warmup (JIT compile)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.std(times))


def sample_tangent(manifold, x0, sigma: float = 0.3, seed: int = 42):
    """Return a normalised horizontal tangent vector scaled by sigma."""
    rng = jax.random.PRNGKey(seed)
    v = jax.random.normal(rng, x0.shape)
    v_hp = manifold.horizontal_projection_tvector(x0, v[:, :, None])
    nrm = manifold.norm(x0, v_hp)
    nrm_safe = jnp.maximum(nrm, 1e-8)[:, :, :, None, None]
    return sigma * v_hp[:, :, 0] / nrm_safe[:, :, 0]     # (N, 1, n, d)


# ---------------------------------------------------------------------------
# Section 1: s_geodesic timing — prelog vs s_log
# ---------------------------------------------------------------------------

def benchmark_geodesic_timing(manifold, x, y, n_repeats: int = 3):
    print(f"\n--- 1. s_geodesic timing: prelog vs s_log (n={x.shape[2]}) ---")

    tau_half = jnp.array([0.5])

    def fn_prelog():
        return manifold.s_geodesic(x, y, tau_half, tol=1e-3,
                                   use_separation_grad=True).block_until_ready()

    def fn_slog():
        return manifold.s_geodesic(x, y, tau_half, tol=1e-3,
                                   use_separation_grad=False).block_until_ready()

    t_prelog, std_prelog = measure_time(fn_prelog, n_repeats)
    t_slog,   std_slog   = measure_time(fn_slog,   n_repeats)

    speedup = t_slog / max(t_prelog, 1e-9)
    print(f"  s_geodesic (prelog):  {t_prelog*1000:8.1f} ms  ± {std_prelog*1000:.1f} ms")
    print(f"  s_geodesic (s_log):   {t_slog*1000:8.1f} ms  ± {std_slog*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.1f}×")

    return t_prelog, t_slog, speedup


# ---------------------------------------------------------------------------
# Section 2: s_exp timing — prelog vs s_log
# ---------------------------------------------------------------------------

def benchmark_s_exp_timing(manifold, x0, n_repeats: int = 3, sigma: float = 0.3):
    print(f"\n--- 2. s_exp timing: prelog vs s_log (n={x0.shape[2]}, sigma={sigma}) ---")

    X = sample_tangent(manifold, x0, sigma=sigma)
    K = int(0.25 * float(jnp.max(manifold.norm(x0, X[:, :, None])))) + 1
    print(f"  K (geodesic doubling steps) = {K}")

    def fn_prelog():
        return manifold.s_exp(x0, X, tol=1e-3,
                              use_separation_grad=True).block_until_ready()

    def fn_slog():
        return manifold.s_exp(x0, X, tol=1e-3,
                              use_separation_grad=False).block_until_ready()

    t_prelog, std_prelog = measure_time(fn_prelog, n_repeats)
    t_slog,   std_slog   = measure_time(fn_slog,   n_repeats)

    speedup = t_slog / max(t_prelog, 1e-9)
    print(f"  s_exp (prelog):  {t_prelog*1000:8.1f} ms  ± {std_prelog*1000:.1f} ms")
    print(f"  s_exp (s_log):   {t_slog*1000:8.1f} ms  ± {std_slog*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.1f}×")

    target_ms = 50.0
    status = "✓ TARGET MET" if t_prelog * 1000 <= target_ms else f"✗ target={target_ms}ms NOT MET"
    print(f"  Phase 2.5 target (≤{target_ms}ms):  {status}")

    return t_prelog, t_slog, speedup


# ---------------------------------------------------------------------------
# Section 3: Geodesic accuracy vs tolerance
# ---------------------------------------------------------------------------

def benchmark_geodesic_accuracy(manifold, x, y):
    """
    Compare geodesic accuracy for prelog vs s_log on two scenarios:
    (a) Nearby conformations (as used in s_exp doubling steps) — the important case.
    (b) Far-apart conformations — shows degradation for large distances.

    The prelog gradient is a flat (Euclidean) approximation to the Riemannian gradient,
    accurate near-diagonal (small w^delta). For s_exp, inputs are always nearby
    (step X/K with K ≥ 1), so accuracy is good where it matters.
    """
    print(f"\n--- 3. Geodesic accuracy: prelog vs s_log (n={x.shape[2]}) ---")

    tau_half = jnp.array([0.5])

    # (a) Nearby conformations — representative of s_exp usage
    rng = jax.random.PRNGKey(7)
    v = jax.random.normal(rng, x.shape)
    v_hp = manifold.horizontal_projection_tvector(x, v[:, :, None])
    nrm = manifold.norm(x, v_hp)
    nrm_safe = jnp.maximum(nrm, 1e-8)[:, :, :, None, None]
    X = 0.5 * v_hp[:, :, 0] / nrm_safe[:, :, 0]
    y_near = manifold.align_mpoint(x + X)
    dist_near = float(manifold.s_distance(x, y_near)[0, 0, 0])

    # (b) Far conformations — shows behaviour outside normal s_exp range
    dist_far = float(manifold.s_distance(x, y)[0, 0, 0])

    print(f"  (a) Nearby: w^delta(x, y_near) = {dist_near:.4f}  [s_exp use case]")
    print(f"  (b) Far:    w^delta(x, y_far)  = {dist_far:.4f}  [direct geodesic interpolation]")

    tol_vals = [1e-1, 1e-2, 1e-3, 1e-4]

    for label, xa, ya, dist in [("nearby", x, y_near, dist_near), ("far", x, y, dist_far)]:
        z_ref = manifold.s_geodesic(xa, ya, tau_half, tol=1e-5, use_separation_grad=False)
        print(f"\n  Scenario: {label} (w^delta={dist:.3f})")
        print(f"  {'tol':>8}  {'prelog error':>16}  {'s_log error':>16}  {'rel prelog':>12}")
        print("  " + "-" * 60)
        for tol in tol_vals:
            z_prelog = manifold.s_geodesic(xa, ya, tau_half, tol=tol, use_separation_grad=True)
            z_slog   = manifold.s_geodesic(xa, ya, tau_half, tol=tol, use_separation_grad=False)
            err_p = float(manifold.s_distance(z_ref, z_prelog)[0, 0, 0])
            err_s = float(manifold.s_distance(z_ref, z_slog)[0, 0, 0])
            rel   = err_p / max(dist, 1e-8)
            print(f"  {tol:>8.0e}  {err_p:>16.6f}  {err_s:>16.6f}  {rel:>12.4f}")

    print()
    print("  Note: prelog accuracy degrades for large w^delta (far conformations).")
    print("  In s_exp, all doubling steps use nearby inputs — prelog is accurate there.")


# ---------------------------------------------------------------------------
# Section 4: Score target accuracy (approx ambient noising vs exact s_exp)
# ---------------------------------------------------------------------------

def benchmark_score_target(manifold, x0, sde):
    """
    Compare score targets from ambient-noised x_t vs exact s_exp x_t.

    For the ambient noising convention (Option A), x_t = x_0 + sigma*v_h.
    For exact s_exp, x_t = s_exp(x_0, sigma*v_h).

    Both use s_log(x_t, x_0) as the score target direction.
    """
    print(f"\n--- 4. Score target accuracy: ambient vs s_exp (n={x0.shape[2]}) ---")
    sigma_vals = [0.05, 0.1, 0.2, 0.4, 0.8]

    print(f"  {'sigma':>6}  {'cos(s_ambient, s_exact)':>24}  "
          f"{'|s_ambient|/|s_exact|':>22}  {'verdict':>8}")
    print("  " + "-" * 68)

    rng = jax.random.PRNGKey(99)

    for sigma in sigma_vals:
        v = jax.random.normal(rng, x0.shape)
        v_hp = manifold.horizontal_projection_tvector(x0, v[:, :, None])
        nrm = manifold.norm(x0, v_hp)
        nrm_safe = jnp.maximum(nrm, 1e-8)[:, :, :, None, None]
        tangent = float(sigma) * v_hp[:, :, 0] / nrm_safe[:, :, 0]

        x_t_exact   = manifold.s_exp(x0, tangent, use_separation_grad=True)
        x_t_ambient = x0 + tangent

        log_exact   = manifold.s_log(x_t_exact,   x0)[0, 0, 0].flatten()
        log_ambient = manifold.s_log(x_t_ambient,  x0)[0, 0, 0].flatten()

        cos_sim  = float(jnp.dot(log_exact, log_ambient) /
                         (jnp.linalg.norm(log_exact) * jnp.linalg.norm(log_ambient) + 1e-12))
        mag_ratio = float(jnp.linalg.norm(log_ambient) / (jnp.linalg.norm(log_exact) + 1e-12))
        verdict  = "OK" if cos_sim > 0.95 else "BAD"

        print(f"  {sigma:>6.2f}  {cos_sim:>24.4f}  {mag_ratio:>22.4f}  {verdict:>8}")
        rng, _ = jax.random.split(rng)

    print()
    print("  Interpretation:")
    print("  - cos ~ 1.0: ambient and exact score targets agree in direction")
    print("  - mag_ratio ~ 0.5: ambient x_t gives ~half magnitude (K=1 doubling effect)")
    print("  - Direction agreement means both conventions train equivalent models")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Use full 214-atom AK data. Default: n=10 synthetic.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of timing repeats (default 3)")
    args = parser.parse_args()

    fast = not args.full
    x0, y0 = load_frames(fast=fast)
    n, d = x0.shape[2], x0.shape[3]
    label = "adenylate kinase (full, n=214)" if not fast else f"synthetic (fast, n={n})"

    manifold = ShapeManifold(dim=d, numpoints=n, alpha=1.0, base=np.array(x0[0, 0]))
    sde      = ManifoldVP(manifold)

    # Align both frames to base
    x0 = manifold.align_mpoint(x0)
    y0 = manifold.align_mpoint(y0)

    print("=" * 65)
    print("s_exp Optimisation Benchmark — Phase 2.5")
    print(f"Data: {label}")
    print("Gradient strategy: prelog (separation) vs s_log (Riemannian)")
    print("=" * 65)

    benchmark_geodesic_timing(manifold, x0, y0, n_repeats=args.repeats)
    benchmark_s_exp_timing(manifold, x0, n_repeats=args.repeats)
    benchmark_geodesic_accuracy(manifold, x0, y0)
    benchmark_score_target(manifold, x0, sde)

    print("\n" + "=" * 65)
    print("Benchmark complete.")
    print("=" * 65)
