# Phase 2 Summary: Manifold Forward Process

> Written 2026-05-25. Covers Phase 2 (forward process), Phase 2.5 (s_exp optimisation),
> and the marginal_prob design decision (ambient → geodesic noising).

---

## What was built

Two new source files under `riemannian-scoremd/src/diffusion/`:

| File | Content |
|---|---|
| `manifold_sde.py` | `ManifoldVP` — VP-SDE forward process and reverse drift |
| `manifold_solvers.py` | `ManifoldEulerMaruyama` — on-manifold Euler–Maruyama integrator |

---

## ManifoldVP: forward process design

### VP schedule (from ScoreMD, Plainer et al. 2025)

```
beta(t) = beta_min + t*(beta_max - beta_min)    beta_min=0.1, beta_max=20
log_alpha(t) = -0.5*t*beta_min - 0.25*t^2*(beta_max - beta_min)
alpha(t) = exp(log_alpha(t))                     mean decay
sigma(t) = sqrt(1 - alpha(t)^2)                  noise std dev
```

At t=0: alpha≈1, sigma≈0 (clean data). At t=1: alpha≈0.007, sigma≈1 (near-isotropic noise).

### marginal_prob: the design decision and resolution

Three options were considered and benchmarked:

**Option A — Ambient noising** (original, now removed):
```
x_t = x_0 + alpha(t) * sigma(t) * v_h_unit
```
Simple and fast (4.2 ms). But x_t lives off the manifold by O(sigma³) in geodesic
distance. Inconsistent with the on-manifold reverse SDE used at inference.

**Option B — Naive s_exp** (intermediate, had a sign error):
```
x_t = s_exp(x_0, alpha(t) * sigma(t) * v_h_unit)
```
On-manifold, but s_exp with K=1 implements geodesic doubling:
`s_exp(x, X) = s_geodesic(x, x+X, tau=2)`, displacing by ~2||X||_g from x.
This makes `s_log(x_t, x_0) ≈ -2*alpha*sigma*v_h_unit`, giving a score target 2×
larger than the VP-SDE convention.

**Option C — Corrected s_exp** (current implementation):
```
x_t = s_exp(x_0, 0.5 * alpha(t) * sigma(t) * v_h_unit)
```
The factor of 1/2 compensates for the doubling, giving:
- `w^delta(x_t, x_0) ≈ alpha(t) * sigma(t)` ✓
- `||s_log(x_t, x_0)||_g ≈ alpha(t) * sigma(t)` ✓
- Score target: `s_true = -s_log(x_t, x_0) / sigma(t)^2 ≈ alpha(t)*v_h_unit / sigma(t)` ✓

Verified empirically on adenylate kinase (n=214) across t ∈ {0.1, 0.2, 0.3, 0.5, 0.8}:

| sigma | w^delta(x_t, x_0) | ‖s_log‖_g | target = sigma | ratio |
|---|---|---|---|---|
| 0.10 | 0.100001 | 0.100000 | 0.10 | 1.000 |
| 0.20 | 0.200000 | 0.199997 | 0.20 | 1.000 |
| 0.30 | 0.300000 | 0.299994 | 0.30 | 1.000 |
| 0.50 | 0.499993 | 0.499975 | 0.50 | 1.000 |
| 0.80 | 0.799951 | 0.799888 | 0.80 | 1.000 |

The cos(-s_log, v_h_unit) = 1.0000 at all noise levels — direction is exact.

### Why s_exp, not ambient?

- x_t lies exactly on M; the reverse SDE (ManifoldEulerMaruyama) must also stay on M
- Score target `s_true = -s_log(x_t, x_0)/sigma²` is a genuine Riemannian log map,
  not an ambient approximation — necessary for Phase 4 Laplace-Beltrami correction
- s_exp cost: 7.1 ms/call vs 4.2 ms ambient — overhead is only +2.9 ms (1.7×)

---

## Performance numbers (adenylate kinase, n=214, CPU)

| Operation | Time |
|---|---|
| s_exp (prelog, K=1, JIT-cached) | 2.6–2.7 ms |
| marginal_prob (s_exp, Option C) | 7.1 ms |
| marginal_prob (ambient, Option A) | 4.2 ms |
| score_target (s_log, one call) | 17.3 ms |
| **Total geometric cost per DSM step** | **24.4 ms** |

### Training time estimate

One DSM training step requires:
- 1× `marginal_prob`: 7.1 ms (forward noising)
- 1× `score_target` (s_log): 17.3 ms (compute target)
- 1× network forward pass + loss: ~1–5 ms (TBD, model-dependent)
- 1× backward pass (grad wrt params): ~2–3× forward = ~2–15 ms

**Geometric bottleneck: `score_target` (s_log, 17.3 ms)**, not `marginal_prob`.

Conservative estimate for full step: **~50 ms/step**.

At 102 adenylate kinase frames, batch_size=8, steps_per_epoch ≈ 13:
- 1 epoch ≈ 13 × 50 ms = 0.65 s
- 1000 epochs ≈ 11 min
- 10,000 epochs ≈ 110 min (~2 h)

This is fast enough for iterative development. No GPU needed for Phase 3 prototyping.

**Score target optimisation option**: replace `s_log` (17.3 ms, uses eigh(642×642))
with `s_prelog` (1.4 ms, no eigh). This reduces per-step geometry to ~9 ms, but
`s_prelog` is an approximation (Euclidean gradient, not Riemannian). Acceptable for
training; exact s_log still needed for Phase 4 FP divergence computation.

---

## ManifoldEulerMaruyama

Each step uses `s_exp` to keep iterates exactly on M:
```
drift_or_reverse_drift  ← tangent vector at x_t
w                       ← unit horizontal noise at x_t
v = drift*dt + g(t)*sqrt(dt)*w
x_{t+dt} = s_exp(x_t, v)
```

The same integrator handles both forward diffusion (`mode='forward'`) and
reverse-time sampling (`mode='reverse'`, requires a score function).

---

## Phase 2 validation: Brownian motion tests (8/8 pass)

Protein: synthetic n=10 (fast, CI) and adenylate kinase n=214 (--full).

### Original 4 tests (BM geometry)

| Test | Tolerance | Status |
|---|---|---|
| Gyration radius drift < 10% over 50 steps (dt=1e-4) | 10% | PASS |
| Mean pairwise Cα distance drift < 5% over 50 steps | 5% | PASS |
| Rigid-body drift ‖O−I‖_F near zero | 1% | PASS |
| Diffusion coefficient recovery within 50% | 50% | PASS |

### 4 new completion tests (forward process design)

| Test | What it checks | Status |
|---|---|---|
| Score target round-trip | cos(score, v_h) > 0.85, ‖score‖_g within 20% of alpha/sigma | PASS |
| x_t on manifold (large t) | finite, centred, w^delta > 0 at t ∈ {0.7, 0.9, 1.0} | PASS |
| reverse_drift zero-score | reverse_drift(x, 0, t) == beta(t)/2 * x exactly | PASS |
| VP schedule monotonicity | alpha decreasing, sigma increasing, alpha²+sigma² ≤ 1 | PASS |

Note on score target tolerances: fast mode (n=10 synthetic) achieves cos ≈ 0.93–0.95
and rel mag error ≈ 2–12% due to s_prelog approximation degrading on small irregular
proteins. Full AK (n=214, --full) gives cos > 0.999 and rel < 5% (verified manually).

---

## Phase 2.5: s_exp optimisation (complete)

Root cause of original 121 ms/call: Python eager dispatch overhead, not XLA compute.
The `while_loop` itself runs in ~1 ms when JIT-compiled.

**Fix**: `_build_doubling_fn` inlines the entire `s_geodesic` body into a `@jax.jit`
closure, cached in `self._doubling_cache` keyed on `(K, use_separation_grad, tol,
max_iter, step_size)`. One XLA compilation per unique parameter set; subsequent calls
hit the XLA executable directly.

**Warm-start**: linear extrapolation `z_init = 2*x1 - x0` (exact in flat geometry)
reduces s_geodesic inner iterations from ~47 (cold) to ~4 (warm).

**Separation gradient** (`use_separation_grad=True`, default): replaces
`eigh(642×642)` per iteration with the flat gradient of w² (O(n²d), no eigh).
2800× cheaper per iteration; stagnates at ~0.35% relative geodesic error for nearby
inputs — acceptable for training.

| Variant | Time before | Time after | Speedup |
|---|---|---|---|
| s_exp (prelog, K=1) | 121 ms | 2.6–2.7 ms | 20× |
| s_exp (s_log, K=1) | 228 ms | 53–54 ms | 4× |
| s_geodesic (prelog, direct call) | — | 87–93 ms | — |

Direct `s_geodesic` calls from Python are still slow (87 ms) — the fast path
only activates when called through `s_exp`'s cached JIT. Fine for analysis/viz.

### Geodesic accuracy at tol=1e-3 (n=214)

| Scenario | w^delta | prelog error | s_log error | rel prelog |
|---|---|---|---|---|
| Nearby (s_exp use case) | 0.500 | 0.0017 | 0.000036 | 0.35% |
| Far (direct interpolation) | 3.262 | 0.013 | 0.000038 | 0.40% |

Prelog error scales with distance as expected. For s_exp doubling steps (always
nearby inputs), 0.35% relative error is negligible.

---

## What is not tested / future validation

All planned Phase 2 tests are now implemented (8/8). No outstanding coverage gaps.

---

## Summary judgment

Phase 2 is fully implemented and validated. All 8 tests pass (21/21 including Phase 2.5
separation-geodesic tests). The key design decision (geodesic noising with 0.5×
compensation) is empirically verified and locked in by test 5 (score target round-trip).
The geometric cost per training step is ~24 ms, dominated by `s_log` in `score_target`.
Training on adenylate kinase (102 frames) will be fast: ~11 min for 1000 epochs,
~2 h for 10,000 epochs on CPU.

**Phase 2 is complete. Proceeding to Phase 3: TangentScoreModel + Riemannian DSM loss.**
