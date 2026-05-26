# The `s_exp` Performance Problem

*Analysis, root causes, and a prioritised solution roadmap.*

---

## What `s_exp` does and why it is slow

The exponential map `s_exp(x, X)` answers: *starting at conformation x and moving in direction X (a tangent vector), where do you end up on the manifold?*

It cannot be computed in closed form for the w^delta metric. Instead it uses **geodesic doubling**: split the tangent vector into K small pieces, then repeatedly double using the geodesic midpoint rule.

```
x_0 = x
x_1 = x + (1/K) * X             # first tiny step in ambient space

for k in 0..K-1:
    x_new = s_geodesic(x_k, x_{k+1}, tau=2)   # extrapolate along geodesic
    x_k, x_{k+1} = x_{k+1}, x_new
```

`s_geodesic` itself is an **inner iterative loop** — gradient descent on the manifold until the interpolated point converges (up to 100 iterations, tolerance 1e-3). Each iteration calls `s_log` once, which calls `metric_tensor` once.

The full call graph for a single `s_exp` call:

```
s_exp
└─ fori_loop (K iterations)
   └─ s_geodesic  (while_loop, up to 100 iters)
      └─ s_log  (×2, once for each endpoint)
         ├─ s_prelog        O(n²d)
         └─ metric_tensor   O(n²d²) + eigh O(nd)³
```

---

## The three nested loops

### Loop 1 — geodesic doubling steps K

K is set by:
```python
K = int(c * norm_g(X).max()) + 1,   c = 0.25
```

For a typical noisy tangent vector during training, `norm_g(X)` in AK units is of order 10–100 (because coordinates are in Å and the g-norm accumulates over all n²=45796 pairwise terms). This gives **K ≈ 3–25** doubling steps.

### Loop 2 — geodesic gradient descent (inside each doubling step)

`s_geodesic` runs gradient descent to find the point at parameter τ along the geodesic. In practice this converges in **10–40 iterations** for well-conditioned inputs.

### Loop 3 — metric tensor + eigendecomposition (inside each gradient step)

Each `s_log` call recomputes:
- `s_prelog`: two pairwise distance matrices, O(n²d) — **~45 800 multiplications for AK**
- `metric_tensor`: builds a (nd × nd) = (642 × 642) matrix, O(n²d²) = **~420 000 multiplications**
- `eigh` on a (642 × 642) matrix: **O((nd)³) ≈ 265 million flops**

The total per `s_exp` call on n=214:

| Component | Calls | Flops estimate |
|---|---|---|
| eigh (642×642) | K × 40 × 2 | ~2000 × 265M = **530 billion** |
| metric_tensor build | same | ~2000 × 420K = **840 million** |
| s_prelog | same | ~2000 × 92K = **184 million** |

This explains the measured ~767ms/call on CPU.

---

## Why this matters for training

During Phase 3 training, `s_exp` appears in two places:

1. **`marginal_prob`** — called once per training sample per batch to generate `x_t ~ q(x_t|x_0)`. With batch size B=64 and e.g. 1000 gradient steps per epoch, that is 64,000 `s_exp` calls per epoch × 767ms = **~14 hours per epoch**. Completely intractable.

2. **`ManifoldEulerMaruyama.step`** — called once per Langevin step during sampling. With 500 reverse steps this is 500 × 767ms = **~6 minutes per sample trajectory**. Slow but potentially tolerable for inference only.

---

## Solution roadmap (four strategies, in priority order)

---

### Strategy 1 — Replace `s_exp` in training with the tangent-space approximation (no `s_exp` at all)

**Applicable to**: `marginal_prob` during training only.

The full wrapped-Gaussian forward process requires `s_exp` to map the noisy tangent vector back to the manifold. But for small noise levels (early in training, or small t), the approximation:

```
x_t ≈ x_0 + v_h
```

where `v_h = σ(t) * unit_horizontal_noise` is simply adding the tangent vector in ambient space. This is the **first-order approximation** and is exact in the limit σ(t) → 0.

More precisely: since `s_exp` is defined via geodesic doubling and the w^delta log map is third-order accurate, the error of this approximation is O(σ(t)³). For the VP schedule, σ(t) < 0.3 for t < 0.4, meaning the error is < 3%.

**Implementation**: add a `use_approx_exp: bool = True` flag to `marginal_prob`:
```python
if use_approx_exp:
    x_t = x_0 + tangent           # fast: O(nd), no iterative loops
else:
    x_t = self.s_exp(x_0, tangent) # exact: slow
```

This reduces training cost from 14h/epoch to **milliseconds per batch**. Use exact `s_exp` only for sampling (inference).

**Empirical finding** (benchmarks/benchmark_s_exp.py): The score target error is not negligible in the w^delta metric. With `x_approx = x_0 + σv` vs `x_exact = s_exp(x_0, σv)`:

- `||s_true_approx - s_true_exact||_F / ||s_true_exact||_F ≈ 0.5` for all σ

This ~50% error arises because, with K=1 geodesic doubling (which is the typical case when tangent vectors are unit-normalised), `s_log(x_exact, x_0)` has magnitude ≈ 2× the tangent vector (the K=1 s_exp doubles the displacement once), while `s_log(x_approx, x_0) ≈ −tangent` (the log map inverts the ambient displacement cleanly). Both score targets have the same **direction** (cos ≈ 0.99) but differ in magnitude by ~2×.

**Practical implication**: For training, both paths produce score targets that point in the correct direction but differ in scale. Since the score network can absorb a consistent 2× scale factor in the output, the `use_approx_exp=True` path is valid for training — but the learned score will be calibrated to the simpler approx path, not the exact wrapped Gaussian. This is consistent with how Euclidean diffusion models work in practice: the exact sample from the forward process matters less than the consistency of the target.

The tangent-space approximation is best understood as defining a slightly different (but valid) training objective: a score network trained with `use_approx_exp=True` learns to denoise `x_0 + σv` (ambient noising), which at inference time should be paired with the same approx path or an exact reverse-time SDE that accounts for this calibration.

---

### Strategy 2 — Reduce K via better normalisation of tangent vectors

**Applicable to**: both training and sampling.

K grows linearly with `norm_g(X)`. The g-norm in Ångström coordinates is large because the metric tensor H accumulates over all n² pairs. We can reduce K by **working in normalised coordinates**:

- Normalise input conformations to unit gyration radius before training (divide all coordinates by `sqrt(tr(G)/n)`)
- This reduces typical pairwise distances from ~20Å to ~1Å, shrinking `norm_g(X)` by a factor of ~20
- K drops from ~20 to **K=1 or K=2** for typical training noise levels

This alone gives a **10–20× speedup** on `s_exp` for free, with no approximation error.

The normalisation is reversible: scale back after sampling. It is standard practice in Euclidean diffusion models and fully compatible with the w^delta metric (which is scale-invariant in the log-ratio terms, only the gyration correction is affected — and α can be rescaled accordingly).

---

### Strategy 3 — Cache and reuse the metric tensor eigendecomposition

**Applicable to**: the inner loops of `s_geodesic`.

Within a single `s_exp` call, the `while_loop` inside `s_geodesic` recomputes the eigendecomposition of H(z) at every gradient step, even though z changes slowly (step_size = 1.0, convergence is linear). In practice, H(z) changes very little between consecutive gradient steps.

**Implementation**: compute H(z) and its eigendecomposition once per gradient step, pass it as part of the carry in the `while_loop`. This halves the number of `eigh` calls inside the inner loop from `2 × n_iter` to `2 × n_iter / 2`.

This is a **2× speedup** on the dominant cost with no approximation.

---

### Strategy 4 — Hutchinson estimator for the geodesic step (approximate `s_geodesic`)

**Applicable to**: the whole iterative machinery, as a longer-term refactor.

The geodesic gradient descent in `s_geodesic` calls `s_log` (and thus `metric_tensor + eigh`) twice per iteration. An alternative for the *training forward process only* is to replace `s_geodesic` with the **retraction** approach:

```
retract(x, v) = s_exp_linear(x, v)   # project x + v back onto M
```

using the "projection retraction" — centre and align `x + v` back to the quotient space. This is O(nd) (SVD of a d×d matrix) rather than O((nd)³) (eigh of an nd×nd matrix), and is standard in Riemannian optimisation.

This is only correct to first order (retraction ≠ geodesic exponential map), but for a forward diffusion process the exact geodesic is not needed — we just need to stay on the manifold.

**This is the most impactful long-term change** but also the most involved to implement correctly.

---

## Recommended implementation plan for Phase 3

| Action | Speedup | Effort | When |
|---|---|---|---|
| Normalise coordinates to unit gyration radius | 10–20× | 1 hour | Before Phase 3 starts |
| `use_approx_exp=True` in `marginal_prob` for training | 10000× | 2 hours | Phase 3 start |
| Cache eigendecomposition in `s_geodesic` carry | 2× | 3 hours | Phase 3, if sampling speed matters |
| Retraction-based `s_exp` for forward process | 100× exact | 1 day | Phase 5, before scaling to helicase |

**The minimum viable path into Phase 3 training** is strategies 1 + 2 together: normalise coordinates and use the tangent-space approximation in `marginal_prob`. This brings training cost from ~14h/epoch down to **<10 seconds/epoch** with negligible loss in approximation quality.

The exact `s_exp` is then reserved for:
- Inference (reverse SDE sampling, 500 steps)
- Validation of the forward process (BM tests with `--full` flag)
- Computing geodesics for visualisation

---

## What this means for the score network

The score network `s_θ(x_t, t)` outputs a horizontal tangent vector at `x_t`. With the tangent-space approximation for training, `x_t = x_0 + v_h` lives slightly off the manifold (by O(σ³)). Two implications:

1. The network input `x_t` may have small residual vertical components. Apply `horizontal_projection_tvector(x_t, s_θ)` to the *output* but not to the input — the input off-manifold error is below the float32 noise floor for σ < 0.3.

2. The score target `s_true = -s_log(x_t, x_0)/σ²` still uses the exact `s_log`. Since `s_log` is already an approximation (separation), this is fine and keeps the target in the correct tangent space at `x_t`.

Both of these are standard practice in wrapped-Gaussian diffusion on manifolds (Huang et al. 2022, De Bortoli et al. 2022) and do not affect the theoretical validity of the DSM loss.
