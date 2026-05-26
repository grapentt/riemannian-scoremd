# s_exp Optimisation Plan — Phase 2.5

*Why it matters, what is slow, how to fix it, and what the Diepeveen 2025 paper contributes.*

---

## 1. Goal and framing

The project goal is to train a diffusion model **on the actual Riemannian manifold** — meaning `s_exp` must appear in the forward process, not just the reverse SDE. Without this:
- Training uses ambient noising (`x_t = x_0 + sigma*v_h`), which bypasses manifold geometry in the forward process entirely.
- The w^delta metric then only enters through the score target direction (`s_log`) and the loss norm (`||·||_g`).
- The model reduces to a mildly improved Xu 2026, not a genuinely new geometric approach.

**Target**: `s_exp` ≤ 50ms/call on n=214 CPU → batch training at ~3–5s/step → tractable.

---

## 2. Final state (Phase 2.5 COMPLETE)

### What was done

**Task 2.5.1**: `s_geodesic` has `use_separation_grad: bool = True`. The while_loop body calls `s_prelog(z, x)` and `s_prelog(z, y)` instead of `s_log`. Eliminates eigh entirely.

**Task 2.5.2 + 2.5.3**: Warm-start + JIT caching. **s_exp = 2.6ms on n=214.** Target (≤50ms) exceeded by 19×.

### What the actual bottleneck was (important lesson for future work)

The 121ms pre-fix timing was **not** the while_loop computation. It was **Python eager dispatch overhead**.

Diagnosis:
- `s_geodesic` called directly from Python (no `@jax.jit`): ~90ms (Python overhead, not XLA)
- `s_geodesic` JIT-wrapped externally: **1ms** (warm) / **10ms** (cold 47 iters)
- `while_loop` body alone JIT-compiled: ~0.25ms/iteration

`s_exp` uses `int(c * float(jnp.max(nrm))) + 1` to compute K, making it impossible to `jax.jit` directly. The old `fori_loop` called `s_geodesic` as a Python method, incurring ~22ms/call Python dispatch overhead × K calls = ~22ms total for K=1, but **the while_loop inside** was running eagerly too (each iteration = separate Python dispatch).

### Final implementation

**`_build_doubling_fn`**: a factory method that inlines all of `s_geodesic`'s content directly into a `@jax.jit`-decorated function, then caches the compiled result in `self._doubling_cache` keyed on `(K, use_separation_grad, tol, max_iter, step_size)`. One XLA compilation per unique parameter set. Subsequent calls hit the XLA cache at ~1ms.

**Warm-start**: linear extrapolation `z_init = 2*x1 - x0` for tau=2 (flat-geometry exact solution). Reduces iterations 47 (cold) → 4 (warm). Cross-step warm-start: `z_next = 2*x_new - xk` for subsequent K steps.

### Final measured costs on n=214 CPU

| Component | Cost |
|---|---|
| `s_prelog(z, x)` — single call | 1.4 ms |
| `s_log(z, x)` — single call (1 eigh + einsum) | 15.5 ms |
| `eigh(642×642)` alone | 11.8 ms |
| `while_loop` JIT-compiled, warm (4 iters) | **1.0 ms** |
| `while_loop` JIT-compiled, cold (47 iters) | **10.0 ms** |
| `s_geodesic` direct Python call (warm) | 80 ms (Python dispatch overhead) |
| `s_exp` prelog, K=1, warm-start + JIT | **2.6 ms** ← final |
| `s_exp` s_log, K=1, warm-start + JIT | 53 ms |
| Speedup prelog vs s_log | 20.3× |

### Accuracy on n=214 (unchanged from 2.5.1)

| Input type | Relative geodesic error (tol=1e-3) |
|---|---|
| Nearby (w^delta ≈ 0.5) — s_exp use case | **0.35%** |
| Far (w^delta ≈ 3.3) — direct geodesic queries | 0.4% |

Score target direction agreement: cos = 1.0000 at all sigma values on n=214.

---

## 3. What was needed and why it worked

The key was identifying that Python eager dispatch, not the mathematical computation, dominated. Two changes together fixed it:

1. **Warm-start** (linear extrapolation): iterations 47→4, reducing `while_loop` time 10ms→1ms
2. **JIT caching** (`_build_doubling_fn`): eliminates ~22ms/call Python dispatch, reducing total from ~90ms to ~3ms

The `s_prelog` substitution (Task 2.5.1) was necessary for accuracy reasons (avoids eigh in while_loop) and provides a 20× speedup over s_log, but the dominant gain was the JIT fix.

### Training feasibility (achieved)

| Forward process | s_exp cost | Step time (batch 8) | Status |
|---|---|---|---|
| Ambient noising (fallback) | < 1 ms | ~1–2s | Off-manifold |
| Exact s_exp, post-fix | **2.6 ms** | ~1–3s | **Goal: on-manifold** ✓ |
| Exact s_exp, pre-fix | 121 ms | ~25s | Was too slow |

---

## 7. Diepeveen 2025 paper analysis (arXiv:2410.01950)

This paper by Diepeveen, Batzolis, Shumaylov, Schönlieb (2025) is the **same first author as the w^delta paper** but addresses a completely different problem. Reading it carefully is important for understanding what it does and does not contribute to our project.

### What the paper does

It constructs a **data-driven Riemannian geometry** for distributions of the form:
```
p(x) ∝ exp(-ψ(φ(x)))
```
where ψ is strongly convex and φ is a diffeomorphism (normalizing flow). For the special case ψ(x) = ½ xᵀA⁻¹x (diagonal A), **all manifold maps are closed-form** (Prop 1, Eqs 5–8):

```
distance:   d(x,y) = ||A⁻¹(φ(x) - φ(y))||₂                              (5)
geodesic:   γ(t)   = φ⁻¹((1-t)φ(x) + tφ(y))                             (6)
exp map:    exp_x(Ξ) = φ⁻¹(φ(x) + D_x φ[Ξ])                             (7)
log map:    log_x y  = D_{φ(x)} φ⁻¹ [φ(y) - φ(x)]                       (8)
```

The Riemannian autoencoder (RAE) uses this to discover the intrinsic dimension of data — the number of latent dimensions needed to reconstruct x to precision ε.

### What it does NOT do

- It does **not** define or use the w^delta metric
- It does **not** provide faster geodesic computation for analytically-defined metrics like ours
- It does **not** derive the Laplace-Beltrami divergence for any specific metric
- It does **not** train a score network on a manifold — the "score" connection is conceptual (the pullback metric is related to the score function gradient when φ is approximately isometric)

### Why it is not directly applicable to our project

Our manifold is defined by **explicit physics-based geometry** (pairwise log-distances + gyration tensor). We do not need to learn the geometry from data — we already have it, analytically and exactly. The pullback framework is designed precisely for the case where you do **not** have an analytical metric and must learn one.

The paper explicitly acknowledges this distinction (p.2): "scalability of manifold mappings was completely circumvented by Diepeveen (2024) and de Kruiff et al. (2024) by using pullback geometry. However, here learning a suitable (and stable) pullback geometry suffers from challenges regarding scalability of the training algorithm."

This confirms: **our approach (analytical w^delta metric) is the scalable one**; their approach (learned pullback) trades manifold map speed for training cost.

### What IS useful: sliced Jacobian-vector products (Phase 4)

The one technique from this paper with direct applicability is the **sliced isometry regularization** (Appendix I.2), which scales the computation of `||Dφ||²` via random Jacobian-vector products instead of the full Jacobian.

For our **Phase 4 Fokker-Planck loss**, we need:
```
div_M(s) = div_flat(s) - ⟨s, ∇ log√det g(x)⟩
```

The term `∇ log det g(x)` requires differentiating through `metric_tensor(x)`, which naively costs O(n⁴d²) — the full Jacobian of g w.r.t. x. The sliced JVP approach gives an unbiased estimator:

```python
# Sliced estimator for tr(J_s(x)):
def sliced_divergence(score_fn, x, n_slices=8):
    eps = jax.random.normal(key, (n_slices, *x.shape))  # random directions
    eps = eps / jnp.linalg.norm(eps, axis=..., keepdims=True)
    # JVP: ∂s/∂x · eps — cost O(nd) per slice, not O((nd)²)
    _, jvp = jax.jvp(score_fn, (x,), (eps,))
    return jnp.mean(jnp.sum(eps * jvp, axis=...))
```

This is the same trick used in ScoreMD's Hutchinson estimator, now applied to the manifold correction term. **Action item for Phase 4, not Phase 2.5.**

### Connection to our score network design (Phase 3)

The paper provides a useful conceptual frame: for a well-trained score model, the score Jacobian `D_x ∇ log p(x)` approximates `-g(x)` (the negative metric tensor). This means:

- The **score encodes the metric** implicitly
- A score network with output in the horizontal tangent space will, if well-trained, induce the w^delta metric implicitly through its gradient structure

This validates our Phase 3 architecture choice: projecting the score output to the horizontal tangent space via `horizontal_projection_tvector` is not just enforcing gauge — it is aligning the score with the intrinsic geometry.

### Summary table

| Paper idea | Applicable to us? | When |
|---|---|---|
| Pullback metric from NF | No — we have analytic metric | — |
| Riemannian autoencoder (RAE) | No — we don't need dim reduction | — |
| Closed-form geodesics via NF | No — different geometric setup | — |
| Sliced JVP for Jacobian computation | **Yes** | Phase 4: FP divergence |
| Score ↔ metric tensor connection | **Yes** (conceptual) | Phase 3: architecture intuition |
| Anisotropic base distribution | Not needed — metric handles anisotropy | — |
| Isometry + volume loss for NF training | No — not training a flow | — |

---

## 8. Accuracy and correctness summary (n=214, measured)

| Method | Relative geodesic error (nearby) | Score dir cos | Notes |
|---|---|---|---|
| `s_log` exact | reference | 1.0000 | Riemannian gradient, slow |
| `s_prelog` tol=1e-3 (nearby) | **0.04%** | 1.0000 | s_exp use case, fast |
| `s_prelog` tol=1e-3 (far) | 0.4% | 1.0000 | Direct geodesic queries |
| Ambient noising (no s_exp) | N/A | 1.0000 | Current training fallback |

The prelog approximation is accurate enough for training. After warm-starting, both the timing and accuracy will be well within acceptable bounds for a fully geometric diffusion model.

---

## 9. Future optional direction: Riemannian Flow Matching

After Phase 3 (diffusion baseline), Riemannian Flow Matching on the w^delta manifold is a high-value optional extension.

**Key idea**: Replace the VP-SDE forward process with a geodesic OT interpolant `γ_{x₀,x_noise}(t)` computed via `s_exp`/`s_geodesic`. The network learns the tangent velocity field instead of the score. The reparameterization to the score is `∇_x log p_t(x) = 1/(1-t) · log_{x_t}(x₁)`.

**Motivation from ScoreMD A.4**: Flat FM fails near t≈0 "likely because the stochasticity inherent to diffusion models improves generalization." On the w^delta manifold, geodesic interpolants stay on manifold — no physically forbidden conformations appear during training, which may smooth the loss landscape at t=0 and improve generalization. Hypothesis, not proven.

**Efficiency**: 1–10 NFEs vs 100–500 for diffusion → 10–100× faster sampling. At ≤15ms/s_exp (post warm-start), 10 NFEs ≈ 150ms/sample.

**Constraint**: Requires `s_exp` during training (for geodesic interpolant). Gated on Task 2.5.2.

**FP consistency**: Phase 4 Laplace-Beltrami FP regularization applies directly to FM too.

**Papers**:
- Lipman et al. 2023 — Flow Matching (ICLR) — flat FM
- Klein, Krämer, Noé 2023 — Equivariant Flow Matching (NeurIPS) — SE(3)-quotient FM, directly analogous
- Chen & Lipman 2024 — Flow Matching Guide (arXiv:2412.06264) — includes Riemannian FM formalism
- Köhler, Klein, Noé 2023 — FM for CG-MD without forces (JCTC) — same application domain

**Recommended plan**: Phase 3b ablation after diffusion baseline. Same network, same evaluation. Paper table: (a) Euclidean diffusion, (b) Riemannian diffusion, (c) Riemannian FM.

---

## 10. Success criterion

Phase 2.5 is complete when:
- `s_exp` on n=214 runs in ≤ 50ms/call (verified by `benchmarks/benchmark_s_exp.py --full`)
- `tests/test_brownian_motion.py --full` passes (4/4)
- `tests/test_separation_geodesic.py --full` passes (5/5)
- `ManifoldVP.marginal_prob` updated to use `s_exp` (forward process on actual manifold)
