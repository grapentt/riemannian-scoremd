# Project Exposé

**Riemannian Diffusion Models on the $w^\delta$-Protein Shape Manifold**
*Physics-Informed Generative Modeling and Consistent Simulation for Coarse-Grained Protein Conformations*

*(Proof of Concept — Adenylate Kinase)*

---

## The Challenge

Modern diffusion and flow-matching models have dramatically advanced molecular generation, yet the vast majority operate in flat Euclidean space or generic SE(3) quotients. These approaches treat protein conformations as unstructured point clouds and therefore fail to capture the **intrinsic non-linear geometry** imposed by the underlying energy landscape. As a consequence, they suffer from poor generalization across different proteins, generate unphysical or high-energy conformations during large-scale transitions, and produce score functions that break down when repurposed as force fields for actual molecular dynamics simulation (Plainer et al., 2025). Even recent quotient-space diffusion methods (Xu et al., 2026) focus primarily on symmetry removal rather than embedding the physics of the energy landscape itself.

---

## Our Approach

We introduce the first diffusion (or stochastic flow-matching) generative model that operates **directly on a physics-informed Riemannian manifold** specifically engineered for protein conformational dynamics.

At its core is the smooth quotient manifold $M = P(d,n)/E(d)$ of centered, non-colliding point clouds modulo rigid-body motions (Diepeveen et al., 2024, Theorem 4.2). We equip this manifold with the **$w^\delta$-metric** (Eq. 17 in the original work): a complete, energy-landscape-derived separation metric constructed from logarithmic ratios of pairwise inter-atomic distances plus a radius-of-gyration term. This metric was deliberately reverse-engineered so that its geodesics closely approximate the energy-minimizing paths observed in molecular dynamics trajectories.

Crucially, we exploit the **separation construction** (Diepeveen et al., Definition 3.1 and Theorems 3.2–3.4). This mathematical object provides provably accurate, third-order approximations to the Riemannian logarithmic map and enables the computation of geodesics and exponential maps via simple, linearly convergent Riemannian gradient descent. Making these primitives fast enough for training-time use was the central computational challenge of this PoC (see below).

We further strengthen the model by integrating state-of-the-art consistency techniques from Plainer et al. (2025): Fokker–Planck regularization to enforce dynamical consistency between the learned score and the probability flow, together with conservative (energy-based) score parameterization. The resulting score function can be used not only for high-fidelity generation but also directly as a physically meaningful force field for stable Langevin dynamics *on the manifold itself*.

---

## Scope and PoC Design

This proof of concept targets two proteins: **BBA (beta3s, n=28)** as the primary scientific validation target (enabling direct PMF comparison to ScoreMD's reported benchmarks), and **adenylate kinase (AK, n=214)** as the w^delta showcase protein for the open↔closed conformational transition. BBA uses DE Shaw 325 μs MD data; AK uses the 102 frames from Diepeveen (2024) augmented with Atlas MD trajectories.

The code lives in `riemannian-scoremd/` and is written entirely in JAX. It ports Diepeveen's PyTorch manifold primitives to JAX (`src/manifold/pointcloud_jax.py`) and implements the diffusion forward process (`src/diffusion/manifold_sde.py`, `manifold_solvers.py`). The score network (Phase 3) and Fokker-Planck loss (Phase 4) are next.

**Phases**:
- **Phase 1** ✓ — JAX port of all 20 manifold primitives; 18 parity tests pass at 1e-5
- **Phase 2** ✓ — Manifold forward process (wrapped-Gaussian Brownian motion via `s_exp`, with 0.5× doubling compensation); 8/8 BM tests pass
- **Phase 2.5** ✓ — `s_exp` optimisation: **2.6 ms/call on n=214** (target ≤50 ms, achieved 20×). Root cause of original 121ms cost was Python eager dispatch overhead, not XLA compute. Fix: `_build_doubling_fn` inlines `s_geodesic` body into a `@jax.jit` closure, cached in `self._doubling_cache` keyed on (K, tol, max_iter, step_size). Warm-start (linear extrapolation `z=2x₁−x₀`) cuts iterations 47→4.
- **Phase 2.6** (gate) — Data acquisition: DE Shaw chignolin + BBA + Atlas MD AK; pipeline validation tests
- **Phase 3** (next after 2.6) — Tangent-space score network (`TangentScoreModel`): 4-layer MLP outputting horizontal tangent vectors; develop on chignolin, validate on BBA
- **Phase 4** — Manifold Fokker-Planck loss: full Laplace-Beltrami correction derived from Kolmogorov forward equation on Riemannian manifold
- **Phase 5** — End-to-end sampling + baseline comparison vs Xu 2026

---

## Key Advantages

**Physics fidelity from the start.** The metric itself encodes the protein energy landscape, so generated trajectories naturally follow realistic, energy-minimizing paths rather than generic interpolations.

**Dramatic effective dimension collapse.** Realistic conformational ensembles lie on extremely low-dimensional submanifolds within $M$ (e.g., 636-dimensional adenylate kinase conformations collapse to an effective 1D manifold, while 1764-dimensional helicase conformations collapse to approximately 7D). Our score network therefore learns and samples in a much lower-dimensional tangent space.

**Computational efficiency.** Separation-based primitives, combined with JIT caching and warm-starting, reduce geodesic operations to ~2.6 ms per call — fast enough for use in every training step of the forward process.

**Dual utility.** The same trained model supports both unconditional/conditional generation of new conformations *and* consistent, long-timescale molecular dynamics simulation directly on the manifold.

---

## Optional Future Direction: Riemannian Flow Matching

After establishing the diffusion baseline (Phase 3), Riemannian Flow Matching (FM) on the $w^\delta$ manifold is a high-value optional extension.

Standard FM in flat space replaces the noisy SDE forward process with deterministic straight-line interpolants between data and noise. Near $t \approx 0$, the model must recover an exact deterministic vector field with no stochastic smoothing — ScoreMD Appendix A.4 found this causes worse generalization near $t=0$ compared to diffusion.

On the $w^\delta$ manifold, flat interpolants go off-manifold. **Riemannian FM** replaces them with geodesic interpolants $\gamma_{x_0, x_\text{noise}}(t)$ computed via `s_exp`/`s_geodesic`. The on-manifold paths expose the network only to physically valid conformations during training, potentially smoothing the $t=0$ loss landscape. The score/velocity relationship becomes $\nabla_x \log p_t(x) = \frac{1}{1-t} \log_{x_t}(x_1)$.

**Efficiency**: Riemannian FM needs 1–10 NFEs vs 100–500 for diffusion → 10–100× faster sampling. At 2.6 ms/`s_exp`, 10 NFEs ≈ 26 ms/sample. This gate required `s_exp` ≤ 50 ms, now cleared.

**Relevant papers**:
- Lipman et al. (2023). *Flow Matching for Generative Modeling.* ICLR 2023. arXiv:2210.02747
- Klein, Krämer, Noé (2023). *Equivariant Flow Matching.* NeurIPS 2023 — FM on SE(3)-quotient spaces; direct analogue of our setting
- Chen & Lipman (2024). *Flow Matching Guide and Code.* arXiv:2412.06264 — includes Riemannian FM formalism
- Köhler, Klein, Noé (2023). *Flow-Matching: Efficient CG-MD without forces.* JCTC — same application domain

Recommended plan: implement Phase 3 diffusion baseline first, then add FM as a Phase 3b ablation using the same architecture. Ablation table: (a) Euclidean diffusion, (b) Riemannian diffusion, (c) Riemannian FM.

---

## Expected Impact

This project delivers a new class of generative models that finally respect the true geometry of the protein energy landscape. The resulting open-source library will combine the Diepeveen manifold primitives with ScoreMD-style consistency tools, providing a reusable foundation for physics-informed generative modeling in structural biology. Applications range from accelerated transition-path sampling and enhanced-sampling MD to protein design and the creation of high-quality collective variables for large-scale simulations.

In short, we are not merely removing symmetries or learning geometry from data — we are performing **generative modeling that is intrinsically grounded in the physics of the energy landscape**. This represents a principled advance over both generic quotient diffusion and Euclidean coarse-grained approaches.

---

## References

- Diepeveen et al. (2024). *Riemannian geometry for efficient analysis of protein dynamics data.* arXiv:2308.07818
- Plainer et al. (2025). *Molecular dynamics with energy-based diffusion models.* NeurIPS 2025. arXiv:2506.17139
- Xu et al. (2026). *Quotient-space diffusion models.* ICLR 2026. arXiv:2604.21809
- Lipman et al. (2023). *Flow Matching for Generative Modeling.* ICLR 2023. arXiv:2210.02747
- Klein, Krämer, Noé (2023). *Equivariant Flow Matching.* NeurIPS 2023.
- Chen & Lipman (2024). *Flow Matching Guide and Code.* arXiv:2412.06264
