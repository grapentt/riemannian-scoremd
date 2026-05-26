# Riemannian Diffusion on the $w^\delta$-Protein Shape Manifold

*Physics-informed generative modeling and consistent simulation for coarse-grained protein conformations.*

---

## The challenge

Modern diffusion and flow-matching models have dramatically advanced molecular generation, yet nearly all of them operate in flat Euclidean space or generic SE(3) quotients. These approaches treat protein conformations as unstructured point clouds and therefore fail to capture the **intrinsic non-linear geometry** imposed by the underlying energy landscape. Three concrete failure modes follow:

- Generated conformations during large-scale transitions are often unphysical or high-energy
- Score functions trained in flat space break down when repurposed as force fields for actual molecular dynamics (Plainer et al., 2025)
- Poor generalization across proteins with different sizes and topologies

Even recent quotient-space diffusion methods (Xu et al., 2026) address symmetry removal without embedding the physics of the energy landscape itself.

---

## The approach

We build the first diffusion generative model that operates **directly on a physics-informed Riemannian manifold** engineered for protein conformational dynamics.

### The manifold

The configuration space is the smooth quotient manifold

$$M = P(d,n)/E(d)$$

of centered, non-colliding Cα point clouds modulo rigid-body motions (Diepeveen et al., 2024, Theorem 4.2). We equip $M$ with the **$w^\delta$-metric** — a complete, energy-landscape-derived separation metric built from logarithmic ratios of pairwise inter-atomic distances plus a radius-of-gyration correction term:

$$w^\delta([X],[Y])^2 = \frac{1}{2}\sum_{ij} \left(\log \frac{\|x_i - x_j\|^2}{\|y_i - y_j\|^2}\right)^2 + \delta \left(\log \frac{\det G_X}{\det G_Y}\right)^2$$

This metric was reverse-engineered so that its geodesics closely approximate the energy-minimizing paths observed in MD trajectories.

### Tractable geometry via separation

A key ingredient is the **separation construction** (Diepeveen et al., Definition 3.1 and Theorems 3.2–3.4), which provides provably accurate (third-order), closed-form approximations to the Riemannian log map and enables geodesic and exponential map computations via simple, linearly convergent Riemannian gradient descent. This eliminates the usual computational bottleneck of manifold diffusion.

### Fokker–Planck consistency on the manifold

We integrate the state-of-the-art consistency techniques of Plainer et al. (2025): Fokker–Planck regularization that enforces dynamical consistency between the learned score and the probability flow, and a conservative (energy-based) score parameterization. The Euclidean FP residual must be corrected for the manifold geometry via the Laplace–Beltrami operator:

$$\nabla_M \cdot s = \nabla_{\mathbb{R}^{nd}} \cdot s - s \cdot \tfrac{1}{2}\,\nabla_x \log \det g(x)$$

where $g(x)$ is the $w^\delta$ metric tensor — the core theoretical contribution of this project.

The resulting score can be used both for **generation** and directly as a physically meaningful force field for stable **Langevin dynamics on the manifold**.

---

## Key advantages

**Physics fidelity from the start.** The metric encodes the protein energy landscape, so generated trajectories naturally follow realistic, energy-minimizing paths.

**Dramatic effective dimension collapse.** Realistic conformational ensembles lie on extremely low-dimensional submanifolds of $M$. For adenylate kinase (214 Cα atoms, $\dim M = 636$), the ensemble collapses to an effective 1D manifold along the open/closed transition. The score network learns and samples in this much smaller tangent space.

**Computational efficiency.** Separation-based primitives reduce expensive geodesic operations to lightweight gradient-descent steps, enabling training and sampling on modest hardware.

**Dual utility.** The same trained model supports unconditional/conditional generation *and* consistent long-timescale molecular dynamics directly on the manifold.

---

## Relation to prior work

| | Geometry | Energy-informed metric | FP consistency |
|---|---|---|---|
| ScoreMD (Plainer et al., 2025) | Flat $\mathbb{R}^n$ | No | Yes |
| Quotient Diffusion (Xu et al., 2026) | $\mathbb{R}^{3n}/SE(3)$ | No | No |
| **This work** | $P(d,n)/E(d)$ with $w^\delta$ | **Yes** | **Yes** |

---

## Repository structure

```
src/
  manifold/
    pointcloud_jax.py    — ShapeManifold: the w^delta manifold in JAX
  diffusion/             — (Phase 2) ManifoldVP forward process, Euler–Maruyama
  models/                — (Phase 3) TangentScoreModel, conservative parameterization
  training/              — (Phase 3) Riemannian DSM loss, training loop
  sampling/              — (Phase 5) reverse SDE, transition path generation
  evaluation/            — (Phase 5) metrics, baseline comparison

tests/
  test_port_parity.py    — numerical correctness checks for ShapeManifold
  test_brownian_motion.py — (Phase 2) BM validation: gyration, distances, diffusion coeff
```

---

## Development status

| Phase | Description | Status |
|---|---|---|
| 0 | Environment setup, reference reproduction | Complete |
| 1 | $w^\delta$ `ShapeManifold` in JAX, correctness verified | **Complete** |
| 2 | Manifold forward process (wrapped-Gaussian BM) | Next |
| 3 | Score network in tangent bundle, Riemannian DSM loss | — |
| 4 | Manifold Fokker–Planck loss (Laplace–Beltrami correction) | — |
| 5 | End-to-end sampling, transition paths, evaluation | — |
| 6 | Scaling to helicase, ablations, arXiv preprint | — |

---

## Running correctness checks

Requires a Python environment with both `torch` (for the reference) and `jax`. The tested configuration is `/tmp/torch_refs_venv`:

```bash
# Recreate if /tmp is cleared:
python3.11 -m venv /tmp/torch_refs_venv
pip install "torch==2.0.1" "numpy<2" "jax[cpu]==0.4.30" "jaxlib==0.4.30"

# Run:
/tmp/torch_refs_venv/bin/python tests/test_port_parity.py
```

All 18 geometric operations pass at max|diff| < 1e-4 (float32). All `jax.jit` checks pass.

---

## References

- Diepeveen et al. (2024). *Riemannian geometry for efficient analysis of protein dynamics data.* arXiv:2308.07818
- Plainer et al. (2025). *Molecular dynamics with energy-based diffusion models.* NeurIPS 2025. arXiv:2506.17139
- Xu et al. (2026). *Quotient-space diffusion models.* ICLR 2026. arXiv:2604.21809
- Huang et al. (2022). *Riemannian diffusion models.* NeurIPS 2022. arXiv:2208.07949
- Song et al. (2021). *Score-based generative modeling through stochastic differential equations.* ICLR 2021. arXiv:2011.13456
