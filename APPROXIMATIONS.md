# Approximation Map: riemannian-scoremd

*Last updated: 2026-05-29. Phases 3.65–3.67 settled. All numbers reflect post-bugfix code (metric_tensor transpose fix). Run `scripts/approximation_audit.py` to re-verify numbers.*

This document catalogues every place in the pipeline where an approximation is made instead of an exact computation, why the choice was made, and what the measured cost is on real BBA data ($n=28$, $\alpha=1.0$, 63k frames).

A background section first defines all geometric quantities precisely. The approximation entries (A–D) that follow reference these definitions.

---

## Background: geometric setup

> **Note (2026-05-28 unification pass)**: The complete geometric background (quotient $\mathcal{M} = \mathrm{P}(d,n)/\mathrm{E}(d)$, vertical/horizontal spaces and the two projectors, $w^\delta$ separation, explicit derivation of $H = A + \delta B$, prelog vs. s_log, s_exp, diffusion/DSM on the manifold, and the critical H-vs-G vertical-space mismatch) is now in the single source of truth:
>
> **→ `riemannian-scoremd/THEORY.md`** (unified v2.0).
>
> This document (`APPROXIMATIONS.md`) is the **canonical numerical companion only**. It contains the four measured-impact sections (A–D), the approximation hierarchy, and the implementation status table. Every number below is anchored in the derivations of the unified THEORY. Read the two documents together.

### The protein shape manifold (short summary — full derivation in THEORY §§1–4)

A protein conformation with $n$ Cα atoms is a matrix $X \in \mathbb{R}^{n \times d}$ (rows = atom positions, $d = 3$). Two matrices $X$ and $Y$ represent the **same shape** if one is a rigid-body image of the other: $Y = XO^\top + \mathbf{1}t^\top$ for some rotation $O \in \mathrm{SO}(d)$ and translation $t \in \mathbb{R}^d$.

The space of distinct point clouds is:

$$
\mathrm{P}(d,n) = \bigl\{ X \in \mathbb{R}^{n \times d} \mid x_i \neq x_j \text{ for all } i \neq j \bigr\}
$$

The Euclidean group $\mathrm{E}(d) = \mathrm{SO}(d) \ltimes \mathbb{R}^d$ acts on it by $(O, t) \cdot X = XO^\top + \mathbf{1}t^\top$. The **shape manifold** is the quotient:

$$
\mathcal{M} = \mathrm{P}(d,n)/\mathrm{E}(d)
$$

By Theorem 4.2 of Diepeveen (2024), $\mathcal{M}$ is a smooth manifold of dimension:

$$
\dim \mathcal{M} = nd - \frac{d(d+1)}{2}
$$

The subtracted term $d(d+1)/2 = d$ translations $+ \, d(d-1)/2$ rotations counts the rigid-body degrees of freedom. For $n=214$ (adenylate kinase), $d=3$: $\dim \mathcal{M} = 642 - 6 = 636$. For $n=28$ (BBA), $d=3$: $\dim \mathcal{M} = 84 - 6 = 78$.

We write $[X]$ for the equivalence class of $X$. In practice we always work with a chosen representative $X \in [X]$ (e.g. centered coordinates) and lift all computations to $\mathrm{P}(d,n)$.

### Two $\mathrm{E}(d)$-invariant quantities

Two quantities built from $X$ are invariant under $\mathrm{E}(d)$ and serve as building blocks for the metric.

**Pairwise squared distances**: for residues $i, j$,

$$
r^2_{ij}(X) = \|x_i - x_j\|^2
$$

Rotating or translating $X$ leaves all pairwise distances unchanged, so $r_{ij}$ is well-defined on $[X]$.

**Gyration matrix**: the $d \times d$ symmetric positive-definite matrix

$$
G_X = \sum_i (x_i - \bar{x})(x_i - \bar{x})^\top, \qquad \bar{x} = \frac{1}{n}\sum_i x_i
$$

Under $(O,t) \cdot X$: $G_X \mapsto O G_X O^\top$, so $\det G_X$ is invariant. Geometrically, $G_X$ is the covariance matrix of the atom positions; its eigenvalues are the squared principal radii of gyration.

### The $w^\delta$ distance (Corollary 5.2.1, Eq. 17)

Instead of the quotient of the Euclidean metric, Diepeveen (2024) defines a distance on $\mathcal{M}$ directly from the two invariants above:

$$
w^\delta([X],[Y])^2
= \frac{1}{2}\sum_{i \neq j} \left[\log\frac{r^2_{ij}(X)}{r^2_{ij}(Y)}\right]^{\!2}
+ \delta \left[\log\frac{\det G_X}{\det G_Y}\right]^{\!2}
$$

The $\tfrac{1}{2}$ accounts for the symmetric double-counting $(i,j)=(j,i)$, so this equals

$$
\sum_{i < j} \left[\log\frac{r_{ij}(X)}{r_{ij}(Y)}\right]^{\!2}
+ \delta \left[\log\frac{\det G_X}{\det G_Y}\right]^{\!2}
$$

The parameter $\delta$ (called `alpha` in the code) weights the gyration term. Default: $\delta = 0.1$; adenylate kinase experiments use $\delta = 1.0$.

**Interpretation of each term:**
- *Pairwise term*: penalises log-ratio deviations of inter-residue separations. It is scale-invariant: uniform scaling $X \to \lambda X$ contributes $2\log\lambda$ per pair equally.
- *Gyration term*: penalises differences in overall "spread" ($\det G$ is the squared volume of the ellipsoid of inertia). Setting $\delta > 0$ breaks scale invariance and adds sensitivity to global size changes.

**$w^\delta$ is a complete metric** (Corollary 5.2.1): it satisfies all metric axioms (positivity, symmetry, triangle inequality) and induces the correct topology on $\mathcal{M}$. It is not merely an approximation to some pre-existing "true" distance — it is a bona fide distance function in its own right.

**The metric tensor $H$ is derived from $w^\delta$** (not the other way around): $H$ is the Hessian of $\tfrac{1}{2}(w^\delta)^2$ along the diagonal, or equivalently, the object defined by property (iii) of a separation. The geodesic distance $d_\mathcal{M}$ is then the distance you get by *integrating path lengths with $H$* — it is a third derived object, harder to compute than $w^\delta$ itself.

**These two distances agree to third order** (Theorem 3.2, Eq. 1):

$$
w^\delta([X],[Y])^2 = d_{\mathcal{M}}([X],[Y])^2 + \mathcal{O}\!\left(d_{\mathcal{M}}([X],[Y])^3\right)
\qquad \text{as } [Y] \to [X]
$$

They share the same local geometry (same $H$ at every point) but differ at third order when moving away from a point — the same way any two metrics with a common Hessian at the diagonal must diverge at cubic order.

### The metric tensor $H = A + \delta B$ (Theorem 5.3, Eqs. 18–21)

The metric tensor $H$ is the Hessian of $\tfrac{1}{2}(w^\delta)^2$ evaluated along the diagonal, restricted to the horizontal space. Integrating path lengths with $H$ gives the geodesic distance $d_\mathcal{M}$, which differs from $w^\delta$ only at third order (Theorem 3.2).

At a representative $X$, $H$ is an $nd \times nd$ symmetric positive semi-definite matrix, written in $d \times d$ blocks $H_{ij}$ (one block per pair of residues $i,j \in \{1,\ldots,n\}$). It decomposes as $H = A + \delta B$.

**Component $A$** — separation part. Block $(i,j)$:

$$
A_{ij} = \begin{cases}
\displaystyle\sum_{k \neq i} \frac{(x_i - x_k)(x_i - x_k)^\top}{r^4_{ik}(X)} & i = j \\[10pt]
\displaystyle -\frac{(x_i - x_j)(x_i - x_j)^\top}{r^4_{ij}(X)} & i \neq j
\end{cases}
$$

This has a graph-Laplacian structure on the complete residue graph: each edge $(i,j)$ contributes a rank-1 outer product weighted by $r^{-4}$.

**Component $B$** — gyration correction part. Block $(i,j)$:

$$
B_{ij} = 4\,\bigl(G_X^{-1}(x_i - \bar{x})\bigr) \otimes \bigl(G_X^{-1}(x_j - \bar{x})\bigr)
$$

where $\otimes$ denotes the outer product. $B$ has rank at most $d$: it is a sum of $d$ rank-1 terms, one per spatial dimension.

$H$ is positive semi-definite. Its kernel is exactly the $d(d+1)/2$-dimensional **vertical space** (see below).

### Horizontal and vertical tangent spaces

The tangent space at $X \in \mathrm{P}(d,n)$ splits under the metric $H$ into two orthogonal subspaces.

**Vertical space** $\mathcal{V}_X$ — infinitesimal rigid-body motions (Appendix B.3, Eq. 66):

$$
\mathcal{V}_X = \bigl\{ SX + \mathbf{1}t^\top \;\big|\; S + S^\top = 0 \in \mathbb{R}^{d \times d},\; t \in \mathbb{R}^d \bigr\}
$$

These are infinitesimal rotations (anti-symmetric $S$, $d(d-1)/2$ dimensions) and translations ($t$, $d$ dimensions). A vector in $\mathcal{V}_X$ moves $X$ along its $\mathrm{E}(d)$ orbit — it changes the representative but not the shape. $\dim \mathcal{V}_X = d(d+1)/2 = 6$ for $d=3$.

**Horizontal space** $\mathcal{H}_X$ — shape deformations:

$$
\mathcal{H}_X = \bigl\{ \Xi \in \mathbb{R}^{n \times d} \;\big|\; \langle \Xi, V \rangle_H = 0 \text{ for all } V \in \mathcal{V}_X \bigr\}
$$

Concretely: $\sum_i \xi_i = 0$ (zero net translation) and $\sum_i \xi_i x_i^\top = \bigl(\sum_i \xi_i x_i^\top\bigr)^\top$ (zero net torque). $H$ restricted to $\mathcal{H}_X$ is positive definite.

The projection onto $\mathcal{H}_X$ is `horizontal_projection_tvector` in the PyTorch code, called `project_G` in this codebase. It removes the vertical component from any ambient tangent vector.

### The exact Riemannian log map (intractable)

Given shapes $[X]$ and $[Y]$, the exact log map $\log_{[X]}([Y]) \in \mathcal{H}_X$ is the initial velocity of the geodesic from $[X]$ to $[Y]$:

$$
\gamma(0) = [X], \quad \gamma(1) = [Y], \quad
\dot{\gamma}(0) = \log_{[X]}([Y]), \quad
\|\dot{\gamma}(0)\|_H = d_{\mathcal{M}}([X],[Y])
$$

Computing this requires solving a boundary-value ODE on $\mathcal{M}$. No closed form is known for the $w^\delta$ manifold. All three quantities s_prelog, s_log, and s_exp are approximations to — or building blocks for — this exact object.

### s_prelog: the flat gradient of $\tfrac{1}{2}(w^\delta)^2$

On any Riemannian manifold, the true Riemannian distance satisfies (away from the cut locus):

$$
-\tfrac{1}{2}\,\nabla_p\, d_\mathcal{M}(p,q)^2 = \log_p(q)
$$

This is a standard result of Riemannian geometry, not specific to flat space (in $\mathbb{R}^n$ it simply gives $q - p$).

**Why this identity does not help directly.** Both sides are simultaneously intractable. To evaluate the left-hand side you need $d_\mathcal{M}$; to compute $d_\mathcal{M}$ you must already solve the geodesic boundary-value problem — the very problem you are trying to avoid. The identity is circular: it tells you that *if* you could compute $d_\mathcal{M}$, its Riemannian gradient would give you $\log_p$. It provides no shortcut.

**The paper's move** (Definition 3.1(iii), Eq. 3): replace $d_\mathcal{M}$ with the separation $w^\delta$ — a complete metric with the same local geometry but a closed-form expression. The approximate log map is then defined as the Riemannian gradient of $\tfrac{1}{2}(w^\delta)^2$:

$$
\log_p^w(q) \;:=\; -\,\mathrm{grad}_H\,\tfrac{1}{2}\,w^\delta(p,q)^2
\;=\; -\,H^{-1}\,d\!\left[\tfrac{1}{2}\,w^\delta(p,q)^2\right] \;\in\; T_p\mathcal{M}
$$

The error is $\mathcal{O}(d_\mathcal{M}^2)$ (Corollary 3.2.1): because $w^\delta$ and $d_\mathcal{M}$ share the same Hessian, their gradients agree to first order, and the cubic discrepancy in the distances propagates to a quadratic error in the log map.

**Computing the Riemannian gradient: raising the index.** The second equality above is the key step. For any smooth $f: \mathcal{M} \to \mathbb{R}$, the differential $df \in T^*_p\mathcal{M}$ is a covector: it has components $\partial_i f$ and lives in the *cotangent* space (lower index). The Riemannian gradient $\mathrm{grad}_g f \in T_p\mathcal{M}$ is the unique tangent *vector* satisfying

$$
g(\mathrm{grad}_g f,\, v) = df(v) \quad \forall\, v \in T_p\mathcal{M},
$$

which gives $(\mathrm{grad}_g f)^i = g^{ij}\partial_j f$, or in matrix form $g^{-1}df$. This is called **raising the index** or applying the **sharp isomorphism** $\sharp: T^*\mathcal{M} \to T\mathcal{M}$. In physics notation, $df$ has a lower index ($\omega_i$) and $\mathrm{grad}_g f$ has an upper index ($v^i = g^{ij}\omega_j$).

This decomposes the computation into two steps:

1. **Compute the exterior derivative** $d\!\left[\tfrac{1}{2}(w^\delta)^2\right]$ — a covector, obtainable by ordinary partial differentiation of the closed-form $w^\delta$ formula. This is `s_prelog`. Cost: $\mathcal{O}(n^2 d)$, no metric inversion.

2. **Raise the index** with $H^{-1}$ — apply the sharp isomorphism to convert the covector to a tangent vector. This is `s_log`. Cost: $\mathcal{O}((nd)^3)$, requires eigendecomposition of $H$.

**Explicit formula for s_prelog** (Appendix D.3, Eq. 119). For representatives $X \in [X]$, $Y \in [Y]$, the $i$-th block (residue $i$) is:

$$
\mathrm{prelog}_i(X,Y)
= -\sum_{j \neq i} \log\frac{r_{ij}(X)}{r_{ij}(Y)} \cdot \frac{x_i - x_j}{r^2_{ij}(X)}
\;-\; 2\delta\,\log\frac{\det G_X}{\det G_Y} \cdot G_X^{-1}(x_i - \bar{x})
$$

The first sum is the chain rule through the pairwise log-ratio terms; the second is the chain rule through the gyration log-ratio term.

**Note on horizontality**: because $w^\delta$ is $\mathrm{E}(d)$-invariant, differentiating $w^\delta((O,t)\cdot X, Y) = w^\delta(X,Y)$ at the identity shows that the exterior derivative is automatically Euclidean-orthogonal to all vertical vectors. So s_prelog already lives in $\mathcal{H}_X^E = \mathcal{V}_X^{\perp_E}$ without any projection. The `project_G` call in the training pipeline is numerical cleanup only (floating-point leakage $< 10^{-7}$).

### s_log: the approximate Riemannian log map (Eq. 119)

`s_log` is step 2 of the decomposition above: it applies $H^{-1}$ to s_prelog, raising the covector index to produce the full Riemannian gradient:

$$
\log_{[X]}^w([Y]) = H^\dagger_{\mathrm{horiz}} \cdot \mathrm{prelog}(X,Y)
$$

where $H^\dagger_{\mathrm{horiz}}$ is the pseudo-inverse of $H$ restricted to $\mathcal{H}_X$. In practice this is computed via eigendecomposition of $H$:

$$
H = Q\Lambda Q^\top
\quad\Longrightarrow\quad
H^\dagger_{\mathrm{horiz}} = Q_{\mathrm{horiz}}\,\Lambda_{\mathrm{horiz}}^{-1}\,Q_{\mathrm{horiz}}^\top
$$

where "horiz" means we keep only the $nd - d(d+1)/2$ largest eigenvectors, discarding the $d(d+1)/2$ near-zero eigenvectors that span $\mathcal{V}_X$.

**Accuracy** (Corollary 3.2.1, Eq. 4):

$$
\bigl\|\log_{[X]}([Y]) - \log_{[X]}^w([Y])\bigr\|_H
= \mathcal{O}\!\left(d_{\mathcal{M}}([X],[Y])^2\right)
\qquad \text{as } [Y] \to [X]
$$

The approximate log map matches the exact one to **second order** in the geodesic distance.

**Computational cost**: $\mathcal{O}((nd)^3)$ — requires eigendecomposition of the $nd \times nd$ matrix $H$. For $n=28$ (BBA): an $84 \times 84$ matrix; for $n=214$ (adenylate kinase): a $642 \times 642$ matrix.

**Relationship between s_prelog and s_log**: s_prelog is the "raw" ambient-space gradient; s_log applies the metric correction $H^{-1}$ to convert it to a proper Riemannian tangent vector. They agree when $H \approx I$; they differ substantially when $H$ has large condition number (which it does on real data — see approximation A).

### s_exp: the approximate exponential map

The exact exp map $\exp_{[X]}(V)$ — follow the geodesic with initial velocity $V \in \mathcal{H}_X$ for unit time — also has no closed form.

`s_exp` uses an iterative **geodesic doubling** scheme (Eqs. 5–6 of the paper). The key observation is that the midpoint $[Z]$ of the geodesic segment from $[X]$ to $[Y]$ is the minimiser:

$$
[Z] = \underset{[R]}{\arg\min}\;\bigl[w([X],[R])^2 + w([R],[Y])^2\bigr]
$$

Starting from $X$ and a direction $V$, we set an initial target $Y$ (via a flat Euler step) and then repeatedly refine the midpoint by gradient descent on this criterion. `s_geodesic(X,Y)` implements one such step; `s_exp` calls it in a doubling loop until convergence.

The gradient of the midpoint criterion is $-\tfrac{1}{2}\nabla w^2$, i.e. s_prelog — this is why s_prelog appears inside `s_geodesic`.

### Diffusion and score matching (brief background)

A **diffusion model** defines a forward noising process that gradually turns a data sample $x_0$ into Gaussian noise:

$$
x_t = \alpha(t)\,x_0 + \sigma(t)\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I), \qquad t \in [0,1]
$$

where $\alpha(t) \to 1$, $\sigma(t) \to 0$ as $t \to 0$ (nearly clean data) and $\alpha(t) \to 0$, $\sigma(t) \to 1$ as $t \to 1$ (pure noise). The schedule $(\alpha, \sigma)$ is fixed in advance (variance-preserving in this codebase).

The **score function** is the gradient of the log-density of $x_t$:

$$
s^*(x, t) = \nabla_x \log p_t(x)
$$

A neural network $s_\theta$ is trained to approximate $s^*$. The **denoising score matching (DSM)** loss is:

$$
\mathcal{L} = \mathbb{E}_{t,\,x_0,\,x_t}\!\left[\,\beta(t)\,\bigl\|s_\theta(x_t, t) - s_{\mathrm{true}}(x_t, x_0)\bigr\|^2\right]
$$

where $s_{\mathrm{true}}$ is the **score target** — the direction pointing from the noised $x_t$ back to the clean $x_0$. For Gaussian noise in flat space: $s_{\mathrm{true}} = -(x_t - \alpha(t)x_0)/\sigma(t)^2$.

On the manifold $\mathcal{M}$, $x_t$ is sampled by geodesic noising via s_exp, and the score target becomes:

$$
s_{\mathrm{true}} = -\frac{\log_{x_t}(x_0)}{\sigma(t)^2}
$$

i.e., the (negative) log map from the noised point back toward the clean point, rescaled. Since the exact log map is intractable, we use s_log or s_prelog as a substitute — this is approximation A.

---

> **Design principle**: The goal is to use exact geometric primitives everywhere. Every approximation in this document is a *temporary compromise* forced by a concrete, measured numerical or computational blocker. Each entry below states precisely what the blocker is, what would need to change to remove it, and how to reproduce the measurement that established the blocker. When a blocker is resolved, the approximation should be removed.

## Quick reference

| # | Approximation | Where | Measured blocker | To remove |
|---|---|---|---|---|
| C | Ambient Gaussian noising instead of geodesic s_exp | removed — s_exp now used | ~25×σ off-manifold error at t=0.5 | **Done** |
| A | `prelog` instead of `s_log` in `score_target` | `manifold_sde.py:score_target` (smoke-test path only) | eigh per sample in Python loop ~12× slower; production path already uses s_log | **Already s_log in production** |
| D | Euclidean norm instead of g-norm in DSM loss | `score_loss.py:riemannian_dsm_loss_from_noised` | cond(H) ~2.4e6, NaN gradients; scale-invariant, normalization doesn't help | Requires preconditioned optimizer |
| B | `prelog` gradient in `s_geodesic` inner loop | `pointcloud_jax.py:s_geodesic` | eigh of (nd×nd) inside JIT fori_loop: O((nd)³) per step × K×max_iter steps | AK-scale preconditioned s_log, or separate s_geodesic from JIT boundary |

Reproduce all numbers: `python riemannian-scoremd/scripts/approximation_audit.py`

---

## C. "Not using exp" — ambient vs geodesic noising

**Status: resolved. `s_exp` is used.**

The original concern (Phase 2) was whether computing the forward noising as a flat Gaussian in $\mathbb{R}^{nd}$ was acceptable:
```
x_t_flat = alpha(t) * x0 + sigma(t) * eps       # Euclidean, off-manifold
x_t_geod = s_exp(x0, sigma(t) * v_h_unit)       # Geodesic, on-manifold  ← current
```

**Measured on BBA** (64 frames, 5 time points):

| $t$ | $\sigma(t)$ | $w^\delta(x_t^{\mathrm{flat}},\, x_t^{\mathrm{geod}})$ | ratio / $\sigma$ |
|---|---|---|---|
| 0.10 | 0.322 | 1.56 | **4.8×** |
| 0.30 | 0.777 | 9.78 | **12.6×** |
| 0.50 | 0.960 | 23.5 | **24.5×** |
| 0.70 | 0.996 | 35.8 | **36.0×** |
| 0.90 | 1.000 | 38.4 | **38.4×** |

The flat $x_t$ lands **25–38 noise-scales away** from the manifold-geodesic $x_t$ at moderate-to-large $t$. This is catastrophic: the score model would be trained on conformations that are not geometrically related to real proteins and the learned score field would have no meaning on the actual manifold. This was always a hard requirement, not a tradeoff.

**Cost of using s_exp**: 2.6 ms/call on n=214 CPU (JIT-cached, warm-start). Acceptable.

---

## A. `s_log` vs `prelog` in `score_target`

**Goal**: Use `s_log` (the true Riemannian gradient of $\tfrac{1}{2}(w^\delta)^2$) everywhere.

**Status**: `s_log` IS the default and IS used in all production paths. The only remaining use of `prelog` is in `train()`, the online smoke-test training loop — not in precomputed data generation or actual training runs.

**The prelog path exists for one narrow reason**: `train()` calls `prepare_batch` in a Python loop, once per sample. `s_log` requires eigh of the $(nd \times nd)$ metric tensor $H$ per sample. For BBA ($nd=84$) this is ~12× slower per call in a tight Python loop, making a quick smoke test take ~12× longer. `train()` explicitly passes `use_slog=False` with a comment. All production training goes through `precompute_noised_data.py` + `train_from_precomputed`, which calls `score_target` once per frame at precompute time — cost is amortized, and `s_log` is used.

**To remove entirely**: Remove the `use_slog=False` override in `train()`. Accept the ~12× slower smoke-test loop, or replace `train()` with a precompute-first workflow. This is a developer-convenience tradeoff, not a geometric one.

### What the approximation costs (measured)

Reproduce with `scripts/approximation_audit.py` §A:

| Quantity | Value |
|---|---|
| $\cos(s_{\mathrm{prelog},h},\, s_{\log})$ | **0.81 ± 0.05** |
| Angular difference | **~35°** |
| Norm ratio $\|s_{\mathrm{prelog},h}\| / \|s_{\log}\|$ | **0.18** (prelog ≈ 6× smaller) |

`s_log = H^{-1} \cdot \mathrm{prelog}` is the true Riemannian gradient: it raises the covector index, correcting for the non-uniform eigenvalue structure of $H$. `prelog + project_G` is the horizontal Euclidean gradient — it ignores the metric's direction rescaling. Both share the same DSM fixed point (the model trains to the same score function in expectation), but per-sample training targets differ by ~35°.

### Status of `s_log` after transpose bug fix

The `metric_tensor` transpose bug (see THEORY.md §12) has been fixed. After the fix:
- $H$ is PSD on all tested BBA frames (float64 noise only, min eigenvalue ~-1.6e-10)
- The H-eigenvector cut cleanly separates $\mathcal{V}_X$: rotation residuals $\|H v_{\rm rot}\|/\|v_{\rm rot}\| \approx 4 \times 10^{-9}$
- `s_log` vertical leakage under `project_G`: $4.3 \times 10^{-8}$ (machine zero, no repair needed)

**`s_log` is now geometrically valid.** The pre-fix claims ("H is indefinite") were wrong: the indefiniteness was entirely due to the transpose bug, present in both implementations and therefore invisible to parity tests. See THEORY.md §12 for the full root-cause analysis.

---

## B. `prelog` gradient in `s_geodesic` inner loop

**Goal**: Use `s_log` (the true Riemannian gradient of $w^\delta$) as the gradient step in `s_geodesic`, making geodesic computation fully metric-aware.

**Status**: `prelog` is used instead. This is a genuine active approximation. It cannot be removed without solving a concrete computational blocker.

**The exact geometric primitive**: The midpoint $[Z]$ of the geodesic from $[X]$ to $[Y]$ is the minimiser of $f(Z) = w([X],Z)^2 + w(Z,[Y])^2$. The true Riemannian gradient step is:
$$
Z \leftarrow \exp_Z\!\left(-\eta \operatorname{grad}_g f(Z)\right)
= Z - \eta H^{-1} \nabla_E f(Z) + O(\eta^2)
$$
i.e., the gradient $\nabla_E f = -\mathrm{prelog}(Z,X) - \mathrm{prelog}(Z,Y)$ must be raised with $H^{-1}$ before the step. This is exactly `s_log`.

**The approximation used**: The fast path skips $H^{-1}$ and steps directly in the Euclidean gradient direction:
$$
Z \leftarrow Z + \eta \left[(1-\tau)\,\mathrm{prelog}(Z,X) + \tau\,\mathrm{prelog}(Z,Y)\right]
$$
Both converge to the same geodesic midpoint (same minimum of $f$), but the Euclidean gradient step does not respect the metric's curvature, taking larger steps in low-curvature directions and smaller in high-curvature ones.

**Why we cannot use s_log here (the concrete blocker)**: `s_geodesic` runs inside `_build_doubling_fn`, a `jax.lax.fori_loop` compiled into a single XLA kernel by `jax.jit`. Every operation inside the loop must be traced statically. `s_log` requires `jnp.linalg.eigh` of the $(nd \times nd)$ matrix $H$ at every iteration. The cost is:

$$
\text{cost per doubling step} = K \times \text{max\_iter} \times \mathcal{O}((nd)^3)
$$

For BBA ($nd = 84$): $84^3 \approx 590\,000$ — marginal, ~12× slower per step but potentially feasible.  
For adenylate kinase ($nd = 642$): $642^3 \approx 265\,000\,000$ — per doubling step, before even counting $K$ and max\_iter. This would make a single `s_exp` call take seconds inside XLA. Completely prohibitive.

Since the codebase must scale to AK ($n=214$), the exact path cannot be used inside the JIT boundary as currently structured.

**What would need to change to remove this approximation**:

1. **For BBA-only**: Set `use_separation_grad=False` in `_build_doubling_fn`. Measure the per-step cost inside JIT and verify total `s_exp` latency remains acceptable. `scripts/approximation_audit.py §B` gives the geometric cost; timing is measured there too.

2. **For AK-scale**: Either (a) precompute $H^{-1}$ outside the loop and pass it as a static argument (valid if $x$ changes slowly across doubling steps — not generally true), or (b) use a diagonal or low-rank approximation to $H^{-1}$ that avoids the full eigh, or (c) restructure `s_exp` to call `s_geodesic` from Python (outside JIT), accepting the ~80ms per call that this incurs (feasible only at precompute time, not during training).

### What the approximation costs (measured)

Reproduce with `scripts/approximation_audit.py` §B (16 BBA frame pairs, tol=$10^{-3}$):

| Metric | Value |
|---|---|
| $w^\delta(z_{\mathrm{fast}},\, z_{\mathrm{exact}})$ at midpoint | **1.11 ± 0.3** |
| as % of base distance $d(x, y)$ | **13.8%** (max 25%) |
| Fast path timing (cold, Python eager) | ~150 ms |
| Exact path timing (cold, Python eager) | ~108 ms |

Note: both are slow when called outside JIT. Inside `s_exp`'s JIT cache the fast path runs in ~1 ms per doubling step.

**Impact on training**: `s_geodesic` is never called directly in the training or sampling pipeline — it is only used as the inner loop of `s_exp`, which is itself called from two places:

1. **`ManifoldVP.marginal_prob`** ([manifold_sde.py:137–146](src/diffusion/manifold_sde.py#L137)): computes $x_t = s_\exp(x_0, \tfrac{1}{2}\alpha\sigma v)$ during training (or precompute). The 14% deviation means noised points $x_t$ land ~14% off the exact geodesic position — this affects which region of the manifold the model is trained on, but not the score target direction at that $x_t$.

2. **`ManifoldEulerMaruyama.step`** ([manifold_solvers.py:83–111](src/diffusion/manifold_solvers.py#L83)): each reverse-diffusion step uses `s_exp` to move along the score direction during sampling. The same 14% deviation applies here, potentially compounding over many steps.

The downstream impact on model quality (whether the learned score field or generated samples are meaningfully degraded) has not been isolated — this requires a controlled ablation comparing training and sampling runs with fast vs exact paths.

---

## D. Euclidean norm vs $g$-norm in DSM loss

**Goal**: Use the Riemannian $g$-norm everywhere in the DSM loss, weighting the score residual by the full metric tensor $H$ at each $x_t$.

**Status**: Euclidean norm is used instead. This is a genuine active approximation. It cannot be removed without solving a concrete numerical blocker.

**The exact geometric primitive**: The theoretically correct DSM loss measures the score residual in the Riemannian inner product induced by $H$:

$$
\mathcal{L}_g = \mathbb{E}\!\left[\,\beta(t)\,\bigl\|s_\theta(x_t) - s_{\mathrm{true}}\bigr\|^2_{g(x_t)}\right]
= \mathbb{E}\!\left[\,\beta(t)\,(s_\theta - s_{\mathrm{true}})^\top H(x_t) (s_\theta - s_{\mathrm{true}})\right]
$$

where the residual is taken in the horizontal tangent space. This is the natural objective on a Riemannian manifold: it weights the score error by the local geometry at $x_t$, giving larger penalty to errors in high-curvature directions of $w^\delta$ space.

**The approximation used**: The Euclidean norm on the projected horizontal residual:

$$
\mathcal{L}_E = \mathbb{E}\!\left[\,\beta(t)\,\bigl\|\mathrm{project}_G(s_\theta(x_t)) - \mathrm{project}_G(s_{\mathrm{true}})\bigr\|^2_E\right]
$$

This is `riemannian_dsm_loss_from_noised` with `use_riemannian_norm=False` (the default).

**Why we cannot use the g-norm here (the concrete blocker)**: The g-norm path (`use_riemannian_norm=True`) exists in `score_loss.py` but causes NaN gradients during backpropagation in practice. The root cause is the extreme condition number of $H$:

- Mean cond($H$) on BBA data: **~2.4×10⁶** (post-bugfix, float64)
- Max cond($H$) on BBA data: **~2.5×10⁷**

Using $H$ directly in the loss means the smallest horizontal eigenvalue (~1×10⁻⁶ of the largest) gets amplified by ~10⁶ when $H^{-1}$ is applied in the backward pass. This produces gradient magnitudes of order 10⁶ for the low-curvature directions, causing numerical overflow and NaN propagation.

**Critical**: the condition number is **scale-invariant** (it is a ratio of eigenvalues — each eigenvalue scales as $\lambda^{-4}$ under $x \to \lambda x$, so the ratio is unchanged). Normalizing frames to unit gyration radius does not reduce cond($H$). This was verified directly on BBA data — normalization leaves the condition number unchanged.

**What would need to change to remove this approximation**:

1. **Preconditioned optimizer**: Use a preconditioned gradient step that normalizes by $H^{-1}$ — in effect, use the $g$-norm loss but divide each gradient direction by its eigenvalue before the optimizer step. This requires forming $H^{-1}$ at each training step, cost $\mathcal{O}((nd)^3)$ per sample. For BBA ($nd=84$): feasible per step. For AK ($nd=642$): prohibitive.

2. **Diagonal or low-rank $H$ approximation**: Replace $H$ in the loss with a diagonal or low-rank approximation that has acceptable condition number ($< 10^3$). For example, use only the $A$ component (ignoring $\delta B$, which has rank at most $d$) and regularize the diagonal. This would preserve the metric-aware weighting qualitatively while keeping gradients finite.

3. **Eigenvalue clamping in loss**: Compute the eigendecomposition of $H$ but clamp eigenvalues to $[\varepsilon_{\min}, \varepsilon_{\max}]$ with $\varepsilon_{\min} / \varepsilon_{\max}$ chosen to make the loss gradients numerically stable (e.g. $10^3$). This is a principled compromise between the exact $g$-norm and the flat Euclidean norm.

### What the approximation costs (measured)

Reproduce with `scripts/approximation_audit.py` §D (20 BBA frames, post-bugfix):

| Quantity | Value |
|---|---|
| Mean cond($H$) horizontal subspace | **~2.4×10⁶** |
| Max cond($H$) horizontal subspace | **~2.5×10⁷** |
| Change under unit-gyration normalization | **none** (scale-invariant) |
| g-norm loss (`use_riemannian_norm=True`) | ✗ NaN gradients |
| Euclidean loss (default) | ✓ stable |
| Fixed point | ✓ **same** for both |

Both $\mathcal{L}_g$ and $\mathcal{L}_E$ share the same fixed point: the true horizontal score $s^*$. The difference is **training dynamics only**: $\mathcal{L}_g$ would focus gradient signal on high-curvature directions of $w^\delta$ space; $\mathcal{L}_E$ distributes it evenly across all horizontal directions. The practical impact on convergence rate and sample quality has not been isolated — this would require a controlled experiment with a preconditioned optimizer once the numerical blocker is resolved.

---

## Approximation hierarchy: goals, blockers, and current status

The four items below are ordered by impact. Items C and A are fully or nearly resolved; B and D have active blockers that require specific engineering work to remove.

**C. Geodesic noising (resolved)**  
Goal: noise $x_0 \to x_t$ along the manifold geodesic. Status: **done** — `s_exp` is used. Flat noising put $x_t$ ~25$\sigma$ off-manifold (catastrophic); no tradeoff. Reproduce: `approximation_audit.py §C`.

**A. `s_log` in score_target (resolved for production; narrow fallback remains)**  
Goal: use `s_log` (true Riemannian gradient) as the score target everywhere. Status: **done for all production paths**. The only remaining `prelog` use is in `train()` (smoke-test online loop), where eigh per sample in a Python loop is ~12× slower; this is a developer-convenience tradeoff, not a geometric one. Precomputed data generation and all real training runs use `s_log`. Reproduce: `approximation_audit.py §A`.

**D. $g$-norm in DSM loss (active blocker: cond(H) ~2.4e6)**  
Goal: use Riemannian $g$-norm weighting in $\mathcal{L}_{\mathrm{DSM}}$. Status: **blocked** — cond($H$) ~2.4e6 (mean), scale-invariant; `use_riemannian_norm=True` produces NaN gradients. Both objectives share the same fixed point; training dynamics differ. Unblocking requires a preconditioned optimizer, diagonal $H$ approximation, or eigenvalue clamping. Reproduce: `approximation_audit.py §D`.

**B. `s_log` in `s_geodesic` inner loop (active blocker: O((nd)³) inside JIT)**  
Goal: use `s_log` (Riemannian gradient step) inside the geodesic doubling loop. Status: **blocked** — eigh of $(nd \times nd)$ inside a `jax.lax.fori_loop` is $\mathcal{O}((nd)^3)$ per XLA step. For AK ($nd=642$): ~265M ops per step, prohibitive. Measured midpoint deviation: ~14%. Unblocking requires restructuring the JIT boundary or a scalable $H^{-1}$ approximation. Reproduce: `approximation_audit.py §B`.

---

## Implementation status per approximation

| # | Goal | Where | Blocker | Current |
|---|---|---|---|---|
| C | Geodesic noising | `manifold_sde.py:marginal_prob` | **None — resolved** | **Exact (s_exp)** |
| A | `s_log` score target | `manifold_sde.py:score_target` | Developer convenience in `train()` only | **s_log default**; `use_slog=False` in `train()` only |
| D | $g$-norm DSM loss | `score_loss.py:riemannian_dsm_loss_from_noised` | cond($H$) ~2.4e6, scale-invariant, NaN grads | **Euclidean** (`use_riemannian_norm=True` blocked) |
| B | `s_log` in geodesic | `pointcloud_jax.py:s_geodesic` | $\mathcal{O}((nd)^3)$ per XLA step, prohibitive for AK | **prelog** (`use_separation_grad=False` blocked for AK) |
