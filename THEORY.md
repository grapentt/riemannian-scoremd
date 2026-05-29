# Unified Theory and Implementation Reference: Riemannian Diffusion on the w^δ Protein Shape Manifold

**This is the single source of truth for all mathematical, geometric, and implementation-theory content in the project.**

**Version:** 3.0 (2026-05-28 — transpose bug fix applied; §12 rewritten as root-cause analysis; §13 updated)  
**Previous versions:** v1.3 (2026-05-26) — see git history for the pre-unification state.

**Unification notice**: This document now incorporates unique analytical content previously scattered across `theory_and_architecture.md`, `problem.md`, `bug_diagnostic.md`, `H_REPAIR.md`, and `explainer_targets_and_gradients.md`. Those documents have been marked SUPERSEDED (full content retained for git history) and their distinctive contributions have been merged or excerpted here. See the list at the end of this header.

**Prerequisites:** Differential geometry at the level of Wendl, *Differential Geometry II* (Sommer 2022). All statements are derived from first principles unless explicitly noted as "implementation reality on real data."

**Cross-references**:
- `APPROXIMATIONS.md` — the canonical numerical companion (all measured tables, costs, and the approximation hierarchy). This THEORY document contains the derivations; APPROXIMATIONS contains the measured reality on BBA/chignolin data.
- For project plan and current status: `PLAN.md` and root `CURRENT_STATE.md`.
- Historical context for specific runs: the archived documents listed below + `riemannian-scoremd/runs/`.

---

## How to read this document

1. **Start with §12** if you are debugging `metric_tensor`, projection, loss, or want to understand the Phase 3.6 BBA training failure and the transpose bug.
2. **§1–6** for the mathematical foundations (quotient, w^δ, explicit H, separation property, prelog).
3. **§7–9** for the practical algorithms (s_log, s_exp) and why we use the approximations we do.
4. **§10–11** for diffusion/DSM setup and computational trade-offs.
5. **§13** for the paths to a full Riemannian treatment (Phase 4 FP loss).
6. **APPROXIMATIONS.md** (in parallel) for every number that appears in the tables or cost claims.

---

## Documents superseded by this unification (2026-05-28)

All of the following retain their full original content for git history and blame. Only this THEORY.md + APPROXIMATIONS.md should be considered living references.

- `theory_and_architecture.md` — pedagogical diffusion derivation + implementation traces + Q&A (selected excerpts merged).
- `problem.md` — geometric implication chain analysis + root-cause hypotheses + "why Diepeveen still worked" + 3-level resolution ladder (core analysis merged into §1 and the new resolution subsection).
- `bug_diagnostic.md` — complete evidence package for the Phase 3.6 contradiction (tables and "why tests missed it" now in §12).
- `H_REPAIR.md` — audit results on clipping vs. subspace alignment (now superseded: clipping was never needed; the correct fix was the transpose permutation).
- `explainer_targets_and_gradients.md` — beginner tutorial (largely duplicated; basic motivation integrated into §10).
- `phase2_summary.md` — historical achievement log (enduring design rationale excerpted into §§9–11).

---

**Original v1.3 header preserved for provenance**:

**Version:** 1.3 (revised 2026-05-26 — corrections to §1, §7, §10, §11; added §12 on the H-vs-G vertical space bug)  
**Based on:** colleague's v1.2 (2026-05-28). Revisions maintain the proof structure but correct several claims that are wrong on real data or in the codebase.  
**Prerequisites:** Differential geometry at the level of Wendl, *Differential Geometry II* (Sommer 2022). All statements are derived from first principles.  
**Cross-references:** See `APPROXIMATIONS.md` for measured numerical values of every approximation discussed here.

---

## 1. The shape manifold as a quotient

Let $d=3$ and $n \ge d+1$. A conformation is a matrix $X = (x_1^\top, \dots, x_n^\top)^\top \in \mathbb{R}^{n \times d}$ with distinct atom positions $x_i \neq x_j$.

**Definition 1.1.** The *point-cloud space* is the open subset
$$
\mathrm{P}(d,n) = \{X \in \mathbb{R}^{n \times d} \mid x_i \neq x_j\ \forall i \neq j\} \subset \mathbb{R}^{nd}.
$$
It is a smooth manifold of dimension $nd$ (open subset of Euclidean space).

The **Euclidean group** $G = \mathrm{E}(d) = \mathrm{SO}(d) \ltimes \mathbb{R}^d$ acts smoothly on $\mathrm{P}(d,n)$ by
$$
g \cdot X := XO^\top + \mathbf{1}t^\top, \qquad g = (O, t) \in G.
$$
This action is free and proper. By the quotient manifold theorem (Wendl §3.3), the orbit space
$$
\mathcal{M} := \mathrm{P}(d,n)/G
$$
is a smooth manifold of dimension
$$
\dim \mathcal{M} = nd - \frac{d(d+1)}{2}.
$$
We write $[X]$ for the equivalence class of $X$. In computations we choose a representative $X \in [X]$ (e.g. centered coordinates) and lift to $\mathbb{R}^{n \times d}$.

### Tangent spaces and the horizontal-vertical split

The tangent space at $X \in \mathrm{P}(d,n)$ is $T_X \mathrm{P}(d,n) \simeq \mathbb{R}^{n \times d}$.

**Vertical space** (infinitesimal rigid-body motions):
$$
\mathcal{V}_X := \bigl\{SX + \mathbf{1}t^\top \;\big|\; S \in \mathfrak{so}(d),\; t \in \mathbb{R}^d\bigr\}.
$$
These vectors move $X$ along its $G$-orbit without changing the shape. $\dim \mathcal{V}_X = d(d+1)/2 = 6$ for $d=3$.

There are **two natural horizontal spaces** corresponding to two different inner products on $\mathbb{R}^{n \times d}$:

**Euclidean horizontal space** $\mathcal{H}_X^E$: the Euclidean orthogonal complement of $\mathcal{V}_X$,
$$
\mathcal{H}_X^E = \mathcal{V}_X^{\perp_E} = \bigl\{\Xi \in \mathbb{R}^{n \times d} \;\big|\; \sum_i \xi_i = 0 \text{ and } \sum_i \xi_i x_i^\top = \bigl(\sum_i \xi_i x_i^\top\bigr)^\top \bigr\}.
$$

**Riemannian horizontal space** $\mathcal{H}_X^H$: the $H$-orthogonal complement of $\mathcal{V}_X$,
$$
\mathcal{H}_X^H = \bigl\{\Xi \in \mathbb{R}^{n \times d} \;\big|\; \langle \Xi, V \rangle_H = 0\ \forall V \in \mathcal{V}_X\bigr\}.
$$

**Theorem 1.2** (Diepeveen 2024, §B.3). $\ker H_X = \mathcal{V}_X$, so $H_X$ is positive definite on $\mathcal{H}_X^H$. Moreover, $\mathcal{H}_X^H = \mathcal{H}_X^E$ when $H_X$ is positive semi-definite with kernel exactly $\mathcal{V}_X$.

**Note** (see §12 for the full history): An earlier implementation bug caused $H_X$ to appear to have 4–5 negative eigenvalues on real folded protein data. After the bug was fixed (§12), $H_X$ is PSD up to float64 rounding noise (~1e-10) on all tested BBA frames. Theorem 1.2 holds numerically.

The codebase provides two projectors:
- `horizontal_projection_tvector` (`project_G`): projects onto $\mathcal{H}_X^E$ using explicit gyration-matrix generators (§B.3 of the paper). **This is the canonical projector used throughout the training pipeline.**
- The H-eigendecomposition cut inside `s_log`: implicitly projects onto the span of the top-$(nd - 6)$ eigenvectors of $H_X$. **After the bug fix, this agrees with `project_G` at machine precision.**

**Important clarification on the direction of implication** (from the geometric analysis previously in `problem.md` §1.1). A Riemannian metric $g$ on the *total space* $\mathrm{P}(d,n)$ induces a canonical horizontal complement $H_X = (V_X)^\perp_g$. When $g$ is additionally $E(d)$-invariant, it descends to a well-defined metric on the quotient $\mathcal{M}$. The converse does **not** hold: a metric on the quotient alone does not determine a unique horizontal distribution on the total space. There are infinitely many complementary subspaces to $V_X$ that project isomorphically onto $T_{[X]}\mathcal{M}$. Declaring an orthogonal complement requires an inner product on $T_X\mathrm{P}(d,n)$, which is additional data not provided by the quotient metric. The correct implication chain is therefore

$$
\underbrace{g \text{ on } \mathrm{P}(d,n)}_{\text{total space}} \xrightarrow{\;\text{orth. complement}\;} H_X \quad \xrightarrow{\;E(d)\text{-invariance}\;} \tilde{g} \text{ on } \mathcal{M}.
$$

The arrow runs only left-to-right. This is why the concrete realization of $H$ via `metric_tensor` (an ambient $nd \times nd$ bilinear form) plus a subsequent projection choice is not merely an implementation detail — it is the step that actually *selects* which horizontal distribution we are using. When that choice (H-eigen cut) differs from the explicit group-theoretic basis (`project_G`), we are no longer working with a single coherent geometric object.

---

## 2. Two $G$-invariant building blocks

**Pairwise distances:** $r_{ij}(X) = \|x_i - x_j\|$. These are invariant under $G$ (translations shift all atoms equally; rotations preserve norms).

**Gyration matrix:** $G_X = \sum_i (x_i - \bar{x})(x_i - \bar{x})^\top$, $\bar{x} = n^{-1}\sum_i x_i$. Under $(O,t) \cdot X$: $G_X \mapsto O G_X O^\top$, so $\det G_X$ is $G$-invariant.

---

## 3. The $w^\delta$ function

**Definition 3.1.** For $[X], [Y] \in \mathcal{M}$:
$$
w^\delta([X],[Y])^2 := \sum_{i<j} \Bigl[\log\frac{r_{ij}(X)}{r_{ij}(Y)}\Bigr]^2 + \delta \Bigl[\log\frac{\det G_X}{\det G_Y}\Bigr]^2, \qquad \delta > 0.
$$
This is well-defined (depends only on $[X]$ and $[Y]$), non-negative, and zero iff $[X] = [Y]$.

The parameter $\delta$ (called `alpha` in the code) weights the gyration term. Default: $\delta = 0.1$; adenylate kinase and BBA experiments use $\delta = 1.0$.

---

## 4. The Riemannian metric: explicit derivation of $H = A + \delta B$

Fix $q = [Y]$ and define $f_q : \mathrm{P}(d,n) \to \mathbb{R}$ by
$$
f_q(X) = \tfrac{1}{2} w^\delta([X],[Y])^2.
$$

The metric tensor at $p = [X] = [Y]$ is the restriction of $\operatorname{Hess}_X f_q$ to horizontal vectors. In block-matrix notation (blocks indexed by residues $i,j$):

**4.1 Pairwise contribution (matrix $A$)**

For one pair $i < j$, the $\ell$-th summand is $\frac{1}{2}[\log r_{ij}(X)/r_{ij}(Y)]^2$. At $X = Y$:
$$
\nabla_{x_i}\!\left(\log r_{ij}(X)\right) = \frac{x_i - x_j}{r_{ij}^2}.
$$
The Hessian of each summand at $X=Y$ contributes rank-1 updates that sum over all pairs to:
$$
A_{ij} = \begin{cases}
\displaystyle\sum_{k \neq i} \frac{(x_i - x_k)(x_i - x_k)^\top}{r_{ik}^4} & i = j, \\[10pt]
\displaystyle -\frac{(x_i - x_j)(x_i - x_j)^\top}{r_{ij}^4} & i \neq j.
\end{cases}
$$
$A$ has the graph-Laplacian structure of the complete residue graph.

**4.2 Gyration contribution (matrix $B$)**

For $h(X) = \frac{\delta}{2}[\log \det G_X - \log \det G_Y]^2$, using $\nabla_{x_i} \log \det G_X = 2G_X^{-1}(x_i - \bar{x})$:
$$
B_{ij} = 4\bigl(G_X^{-1}(x_i - \bar{x})\bigr) \otimes \bigl(G_X^{-1}(x_j - \bar{x})\bigr).
$$
$B$ has rank at most $d$ (outer product of centred-and-whitened positions).

Thus $H_X = A_X + \delta B_X$ and
$$
g_{[X]}(\Xi, \Psi) = \Xi : H_X \Psi := \sum_{i,j} \xi_i^\top (H_X)_{ij} \psi_j.
$$

**Theorem 4.2.** (Diepeveen 2024, Theorem 5.3) $H_X$ is positive semi-definite with $\ker H_X = \mathcal{V}_X$, and positive definite on $\mathcal{H}_X$. Hence $(\mathcal{M}, g)$ is a Riemannian manifold.

**Remark 4.3** (historical indefiniteness — implementation bug, now fixed). Theorem 4.2 holds both in theory and in practice after the transpose fix (§12). An earlier version of `metric_tensor` used the wrong d-index permutation when reshaping the $(N,M,n,n,d,d)$ tensor to $(N,M,nd,nd)$ form: `.permute(0,1,2,5,3,4)` instead of the correct `.permute(0,1,2,4,3,5)`. This made the flat matrix row index `i*d+b` instead of `i*d+a`, creating a matrix inconsistent with the $G$-basis generators (which use `i*d+a` indexing). The corrupted $H$ had 4–5 apparent negative eigenvalues (min $\approx -0.015$) per BBA frame. After the fix: $H$ has at most ~0.4 negative eigenvalues/frame at magnitudes ~1.6e-10 (float64 rounding noise); all rotation residuals $\|H v_{\rm rot}\|/\|v_{\rm rot}\| < 10^{-8}$. See §12 for the full root-cause analysis.

---

## 5. $w^\delta$ is a separation w.r.t. the Riemannian distance $d_g$

**Theorem 5.1.** (Corollary 3.2.1, Eq. 4)
$$
w^\delta(p,q)^2 = d_g(p,q)^2 + \mathcal{O}(d_g(p,q)^3) \qquad \text{as } d_g(p,q) \to 0.
$$

**Proof.** Fix $p$. Along the unit-speed geodesic $\gamma(t)$ from $p = \gamma(0)$ to $q = \gamma(r)$ with $r = d_g(p,q)$, define $f(t) = \frac{1}{2} w^\delta(\gamma(t), p)^2$. By definition of $g$: $f(0) = 0$, $f'(0) = 0$, $f''(0) = 1$. Taylor with Lagrange remainder:
$$
f(r) = \tfrac{1}{2} r^2 + \tfrac{1}{6} f'''(\xi) r^3 = \tfrac{1}{2} d_g(p,q)^2 + \mathcal{O}(r^3).
$$
Hence $w^\delta(p,q)^2 = d_g(p,q)^2 + \mathcal{O}(d_g(p,q)^3)$. $\square$

---

## 6. s_prelog: the flat (Euclidean) representation of $df$

**Definition 6.1.**
$$
\mathrm{prelog}(X,Y) := \nabla_X^E \tfrac{1}{2} w^\delta([X],[Y])^2.
$$

**Explicit formula** (Appendix D.3, Eq. 119):
$$
\mathrm{prelog}_i(X,Y)
= -\sum_{j \neq i} \log\frac{r_{ij}(X)}{r_{ij}(Y)} \cdot \frac{x_i - x_j}{r_{ij}^2(X)}
\;-\; 2\delta \log\frac{\det G_X}{\det G_Y} \cdot G_X^{-1}(x_i - \bar{x}).
$$

**Proposition 6.2** (automatic Euclidean-horizontality). $\mathrm{prelog}(X,Y) \in \mathcal{H}_X^E$ without any projection.

*Proof.* Differentiate the $G$-invariance $w^\delta((O,t) \cdot X, Y) = w^\delta(X, Y)$ at the identity in the direction $V \in \mathcal{V}_X$. The derivative is $\langle \nabla_X^E w^\delta, V \rangle_E = 0$ for all $V \in \mathcal{V}_X$. $\square$

In the codebase, `project_G` is called after `s_prelog` as a numerical cleanup step only. Measured floating-point leakage: $< 10^{-7}$ (see `APPROXIMATIONS.md §A`).

The distinction between `prelog` and `s_log` is developed rigorously in §7 below: `prelog` is the coordinate representation of $df$ via the flat musical isomorphism, while `s_log` is the coordinate representation of the true Riemannian gradient $\operatorname{grad}_g f$.

### 6.1 Coordinate identification: prelog as the flat representation of $df$

Let $f = \frac12 (w^\delta)^2$, viewed as a smooth real-valued function on the total space $\mathrm{P}(d,n)$ (or, equivalently, as a $G$-invariant function on the quotient $\mathcal{M}$). Its exterior derivative $df$ is a covector field: at each $X$ we have a linear functional
$$
df_X : T_X \mathrm{P}(d,n) \to \mathbb{R}.
$$

The ambient Euclidean metric $\langle \cdot, \cdot \rangle_E$ on $\mathbb{R}^{nd}$ induces a canonical vector bundle isomorphism (the flat musical isomorphism)
$$
\flat_E : T\mathrm{P}(d,n) \xrightarrow{\;\sim\;} T^*\mathrm{P}(d,n), \qquad
\sharp_E : T^*\mathrm{P}(d,n) \xrightarrow{\;\sim\;} T\mathrm{P}(d,n).
$$
Under this identification, the covector $df_X$ corresponds to a unique tangent vector $\nabla^E f(X) \in T_X \mathrm{P}(d,n)$ characterized by
$$
\langle \nabla^E f(X), V \rangle_E = df_X(V) \qquad \text{for all } V \in T_X \mathrm{P}(d,n).
$$
In other words, $\nabla^E f$ is the vector representation of the covector $df$ with respect to the flat metric. This is precisely what `s_prelog` computes:
$$
\mathrm{prelog}(X,Y) = \nabla_X^E f = \sharp_E \bigl( df_X \bigr).
$$
Thus `prelog` is *not* yet a Riemannian object; it is the coordinate representation of $df$ in the trivialization of the tangent bundle coming from the embedding $\mathrm{P}(d,n) \hookrightarrow \mathbb{R}^{n \times d}$.

(The $G$-invariance of $w^\delta$ immediately implies that this vector is Euclidean-horizontal, as stated in Proposition 6.2.)

---

## 7. s_log: the approximate Riemannian log map

On a Riemannian manifold $(\mathcal{M}, g)$ the **Riemannian gradient** of a smooth function $f$ is the vector field $\operatorname{grad}_g f$ defined by the characterizing property
$$
g\bigl( \operatorname{grad}_g f,\, V \bigr) = df(V) \qquad \text{for all tangent vectors } V.
$$
In other words, $\operatorname{grad}_g f = \sharp_g (df)$, where $\sharp_g : T^*\mathcal{M} \to T\mathcal{M}$ is the musical isomorphism (index-raising) induced by $g$.

### 7.1 Coordinate realization in ambient $\mathbb{R}^{nd}$

The total space $\mathrm{P}(d,n)$ carries a global trivialization of its tangent bundle coming from the ambient embedding into $\mathbb{R}^{n \times d} \simeq \mathbb{R}^{nd}$. In this trivialization every tangent vector (and every covector) is represented by an ordinary matrix in $\mathbb{R}^{n \times d}$.

With respect to this trivialization the Riemannian metric $g$ (whose abstract definition is the Hessian of $\frac12 w^\delta$ restricted to horizontal vectors) is represented at each point $X$ by the bilinear form whose matrix is exactly $H_X = \operatorname{metric\_tensor}(X)$. Consequently the abstract equation $g(\operatorname{grad}_g f, V) = df(V)$ becomes the linear algebra problem
$$
\langle H_X \cdot (\operatorname{grad}_g f),\, V \rangle_E = \langle \operatorname{prelog}(X),\, V \rangle_E
$$
for all $V$, or simply
$$
H_X \cdot (\operatorname{grad}_g f) = \operatorname{prelog}(X)
$$
in the ambient coordinates.

Solving for the gradient therefore requires inverting (or pseudo-inverting) $H_X$:
$$
\operatorname{grad}_g f(X) = H_X^\dagger \,\operatorname{prelog}(X).
$$
This is precisely the definition of `s_log`:

**Definition 7.1 (Coordinate formula).**
$$
\mathrm{s\_log}(X,Y) := H_X^\dagger\,\mathrm{prelog}(X,Y).
$$
In practice the pseudo-inverse is realized by the eigendecomposition $H_X = Q \Lambda Q^\top$, discarding the bottom `vert_dim` eigenvectors (the putative vertical space) and inverting only on the horizontal block:
$$
H_X^\dagger = Q_{\mathrm{horiz}}\,\Lambda_{\mathrm{horiz}}^{-1}\,Q_{\mathrm{horiz}}^\top.
$$

The approximate logarithmic map is then defined by $\log_{[X]}^w([Y]) := -\mathrm{s\_log}(X,Y)$. This is the natural training target that appears in the Riemannian score-matching loss (see §10).

### 7.2 Why the dagger, and why it is delicate

The dagger appears for two independent reasons:

1. **Kernel on the total space.** Even in the ideal case, $H_X$ is only positive semi-definite on the full ambient tangent space $T_X \mathrm{P}(d,n)$, with $\ker H_X = \mathcal{V}_X$ (the infinitesimal generators of the $E(d)$ action). We are only entitled to invert $H_X$ on a complementary horizontal subspace.

2. **Quotient vs. total-space geometry.** The metric that ultimately interests us lives on the quotient $\mathcal{M}$. The horizontal distribution used to realize the quotient metric must be chosen consistently. In the implementation the choice is made by the explicit $G$-basis projector `horizontal_projection_tvector` (see §1 and §12). The eigencut that defines $H_X^\dagger$ inside the original `s_log` is an *implicit* choice of horizontal complement; on real data these two choices do not coincide. This is the root of the Phase 3.6 inconsistency (detailed in §12).

The key point is that `s_log` is the first place in the pipeline where the *Riemannian* metric $H$ (rather than the flat Euclidean structure) is used to convert the covector $df$ into a tangent vector. Everything before this step (including the entire forward noising process when using `prelog`) can be viewed as operating with the flat identification $\sharp_E$.

**Theorem 7.2.** (Corollary 3.2.1) $\|\log_{[X]}([Y]) - \log_{[X]}^w([Y])\|_H = \mathcal{O}(d_g([X],[Y])^2)$ as $[Y] \to [X]$.

### 7.3 Practical caveats and computational cost

The construction in Definition 7.1 relies on two assumptions that are *not* automatic:

- The bottom `vert_dim` eigenvectors of the numerically realized matrix $H_X$ must span the true vertical space $\mathcal{V}_X$ generated by the $E(d)$ action.
- $H_X$ must be positive semi-definite on the orthogonal complement of that vertical space.

**When these assumptions fail.** On real folded protein data with $\delta = 1.0$, the bottom-6 eigenvectors of $H_X$ and the explicit $G$-basis generators of $\mathcal{V}_X$ span subspaces that differ by a non-trivial rotation:
$$
\|\mathrm{G\text{-}vert}^\top \cdot \mathrm{H\text{-}vert}\|_F = 2.43 \quad \text{(ideal value } \sqrt{6} \approx 2.449\text{)}.
$$
Consequently the eigencut that defines $H_X^\dagger$ inside the original `s_log` produces a vector that still contains a small vertical component with respect to the projector actually used everywhere else in the training pipeline (`horizontal_projection_tvector`). This is the origin of the Phase 3.6 training pathology (see §12).

**Computational cost:** $\mathcal{O}((nd)^3)$ — full eigendecomposition of the $nd \times nd$ matrix $H_X$ at every call. This is the dominant reason the production training path uses `prelog + project_G` instead of `s_log`.

---

## 8. Error analysis: $\|\log - \log^w\|_H = \mathcal{O}(r^2)$

Let $\psi = \frac{1}{2}w^2$, $\phi = \frac{1}{2}d_g^2$, $e = \psi - \phi = \mathcal{O}(r^3)$ (Theorem 5.1). Then $de(V) = \mathcal{O}(r^2)\|V\|$.
$$
g(\operatorname{grad}_g \phi - \operatorname{grad}_g \psi,\, V) = de(V) = \mathcal{O}(r^2)\|V\|.
$$
Choosing $V = \operatorname{grad}_g \phi - \operatorname{grad}_g \psi$ gives $\|\operatorname{grad}_g \phi - \operatorname{grad}_g \psi\|_H = \mathcal{O}(r^2)$.

This bound holds under the assumption that $H$ is positive definite on $\mathcal{H}_X$. After the transpose bug fix (§12), $H$ is positive definite on $\mathcal{H}_X$ for all tested BBA frames and the bound applies.

---

## 9. Approximate exponential map (s_exp)

The exact exp map $\exp_{[X]}(V) = $ "follow the geodesic with initial velocity $V \in \mathcal{H}_X$ for unit time" has no closed form. `s_exp` uses iterative **geodesic doubling** based on the midpoint property (Eqs. 5–6):
$$
[Z] = \underset{[R]}{\arg\min}\;[w([X],[R])^2 + w([R],[Y])^2].
$$
The gradient of this criterion is $-\frac{1}{2}\nabla w^2 = \mathrm{prelog}$, so the inner loop (`s_geodesic`) uses `prelog` gradient steps.

**JIT architecture:** `s_exp` cannot be directly `jax.jit`'d due to data-dependent loop count. The codebase uses a factory method `_build_doubling_fn(K)` that inlines the full loop as a Python closure with $K$ as a compile-time constant, cached in `_doubling_cache`. This achieves ~1 ms per call (warm, JIT-cached) vs ~80 ms (Python eager dispatch).

---

## 10. Diffusion and score matching on $\mathcal{M}$

### Forward process

The VP-SDE forward process uses geodesic noising:
$$
x_t = \mathrm{s\_exp}(x_0,\; \sigma(t) \cdot v_{h,\mathrm{unit}})
$$
where $v_{h,\mathrm{unit}} \in \mathcal{H}_{x_0}^E$ is a unit horizontal noise vector. Flat Gaussian noising ($x_t = \alpha(t) x_0 + \sigma(t) \varepsilon$) is **not used** — it lands $\sim 25\sigma$ off-manifold at moderate $t$ (see `APPROXIMATIONS.md §C`).

### Score target (current implementation)

The theoretically exact Riemannian DSM target would be:
$$
s_{\mathrm{true}}^{\mathrm{exact}} = -\frac{\log_{x_t}^w(x_0)}{\sigma(t)^2} = -\frac{H_{x_t}^\dagger\,\mathrm{prelog}(x_t, x_0)}{\sigma(t)^2}.
$$

**Two score target options are implemented** (Phase 3.65 settled — see APPROXIMATIONS.md §A):

`use_slog=True` (**default** — all production paths):
$$
s_{\mathrm{true}} = -\frac{H^{-1}\,\mathrm{prelog}(x_t, x_0)}{\sigma(t)^2}
$$

`use_slog=False` (**fast fallback** — smoke-test `train()` loop only):
$$
s_{\mathrm{true}} = -\frac{\mathrm{project}_G\bigl(\mathrm{prelog}(x_t, x_0)\bigr)}{\sigma(t)^2}
$$

`s_log` is the geometrically principled choice (true Riemannian gradient, $\mathcal{O}((nd)^3)$). `prelog` is used explicitly in the online `train()` loop where eigh per sample in a Python loop is prohibitively slow; all production training goes through `precompute_noised_data.py` + `train_from_precomputed` and never calls `score_target` at training time. See APPROXIMATIONS.md §A for measurements.

### DSM loss

$$
\mathcal{L} = \mathbb{E}_{t, x_0, x_t}\!\left[\,\beta(t)\,\bigl\|\mathrm{project}_G(s_\theta(x_t, t)) - s_{\mathrm{true}}\bigr\|_E^2\right]
$$

The Euclidean norm is used instead of the $g$-norm because the condition number of $H$ on BBA data is $\sim 2.4 \times 10^6$ (mean, post-bugfix), scale-invariant — normalization doesn't help. NaN gradients result from g-norm. Phase 3.66 settled. See APPROXIMATIONS.md §D.

---

## 11. Computational aspects

| Operation | Cost | Notes |
|---|---|---|
| `s_prelog` | $\mathcal{O}(n^2 d)$ | No eigendecomposition; numerically robust |
| `s_log` | $\mathcal{O}((nd)^3)$ | Requires eigh of $nd \times nd$ matrix; now valid (H is PSD after bug fix) |
| `s_exp` (JIT-warm) | ~1 ms ($n=214$, CPU) | Factory-cached closure; 2.6 ms total including overhead |
| `metric_tensor` H | $\mathcal{O}(n^2 d)$ | Pairwise outer products + gyration correction |
| `horizontal_projection_tvector` | $\mathcal{O}(n d^2)$ | Gyration eigenvectors + skew generators; always valid |

**`s_log` is the default** (`use_slog=True` in `score_target`, `use_slog=True` in `prepare_batch`). The `prelog + project_G` path (`use_slog=False`) is a fast fallback used explicitly in `train()` (online smoke-test loop) where eigh per sample in a Python loop is prohibitively slow. All production training uses `precompute_noised_data.py` + `train_from_precomputed` and never calls `score_target` at training time. See APPROXIMATIONS.md §A.

**Updated status of regularization options for a full Riemannian treatment** (Phase 4):
- Eigenvalue clipping: no longer needed for basic correctness; may still be useful as defensive programming.
- Reduce $\delta$: not needed for PSD; still affects eigenvalue spread.
- Normalize conformations to unit gyration radius: does not change condition number (scale-invariant); does not help g-norm stability (Phase 3.66 settled).
- Use the sliced JVP divergence estimator from Diepeveen (2025) for scalable FP loss computation.

---

## 12. The `metric_tensor` transpose bug and its downstream effects

This section documents two interleaved events: a silent implementation bug that corrupted $H$ for the entire project history, and a Phase 3.6 training failure that was caused (though not solely) by its consequences. The bug has been fixed; this section is a permanent record of the root cause, why it was hard to find, and what was affected.

### 12.1 The transpose bug

**Root cause.** `metric_tensor(asmatrix=True)` computes the $(N,M,n,n,d,d)$ tensor $(A + \delta B)_{ij,ab}$ and then reshapes it to an $(N,M,nd,nd)$ matrix. The correct reshape interleaves spatial index $a$ as the fast axis within each atom block $i$, giving flat row index $k = i \cdot d + a$. The original code used:

```python
# WRONG — swaps the d-indices
H = (A + alpha * B).transpose(0, 1, 2, 5, 3, 4).reshape(N, M, n*d, n*d)
```

The permutation `(0,1,2,5,3,4)` maps $(N,M,n_i,n_j,d_a,d_b) \to (N,M,n_i,d_b,n_j,d_a)$, making the flat row index $i \cdot d + b$ instead of $i \cdot d + a$. The correct code is:

```python
# CORRECT — flat row = i*d + a
H = (A + alpha * B).transpose(0, 1, 2, 4, 3, 5).reshape(N, M, n*d, n*d)
```

The same bug was present in both the JAX port and the original PyTorch reference (`diepeveen2024/src/manifolds/pointcloud.py`). Both have been fixed.

**Why the parity tests never caught it.** All `test_port_parity.py` tests compare JAX output against PyTorch output. Both implementations had the *same* wrong transpose, so they agreed perfectly (relative error $< 10^{-4}$) while both produced the wrong matrix. The tests validated mutual consistency, not absolute correctness.

**Why translations passed under both conventions.** Translation generators $\xi^{(k)}_i = \delta_{k \cdot} e_a$ are symmetric: all atoms shift by $e_a$ in direction $k$. Swapping the $d$-indices inside each atom block leaves a translation generator invariant (both $i \cdot d + a$ and $i \cdot d + b$ indexings give the same result for the uniform-shift structure). Rotation generators are not symmetric under this swap, so they expose the inconsistency.

**Why the $\alpha=0$ tetrahedron test passed.** At $\alpha=0$ only the $A$ term contributes. The $A$ tensor involves symmetric outer products of pairwise difference vectors over the squared distances. For a regular tetrahedron (high symmetry), $A$ is numerically symmetric under d-index relabeling. The $B = 4 y_i \otimes y_j$ term (which mixes coordinates via gyration eigenvectors) is what exposes the swap — it is only present at $\alpha > 0$.

**Measured effect of the fix** (BBA $n=28$, $\delta=1.0$, float64):

| Quantity | Before fix | After fix |
|---|---|---|
| Rotation residuals $\|H v_{\rm rot}\|/\|v_{\rm rot}\|$ | ~1e-2 | ~4e-9 |
| Negative eigenvalue count/frame | 3–5 (min ~-0.011) | ~0.4 at ~-1.6e-10 |
| `s_log` vertical leakage (no repair) | 4.5% | 4.3e-8 |
| Quadratic form consistency (matrix vs 6D tensor) | rel_err 1.8e-3 | rel_err 3.2e-16 |
| Angular difference prelog vs `s_log` | 35.9° | ~35° |

The last row is essentially unchanged after the fix. The pre-fix `s_log` happened to operate on a corrupted $H$ whose distortion partially mimicked the Riemannian correction — yielding a similar angle by coincidence. The measured ~35° reflects the true Riemannian correction on the correct $H$.

### 12.2 The Phase 3.6 training failure

**Setup.** Before the transpose bug was discovered, training data for the BBA Phase 3.6 run was precomputed via `scripts/precompute_noised_data.py`. The `score_target` at that time used:

```python
# Old (with buggy H in s_log):
log_map = self.manifold.s_log(x_t, x_0)   # H-eigenvector cut on corrupted H
return -log_map / sigma_t**2
```

The buggy $H$ had 4–5 negative eigenvalues. The "bottom-6 eigenvectors" cut used to identify vertical directions sat inside the negative-eigenvalue region and did not span $\mathcal{V}_X$ at all. The result: `s_log`-based targets had ~1.3% vertical leakage relative to the G-basis projector used in the loss.

**Symptom.** The training run produced:
- Loss: 105.5 (well below zero-output baseline 177 — loss was going *down*)
- Cosine similarity between predicted and true scores: $\approx 0$ at all $t$

The model successfully minimized the DSM loss on distorted targets. Loss descent was misleading: the baseline 177 was also computed from the distorted targets, so the loss appeared to be making progress. The trained score pointed in random directions relative to the true denoising direction.

**Fix for the training pipeline.** Replace `s_log` with `prelog + project_G` in `score_target` (`manifold_sde.py`) as an immediate safe fallback:

```python
# New (correct — uses G-basis, same projector as noise and loss):
prelog = self.manifold.s_prelog(x_t, x_0)
prelog_h = self.manifold.horizontal_projection_tvector(x_t, prelog)
return -prelog_h / sigma_t**2
```

This fix is valid regardless of whether $H$ is PSD or not, because it never uses `s_log` or H's eigenstructure. After the fix: vertical leakage $< 10^{-7}$ (machine zero).

**Current status.** The transpose bug fix makes `s_log` geometrically valid again. `score_target` now accepts `use_slog: bool = False`. The precompute script calls it with `use_slog=True` — `s_log` is the default for precomputed data generation (Phase 3.65 settled). The `prelog + project_G` fast path remains available for online training. Precomputed data from the Phase 3.6 run must be regenerated before re-training.

### 12.3 Lesson: How to test absolute correctness, not just parity

The failure mode here — both implementations sharing the same bug, parity tests passing, absolute correctness never checked — is a general trap. The diagnostic that finally found it:

1. Compute the 6 explicit rigid-body generators $v_{\rm trans}^{(k)}$, $v_{\rm rot}^{(\ell)}$ from the group action (not from H eigenvectors).
2. Measure $\|H v\|/\|v\|$ for each generator.
3. Expect machine-precision results ($< 10^{-8}$) since $v \in \ker H$ by Theorem 4.2.

This diagnostic is the definitive absolute-correctness check for `metric_tensor` and should be run whenever `metric_tensor` is modified (see §Q&A below).

---

## 13. Paths to Full Resolution (Future Work)

The pragmatic repair (prelog + explicit `project_G` in `score_target`, Euclidean norm on the projected residual) restores internal consistency for denoising score matching on the current data regime. After the transpose bug fix, $H$ is PSD and `s_log` is geometrically valid. However, the Euclidean-norm loss still does not realize the full Riemannian $g$-norm loss (required for Phase 4), because the condition number of $H$ (~8e7 on BBA) causes NaN gradients in `manifold.inner`.

A theoretically satisfactory implementation must satisfy, at minimum, the following invariants for every conformation $X$ used in training or sampling:

1. The concrete bilinear form realized by `metric_tensor` (after restriction to the horizontal space defined by the group action) must be positive semi-definite. **✓ Satisfied after transpose bug fix.**
2. The horizontal projection used to generate score targets, noise, and residuals must be the orthogonal complement, with respect to that bilinear form, of the vertical space $V_X$ coming from the $E(d)$ action. **✓ Satisfied: `project_G` (G-basis) is used everywhere in the training pipeline.**
3. `s_log` (when needed) must be the inverse of the metric on the correct horizontal space, not a spectral truncation of an ambient indefinite matrix. **✓ Satisfied after transpose bug fix: H is PSD, `s_log` eigenvector cut is now correct.**

The remaining gap is that the training pipeline does not yet use the Riemannian $g$-norm in the DSM loss or a proper manifold FP residual for Phase 4.

**Level 1 — Pragmatic (current production path for Phase 3 training)**  
For any operation that must produce horizontal tangent vectors:
1. Compute the flat gradient of $w^2$ via `s_prelog`.
2. Immediately project onto the explicit group-theoretic horizontal space using `horizontal_projection_tvector`.
3. (Optional) Apply any desired metric-aware scaling *after* this projection.
This is what the repaired `score_target` does. Valid, internally consistent, and numerically stable.

**Level 2 — Proper Horizontal Restriction of the Metric Operator (medium effort)**  
Enable the Riemannian $g$-norm loss. The condition number of $H$ on BBA frames is ~2.4e6 (mean, post-bugfix), scale-invariant — normalization does not reduce it (Phase 3.66 settled, Euclidean norm is the production choice). Options if this is revisited in future:
- Use a preconditioned optimizer that normalizes gradient directions by $H$.
- Use `s_log = H^{-1} \cdot \mathrm{prelog}` after projection, inverting only on the horizontal block.
This would bring the implementation to a true Riemannian DSM loss with the correct metric weighting.

**Level 3 — Manifold Fokker–Planck Loss (required for Phase 4)**  
The log-density residual on a Riemannian manifold involves the Laplace–Beltrami operator $\Delta_{\mathcal{M}} \log p$, not merely a corrected divergence. This requires:
1. A consistent horizontal distribution (✓ now available).
2. A PSD metric operator (✓ now available after bug fix).
3. Deriving the full Fokker–Planck residual from the Kolmogorov forward equation on $(\mathcal{M}, g)$ (analytic work, Phase 4 hard blocker — see CLAUDE.md §FP warning).

**Recommended immediate validation step before investing in Level 2**  
Reproduce the core Diepeveen et al. 2024 experiments (w^δ-geodesics between distant adenylate kinase frames, barycentre + rank-1 approximation) using the current JAX implementation on the original 102-frame data. If RMSD remains comparable to the paper's ~3.85 Å threshold and the dominant biological transition is recovered, the practical geometry remains useful.

---

## 14. Summary of approximations

See `APPROXIMATIONS.md` for full numerical details. Brief summary (measured on BBA $n=28$, $\delta=1.0$, 63k frames):

| Approximation | Location | Impact | Status |
|---|---|---|---|
| Geodesic (s_exp) vs flat noising | `marginal_prob` | ~25× off-manifold error — non-negotiable | **s_exp used** |
| prelog vs s_log in score_target | `score_target` | ~35° angle, prelog ≈ 0.18× norm of s_log | **s_log for precompute (`use_slog=True`); prelog for online** |
| Euclidean vs $g$-norm loss | DSM loss | Same fixed point; cond# ~2.4e6, scale-invariant | **Euclidean (NaN with g-norm, Phase 3.66 settled)** |
| Fast (prelog) vs exact (s_log) in s_geodesic | `s_geodesic` inner loop | 14% midpoint deviation | **Fast path (s_log O((nd)³) in JIT, Phase 3.67 settled)** |

---

## 15. Why the prefix "s_"?

In score-based generative modelling the neural network is $s_\theta$ (the learned score). The symbols `s_prelog`, `s_log`, `s_exp`, `s_geodesic`, `s_distance` are the separation-derived analogues of the standard Riemannian operations (gradient of $\frac{1}{2}d^2$, log map, exp map, geodesic, distance), specialized to the $w^\delta$ separation. The prefix "s_" signals "separation-derived" — they agree with the exact Riemannian operations to third order (Theorem 5.1) but are computable in closed form or via simple iteration.

---

## Appendix: Selected Questions and Answers

The following Q&A items were excerpted and lightly adapted from the pedagogical companion `theory_and_architecture.md` (now superseded). They address the most common points of confusion that arose during development.

**Q: What should the score actually be on this manifold?**

The Riemannian score is the vector field $\operatorname{grad}_g \log p_t$ on $\mathcal{M}$, i.e., $\sharp_g (d \log p_t)$. Equivalently, it is the unique horizontal vector field $s$ satisfying
$$
g(s, V) = d(\log p_t)(V)
$$
for all horizontal $V$. For small noise, Varadhan's lemma gives the asymptotic $t \cdot \operatorname{grad}_g \log p_t(x_t \mid x_0) \to -\log_{x_t}(x_0)$. The natural training target is therefore a scaled negative logarithmic map. Because the true log map is intractable, we use separation-derived surrogates. `prelog` realizes the *flat* identification $\sharp_E(df)$, while `s_log` realizes the *Riemannian* identification $\sharp_g(df) = \operatorname{grad}_g f$ (see §§6.1 and 7.1).

**Q: Why do we ever use the weaker `prelog` (flat gradient of ½w²) instead of the more "Riemannian" `s_log`?**

`s_log` is the default (`use_slog=True`) and the correct choice for all production use. `prelog + project_G` (`use_slog=False`) is a fast fallback used in exactly one place: the online `train()` smoke-test loop, where `prepare_batch` is called once per sample in a Python loop and eigh of an `(nd × nd)` matrix per sample is ~12× slower. The production training path (`precompute_noised_data.py` + `train_from_precomputed`) calls `score_target` only once per frame at precompute time — the cost is amortized and s_log is used. The ~35° per-sample angular gap is real — `s_log = H^{-1} \cdot \mathrm{prelog}` is the true Riemannian gradient while `prelog + project_G` is the horizontal Euclidean gradient — but both share the same DSM fixed point.

Historical note: before the transpose bug fix, `s_log` produced targets with ~1.3% vertical leakage because the corrupted $H$ did not have $\ker H = \mathcal{V}_X$. The Phase 3.6 training failure was caused by this. After the fix, `s_log` is the default.

**Q: Does the current model "diffuse on the w^δ Riemannian manifold"?**

After the transpose bug fix, the geometric foundation is consistent: $H$ is PSD with $\ker H = \mathcal{V}_X$, the G-basis projector and H-eigenvector cut agree at machine precision, and `s_log` is a valid Riemannian log map. For the forward noising process and the DSM loss: yes, the model diffuses on the w^δ manifold in a well-defined sense. For any claim that requires the Laplace–Beltrami operator or a consistent Fokker–Planck residual on $(\mathcal{M}, g)$: not yet (Phase 4 analytic derivation still needed). See §13.

**Q: What is the single most important thing to remember when touching projection or loss code?**

Always keep the same projector throughout noise generation, score targets, and loss: either all use `project_G` (G-basis, current production) or all use the H-eigenvector cut (now valid after transpose fix, but slower). Mixing them creates inconsistent training targets. The Phase 3.6 training run (loss 105, cosine ≈ 0) was the direct consequence of mixing the H-eigenvector cut in `score_target` with `project_G` in the noise and loss — compounded by the then-present transpose bug that made the H-eigenvector cut wrong to begin with.

The absolute-correctness check for `metric_tensor` is: compute explicit rigid-body generators from the group action and verify $\|H v\|/\|v\| < 10^{-8}$ for each (see §12.3).

**Q: What does this mean for Phase 4 (manifold FP loss)?**

After the transpose fix, the geometric prerequisites for Phase 4 are satisfied: $H$ is PSD, $\ker H = \mathcal{V}_X$, and the horizontal distribution is well-defined. The remaining blocker is analytic: the log-density residual on a Riemannian manifold involves the Laplace–Beltrami operator $\Delta_{\mathcal{M}} \log p$, not merely a corrected divergence. The derivation must be done from first principles (Hsu 2002 §3.2) — patching the Euclidean FP formula will miss terms from the Kolmogorov forward equation on $(\mathcal{M}, g)$.

---

*End of unified THEORY.md (v3.0, 2026-05-28 — transpose bug fix applied, §12 rewritten).*
