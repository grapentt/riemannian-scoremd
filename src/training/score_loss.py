"""
Riemannian denoising score matching loss (Task 3.3).

The standard (Euclidean) DSM loss for VP-SDE is:

    L_DSM = E_{t, x0, x_t} [ ||s_theta(x_t, t) - s_true(x_t, x0, t)||^2 ]

On the w^delta manifold the squared norm is measured in the Riemannian
metric at x_t:

    L_DSM = E [ ||s_theta(x_t, t) - s_true||^2_{g(x_t)} ]

where:
  s_true = -s_log(x_t, x0) / sigma(t)^2          (ManifoldVP.score_target)
  g(x_t) = metric_tensor(x_t)                     (ShapeManifold.metric_tensor)
  ||v||^2_g = inner(x_t, v, v)                    (ShapeManifold.inner)

Both s_theta and s_true are projected to the horizontal tangent space
at x_t before computing the norm.

---

VMAP NOTE: s_exp and s_log cannot be vmapped because they call
    K = int(c * float(jnp.max(norm))) + 1
which requires a concrete Python int and is incompatible with jax.vmap's
abstract tracing.

The loss is therefore split in two phases:
  Phase A (Python loop, outside JIT): marginal_prob + score_target per sample
    → produces x_t_batch and s_true_batch as concrete JAX arrays.
  Phase B (JIT-friendly): score network forward + horizontal projection +
    g-norm residual. horizontal_projection_tvector is vmap-safe.

The training loop uses this split: Phase A outside jax.jit, Phase B inside.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Phase A (vmapped): batched geodesic noising for GPU-parallel training
# ---------------------------------------------------------------------------

def prepare_batch_vmapped(manifold, sde, x0: jnp.ndarray, t: jnp.ndarray,
                           rng: jax.random.PRNGKey, fixed_K: int = 1):
    """
    Vmapped version of prepare_batch. Runs marginal_prob + score_target for
    the entire batch in a single JIT-compiled GPU kernel — no Python loop.

    Requirements:
      - fixed_K must be a Python int constant (not data-dependent).
        K=1 is valid for BBA (n=28) and chignolin (n=10). For larger proteins,
        verify with: max(int(0.25 * float(manifold.norm(x[i:i+1,None], X[i:i+1,None,None]))) + 1)
        over training frames.
      - Both marginal_prob and score_target must be pure JAX (no Python-level
        data-dependent branches). With fixed_K this is satisfied.

    Speed vs prepare_batch:
      CPU n=28: ~256ms/step (B=32, Python loop)  →  ~3ms/step (vmapped, JIT)
      GPU n=28: ~2900ms/step (Python loop)        →  ~1ms/step (vmapped, single kernel)

    :param x0:     (B, n, d) clean conformations
    :param t:      (B,) diffusion times
    :param rng:    JAX PRNGKey (split per sample inside vmap)
    :param fixed_K: K for s_exp (default 1, valid for BBA/chignolin)
    :return: (x_t, s_true) each (B, n, d)
    """
    B = x0.shape[0]
    rng_keys = jax.random.split(rng, B)

    def _single(xi, ti, ki):
        """Process one sample. xi: (n,d), ti: scalar, ki: PRNGKey."""
        xi_batched = xi[None, None]                              # (1, 1, n, d)
        x_t_i, _, _ = sde.marginal_prob(xi_batched, ti, ki, fixed_K=fixed_K)
        # x_t_i: (1, 1, n, d)
        s_t_i = sde.score_target(x_t_i, xi_batched, ti)         # (1, 1, 1, n, d)
        return x_t_i[0, 0], s_t_i[0, 0, 0]                     # (n,d), (n,d)

    x_t_batch, s_true_batch = jax.vmap(_single)(x0, t, rng_keys)
    return x_t_batch, s_true_batch                              # (B, n, d), (B, n, d)


# ---------------------------------------------------------------------------
# Phase A: geometric preprocessing (Python loop, outside JIT)
# ---------------------------------------------------------------------------

def prepare_batch(manifold, sde, x0: jnp.ndarray, t: jnp.ndarray,
                  rng: jax.random.PRNGKey, use_slog: bool = True):
    """
    Run marginal_prob and score_target for each sample.
    Must be called OUTSIDE jax.jit (s_exp/s_log need concrete K).

    :param x0: (B, n, d)
    :param t:  (B,) diffusion times
    :param use_slog: if True (default), use s_log (Riemannian gradient) for score
        targets — geometrically correct, O((nd)³), recommended for all production use.
        Set False only for smoke tests or online training where eigh cost is prohibitive.
    :return:   (x_t, s_true) each (B, n, d)
    """
    B = x0.shape[0]
    x_t_list = []
    s_true_list = []
    rng_keys = jax.random.split(rng, B)

    for i in range(B):
        xi = x0[i:i+1, None]               # (1, 1, n, d)
        ti = t[i]
        x_t_i, _, _ = sde.marginal_prob(xi, ti, rng_keys[i])              # (1, 1, n, d)
        s_t_i = sde.score_target(x_t_i, xi, ti, use_slog=use_slog)        # (1, 1, 1, n, d)

        x_t_list.append(x_t_i[0, 0])       # (n, d)
        s_true_list.append(s_t_i[0, 0, 0]) # (n, d)

    return jnp.stack(x_t_list), jnp.stack(s_true_list)


# ---------------------------------------------------------------------------
# Phase B: JIT-friendly loss given pre-noised data
# ---------------------------------------------------------------------------

def riemannian_dsm_loss_from_noised(
    score_fn,
    manifold,
    sde,
    x_t: jnp.ndarray,
    s_true: jnp.ndarray,
    t: jnp.ndarray,
    likelihood_weighting: bool = True,
    use_riemannian_norm: bool = False,
) -> jnp.ndarray:
    """
    Riemannian DSM loss given pre-noised x_t and score targets.
    JIT-friendly (no s_exp/s_log calls).

    :param score_fn: (x_flat: (B,nd), t_col: (B,1)) -> (B,nd)
    :param x_t:      (B, n, d) noisy conformations
    :param s_true:   (B, n, d) score targets
    :param t:        (B,) diffusion times
    :param likelihood_weighting: if True, weight each sample by beta(t)
    :param use_riemannian_norm: if True, use g-norm (theoretical, but numerically
           unstable due to near-zero metric eigenvalues). Default False: Euclidean
           norm on projected horizontal residual. Both converge to the same score.
    :return:         scalar loss
    """
    B, n, d = x_t.shape

    # ---- Score prediction ----
    x_flat = x_t.reshape(B, n * d)
    s_pred_flat = score_fn(x_flat, t.reshape(B, 1))   # (B, nd)
    s_pred = s_pred_flat.reshape(B, n, d)

    # ---- Project both to horizontal tangent space ----
    def project_h(xi_1, v_i):
        return manifold.horizontal_projection_tvector(
            xi_1[None],          # (1, 1, n, d)
            v_i[None, None, None],
        )[0, 0, 0]

    s_true_h = jax.vmap(project_h)(x_t[:, None], s_true)
    s_pred_h = jax.vmap(project_h)(x_t[:, None], s_pred)

    # ---- Squared error ----
    residual = s_pred_h - s_true_h

    if use_riemannian_norm:
        # g-norm: theoretically correct but numerically unstable
        # (metric tensor has near-zero eigenvalues; gradients can become NaN)
        def g_norm_sq(xi_1, v_i):
            return manifold.inner(
                xi_1[None], v_i[None, None, None], v_i[None, None, None]
            )[0, 0, 0]
        norms_sq = jax.vmap(g_norm_sq)(x_t[:, None], residual)
    else:
        # Euclidean norm on projected horizontal residual — numerically stable.
        # Equivalent to g-norm up to metric eigenvalue weighting; same fixed point.
        norms_sq = jnp.sum(residual ** 2, axis=(-2, -1))

    # ---- Time weighting: w(t) = beta(t) ----
    if likelihood_weighting:
        norms_sq = norms_sq * jax.vmap(sde.beta)(t)

    return jnp.mean(norms_sq)


# ---------------------------------------------------------------------------
# Flat (Euclidean) baseline: Phase A + B without any manifold geometry
# ---------------------------------------------------------------------------

def prepare_batch_flat(sde, x0: jnp.ndarray, t: jnp.ndarray,
                       rng: jax.random.PRNGKey):
    """
    Euclidean VP-SDE batch preparation — no geodesics, no manifold.
    Fully vmappable (pure JAX).

    Forward process: x_t = alpha(t)*x0 + sigma(t)*eps,  eps ~ N(0,I)
    Score target:    s_true = -(x_t - alpha(t)*x0) / sigma(t)^2
                            = -eps / sigma(t)

    :param x0: (B, n, d)
    :param t:  (B,) diffusion times
    :param rng: JAX PRNGKey
    :return: (x_t, s_true) each (B, n, d)
    """
    B = x0.shape[0]
    eps = jax.random.normal(rng, x0.shape)            # (B, n, d)

    # Per-sample alpha/sigma — vmap over batch
    alpha_t = jax.vmap(sde.alpha)(t)                  # (B,)
    sigma_t = jax.vmap(sde.sigma)(t)                  # (B,)

    a = alpha_t[:, None, None]                         # (B, 1, 1)
    s = sigma_t[:, None, None]                         # (B, 1, 1)

    x_t   = a * x0 + s * eps                          # (B, n, d)
    s_true = -eps / s                                  # (B, n, d)

    return x_t, s_true


def flat_dsm_loss_from_noised(
    score_fn,
    sde,
    x_t: jnp.ndarray,
    s_true: jnp.ndarray,
    t: jnp.ndarray,
    likelihood_weighting: bool = True,
) -> jnp.ndarray:
    """
    Flat (Euclidean) DSM loss — no manifold projection.
    JIT-friendly counterpart to riemannian_dsm_loss_from_noised.

    :param score_fn: (x_flat: (B,nd), t_col: (B,1)) -> (B,nd)
    :param x_t:     (B, n, d)
    :param s_true:  (B, n, d)
    :param t:       (B,)
    :return: scalar loss
    """
    B, n, d = x_t.shape
    x_flat = x_t.reshape(B, n * d)
    s_pred = score_fn(x_flat, t.reshape(B, 1)).reshape(B, n, d)

    norms_sq = jnp.sum((s_pred - s_true) ** 2, axis=(-2, -1))  # (B,)

    if likelihood_weighting:
        norms_sq = norms_sq * jax.vmap(sde.beta)(t)

    return jnp.mean(norms_sq)


# ---------------------------------------------------------------------------
# Convenience: full DSM loss (A + B combined, for testing outside of training)
# ---------------------------------------------------------------------------

def riemannian_dsm_loss(
    score_fn, manifold, sde,
    x0: jnp.ndarray, t: jnp.ndarray, rng: jax.random.PRNGKey,
    likelihood_weighting: bool = True,
) -> jnp.ndarray:
    """
    Full Riemannian DSM loss (Phase A + B). For use in tests and evaluation.
    For training, the loop calls prepare_batch + riemannian_dsm_loss_from_noised
    separately so Phase B can be JIT-compiled independently.
    """
    x_t, s_true = prepare_batch(manifold, sde, x0, t, rng)
    return riemannian_dsm_loss_from_noised(
        score_fn, manifold, sde, x_t, s_true, t, likelihood_weighting
    )
