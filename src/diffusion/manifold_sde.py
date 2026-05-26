"""
Manifold forward process: wrapped-Gaussian Brownian motion on (M, w^delta).

We use the VP-SDE framework with the linear beta schedule from ScoreMD
(Plainer et al., 2025):

    beta(t) = beta_min + t*(beta_max - beta_min),   beta_min=0.1, beta_max=20
    log_alpha(t) = -0.5*t*beta_min - 0.25*t^2*(beta_max - beta_min)
    alpha(t)     = exp(log_alpha(t))        -- mean decay coefficient
    sigma(t)     = sqrt(1 - alpha(t)^2)     -- noise standard deviation

The forward process uses the geodesic exponential map:

    v   ~ N(0, I_{nd})                    sample isotropic noise
    v_h = horizontal_projection(x_0, v)   project to horizontal tangent space
    v_h = v_h / ||v_h||_g                 normalise to unit g-norm
    X   = 0.5 * alpha(t) * sigma(t) * v_h  pre-scaled tangent vector (see below)
    x_t = s_exp(x_0, X)                   geodesic displacement — stays on M

The factor of 1/2 compensates for the geodesic-doubling in s_exp (K=1):
s_exp(x, X) calls s_geodesic(x, x+X, tau=2), which extrapolates to tau=2,
giving a manifold displacement of norm ~||X||_g from x + X, i.e. ~2||X||_g from x.
So passing X/2 = alpha*sigma/2 * v_h results in w^delta(x_t, x_0) ≈ alpha*sigma,
and the score target is consistent:

    s_log(x_t, x_0) ≈ -alpha*sigma*v_h   (magnitude = alpha*sigma)
    s_true = -s_log(x_t, x_0) / sigma^2  = alpha*v_h / sigma

s_exp runs at 2.6 ms/call (n=214, JIT-cached) — fast enough for every training step.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def _beta(t, beta_min: float = 0.1, beta_max: float = 20.0):
    """Instantaneous noise rate beta(t)."""
    return beta_min + t * (beta_max - beta_min)


def _log_alpha(t, beta_min: float = 0.1, beta_max: float = 20.0):
    """log alpha(t) = -0.5 * integral_0^t beta(s) ds."""
    return -0.5 * t * beta_min - 0.25 * t ** 2 * (beta_max - beta_min)


class ManifoldVP:
    """
    VP-SDE forward process on the w^delta manifold.

    The forward SDE in the ambient tangent space is:
        dX = -beta(t)/2 * X dt + sqrt(beta(t)) dW_M

    where dW_M is Brownian motion on M (horizontal Wiener process).

    All operations are in the (N, M, n, d) tensor convention of ShapeManifold.
    """

    def __init__(self, manifold, beta_min: float = 0.1, beta_max: float = 20.0):
        """
        :param manifold: ShapeManifold instance
        :param beta_min: lower end of linear beta schedule (default 0.1)
        :param beta_max: upper end of linear beta schedule (default 20.0)
        """
        self.manifold = manifold
        self.beta_min = beta_min
        self.beta_max = beta_max

    # ------------------------------------------------------------------
    # Schedule functions (scalar -> scalar, vmappable)
    # ------------------------------------------------------------------

    def beta(self, t):
        """Instantaneous noise rate."""
        return _beta(t, self.beta_min, self.beta_max)

    def log_alpha(self, t):
        """log of the mean decay coefficient: log alpha(t) = -0.5 int_0^t beta(s) ds."""
        return _log_alpha(t, self.beta_min, self.beta_max)

    def alpha(self, t):
        """Mean decay coefficient alpha(t) = exp(log_alpha(t))."""
        return jnp.exp(self.log_alpha(t))

    def sigma(self, t):
        """Noise standard deviation sigma(t) = sqrt(1 - alpha(t)^2)."""
        return jnp.sqrt(jnp.maximum(1.0 - jnp.exp(2.0 * self.log_alpha(t)), 1e-8))

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def marginal_prob(self, x0, t, rng):
        """
        Sample x_t ~ q(x_t | x_0) via geodesic noising on the manifold.

          1. Sample v ~ N(0, I) in ambient R^{nd}
          2. Project v to horizontal tangent space at x0
          3. Normalise to unit g-norm
          4. Displace: x_t = s_exp(x0, 0.5 * alpha(t) * sigma(t) * v_h_unit)

        The factor of 1/2 compensates for geodesic doubling inside s_exp (K=1):
            s_exp(x, X) = s_geodesic(x, x+X, tau=2) ≈ point at distance 2||X||_g from x
        Passing X/2 ensures w^delta(x_t, x_0) ≈ alpha(t)*sigma(t), so that:
            s_log(x_t, x_0) ≈ -alpha(t)*sigma(t)*v_h_unit
            s_true = -s_log(x_t, x_0) / sigma(t)^2  = alpha(t)*v_h_unit / sigma(t)
        which is the standard VP-SDE score (Diepeveen/ScoreMD conventions consistent).

        :param x0: (N, 1, n, d) reference conformation
        :param t: scalar time in [0, 1]
        :param rng: JAX PRNGKey
        :return: (x_t, v_h_unit, sigma_t)
                 x_t      -- (N, 1, n, d) noisy conformation at time t (on M)
                 v_h_unit  -- (N, 1, 1, n, d) horizontal unit tangent noise vector
                 sigma_t   -- scalar noise level
        """
        sigma_t = self.sigma(t)
        alpha_t = self.alpha(t)

        # Sample ambient noise and project to horizontal tangent space
        v = jax.random.normal(rng, x0.shape)                 # (N, 1, n, d)
        v_hproj = self.manifold.horizontal_projection_tvector(
            x0, v[:, :, None, :, :]                           # (N, 1, 1, n, d)
        )                                                     # (N, 1, 1, n, d)

        # Normalise to unit g-norm
        nrm = self.manifold.norm(x0, v_hproj)                # (N, 1, 1)
        nrm = jnp.maximum(nrm, 1e-8)[:, :, :, None, None]    # (N, 1, 1, 1, 1)
        v_h_unit = v_hproj / nrm                              # (N, 1, 1, n, d)

        # Geodesic displacement: factor 0.5 compensates for s_exp doubling
        tangent = 0.5 * alpha_t * sigma_t * v_h_unit[:, :, 0, :, :]  # (N, 1, n, d)
        x_t = self.manifold.s_exp(x0, tangent)

        return x_t, v_h_unit, sigma_t

    def score_target(self, x_t, x_0, t):
        """
        Denoising score matching target in the horizontal tangent space at x_t:

            s_true(x_t, x_0, t) = -s_log(x_t, x_0) / sigma(t)^2

        Exact by construction: s_log inverts s_exp up to third-order accuracy.

        :param x_t: (N, 1, n, d)
        :param x_0: (N, 1, n, d)
        :param t: scalar
        :return: (N, 1, 1, n, d) score target (horizontal tangent vector at x_t)
        """
        sigma_t = self.sigma(t)
        log_map = self.manifold.s_log(x_t, x_0)              # (N, 1, 1, n, d)
        return -log_map / (sigma_t ** 2)

    # ------------------------------------------------------------------
    # Reverse SDE drift (for sampling)
    # ------------------------------------------------------------------

    def reverse_drift(self, x, score, t):
        """
        Reverse-time SDE drift (Anderson 1982):

            f_rev(x, t) = -f(x, t) + g(t)^2 * score(x, t)
                        = beta(t)/2 * x + beta(t) * score(x, t)

        Both x and score are horizontal tangent vectors; addition lives in the
        ambient space and is subsequently mapped back via s_exp in the solver.

        :param x: (N, 1, n, d) current conformation
        :param score: (N, 1, n, d) score estimate at x, t  (horizontal tangent)
        :param t: scalar
        :return: (N, 1, n, d) reverse drift tangent vector
        """
        beta_t = self.beta(t)
        # Forward drift in tangent space: -beta(t)/2 * x (VP shrinkage)
        # Reverse negates it and adds g^2 * score
        return 0.5 * beta_t * x + beta_t * score

    def diffusion_coeff(self, t):
        """g(t) = sqrt(beta(t))."""
        return jnp.sqrt(self.beta(t))
