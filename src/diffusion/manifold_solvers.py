"""
Manifold Euler–Maruyama integrator for the VP-SDE on (M, w^delta).

In flat R^n, the Euler–Maruyama step is:
    x_{t+dt} = x_t + drift(x_t, t)*dt + g(t)*sqrt(dt)*eps

On the manifold we cannot add tangent vectors to points. Instead we use the
exponential map to stay on M:

    v_drift = reverse_drift(x_t, score, t) * dt       (tangent at x_t)
    v_noise = g(t) * sqrt(dt) * w                      (tangent at x_t)
    x_{t+dt} = s_exp(x_t, v_drift + v_noise)

where w is a unit horizontal tangent noise vector sampled from the standard
Gaussian on R^{nd} and projected to the horizontal tangent space.

Both forward (diffusion) and reverse (generation) modes are supported.
"""

import jax
import jax.numpy as jnp


class ManifoldEulerMaruyama:
    """
    Euler–Maruyama integrator on the w^delta manifold.

    Replaces flat vector addition with the exp map at each step, ensuring
    that every iterate lies exactly on M and in the correct SE(d) equivalence
    class (up to alignment).

    Usage — forward diffusion (add noise):
        step = ManifoldEulerMaruyama(sde, manifold, mode='forward')
        x_next, rng = step(x, t, dt, rng)

    Usage — reverse sampling (remove noise):
        step = ManifoldEulerMaruyama(sde, manifold, mode='reverse', score_fn=s)
        x_next, rng = step(x, t, dt, rng)
    """

    def __init__(self, sde, manifold, mode: str = 'forward', score_fn=None):
        """
        :param sde: ManifoldVP instance
        :param manifold: ShapeManifold instance
        :param mode: 'forward' (add noise) or 'reverse' (denoise, requires score_fn)
        :param score_fn: callable (x, t) -> tangent vector at x; required for reverse
        """
        assert mode in ('forward', 'reverse'), f"mode must be 'forward' or 'reverse', got {mode}"
        if mode == 'reverse':
            assert score_fn is not None, "score_fn required for reverse mode"
        self.sde = sde
        self.manifold = manifold
        self.mode = mode
        self.score_fn = score_fn

    def _horizontal_noise(self, x, rng):
        """
        Sample a unit horizontal tangent noise vector at x.

        1. Draw v ~ N(0, I) in ambient R^{nd}
        2. Project to horizontal tangent space at x
        3. Normalise to unit g-norm

        :param x: (N, 1, n, d)
        :param rng: JAX PRNGKey
        :return: (N, 1, n, d) unit horizontal noise vector
        """
        v = jax.random.normal(rng, x.shape)                   # (N, 1, n, d)
        v_h = self.manifold.horizontal_projection_tvector(
            x, v[:, :, None, :, :]                            # (N, 1, 1, n, d)
        )[:, :, 0, :, :]                                      # (N, 1, n, d)
        nrm = self.manifold.norm(x, v_h[:, :, None])          # (N, 1, 1)
        nrm = jnp.maximum(nrm[:, :, 0], 1e-8)[:, :, None, None]  # (N, 1, 1, 1)
        return v_h / nrm                                       # (N, 1, n, d)

    def step(self, x, t, dt, rng):
        """
        Single Euler–Maruyama step on the manifold.

        Forward mode:
            drift = -beta(t)/2 * x                            (VP shrinkage)
            v = drift*dt + g(t)*sqrt(dt)*w                    (tangent vector)
            x_next = s_exp(x, v)

        Reverse mode:
            drift = reverse_drift(x, score(x,t), t)
            v = drift*dt + g(t)*sqrt(dt)*w                    (Langevin noise)
            x_next = s_exp(x, v)

        :param x: (N, 1, n, d) current conformation
        :param t: scalar time
        :param dt: scalar step size (positive; for reverse pass in decreasing t
                   call with dt > 0 and manage t externally)
        :param rng: JAX PRNGKey
        :return: (x_next, rng_next)
                 x_next -- (N, 1, n, d) updated conformation on M
                 rng_next -- updated PRNGKey
        """
        rng, noise_rng = jax.random.split(rng)

        g = self.sde.diffusion_coeff(t)                        # scalar
        w = self._horizontal_noise(x, noise_rng)               # (N, 1, n, d)

        if self.mode == 'forward':
            drift = -0.5 * self.sde.beta(t) * x               # (N, 1, n, d)
        else:
            score = self.score_fn(x, t)                        # (N, 1, n, d)
            drift = self.sde.reverse_drift(x, score, t)        # (N, 1, n, d)

        tangent = drift * dt + g * jnp.sqrt(dt) * w            # (N, 1, n, d)
        x_next = self.manifold.s_exp(
            x, tangent,
            base=self.manifold.base_point if self.manifold.has_base_point else None,
        )

        return x_next, rng

    def trajectory(self, x0, ts, rng):
        """
        Integrate a full trajectory over a sequence of time points.

        :param x0: (N, 1, n, d) initial conformation
        :param ts: (T,) array of time points (forward: increasing; reverse: decreasing)
        :param rng: JAX PRNGKey
        :return: (T+1, N, 1, n, d) trajectory including x0
        """
        xs = [x0]
        x = x0
        for i in range(len(ts) - 1):
            t = ts[i]
            dt = float(jnp.abs(ts[i + 1] - ts[i]))
            x, rng = self.step(x, t, dt, rng)
            xs.append(x)
        return jnp.stack(xs, axis=0)                           # (T, N, 1, n, d)
