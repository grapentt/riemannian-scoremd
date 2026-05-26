"""
Tangent-space score network for Riemannian denoising score matching.

Two variants:

  TangentScoreModel (Task 3.1)
    Non-conservative MLP: output is a horizontal tangent vector at x_t.
    Suitable for unconditional generation; does NOT guarantee a stationary
    distribution under Langevin dynamics (no energy function).

  PotentialTangentScoreModel (Task 3.2)
    Conservative variant: output = -grad_x E_theta(x_t, t).
    The negative gradient of a scalar energy network, projected horizontal.
    Required for stable MD simulation (guarantees detailed balance under
    the Langevin dynamics at any fixed t).

Both accept:
  x_flat : (B, n*d)  flattened conformation in Angstrom, already centred
  t      : (B, 1)    diffusion time in [0, 1]

and return:
  score  : (B, n*d)  score estimate in the horizontal tangent space

The horizontal projection step is deferred to the training loss
(score_loss.py) which has access to the ShapeManifold instance.
The models output a raw vector in R^{n*d}; the loss projects it horizontal
before computing ||s_theta - s_true||^2_g.

Time embedding: 4 sinusoidal features (same as ScoreMD BaseDiffusionModel):
  [t - 0.5,  cos(2pi*t),  sin(2pi*t),  -cos(4pi*t)]

Architecture: 4 Dense layers, default hidden_dims=[256, 256, 256, 256],
tanh activation (smooth, bounded — better than ReLU for score functions
near t=1 where ||score||_g is small and we want smooth gradients).
"""

from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Time embedding (shared)
# ---------------------------------------------------------------------------

def sinusoidal_time_embed(t: jnp.ndarray) -> jnp.ndarray:
    """
    4-feature sinusoidal time embedding, same as ScoreMD BaseDiffusionModel.
    :param t: (B, 1) or scalar
    :return:  (B, 4)
    """
    t = t.reshape(-1, 1)
    return jnp.concatenate([
        t - 0.5,
        jnp.cos(2.0 * jnp.pi * t),
        jnp.sin(2.0 * jnp.pi * t),
        -jnp.cos(4.0 * jnp.pi * t),
    ], axis=-1)   # (B, 4)


# ---------------------------------------------------------------------------
# Task 3.1: Non-conservative score network
# ---------------------------------------------------------------------------

class TangentScoreModel(nn.Module):
    """
    Non-conservative 4-layer MLP score network.

    Input:   [x_flat (n*d),  time_embed (4)]  →  (B, n*d + 4)
    Hidden:  Dense(hidden_dims[i])  +  activation  (4 layers)
    Output:  Dense(n*d)  — raw vector, projected horizontal in the loss

    Default: hidden_dims=(256, 256, 256, 256), activation=nn.tanh
    """
    hidden_dims: Sequence[int] = (256, 256, 256, 256)
    activation: callable = nn.tanh

    @nn.compact
    def __call__(self, x_flat: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        :param x_flat: (B, n*d)
        :param t:      (B, 1) or (B,)
        :return:       (B, n*d)
        """
        t_emb = sinusoidal_time_embed(t)         # (B, 4)
        h = jnp.concatenate([x_flat, t_emb], axis=-1)

        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = self.activation(h)

        return nn.Dense(x_flat.shape[-1])(h)     # (B, n*d)


# ---------------------------------------------------------------------------
# Task 3.2: Conservative (energy-based) score network
# ---------------------------------------------------------------------------

class PotentialTangentScoreModel(nn.Module):
    """
    Conservative score network: score = -grad_x E_theta(x_t, t).

    The energy network E_theta maps (x_flat, t) → scalar.
    The score is the negative gradient of E_theta w.r.t. x_flat.
    This guarantees that the score is a gradient field, ensuring
    a stationary distribution exists under Langevin dynamics.

    The gradient is taken BEFORE horizontal projection; the loss
    projects the result horizontal (same as the non-conservative variant).

    Default: hidden_dims=(256, 256, 256, 256), activation=nn.tanh
    """
    hidden_dims: Sequence[int] = (256, 256, 256, 256)
    activation: callable = nn.tanh

    @nn.compact
    def _energy(self, x_flat: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Scalar energy E_theta(x, t). :return: (B, 1)"""
        t_emb = sinusoidal_time_embed(t)
        h = jnp.concatenate([x_flat, t_emb], axis=-1)

        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = self.activation(h)

        return nn.Dense(1)(h)                    # (B, 1)

    def __call__(self, x_flat: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        :param x_flat: (B, n*d)
        :param t:      (B, 1) or (B,)
        :return:       (B, n*d) score = -grad_x E_theta
        """
        # grad w.r.t. x_flat summed over batch
        def energy_sum(x):
            return self._energy(x, t).sum()

        return -jax.grad(energy_sum)(x_flat)
