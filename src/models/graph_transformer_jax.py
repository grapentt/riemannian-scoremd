"""
Graph Transformer score model for Riemannian DSM on protein shape manifolds.

Ported from scoremd/src/scoremd/models/graph_transformer.py (Plainer 2025 / ScoreMD),
which is itself based on lucidrains' graph-transformer-pytorch and the Microsoft
two-for-one-diffusion implementation.

Key design choices vs. the ScoreMD original:
  - Standalone Flax module — no scoremd imports (no BaseDiffusionModel, Dataset, EnergyModel)
  - Interface matches TangentScoreModel: (x_flat: (B,nd), t: (B,1)) → (B,nd)
  - Conservative (energy) parameterization only: score = -∇_x E_θ(x,t)
    Same pattern as PotentialTangentScoreModel in tangent_mlp.py
  - No dropout, no `training` flag — deterministic JIT-friendly
  - Edge features: pairwise differences x_i - x_j (shape B×n×n×d)
    Provides translation-invariant structural inductive bias the MLP lacks
  - Node init: sinusoidal time embedding only (no atom type embeddings —
    BBA has only Cα, all identical type)

Usage:
    model = GraphTransformerScoreModel(n=28, d=3)
    params = model.init(rng, jnp.zeros((B, 84)), jnp.zeros((B, 1)))
    score_fn = lambda x_flat, t: model.apply(params, x_flat, t)
    # score_fn is compatible with riemannian_dsm_loss_from_noised
"""

from typing import Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange, repeat

from models.tangent_mlp import sinusoidal_time_embed


# ---------------------------------------------------------------------------
# Helper modules (ported directly from ScoreMD)
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    """Apply LayerNorm to the first argument before calling fn."""
    fn: Callable

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.LayerNorm()(x)
        return self.fn(x, *args, **kwargs)


class GatedResidual(nn.Module):
    """Gated residual connection: gate = σ(W[x; res; x-res]), out = gate*x + (1-gate)*res."""

    @nn.compact
    def __call__(self, x, res):
        gate_input = jnp.concatenate([x, res, x - res], axis=-1)
        gate = nn.Dense(1, use_bias=False)(gate_input)
        gate = nn.sigmoid(gate)
        return x * gate + res * (1 - gate)


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Multi-head attention with edge bias.

    Nodes attend to each other; the edge features (pairwise differences, projected
    to hidden_dim) bias the key and value projections. This gives the model direct
    access to relative geometry x_i - x_j at every attention step.

    :param heads:    Number of attention heads
    :param dim_head: Dimension per head
    """
    heads: int = 8
    dim_head: int = 64

    @nn.compact
    def __call__(self, nodes, edges, mask=None):
        """
        :param nodes: (B, n, hidden_dim)
        :param edges: (B, n, n, hidden_dim)
        :param mask:  (B, n) bool or None
        :return:      (B, n, hidden_dim)
        """
        h = self.heads
        inner_dim = self.dim_head * h
        scale = self.dim_head ** -0.5

        q = nn.Dense(inner_dim)(nodes)
        k = nn.Dense(inner_dim)(nodes)
        v = nn.Dense(inner_dim)(nodes)
        e_kv = nn.Dense(inner_dim)(edges)

        q = rearrange(q, "b ... (h d) -> (b h) ... d", h=h)
        k = rearrange(k, "b ... (h d) -> (b h) ... d", h=h)
        v = rearrange(v, "b ... (h d) -> (b h) ... d", h=h)
        e_kv = rearrange(e_kv, "b ... (h d) -> (b h) ... d", h=h)

        ek, ev = e_kv, e_kv

        k = rearrange(k, "b j d -> b () j d")
        v = rearrange(v, "b j d -> b () j d")

        # Add edge bias to keys and values
        k = k + ek
        v = v + ev

        sim = jnp.einsum("b i d, b i j d -> b i j", q, k) * scale

        if mask is not None:
            mask2d = rearrange(mask, "b i -> b i ()") & rearrange(mask, "b j -> b () j")
            mask2d = repeat(mask2d, "b i j -> (b h) i j", h=h)
            sim = jnp.where(mask2d, sim, -jnp.finfo(sim.dtype).max)

        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("b i j, b i j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return nn.Dense(nodes.shape[-1])(out)


# ---------------------------------------------------------------------------
# Graph Transformer layer
# ---------------------------------------------------------------------------

class GraphTransformerLayer(nn.Module):
    """
    One graph transformer layer:
      1. Pre-norm attention with edge bias → gated residual
      2. Pre-norm feedforward (hidden_dim × ff_mult → hidden_dim) → gated residual

    :param heads:    Number of attention heads
    :param dim_head: Dimension per head
    :param ff_mult:  Feedforward hidden size multiplier (default 4)
    """
    heads: int = 8
    dim_head: int = 8
    ff_mult: int = 4

    @nn.compact
    def __call__(self, nodes, edges, mask=None):
        """
        :param nodes: (B, n, hidden_dim)
        :param edges: (B, n, n, hidden_dim)
        :param mask:  (B, n) bool or None
        :return:      (B, n, hidden_dim)
        """
        # Attention block
        attn_out = PreNorm(Attention(heads=self.heads, dim_head=self.dim_head))(
            nodes, edges, mask=mask)
        nodes = GatedResidual()(attn_out, nodes)

        # Feedforward block
        hidden = nodes.shape[-1]
        ff_out = PreNorm(
            nn.Sequential([
                nn.Dense(hidden * self.ff_mult),
                nn.gelu,
                nn.Dense(hidden),
            ])
        )(nodes)
        nodes = GatedResidual()(ff_out, nodes)

        return nodes


# ---------------------------------------------------------------------------
# Main model: GraphTransformerScoreModel
# ---------------------------------------------------------------------------

class GraphTransformerScoreModel(nn.Module):
    """
    Conservative graph transformer score model for Riemannian DSM.

    Architecture:
      1. Reshape x_flat (B, n*d) → x (B, n, d)
      2. Edge attributes: x_i - x_j → (B, n, n, d) → project to (B, n, n, hidden_dim)
      3. Node init: [one_hot(n×n) | t_emb(4)] → project to (B, n, hidden_dim)
      4. num_layers GraphTransformerLayer blocks
      5. Energy per node: Dense(1) → (B, n, 1)
      6. Score = -∇_{x_flat} Σ_b Σ_i E_θ(x_b, i, t_b)

    Interface matches TangentScoreModel and PotentialTangentScoreModel:
      __call__(x_flat: (B, n*d), t: (B, 1)) → (B, n*d)

    Default hyperparameters match ScoreMD's BBA "large_potential" config:
      hidden_dim=128, num_layers=3, num_heads=8, dim_head=64
      → ~1.5M parameters (same order as ScoreMD BBA baseline)

    Node features = one-hot atom identity (n×n) concatenated with sinusoidal
    time embedding (4), matching ScoreMD's features=None path (identity matrix).
    This gives each atom a unique positional embedding — critical for memorization
    and generalization on ordered sequences like protein backbones.

    :param n:          Number of atoms (default 28 for BBA)
    :param d:          Spatial dimension (default 3)
    :param hidden_dim: Node/edge feature dimension
    :param num_layers: Number of graph transformer layers
    :param num_heads:  Number of attention heads per layer
    :param dim_head:   Dimension per head (attention inner dim = num_heads * dim_head)
    :param ff_mult:    Feedforward multiplier in each layer
    :param input_scale: Divide x_flat by this before processing (numerical stability)
    """
    n: int = 28
    d: int = 3
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    dim_head: int = 64
    ff_mult: int = 4
    input_scale: float = 6.26

    @nn.compact
    def _energy(self, x_flat: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute per-node energy E_θ(x, t).

        :param x_flat: (B, n*d)
        :param t:      (B, 1)
        :return:       (B, n, 1) per-node energy
        """
        B = x_flat.shape[0]
        x = (x_flat / self.input_scale).reshape(B, self.n, self.d)  # (B, n, d)

        # Edge features: pairwise differences x_i - x_j → (B, n, n, d)
        xa = x[:, :, None, :]   # (B, n, 1, d)
        xb = x[:, None, :, :]   # (B, 1, n, d)
        edge_diff = xa - xb     # (B, n, n, d) — translation-invariant relative geometry
        edges = nn.Dense(self.hidden_dim, name="edge_embedding")(edge_diff)  # (B, n, n, hidden_dim)

        # Node init: one-hot atom identity (n×n) + sinusoidal time embedding (4)
        # Matches ScoreMD's features=None path: identity matrix gives each atom
        # a unique positional embedding — critical for memorization on ordered sequences.
        one_hot = jnp.tile(jnp.eye(self.n)[None], (B, 1, 1))      # (B, n, n)
        t_emb = sinusoidal_time_embed(t)                           # (B, 4)
        t_nodes = jnp.tile(t_emb[:, None, :], (1, self.n, 1))     # (B, n, 4)
        node_feats = jnp.concatenate([one_hot, t_nodes], axis=-1)  # (B, n, n+4)
        nodes = nn.Dense(self.hidden_dim, name="node_embedding")(node_feats)  # (B, n, hidden_dim)

        # All-nodes mask (no masking for BBA — all atoms present)
        mask = jnp.ones((B, self.n), dtype=bool)

        # Graph transformer layers
        for i in range(self.num_layers):
            nodes = GraphTransformerLayer(
                heads=self.num_heads,
                dim_head=self.dim_head,
                ff_mult=self.ff_mult,
                name=f"gt_layer_{i}",
            )(nodes, edges, mask=mask)

        # Per-node energy: (B, n, 1)
        return nn.Dense(1, name="node_decoder")(nodes)

    def __call__(self, x_flat: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute score = -∇_{x_flat} E_θ(x, t).

        :param x_flat: (B, n*d) flattened conformation (centred, in Ångström)
        :param t:      (B, 1) or (B,) diffusion time in [0, 1]
        :return:       (B, n*d) score estimate
        """
        t = t.reshape(-1, 1)   # ensure (B, 1)

        def energy_sum(x_flat: jnp.ndarray) -> jnp.ndarray:
            # Sum over batch and nodes to get a scalar for jax.grad
            return self._energy(x_flat, t).sum()

        return -jax.grad(energy_sum)(x_flat)
