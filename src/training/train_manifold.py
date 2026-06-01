"""
Training loop for Riemannian denoising score matching (Task 3.4).

Usage:
    from training.train_manifold import train, train_from_precomputed

    # Online training (Phase A+B, ~256ms/step at BBA n=28):
    state, history = train(
        model=TangentScoreModel(),
        manifold=manifold,
        sde=ManifoldVP(manifold),
        train_data=x_train,     # (N, n, d), float32, Å, centred
        n_epochs=1000,
        batch_size=64,
        learning_rate=1e-3,
    )

    # Precomputed training (Phase B only, ~0.7ms/step — use for BBA/large datasets):
    state, history = train_from_precomputed(
        model=TangentScoreModel(),
        manifold=manifold,
        sde=ManifoldVP(manifold),
        precomputed_dir="data/precomputed/bba/",   # from scripts/precompute_noised_data.py
        n_epochs=1000,
        batch_size=64,
        learning_rate=3e-4,
    )

The training loop is a minimal but complete implementation that:
  - Uses Optax Adam with cosine learning rate decay
  - Maintains exponential moving average (EMA) of params (weight=0.995)
  - Logs loss every log_every epochs; returns full loss history
  - Samples t ~ Uniform[t_min, t_max] per batch (not per sample, for speed)
  - Saves checkpoints to ckpt_dir if provided (numpy .npz format)

For Phase 3 smoke tests and prototyping, checkpoint saving is optional.
No Hydra / config management — just plain Python function calls.
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from tqdm.auto import tqdm

from training.score_loss import (
    prepare_batch, prepare_batch_vmapped, riemannian_dsm_loss_from_noised,
    prepare_batch_flat, flat_dsm_loss_from_noised,
)


# ---------------------------------------------------------------------------
# EMA helpers (plain pytree, no Flax struct)
# ---------------------------------------------------------------------------

def ema_update(ema_params, new_params, decay: float = 0.995):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p,
        ema_params, new_params,
    )


# ---------------------------------------------------------------------------
# One training step (JIT-compiled)
# ---------------------------------------------------------------------------

def make_train_step(model, manifold, sde, optimizer, likelihood_weighting=True,
                    normalize_targets=False, eps_parameterization=False):
    """
    Returns a JIT-compiled train_step function.
    Phase A (prepare_batch: s_exp/s_log) is called OUTSIDE this step.
    Phase B (network forward + g-norm residual) is JIT-compiled here.
    """

    @jax.jit
    def train_step(params, ema_params, opt_state, x_t, s_true, t_batch):
        """
        One gradient step given pre-noised data from prepare_batch.
        :param x_t:    (B, n, d) noisy conformations
        :param s_true: (B, n, d) score targets
        :param t_batch:(B,) diffusion times
        :return: (params, ema_params, opt_state, loss_scalar)
        """

        def loss_fn(params):
            score_fn = lambda x_flat, t_col: model.apply(params, x_flat, t_col)
            return riemannian_dsm_loss_from_noised(
                score_fn=score_fn,
                manifold=manifold,
                sde=sde,
                x_t=x_t,
                s_true=s_true,
                t=t_batch,
                likelihood_weighting=likelihood_weighting,
                normalize_targets=normalize_targets,
                eps_parameterization=eps_parameterization,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_ema_params = ema_update(ema_params, new_params)

        return new_params, new_ema_params, new_opt_state, loss

    return train_step


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    manifold,
    sde,
    train_data: np.ndarray,
    n_epochs: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    grad_clip: float = 1.0,
    t_min: float = 0.01,
    t_max: float = 0.99,
    ema_decay: float = 0.995,
    likelihood_weighting: bool = True,
    seed: int = 0,
    log_every: int = 100,
    ckpt_dir: Optional[str] = None,
    fixed_K: Optional[int] = 1,
    normalize_gyration: bool = False,
) -> Tuple[dict, list]:
    """
    Train a TangentScoreModel (or PotentialTangentScoreModel) on clean frames.

    :param model:               Flax nn.Module (TangentScoreModel or variant)
    :param manifold:            ShapeManifold instance
    :param sde:                 ManifoldVP instance
    :param train_data:          (N, n, d) float32, Å, centred
    :param n_epochs:            number of full passes over the data
    :param learning_rate:       peak Adam learning rate (cosine decay to 1e-5)
    :param grad_clip:           global gradient norm clip (default 1.0)
    :param t_min / t_max:       diffusion time range (avoids t=0 / t=1 instability)
    :param ema_decay:           EMA weight (0.995 = ScoreMD default)
    :param likelihood_weighting: multiply loss by beta(t) (Song 2021)
    :param seed:                RNG seed
    :param log_every:           print loss every N epochs
    :param ckpt_dir:            if provided, save params.npz there every log_every epochs
    :param fixed_K:             if not None, use vmapped prepare_batch with this fixed K
                                for s_exp (default 1, valid for BBA/chignolin). Set to
                                None to fall back to the original Python-loop prepare_batch
                                (needed if K>1 frames exist in the dataset).
    :param normalize_gyration:  if True, rescale each batch frame to unit gyration radius
                                before computing s_exp and score targets.  Expected to
                                reduce metric tensor condition number and may eliminate H
                                indefiniteness at δ=1.0.  The model trains on normalised
                                coordinates; sampling (Phase 5) should call
                                manifold.denormalize() to recover physical Å units.
                                NOTE: train_from_precomputed loads pre-saved x_t / s_true
                                from disk — normalisation must be applied at precompute
                                time there, not at training time.
    :return: (state_dict, loss_history)
             state_dict has keys: 'params', 'ema_params'
             loss_history: list of (epoch, loss) tuples
    """
    # Auto-detect GPU: use vmapped path on GPU, Python loop on CPU
    if fixed_K == 1:
        has_gpu = any(d.platform == 'gpu' for d in jax.devices())
        if not has_gpu:
            fixed_K = None  # vmapped is slower on CPU — fall back to Python loop

    N, n, d = train_data.shape
    nd = n * d
    n_steps_per_epoch = max(1, N // batch_size)
    total_steps = n_epochs * n_steps_per_epoch

    # ---- Optimizer: Adam + cosine decay + gradient clipping ----
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps,
        alpha=1e-5 / learning_rate,   # final lr = 1e-5
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(schedule),
    )

    # ---- Init model ----
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    dummy_x = jnp.zeros((1, nd))
    dummy_t = jnp.zeros((1, 1))
    params = model.init(init_key, dummy_x, dummy_t)
    ema_params = params
    opt_state = optimizer.init(params)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {n_params:,}")
    if fixed_K is not None:
        print(f"  Phase A: vmapped (fixed_K={fixed_K}) — GPU-parallel batch processing")
    else:
        print(f"  Phase A: Python loop (fixed_K=None) — sequential per-sample")
    if normalize_gyration:
        print(f"  normalize_gyration=True: frames rescaled to unit gyration radius per batch")

    # ---- Build JIT'd step ----
    train_step = make_train_step(model, manifold, sde, optimizer, likelihood_weighting,
                                 normalize_targets=False, eps_parameterization=False)

    # ---- Select batch preparation function ----
    # NOTE: online train() uses use_slog=False (prelog, fast) because s_log (eigh
    # per sample) is prohibitively slow in a Python loop at training time.
    # For production runs, always use precompute_noised_data.py + train_from_precomputed,
    # which runs s_log once at precompute time and reads saved targets during training.
    if fixed_K is not None:
        _prepare = lambda x0_b, t_b, noise_key: prepare_batch_vmapped(
            manifold, sde, x0_b, t_b, noise_key, fixed_K=fixed_K
        )
    else:
        _prepare = lambda x0_b, t_b, noise_key: prepare_batch(
            manifold, sde, x0_b, t_b, noise_key, use_slog=False
        )

    # ---- Training loop ----
    loss_history = []
    epoch_losses = []
    t0_wall = time.time()

    pbar = tqdm(range(n_epochs), desc="training", unit="epoch")
    for epoch in pbar:
        rng, shuffle_key = jax.random.split(rng)
        idx = jax.random.permutation(shuffle_key, N)
        x_shuffled = jnp.array(train_data[np.array(idx)])

        epoch_loss = 0.0
        for step in range(n_steps_per_epoch):
            # Mini-batch
            start = step * batch_size
            x0_batch = x_shuffled[start:start + batch_size]   # (B, n, d)

            # Sample t ~ Uniform[t_min, t_max] per sample
            rng, t_key, noise_key = jax.random.split(rng, 3)
            t_batch = jax.random.uniform(
                t_key, (x0_batch.shape[0],),
                minval=t_min, maxval=t_max
            )

            # Phase A: geodesic noising + score target
            # Optionally normalise to unit gyration radius first
            x0_input = x0_batch
            if normalize_gyration:
                x0_input, _ = manifold.normalize(x0_batch[:, None])
                x0_input = x0_input[:, 0]                  # (B, n, d)
            x_t, s_true = _prepare(x0_input, t_batch, noise_key)

            # Phase B: JIT-compiled gradient step
            params, ema_params, opt_state, loss = train_step(
                params, ema_params, opt_state, x_t, s_true, t_batch
            )
            epoch_loss += float(loss)

        epoch_loss /= n_steps_per_epoch
        epoch_losses.append(epoch_loss)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            elapsed = time.time() - t0_wall
            loss_history.append((epoch + 1, epoch_loss))
            pbar.set_postfix(loss=f"{epoch_loss:.4f}", elapsed=f"{elapsed:.0f}s")
            tqdm.write(f"  epoch {epoch+1:5d}/{n_epochs}  loss={epoch_loss:.6f}  "
                       f"elapsed={elapsed:.1f}s")

            if ckpt_dir is not None:
                _save_checkpoint(ckpt_dir, epoch + 1, params, ema_params)

    # ---- Final checkpoint ----
    if ckpt_dir is not None:
        _save_checkpoint(ckpt_dir, n_epochs, params, ema_params, final=True)

    state_dict = {"params": params, "ema_params": ema_params}
    return state_dict, loss_history


# ---------------------------------------------------------------------------
# Flat (Euclidean) baseline training — same architecture, no manifold geometry
# ---------------------------------------------------------------------------

def make_flat_train_step(model, sde, optimizer, likelihood_weighting=True):
    """JIT-compiled flat DSM train step (Euclidean VP-SDE, no manifold)."""

    @jax.jit
    def train_step(params, ema_params, opt_state, x_t, s_true, t_batch):
        def loss_fn(params):
            score_fn = lambda x_flat, t_col: model.apply(params, x_flat, t_col)
            return flat_dsm_loss_from_noised(
                score_fn=score_fn,
                sde=sde,
                x_t=x_t,
                s_true=s_true,
                t=t_batch,
                likelihood_weighting=likelihood_weighting,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_ema_params = ema_update(ema_params, new_params)

        return new_params, new_ema_params, new_opt_state, loss

    return train_step


def train_flat(
    model: nn.Module,
    sde,
    train_data: np.ndarray,
    n_epochs: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    grad_clip: float = 1.0,
    t_min: float = 0.01,
    t_max: float = 0.99,
    ema_decay: float = 0.995,
    likelihood_weighting: bool = True,
    seed: int = 0,
    log_every: int = 100,
    ckpt_dir: Optional[str] = None,
) -> Tuple[dict, list]:
    """
    Flat (Euclidean) VP-SDE baseline. Identical to train() but with no manifold geometry:
      - x_t = alpha(t)*x0 + sigma(t)*eps  (Gaussian, not geodesic)
      - s_true = -eps / sigma(t)           (Euclidean score target)
      - loss = ||s_pred - s_true||^2       (no horizontal projection, no g-norm)

    Uses the same TangentScoreModel architecture as the Riemannian run for a fair
    architecture-controlled comparison.

    Phase A (prepare_batch_flat) is pure JAX and fully vmappable — much faster
    than the Riemannian path even on CPU. Phase B is identical.
    """
    N, n, d = train_data.shape
    nd = n * d
    n_steps_per_epoch = max(1, N // batch_size)
    total_steps = n_epochs * n_steps_per_epoch

    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps,
        alpha=1e-5 / learning_rate,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(schedule),
    )

    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    dummy_x = jnp.zeros((1, nd))
    dummy_t = jnp.zeros((1, 1))
    params = model.init(init_key, dummy_x, dummy_t)
    ema_params = params
    opt_state = optimizer.init(params)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {n_params:,}")
    print(f"  Phase A: Euclidean (vmapped, no geodesics)")

    train_step = make_flat_train_step(model, sde, optimizer, likelihood_weighting)

    loss_history = []
    t0_wall = time.time()

    for epoch in range(n_epochs):
        rng, shuffle_key = jax.random.split(rng)
        idx = jax.random.permutation(shuffle_key, N)
        x_shuffled = jnp.array(train_data[np.array(idx)])

        epoch_loss = 0.0
        for step in range(n_steps_per_epoch):
            start = step * batch_size
            x0_batch = x_shuffled[start:start + batch_size]

            rng, t_key, noise_key = jax.random.split(rng, 3)
            t_batch = jax.random.uniform(
                t_key, (x0_batch.shape[0],), minval=t_min, maxval=t_max
            )

            # Phase A: Euclidean noising (vmapped, JIT-friendly)
            x_t, s_true = prepare_batch_flat(sde, x0_batch, t_batch, noise_key)

            # Phase B: gradient step
            params, ema_params, opt_state, loss = train_step(
                params, ema_params, opt_state, x_t, s_true, t_batch
            )
            epoch_loss += float(loss)

        epoch_loss /= n_steps_per_epoch

        if (epoch + 1) % log_every == 0 or epoch == 0:
            elapsed = time.time() - t0_wall
            loss_history.append((epoch + 1, epoch_loss))
            print(f"  epoch {epoch+1:5d}/{n_epochs}  loss={epoch_loss:.6f}  "
                  f"elapsed={elapsed:.1f}s")

            if ckpt_dir is not None:
                _save_checkpoint(ckpt_dir, epoch + 1, params, ema_params)

    if ckpt_dir is not None:
        _save_checkpoint(ckpt_dir, n_epochs, params, ema_params, final=True)

    state_dict = {"params": params, "ema_params": ema_params}
    return state_dict, loss_history

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _params_to_numpy(params):
    """Flatten a Flax param tree to a dict of numpy arrays."""
    flat = {}
    for k, v in jax.tree_util.tree_leaves_with_path(params):
        key = "/".join(str(p.key) for p in k)
        flat[key] = np.array(v)
    return flat


def _save_checkpoint(ckpt_dir: str, epoch: int, params, ema_params, final: bool = False):
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = "final" if final else f"epoch_{epoch:06d}"
    path = os.path.join(ckpt_dir, f"ckpt_{tag}.npz")
    flat = _params_to_numpy(params)
    flat_ema = {f"ema/{k}": v for k, v in _params_to_numpy(ema_params).items()}
    np.savez(path, **flat, **flat_ema)


def load_checkpoint(path: str, model: nn.Module, manifold, sde, example_x: np.ndarray):
    """
    Restore params from a saved .npz checkpoint.
    Returns params dict (Flax FrozenDict-compatible nested dict).
    This is a lightweight loader; for full resumable training use orbax.
    """
    data = np.load(path)
    # Reconstruct by running a forward pass to get the param tree shape, then fill in
    nd = example_x.shape[-1] * example_x.shape[-2] if example_x.ndim == 3 else example_x.shape[-1]
    dummy_x = jnp.zeros((1, nd))
    dummy_t = jnp.zeros((1, 1))
    params_template = model.init(jax.random.PRNGKey(0), dummy_x, dummy_t)

    def fill(path_tuple, leaf):
        key = "/".join(str(p.key) for p in path_tuple)
        if key in data:
            return jnp.array(data[key])
        return leaf

    params = jax.tree_util.tree_map_with_path(fill, params_template)
    return params


# ---------------------------------------------------------------------------
# Precomputed training (Phase B only — for large datasets like BBA/AK)
# ---------------------------------------------------------------------------

def train_from_precomputed(
    model: nn.Module,
    manifold,
    sde,
    precomputed_dir: str,
    n_epochs: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    grad_clip: float = 1.0,
    ema_decay: float = 0.995,
    likelihood_weighting: bool = True,
    normalize_targets: bool = False,
    eps_parameterization: bool = False,
    seed: int = 0,
    log_every: int = 100,
    ckpt_dir: Optional[str] = None,
) -> Tuple[dict, list]:
    """
    Train using pre-computed (x_t, s_true, t) pairs from precompute_noised_data.py.
    Phase B only (JIT-compiled network grad step) — ~400× faster than online training.

    Each epoch randomly selects one noise-repeat file from precomputed_dir and
    shuffles it. Multiple repeats provide stochasticity across training.

    :param precomputed_dir: directory containing noised_r{i:02d}.npz files
                             (output of scripts/precompute_noised_data.py)
    :param model: Flax nn.Module (TangentScoreModel or variant)
    :param manifold: ShapeManifold instance (needed for loss computation)
    :param sde: ManifoldVP instance (needed for beta(t) weighting)
    :param normalize_targets: if True, normalize both s_pred and s_true to unit
           Euclidean norm before computing the residual (direction-only loss).
           Removes the zero-output attractor. likelihood_weighting is ignored.
    :param eps_parameterization: if True, predict v_h_unit (unit-g-norm noise
           direction) rather than the raw score. Converts s_true → v_h_unit at
           training time via v = s_true * sigma(t) / alpha(t). No time weighting.
           Recommended long-term fix. likelihood_weighting is ignored.
    :return: (state_dict, loss_history)
    """
    import glob as _glob

    precomputed_dir = Path(precomputed_dir)
    repeat_files = sorted(precomputed_dir.glob("noised_r*.npz"))
    if not repeat_files:
        raise FileNotFoundError(
            f"No noised_r*.npz files found in {precomputed_dir}. "
            "Run scripts/precompute_noised_data.py first."
        )

    # Load first file to get shape
    sample = np.load(repeat_files[0])
    N, n, d = sample["x_t"].shape
    nd = n * d
    n_steps_per_epoch = max(1, N // batch_size)
    total_steps = n_epochs * n_steps_per_epoch

    print(f"  Precomputed dataset: {len(repeat_files)} repeats × {N} frames = "
          f"{len(repeat_files)*N} total noised samples")
    print(f"  n={n} Cα, d={d}, nd={nd}, steps/epoch={n_steps_per_epoch}")

    # Score norm clip threshold: applied per-sample at load time.
    # Prevents rare extreme outliers (e.g. s_true at tiny t → sigma≈0) from
    # exploding gradients. Clip at 10× the expected p99 norm (empirically ~37 for BBA).
    # This does not bias the fixed point — clipped samples are still valid score targets,
    # just with artificially reduced magnitude (equivalent to down-weighting far outliers).
    _SCORE_NORM_CLIP = 500.0

    # ---- Optimizer ----
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps,
        alpha=1e-5 / learning_rate,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(schedule),
    )

    # ---- Init model ----
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    dummy_x = jnp.zeros((1, nd))
    dummy_t = jnp.zeros((1, 1))
    params = model.init(init_key, dummy_x, dummy_t)
    ema_params = params
    opt_state = optimizer.init(params)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {n_params:,}")

    train_step = make_train_step(model, manifold, sde, optimizer, likelihood_weighting,
                                 normalize_targets=normalize_targets,
                                 eps_parameterization=eps_parameterization)

    loss_history = []
    t0_wall = time.time()
    n_repeats = len(repeat_files)

    _running_loss_sum = 0.0
    _running_loss_steps = 0

    pbar = tqdm(range(n_epochs), desc="training", unit="epoch")
    for epoch in pbar:
        rng, shuffle_key = jax.random.split(rng)
        repeat_file = repeat_files[epoch % n_repeats]

        # Load this repeat's data
        data = np.load(repeat_file)
        x_t_np    = data["x_t"].astype(np.float32)    # (N, n, d)
        s_true_np = data["s_true"].astype(np.float32)  # (N, n, d)
        t_np      = data["t"].astype(np.float32)       # (N,)

        # Drop NaN frames (rare: ~3 per 630k across all BBA repeats)
        valid = ~(np.isnan(x_t_np).any(axis=(-2, -1)) |
                  np.isnan(s_true_np).any(axis=(-2, -1)))
        if not valid.all():
            x_t_np, s_true_np, t_np = x_t_np[valid], s_true_np[valid], t_np[valid]

        # Clip extreme s_true outliers by per-sample norm (prevents gradient explosion)
        norms = np.sqrt((s_true_np ** 2).sum(axis=(-2, -1), keepdims=True))  # (N,1,1)
        scale = np.minimum(1.0, _SCORE_NORM_CLIP / (norms + 1e-8))
        s_true_np = s_true_np * scale

        x_t_all    = jnp.array(x_t_np)
        s_true_all = jnp.array(s_true_np)
        t_all      = jnp.array(t_np)
        N_valid    = x_t_all.shape[0]

        # Shuffle
        idx = jax.random.permutation(shuffle_key, N_valid)
        x_t_all    = x_t_all[idx]
        s_true_all = s_true_all[idx]
        t_all      = t_all[idx]

        n_steps = max(1, N_valid // batch_size)
        for step in range(n_steps):
            start = step * batch_size
            x_t_b   = x_t_all[start:start + batch_size]
            s_true_b = s_true_all[start:start + batch_size]
            t_b      = t_all[start:start + batch_size]

            params, ema_params, opt_state, loss = train_step(
                params, ema_params, opt_state, x_t_b, s_true_b, t_b
            )
            step_loss = float(loss)
            _running_loss_sum += step_loss
            _running_loss_steps += 1

        if (epoch + 1) % log_every == 0 or epoch == 0:
            smoothed_loss = _running_loss_sum / max(1, _running_loss_steps)
            _running_loss_sum = 0.0
            _running_loss_steps = 0
            elapsed = time.time() - t0_wall
            loss_history.append((epoch + 1, smoothed_loss))
            pbar.set_postfix(loss=f"{smoothed_loss:.4f}", elapsed=f"{elapsed:.0f}s")
            tqdm.write(f"  epoch {epoch+1:5d}/{n_epochs}  loss={smoothed_loss:.6f}  "
                       f"elapsed={elapsed:.1f}s  repeat={repeat_file.name}")

            if ckpt_dir is not None:
                _save_checkpoint(ckpt_dir, epoch + 1, params, ema_params)

    if ckpt_dir is not None:
        _save_checkpoint(ckpt_dir, n_epochs, params, ema_params, final=True)

    state_dict = {"params": params, "ema_params": ema_params}
    return state_dict, loss_history
