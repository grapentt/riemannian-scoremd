"""
Phase 3.6 — BBA training run.

Trains TangentScoreModel(256×4) on pre-computed BBA noised data for 3000 epochs.
Requires scripts/precompute_noised_data.py to have been run first:

    python scripts/precompute_noised_data.py --protein bba --n-repeats 10

Usage:
    python scripts/train_bba.py
    python scripts/train_bba.py --n-epochs 3000 --lr 3e-4 --ckpt-dir runs/bba_run1/

Gate (Phase 3.6):
    Loss curve should show clear decrease and plateau.
    Qualitative check: final loss < 70% of initial loss.
    (Phase 3.7 will do quantitative score recovery validation.)
"""

import sys
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from manifold.pointcloud_jax import ShapeManifold
from diffusion.manifold_sde import ManifoldVP
from models.tangent_mlp import TangentScoreModel
from training.train_manifold import train_from_precomputed


def main():
    parser = argparse.ArgumentParser(description="Phase 3.6 BBA training run.")
    parser.add_argument("--n-epochs",  type=int,   default=3000)
    parser.add_argument("--batch-size", type=int,  default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--grad-clip",  type=float, default=1.0)
    parser.add_argument("--ema-decay",  type=float, default=0.995)
    parser.add_argument("--hidden-dims", type=int,  nargs="+",
                        default=[256, 256, 256, 256])
    parser.add_argument("--log-every",  type=int,   default=100)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--ckpt-dir",   type=str,
                        default=str(_ROOT / "runs" / "bba_phase36"))
    parser.add_argument("--precomputed-dir", type=str,
                        default=str(_ROOT / "data" / "precomputed" / "bba"))
    args = parser.parse_args()

    # ---- Load manifold ----
    ref_path = _ROOT / "data" / "processed" / "bba_ref.npy"
    x_ref = np.load(ref_path)
    n, d = x_ref.shape[0], x_ref.shape[1]
    print(f"BBA: n={n} Cα, d={d}  →  nd={n*d} dims")

    manifold = ShapeManifold(dim=d, numpoints=n, alpha=1.0, base=x_ref)
    sde = ManifoldVP(manifold)

    # ---- Model ----
    model = TangentScoreModel(hidden_dims=tuple(args.hidden_dims))
    print(f"Architecture: TangentScoreModel({args.hidden_dims})")

    # ---- Train ----
    print(f"\nStarting Phase 3.6 training:")
    print(f"  n_epochs={args.n_epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  precomputed_dir={args.precomputed_dir}")
    print(f"  ckpt_dir={args.ckpt_dir}\n")

    state, history = train_from_precomputed(
        model=model,
        manifold=manifold,
        sde=sde,
        precomputed_dir=args.precomputed_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        likelihood_weighting=True,
        seed=args.seed,
        log_every=args.log_every,
        ckpt_dir=args.ckpt_dir,
    )

    # ---- Summary ----
    first_loss = history[0][1]
    last_loss  = history[-1][1]
    ratio = last_loss / first_loss
    ok = ratio < 0.7 and not np.isnan(last_loss)

    print(f"\n{'='*60}")
    print(f"Phase 3.6 training complete")
    print(f"  Initial loss: {first_loss:.4f}")
    print(f"  Final loss:   {last_loss:.4f}  ({ratio*100:.1f}% of initial)")
    print(f"  Gate (<70%):  {'PASS' if ok else 'FAIL (check loss curve)'}")
    print(f"{'='*60}")

    # Save loss history
    import json
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / "loss_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"\nLoss history saved to {history_path}")
    print("Next: Phase 3.7 score recovery validation on BBA")


if __name__ == "__main__":
    main()
