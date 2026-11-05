"""
Headless multi-asset training.

Usage:
    python train.py --steps 500000                  # resumes checkpoints/multi_latest.pt
    python train.py --steps 1000000 --seed 2 --run seed2 --fresh
"""
import argparse
import time

from trainer import TRAIN_END, TRAIN_START, VAL_END, VAL_START, Hooks, TrainConfig, Trainer, load_basket


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--run", default="multi", help="Checkpoint name prefix")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--episode-length", type=int, default=252)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--gae-lambda", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--reward", default="level", choices=["level", "delta", "mv"])
    p.add_argument("--risk-aversion", type=float, default=1.0)
    p.add_argument("--train-end", default=TRAIN_END)
    p.add_argument("--val-start", default=VAL_START)
    p.add_argument("--val-end", default=VAL_END)
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--fresh", action="store_true", help="Ignore an existing checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(total_steps=args.steps, lr=args.lr, episode_length=args.episode_length,
                      seed=args.seed, device=args.device,
                      gamma=args.gamma, gae_lambda=args.gae_lambda, n_epochs=args.epochs, reward_mode=args.reward,
                      risk_aversion=args.risk_aversion)
    t0 = time.time()

    def on_rollout(tr, step, s):
        eps = tr.episodes[-30:]
        if not eps:
            return
        sr = sum(e["sharpe"] for e in eps) / len(eps)
        bh = sum(e["bnh_sharpe"] for e in eps) / len(eps)
        ex = sum(e["avg_exposure"] for e in eps) / len(eps)
        print(f"step {step:>9,} | ep Sharpe {sr:+.2f} (B&H {bh:+.2f}) | exposure {ex:.2f} | "
              f"kl {s['approx_kl']:.4f} ev {s['explained_variance']:+.2f} std {s['action_std']:.3f} | "
              f"{step / (time.time() - t0):,.0f} steps/s", flush=True)

    val = None if args.no_val else load_basket(start=args.val_start, end=args.val_end, log=lambda *_: None)
    trainer = Trainer(load_basket(start=TRAIN_START, end=args.train_end), cfg, args.ckpt_dir, args.run,
                      resume=not args.fresh, hooks=Hooks(on_rollout=on_rollout), val_datasets=val)
    out = trainer.run()
    print(f"\nDone: {out['steps']:,} steps this run, {out['lifetime_steps']:,} lifetime -> {trainer.ckpt_path}")


if __name__ == "__main__":
    main()
