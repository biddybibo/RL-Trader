"""
Training loop — 500k environment steps with checkpoints every 50k.

Usage:
    python train.py [--steps 500000] [--ckpt-dir checkpoints]
"""
import argparse
import os
import time
import numpy as np

from data.fetch import fetch
from env.trading_env import TradingEnv
from agent.ppo import PPOAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("Fetching training data (AAPL 2015-2022)…")
    df = fetch("AAPL", "2015-01-01", "2022-12-31")
    print(f"  {len(df)} trading days loaded")

    env = TradingEnv(df)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        device=args.device,
    )

    obs, _ = env.reset(seed=args.seed)
    hidden = agent.init_hidden()
    ep_reward, ep_rewards = 0.0, []
    next_ckpt = 50_000
    t0 = time.time()

    print(f"Training for {args.steps:,} steps\n")

    for step in range(1, args.steps + 1):
        action, log_prob, value, hidden = agent.select_action(obs, hidden)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        hx = hidden[0].squeeze().cpu().numpy()
        cx = hidden[1].squeeze().cpu().numpy()
        agent.buffer.add(obs, action, reward, float(done), value, log_prob, hx, cx)
        ep_reward += reward
        obs = next_obs

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
            hidden = agent.init_hidden()

        # PPO update when rollout buffer is full
        if step % args.rollout_steps == 0:
            _, _, last_value, _ = agent.select_action(obs, hidden)
            agent.update(last_value)

        # Logging
        if step % 10_000 == 0:
            mean_r = np.mean(ep_rewards[-20:]) if ep_rewards else 0.0
            elapsed = time.time() - t0
            print(
                f"Step {step:>8,} | ep_reward(20) {mean_r:+.4f} "
                f"| {step / elapsed:,.0f} steps/s"
            )

        # Checkpoint
        if step >= next_ckpt:
            ckpt_path = os.path.join(args.ckpt_dir, f"ppo_{step}.pt")
            agent.save(ckpt_path)
            print(f"  [ckpt] saved -> {ckpt_path}")
            next_ckpt += 50_000

    final_path = os.path.join(args.ckpt_dir, "ppo_final.pt")
    agent.save(final_path)
    print(f"\nTraining complete. Final model -> {final_path}")


if __name__ == "__main__":
    train()
