"""
PPO with an LSTM ActorCritic.

Recurrent-PPO details that matter for correctness:
  * The buffer stores the hidden state *before* each step, so replaying a stored
    sequence from that state reproduces the rollout's log-probs exactly
    (ratio = 1 on the first epoch).
  * episode_starts masks reset the hidden state at episode boundaries inside a
    training sequence, so memory never leaks from one episode into the next.
  * deterministic=True uses the policy mean (for backtests and exports).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

SEQ_LEN = 32
CKPT_FORMAT = 2


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, init_log_std: float = -1.0):
        super().__init__()
        self.hidden_dim = hidden
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        self.critic = nn.Linear(hidden, 1)

        nn.init.orthogonal_(self.encoder[0].weight, gain=np.sqrt(2))
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)  # sigmoid(0) = 50% allocation at start
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs, hidden, episode_starts=None):
        """obs: (batch, seq, obs_dim) or (batch, obs_dim); episode_starts: (batch, seq) or None."""
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        feat = self.encoder(obs)
        h, c = hidden
        if episode_starts is not None and episode_starts[:, 1:].any():
            outs = []
            for t in range(feat.shape[1]):
                keep = (1.0 - episode_starts[:, t]).view(1, -1, 1)
                o, (h, c) = self.lstm(feat[:, t:t + 1], (h * keep, c * keep))
                outs.append(o)
            out = torch.cat(outs, dim=1)
        else:
            if episode_starts is not None:
                keep = (1.0 - episode_starts[:, 0]).view(1, -1, 1)
                h, c = h * keep, c * keep
            out, (h, c) = self.lstm(feat, (h, c))
        mean = torch.sigmoid(self.actor_mean(out))
        std = self.actor_log_std.clamp(-3.0, 0.5).exp().expand_as(mean)
        value = self.critic(out).squeeze(-1)
        return mean, std, value, (h, c)

    def init_hidden(self, batch_size: int = 1, device="cpu"):
        z = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (z, z.clone())


class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, hidden_dim, gamma, gae_lambda, seq_len=SEQ_LEN):
        assert size % seq_len == 0, "rollout_steps must be a multiple of seq_len"
        self.size, self.seq_len, self.hidden_dim = size, seq_len, hidden_dim
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.reset()

    def reset(self):
        n = self.size
        self.obs = np.zeros((n, self.obs_dim), np.float32)
        self.actions = np.zeros((n, self.act_dim), np.float32)
        self.rewards = np.zeros(n, np.float32)
        self.dones = np.zeros(n, np.float32)
        self.starts = np.zeros(n, np.float32)
        self.values = np.zeros(n, np.float32)
        self.log_probs = np.zeros(n, np.float32)
        self.advantages = np.zeros(n, np.float32)
        self.returns = np.zeros(n, np.float32)
        self.hidden_h = np.zeros((n, self.hidden_dim), np.float32)
        self.hidden_c = np.zeros((n, self.hidden_dim), np.float32)
        self.ptr = 0

    @property
    def full(self) -> bool:
        return self.ptr >= self.size

    def add(self, obs, action, reward, done, value, log_prob, h_in, c_in, episode_start):
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.dones[i], self.values[i], self.log_probs[i] = done, value, log_prob
        self.hidden_h[i], self.hidden_c[i], self.starts[i] = h_in, c_in, episode_start
        self.ptr += 1

    def compute_gae(self, last_value: float):
        gae = 0.0
        for t in reversed(range(self.ptr)):
            nv = last_value if t == self.ptr - 1 else self.values[t + 1]
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * nv * nonterminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            self.advantages[t] = gae
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(self, n_seqs_per_batch: int, device):
        sl = self.seq_len
        starts = np.arange(0, (self.ptr // sl) * sl, sl)
        np.random.shuffle(starts)
        T = lambda a: torch.as_tensor(a, device=device)
        for i in range(0, len(starts), n_seqs_per_batch):
            b = starts[i:i + n_seqs_per_batch]
            idx = b[:, None] + np.arange(sl)[None, :]
            yield (
                T(self.obs[idx]), T(self.actions[idx]), T(self.log_probs[idx]),
                T(self.advantages[idx]), T(self.returns[idx]), T(self.starts[idx]),
                (T(self.hidden_h[b]).unsqueeze(0), T(self.hidden_c[b]).unsqueeze(0)),
            )


class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, entropy_coef=0.003, value_coef=0.5, n_epochs=4,
                 batch_size=256, rollout_steps=2048, hidden_dim=128, seq_len=SEQ_LEN,
                 target_kl=0.03, max_grad_norm=0.5, device="cpu"):
        self.config = dict(obs_dim=obs_dim, act_dim=act_dim, lr=lr, gamma=gamma,
                           gae_lambda=gae_lambda, clip_eps=clip_eps, entropy_coef=entropy_coef,
                           value_coef=value_coef, n_epochs=n_epochs, batch_size=batch_size,
                           rollout_steps=rollout_steps, hidden_dim=hidden_dim, seq_len=seq_len,
                           target_kl=target_kl, max_grad_norm=max_grad_norm)
        self.device = torch.device(device)
        self.base_lr, self.gamma = lr, gamma
        self.clip_eps, self.entropy_coef, self.value_coef = clip_eps, entropy_coef, value_coef
        self.n_epochs, self.target_kl, self.max_grad_norm = n_epochs, target_kl, max_grad_norm
        self.n_seqs_per_batch = max(1, batch_size // seq_len)
        self.rollout_steps = rollout_steps
        self.last_loss = 0.0
        self.last_stats: dict = {}

        self.net = ActorCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(rollout_steps, obs_dim, act_dim, hidden_dim, gamma, gae_lambda, seq_len)

    def init_hidden(self):
        return self.net.init_hidden(1, self.device)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, hidden, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        mean, std, value, new_hidden = self.net(obs_t, hidden)
        mean, std = mean[0, 0], std[0, 0]
        dist = Normal(mean, std)
        action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.cpu().numpy(), float(log_prob), float(value[0, 0]), new_hidden

    @torch.no_grad()
    def value(self, obs: np.ndarray, hidden) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        return float(self.net(obs_t, hidden)[2][0, 0])

    def set_lr_fraction(self, frac: float):
        for g in self.optimizer.param_groups:
            g["lr"] = self.base_lr * frac

    def update(self, last_value: float) -> dict:
        buf = self.buffer
        buf.compute_gae(last_value)
        adv = buf.advantages[:buf.ptr]
        buf.advantages[:buf.ptr] = (adv - adv.mean()) / (adv.std() + 1e-8)

        pl, vl, ent, kls, clips, n = 0.0, 0.0, 0.0, [], 0.0, 0
        first_ratio_dev = None
        epochs_run = 0
        for epoch in range(self.n_epochs):
            epoch_kls = []
            for obs_b, act_b, old_lp, adv_b, ret_b, st_b, h_b in buf.get_batches(self.n_seqs_per_batch, self.device):
                mean, std, values, _ = self.net(obs_b, h_b, st_b)
                dist = Normal(mean, std)
                lp = dist.log_prob(act_b).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                logratio = lp - old_lp
                ratio = logratio.exp()
                if first_ratio_dev is None:
                    first_ratio_dev = float((ratio.detach() - 1).abs().max())
                surr = torch.min(ratio * adv_b, ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b)
                actor_loss = -surr.mean()
                value_loss = 0.5 * (values - ret_b).pow(2).mean()
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl = float(((ratio - 1) - logratio).mean())
                    clips += float(((ratio - 1).abs() > self.clip_eps).float().mean())
                epoch_kls.append(kl)
                pl += actor_loss.item(); vl += value_loss.item(); ent += entropy.item(); n += 1
                self.last_loss = loss.item()
            kls.extend(epoch_kls)
            epochs_run = epoch + 1
            if self.target_kl and np.mean(epoch_kls) > self.target_kl:
                break

        v, r = buf.values[:buf.ptr], buf.returns[:buf.ptr]
        explained_var = float(1 - np.var(r - v) / (np.var(r) + 1e-8))
        self.last_stats = {
            "loss": round(self.last_loss, 5),
            "policy_loss": round(pl / max(n, 1), 5),
            "value_loss": round(vl / max(n, 1), 5),
            "entropy": round(ent / max(n, 1), 4),
            "approx_kl": round(float(np.mean(kls)) if kls else 0.0, 5),
            "clip_frac": round(clips / max(n, 1), 4),
            "explained_variance": round(explained_var, 4),
            "action_std": round(float(self.net.actor_log_std.clamp(-3, 0.5).exp().mean()), 4),
            "epochs": epochs_run,
            "first_ratio_dev": first_ratio_dev or 0.0,
        }
        buf.reset()
        return self.last_stats

    # ── Checkpoints ───────────────────────────────────────────────────
    def save(self, path: str, **meta):
        torch.save({"format": CKPT_FORMAT, "config": self.config, "model": self.net.state_dict(),
                    "optim": self.optimizer.state_dict(), "meta": meta}, path)

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(ck, dict) or ck.get("format") != CKPT_FORMAT:
            raise ValueError(f"{path} is an old-format checkpoint (observation/action spec changed); retrain.")
        agent = cls(**ck["config"], device=device)
        agent.net.load_state_dict(ck["model"])
        agent.optimizer.load_state_dict(ck["optim"])
        return agent, ck.get("meta", {})
