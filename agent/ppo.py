"""
PPO with clipped surrogate objective, GAE-lambda advantage, and entropy bonus.
Supports continuous action spaces via a diagonal Gaussian policy.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal init (common PPO practice)
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        feat = self.shared(obs)
        mean = torch.tanh(self.actor_mean(feat))  # bounded to (-1, 1)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(feat).squeeze(-1)
        return mean, std, value

    def get_dist(self, obs):
        mean, std, value = self(obs)
        return Normal(mean, std), value

    def act(self, obs):
        """Sample action and return (action, log_prob, value) as numpy."""
        with torch.no_grad():
            dist, value = self.get_dist(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int, gamma: float, gae_lambda: float):
        self.size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        self.obs      = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.actions  = np.zeros((self.size, self.act_dim), dtype=np.float32)
        self.rewards  = np.zeros(self.size, dtype=np.float32)
        self.dones    = np.zeros(self.size, dtype=np.float32)
        self.values   = np.zeros(self.size, dtype=np.float32)
        self.log_probs = np.zeros(self.size, dtype=np.float32)
        self.advantages = np.zeros(self.size, dtype=np.float32)
        self.returns  = np.zeros(self.size, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_gae(self, last_value: float):
        """GAE-lambda advantage + discounted returns."""
        gae = 0.0
        for t in reversed(range(self.ptr)):
            next_value = last_value if t == self.ptr - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
        # Normalize advantages
        adv = self.advantages[:self.ptr]
        self.advantages[:self.ptr] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def get_batches(self, batch_size: int, device):
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        for start in range(0, self.ptr, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                torch.tensor(self.obs[idx], device=device),
                torch.tensor(self.actions[idx], device=device),
                torch.tensor(self.log_probs[idx], device=device),
                torch.tensor(self.advantages[idx], device=device),
                torch.tensor(self.returns[idx], device=device),
            )


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps

        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self.buffer    = RolloutBuffer(rollout_steps, obs_dim, act_dim, gamma, gae_lambda)
        self.last_loss = 0.0

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, value = self.net.act(obs_t)
        return action[0], log_prob[0], value[0]

    # ------------------------------------------------------------------
    def update(self, last_value: float):
        self.buffer.compute_gae(last_value)

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_lp_b, adv_b, ret_b in self.buffer.get_batches(
                self.batch_size, self.device
            ):
                dist, values = self.net.get_dist(obs_b)
                log_probs = dist.log_prob(act_b).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(log_probs - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (values - ret_b).pow(2).mean()

                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                self.last_loss = loss.item()

        self.buffer.reset()

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()
