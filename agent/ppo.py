"""
PPO with LSTM ActorCritic.
The LSTM gives the agent memory across timesteps so it can learn
market regimes, trends, and multi-day patterns instead of reacting
purely to the current observation.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

SEQ_LEN = 32  # BPTT truncation length


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.hidden_dim = hidden

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.actor_mean    = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic        = nn.Linear(hidden, 1)

        nn.init.orthogonal_(self.encoder[0].weight, gain=np.sqrt(2))
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs, hidden):
        # obs: (batch, seq_len, obs_dim)  OR  (batch, obs_dim) for single step
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        feat = self.encoder(obs)                        # (batch, seq, hidden)
        lstm_out, new_hidden = self.lstm(feat, hidden)  # (batch, seq, hidden)
        mean  = torch.tanh(self.actor_mean(lstm_out))
        std   = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(lstm_out).squeeze(-1)
        return mean, std, value, new_hidden

    def init_hidden(self, batch_size: int = 1, device="cpu"):
        return (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
        )

    def act(self, obs, hidden):
        """Single-step inference; returns numpy arrays + new hidden state."""
        with torch.no_grad():
            mean, std, value, new_hidden = self(obs, hidden)
            mean  = mean.squeeze(1)
            std   = std.squeeze(1)
            value = value.squeeze(1)
            dist     = Normal(mean, std)
            action   = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy(), new_hidden


# ---------------------------------------------------------------------------
# Rollout buffer (stores hidden states for truncated BPTT)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, hidden_dim,
                 gamma, gae_lambda, seq_len=SEQ_LEN):
        self.size       = size
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.reset()

    def reset(self):
        self.obs        = np.zeros((self.size, self.obs_dim),    dtype=np.float32)
        self.actions    = np.zeros((self.size, self.act_dim),    dtype=np.float32)
        self.rewards    = np.zeros(self.size,                    dtype=np.float32)
        self.dones      = np.zeros(self.size,                    dtype=np.float32)
        self.values     = np.zeros(self.size,                    dtype=np.float32)
        self.log_probs  = np.zeros(self.size,                    dtype=np.float32)
        self.advantages = np.zeros(self.size,                    dtype=np.float32)
        self.returns    = np.zeros(self.size,                    dtype=np.float32)
        self.hidden_h   = np.zeros((self.size, self.hidden_dim), dtype=np.float32)
        self.hidden_c   = np.zeros((self.size, self.hidden_dim), dtype=np.float32)
        self.ptr        = 0

    def add(self, obs, action, reward, done, value, log_prob, hx, cx):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.values[self.ptr]    = value
        self.log_probs[self.ptr] = log_prob
        self.hidden_h[self.ptr]  = hx
        self.hidden_c[self.ptr]  = cx
        self.ptr += 1

    def compute_gae(self, last_value: float):
        gae = 0.0
        for t in reversed(range(self.ptr)):
            nv  = last_value if t == self.ptr - 1 else self.values[t + 1]
            nt  = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * nv * nt - self.values[t]
            gae   = delta + self.gamma * self.gae_lambda * nt * gae
            self.advantages[t] = gae
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
        adv = self.advantages[:self.ptr]
        self.advantages[:self.ptr] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def get_batches(self, n_seqs_per_batch: int, device):
        """Yield sequence-aligned mini-batches for truncated BPTT."""
        n          = (self.ptr // self.seq_len) * self.seq_len
        seq_starts = np.arange(0, n, self.seq_len)
        np.random.shuffle(seq_starts)
        sl = self.seq_len

        for i in range(0, len(seq_starts), n_seqs_per_batch):
            batch = seq_starts[i: i + n_seqs_per_batch]
            if len(batch) == 0:
                continue

            obs_b  = np.stack([self.obs[s:s+sl]        for s in batch])
            act_b  = np.stack([self.actions[s:s+sl]    for s in batch])
            lp_b   = np.stack([self.log_probs[s:s+sl]  for s in batch])
            adv_b  = np.stack([self.advantages[s:s+sl] for s in batch])
            ret_b  = np.stack([self.returns[s:s+sl]    for s in batch])
            hx     = np.stack([self.hidden_h[s]         for s in batch])
            cx     = np.stack([self.hidden_c[s]         for s in batch])

            h0 = torch.tensor(hx, device=device).unsqueeze(0)  # (1, batch, hidden)
            c0 = torch.tensor(cx, device=device).unsqueeze(0)

            yield (
                torch.tensor(obs_b,  device=device),
                torch.tensor(act_b,  device=device),
                torch.tensor(lp_b,   device=device),
                torch.tensor(adv_b,  device=device),
                torch.tensor(ret_b,  device=device),
                (h0, c0),
            )


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(
        self,
        obs_dim, act_dim,
        lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_eps=0.2, entropy_coef=0.01, value_coef=0.5,
        n_epochs=10, batch_size=64, rollout_steps=2048,
        hidden_dim=128, seq_len=SEQ_LEN,
        device="cpu",
    ):
        self.device           = torch.device(device)
        self.clip_eps         = clip_eps
        self.entropy_coef     = entropy_coef
        self.value_coef       = value_coef
        self.n_epochs         = n_epochs
        self.n_seqs_per_batch = max(1, batch_size // seq_len)
        self.rollout_steps    = rollout_steps
        self.last_loss        = 0.0

        self.net       = ActorCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer(
            rollout_steps, obs_dim, act_dim, hidden_dim, gamma, gae_lambda, seq_len
        )

    def init_hidden(self):
        return self.net.init_hidden(batch_size=1, device=str(self.device))

    def select_action(self, obs: np.ndarray, hidden):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, value, new_hidden = self.net.act(obs_t, hidden)
        return action[0], log_prob[0], value[0], new_hidden

    def update(self, last_value: float):
        self.buffer.compute_gae(last_value)

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_lp_b, adv_b, ret_b, hidden_b in self.buffer.get_batches(
                self.n_seqs_per_batch, self.device
            ):
                mean, std, values, _ = self.net(obs_b, hidden_b)
                dist      = Normal(mean, std)
                log_probs = dist.log_prob(act_b).sum(-1)
                entropy   = dist.entropy().sum(-1).mean()

                ratio  = torch.exp(log_probs - old_lp_b)
                surr1  = ratio * adv_b
                surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (values - ret_b).pow(2).mean()
                loss       = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                self.last_loss = loss.item()

        self.buffer.reset()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()
