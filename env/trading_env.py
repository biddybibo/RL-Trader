"""
Single-stock, long-only, continuous-allocation trading environment.

Observation (17 floats, all roughly unit-scale):
  Account (3):
    0  position        current stock allocation in [0, 1]
    1  drawdown        (value / episode peak - 1) / 0.1
    2  episode_return  log(value / episode start value) / 0.2
  Market (14): FEATURE_COLUMNS from data/fetch.py, standardised with the fixed
  (center, scale) pairs in FEATURE_SCALING and clipped to ±5. Fixed constants
  (not running statistics) keep train, eval and the browser export identical.

Action (1 float): target allocation, clipped to [0, 1].

Timing: the agent sees day t's close, rebalances at that close, and earns the
return from t to t+1. Costs: 0.1% of traded value plus volume-scaled impact.

Reward ("level"):  log_return − 0.05·|drawdown| − 0.20·|min(0, log_return)|
Reward ("mv"):     log_return − ½·κ·log_return²   (κ = risk_aversion)
Reward ("delta"):  log_return − 0.5·max(0, deepening of drawdown) − 0.20·|min(0, log_return)|

Episodes: with random_start=True each reset begins at a random day and runs
episode_length days (time-limit -> truncated=True, so the learner bootstraps).
With random_start=False the episode covers the whole frame (used for backtests).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data.fetch import FEATURE_COLUMNS

OBS_VERSION = "v2"
OBS_DIM = 17

FEATURE_SCALING = {  # column: (center, scale), from 2010-2022 basket statistics
    "log_return":   (0.0,   0.02),
    "sma_10":       (0.0,   0.035),
    "sma_30":       (0.0,   0.065),
    "volatility":   (0.019, 0.012),
    "rsi":          (0.54,  0.12),
    "macd":         (0.0,   0.024),
    "volume_ratio": (1.0,   0.43),
    "atr_norm":     (0.026, 0.015),
    "hw_proximity": (0.5,   0.3),
    "day_sin":      (0.0,   1.0),
    "day_cos":      (0.0,   1.0),
    "month_sin":    (0.0,   1.0),
    "vix_norm":     (0.62,  0.25),
    "rate_norm":    (0.22,  0.08),
}
OBS_NAMES = ["position", "drawdown", "episode_return"] + FEATURE_COLUMNS


def scale_features(df) -> np.ndarray:
    raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    center = np.array([FEATURE_SCALING[c][0] for c in FEATURE_COLUMNS])
    scale = np.array([FEATURE_SCALING[c][1] for c in FEATURE_COLUMNS])
    return np.clip((raw - center) / scale, -5.0, 5.0).astype(np.float32)


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    TRANSACTION_COST = 0.001
    IMPACT_SCALE = 0.0003
    DD_PENALTY = 0.05
    DOWNSIDE_PENALTY = 0.20
    DD_DELTA_PENALTY = 0.5

    def __init__(self, df, initial_cash: float = 10_000.0,
                 episode_length: int | None = None, random_start: bool = False,
                 reward_mode: str = "level", risk_aversion: float = 1.0):
        super().__init__()
        self.risk_aversion = risk_aversion
        if reward_mode not in ("level", "delta", "mv"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        self.reward_mode = reward_mode
        if len(df) < 3:
            raise ValueError("Need at least 3 rows of data")
        self.dates = df.index
        self.close = df["close"].to_numpy(dtype=np.float64)
        self.vol_ratio = df["volume_ratio"].to_numpy(dtype=np.float64)
        self.features = scale_features(df)
        self.n_steps = len(df)
        self.initial_cash = initial_cash
        self.random_start = random_start
        self.episode_length = episode_length

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self._reset_state(0)

    def _reset_state(self, start: int):
        self.start = start
        span = self.episode_length if self.episode_length else self.n_steps - 1
        self.end = min(start + span, self.n_steps - 1)
        self.t = start
        self.cash = self.initial_cash
        self.shares = 0.0
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        self.episode_turnover = 0.0
        self.values = [self.initial_cash]
        self.allocations: list[float] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = 0
        if self.random_start and self.episode_length and self.n_steps - 1 > self.episode_length:
            start = int(self.np_random.integers(0, self.n_steps - 1 - self.episode_length))
        self._reset_state(start)
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        account = np.array([
            self.position,
            (self.portfolio_value / self.peak_value - 1.0) / 0.1,
            np.log(self.portfolio_value / self.initial_cash) / 0.2,
        ], dtype=np.float32)
        return np.concatenate([np.clip(account, -5.0, 5.0), self.features[self.t]])

    def step(self, action):
        target = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        price = self.close[self.t]

        target_value = target * self.portfolio_value
        delta_value = target_value - self.shares * price
        delta_alloc = abs(target - self.position)
        illiquidity = max(0.5, 2.0 / (self.vol_ratio[self.t] + 1e-8))
        cost = (self.TRANSACTION_COST + self.IMPACT_SCALE * delta_alloc * illiquidity) * abs(delta_value)

        self.shares += delta_value / price
        self.cash -= delta_value + cost
        self.episode_turnover += delta_alloc
        self.allocations.append(target)

        prev_value = self.portfolio_value
        self.t += 1
        next_price = self.close[self.t]
        self.portfolio_value = max(self.shares * next_price + self.cash, 1e-6)
        self.position = self.shares * next_price / self.portfolio_value
        self.values.append(self.portfolio_value)

        log_ret = float(np.log(self.portfolio_value / prev_value))
        prev_dd = prev_value / self.peak_value - 1.0
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = self.portfolio_value / self.peak_value - 1.0
        downside = self.DOWNSIDE_PENALTY * abs(min(0.0, log_ret))
        if self.reward_mode == "level":   # penalise being in drawdown, every day
            reward = log_ret - self.DD_PENALTY * abs(drawdown) - downside
        elif self.reward_mode == "mv":    # mean-variance utility, no drawdown terms
            reward = log_ret - 0.5 * self.risk_aversion * log_ret ** 2
        else:                             # penalise only drawdown getting deeper
            reward = log_ret - self.DD_DELTA_PENALTY * max(0.0, prev_dd - drawdown) - downside

        truncated = self.t >= self.end
        info = {"portfolio_value": self.portfolio_value, "position": self.position,
                "price": next_price, "t": self.t, "log_return": log_ret}
        return self._obs(), reward, False, truncated, info

    # Helpers for charts / summaries
    def episode_prices(self) -> np.ndarray:
        return self.close[self.start:self.t + 1]

    def episode_dates(self):
        return self.dates[self.start:self.t + 1]
