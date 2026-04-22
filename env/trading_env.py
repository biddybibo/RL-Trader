"""
Single-stock continuous-action trading environment.

Observation (17 floats):
  Portfolio state (3):
    0  position       current stock allocation in [0, 1]
    1  cash_ratio     cash / initial_cash
    2  unrealized_pnl (portfolio_value - initial_cash) / initial_cash

  Price / momentum (6):
    3  log_return     previous day log return
    4  sma_10         (sma10 - price) / price
    5  sma_30         (sma30 - price) / price
    6  volatility     20-day rolling std of log returns
    7  rsi            RSI / 100
    8  macd           (ema12 - ema26) / price

  Volume (2):
    9  volume_ratio   today volume / 20d avg volume
   10  atr_norm       ATR(14) / price

  Range / calendar (4):
   11  hw_proximity   (price - 52w_low) / (52w_high - 52w_low)
   12  day_sin        sin(2π × day_of_week / 5)
   13  day_cos        cos(2π × day_of_week / 5)
   14  month_sin      sin(2π × (month-1) / 12)

  Macro (2):
   15  vix_norm       VIX / 30
   16  rate_norm      10Y yield / 10

Action (1 float in [-1, 1]):
  Target portfolio weight; clipped to [0, 1] for long-only.

Reward (Sortino-shaped):
  log_return
  − drawdown_penalty   (0.05 × |drawdown from peak| when in drawdown)
  − downside_penalty   (0.2  × |loss| — asymmetric: losses penalized more)
  Transaction costs and market impact are implicit in portfolio value.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

OBS_DIM = 17

# Market feature columns that must exist in the df (order matters for _obs)
_MARKET_COLS = [
    "log_return", "sma_10", "sma_30", "volatility", "rsi", "macd",
    "volume_ratio", "atr_norm",
    "hw_proximity", "day_sin", "day_cos", "month_sin",
    "vix_norm", "rate_norm",
]


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    TRANSACTION_COST = 0.001   # 0.1% base cost per unit |Δposition|
    IMPACT_SCALE     = 0.0003  # market-impact coefficient (scales with illiquidity)
    DD_PENALTY       = 0.05    # drawdown penalty coefficient
    DOWNSIDE_PENALTY = 0.20    # asymmetric loss penalty coefficient

    def __init__(self, df, initial_cash: float = 10_000.0):
        super().__init__()
        self.df           = df.reset_index(drop=True)
        self.n_steps      = len(df)
        self.initial_cash = initial_cash

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._reset_state()

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.t                  = 0
        self.cash               = self.initial_cash
        self.shares             = 0.0
        self.position           = 0.0
        self.portfolio_value    = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        self.peak_value         = self.initial_cash
        self.episode_turnover   = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        row = self.df.iloc[self.t]
        portfolio_obs = np.array([
            self.position,
            self.cash / self.initial_cash,
            (self.portfolio_value - self.initial_cash) / self.initial_cash,
        ], dtype=np.float32)

        market_obs = np.array(
            [float(row.get(col, 0.0)) for col in _MARKET_COLS],
            dtype=np.float32,
        )
        # clip to prevent exploding values from rare data issues
        market_obs = np.clip(market_obs, -10.0, 10.0)
        return np.concatenate([portfolio_obs, market_obs])

    # ------------------------------------------------------------------
    def step(self, action):
        target_alloc = float(np.clip(action[0], 0.0, 1.0))
        price        = float(self.df.iloc[self.t]["close"])

        # Rebalance to target allocation
        target_stock_value  = target_alloc * self.portfolio_value
        current_stock_value = self.shares * price
        delta_value         = target_stock_value - current_stock_value
        delta_alloc         = abs(target_alloc - self.position)

        # Transaction cost + volume-adjusted market impact
        vol_ratio   = float(self.df.iloc[self.t].get("volume_ratio", 1.0))
        illiquidity = max(0.5, 2.0 / (vol_ratio + 1e-8))  # low vol = higher cost
        tc = (self.TRANSACTION_COST + self.IMPACT_SCALE * delta_alloc * illiquidity) * abs(delta_value)

        self.shares            += delta_value / price
        self.cash              -= delta_value + tc
        self.episode_turnover  += delta_alloc

        # Advance one day
        self.t      += 1
        terminated   = self.t >= self.n_steps - 1
        next_price   = float(self.df.iloc[self.t]["close"])
        self.portfolio_value = self.shares * next_price + self.cash

        # ── Reward ───────────────────────────────────────────────────
        log_ret = float(np.log(self.portfolio_value / (self.prev_portfolio_value + 1e-8)))
        self.prev_portfolio_value = self.portfolio_value
        self.position = (self.shares * next_price) / (self.portfolio_value + 1e-8)

        # Peak tracking for drawdown penalty
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown        = (self.portfolio_value - self.peak_value) / (self.peak_value + 1e-8)
        dd_penalty      = self.DD_PENALTY * abs(min(0.0, drawdown))

        # Asymmetric downside penalty (Sortino-like)
        downside_penalty = self.DOWNSIDE_PENALTY * abs(min(0.0, log_ret))

        reward = log_ret - dd_penalty - downside_penalty

        return self._obs(), reward, terminated, False, {}
