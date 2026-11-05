import numpy as np
import pandas as pd
import pytest

import metrics
from agent.ppo import PPOAgent
from backtest import run_agent
from data.fetch import build_features
from env.trading_env import OBS_DIM, TradingEnv
from trainer import TrainConfig, Trainer


def synthetic_prices(n=800, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    idx = pd.bdate_range("2015-01-01", periods=n)
    px = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                       "volume": rng.uniform(1e6, 2e6, n)}, index=idx)
    vix = pd.Series(rng.uniform(12, 35, n), index=idx)
    tnx = pd.Series(rng.uniform(1.5, 3.5, n), index=idx)
    return px, vix, tnx


@pytest.fixture(scope="module")
def df():
    return build_features(*synthetic_prices())


def test_features_have_no_lookahead():
    px, vix, tnx = synthetic_prices()
    full = build_features(px, vix, tnx)
    cut = px.index[500]
    trunc = build_features(px.loc[:cut], vix.loc[:cut], tnx.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut], trunc, check_freq=False)


def test_obs_shape_and_scale(df):
    env = TradingEnv(df)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert np.abs(obs).max() <= 5.0


def test_full_allocation_tracks_price_minus_cost(df):
    env = TradingEnv(df)
    env.reset()
    for _ in range(env.n_steps - 1):
        env.step(np.array([1.0]))
    price_ret = df["close"].iloc[-1] / df["close"].iloc[0]
    # one entry trade costs ~0.1% plus small impact; nothing else should leak
    assert env.portfolio_value / env.initial_cash == pytest.approx(price_ret * 0.9987, rel=2e-3)


def test_random_start_episodes_are_bounded(df):
    env = TradingEnv(df, episode_length=100, random_start=True)
    starts = set()
    for s in range(5):
        env.reset(seed=s)
        starts.add(env.start)
        steps, trunc = 0, False
        while not trunc:
            *_, trunc, _ = env.step(np.array([0.5]))
            steps += 1
        assert steps == 100
    assert len(starts) > 1


def test_recurrent_replay_reproduces_rollout_logprobs(df):
    """Stored hidden state + episode-start masks must reproduce rollout log-probs (ratio == 1)."""
    agent = PPOAgent(OBS_DIM, 1, rollout_steps=512, n_epochs=1)
    cfg = TrainConfig(total_steps=512, rollout_steps=512, episode_length=45, seed=1, anneal_lr=False)
    Trainer({"SYN": df}, cfg, ckpt_dir=None, agent=agent).run()
    assert agent.last_stats["first_ratio_dev"] < 1e-4


def test_checkpoint_roundtrip_is_deterministic(df, tmp_path):
    agent = PPOAgent(OBS_DIM, 1)
    path = tmp_path / "a.pt"
    agent.save(str(path), lifetime_steps=7)
    loaded, meta = PPOAgent.from_checkpoint(str(path))
    assert meta["lifetime_steps"] == 7
    a = run_agent(agent, df.iloc[:200])["allocations"]
    b = run_agent(loaded, df.iloc[:200])["allocations"]
    np.testing.assert_allclose(a, b)
    assert np.allclose(a, run_agent(agent, df.iloc[:200])["allocations"])  # deterministic


def test_metrics():
    v = np.array([100, 110, 99, 120.0])
    assert metrics.max_drawdown(v) == pytest.approx(-0.1)
    assert metrics.total_return(v) == pytest.approx(0.2)
    assert metrics.sharpe(np.full(10, 100.0)) == 0.0
