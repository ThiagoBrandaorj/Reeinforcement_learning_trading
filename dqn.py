import gymnasium as gym
import numpy as np
import gym_trading_env
from gym_trading_env.downloader import download
import random
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import ccxt
import torch
import torch.nn as nn
import torch.optim as optim

# ========================= CONFIG =========================
class Config:
    GAMMA = 0.99
    NUM_EPISODES = 25
    ALPHA = 1e-3
    EPSILON = 1.0
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01

config = Config()

# ========================= FUNÇÕES AUXILIARES =========================
def reset_ccxt():
    try:
        for ex in ccxt.exchanges:
            cls = getattr(ccxt, ex, None)
            if cls and hasattr(cls, "close"):
                try:
                    cls().close()
                except:
                    pass
        time.sleep(0.2)
    except:
        pass

def safe_download():
    exchange = ccxt.binance()
    exchange.enableRateLimit = True
    try:
        download(
            exchange_names=["binance"],
            symbols=["ETH/USDT"],
            timeframe="1d",
            dir="data",
            since=datetime.datetime(2022,1,1),
            until=datetime.datetime(2025,11,1),
        )
    finally:
        try:
            exchange.close()
        except:
            pass
        del exchange
        time.sleep(0.2)

def load_data():
    print("♻️ Resetando CCXT...")
    reset_ccxt()

    os.makedirs("data", exist_ok=True)
    data_path = "./data/binance-ETHUSDT-1d.pkl"

    if not os.path.exists(data_path):
        print("⬇️ Download seguro dos dados (primeira vez)...")
        safe_download()
    else:
        print("✅ Arquivo de dados já existe. Pulando download...")

    print("📊 Carregando dados do disco...")
    df = pd.read_pickle(data_path)

    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7).max()

    df.dropna(inplace=True)
    print(f"✅ Dados OK: {len(df)} registros\n")
    return df

def reward_function(history):
    portfolio_vals = history["portfolio_valuation"]
    if len(portfolio_vals) < 2:
        return 0
    ratio = portfolio_vals[-1] / portfolio_vals[-2]
    if ratio <= 0:
        return -1
    return np.log(ratio)

def discretize(obs):
    fc, fo, fh, fl, fv, cp, lp = obs

    def quantize(value, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]):
        return int(np.digitize(value, quantiles))

    return (
        quantize(fc),
        quantize(fo - 1),
        quantize(fh - 1),
        quantize(fl - 1),
        quantize(fv),
        int(cp),
        int(lp),
    )

# ========================= DQN =========================
class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=1e-3, epsilon=1.0, eps_decay=0.995, min_eps=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.min_eps = min_eps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNet(state_dim, action_dim).to(self.device)
        self.target_net = DQNNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.memory = []
        self.batch_size = 64
        self.max_mem = 10000
        self.episode_returns = []

    def remember(self, s, a, r, s_, done):
        s = np.array(s, dtype=np.float32)
        s_ = np.array(s_, dtype=np.float32)

        if len(self.memory) >= self.max_mem:
            self.memory.pop(0)
        self.memory.append((s, a, r, s_, done))
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(state)
        return q_vals.argmax().item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, done = zip(*minibatch)

        s = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        s_ = torch.tensor(np.array(s_), dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_vals = self.policy_net(s).gather(1, a).squeeze()
        with torch.no_grad():
            max_next_q = self.target_net(s_).max(1)[0]
            target = r + self.gamma * max_next_q * (1 - done)

        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def run(self, env, num_ep):
        print("=== DQN ===")
        update_target_every = 5

        for ep in range(num_ep):
            obs, _ = env.reset()
            done = False
            ep_ret = 0

            while not done:
                s = np.array(obs, dtype=np.float32)
                a = self.act(s)
                next_obs, r, term, trunc, _ = env.step(a)
                done = term or trunc
                s_ = np.array(next_obs, dtype=np.float32)
                self.remember(s, a, r, s_, done)
                self.learn()
                obs = next_obs
                ep_ret += r

            self.epsilon = max(self.min_eps, self.epsilon * self.eps_decay)
            self.episode_returns.append(ep_ret)

            if (ep + 1) % update_target_every == 0:
                self.update_target()

        return self.episode_returns

# ========================= MAIN =========================
def main(return_results=False):
    print("⚡ Executando apenas DQN")
    start = time.time()

    df = load_data()

    env = gym.make(
        "TradingEnv",
        name="ETHUSD",
        df=df,
        positions=[-1,0,0.25,0.5,0.75,1],
        trading_fees=0.001/100,
        borrow_interest_rate=0.0003/100,
        reward_function=reward_function,
    )

    env.add_metric('Position Changes', lambda h: np.sum(np.diff(h['position'])!=0))
    env.add_metric('Episode Length', lambda h: len(h['position']))

    results = {}

    state_dim = len(discretize(env.reset()[0]))
    action_dim = env.action_space.n

    dqn = DQNAgent(state_dim, action_dim, config.GAMMA, config.ALPHA, config.EPSILON, config.EPSILON_DECAY, config.MIN_EPSILON)
    ret_dqn = dqn.run(env, config.NUM_EPISODES)
    results["DQN"] = {"returns": ret_dqn}

    if return_results:
        env.close()
        return results

    print(f"⏱ Tempo total: {time.time()-start:.2f}s")
    env.close()
    return results

# ========================= EXECUÇÃO MÚLTIPLA =========================
def run_multiple_executions(n_runs=10):
    print(f"\n🚀 Rodando {n_runs} execuções com DQN...\n")
    dqn_metrics = []

    for i in range(n_runs):
        print(f"\n============================")
        print(f"   EXECUÇÃO {i+1}/{n_runs}")
        print(f"============================\n")

        r = main(return_results=True)
        m = np.mean(r["DQN"]["returns"][-5:])
        dqn_metrics.append(m)
        print(f"  DQN: {m:.4f}")

    print("\n📈 MÉDIAS FINAIS:\n")
    print(f"DQN: média={np.mean(dqn_metrics):.4f}, desvio={np.std(dqn_metrics):.4f}, min={np.min(dqn_metrics):.4f}, max={np.max(dqn_metrics):.4f}")
    return dqn_metrics

# ========================= START =========================
if __name__ == "__main__":
    run_multiple_executions(10)
