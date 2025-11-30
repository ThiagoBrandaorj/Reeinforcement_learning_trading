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


# ========================= CONFIGURAÇÕES =========================
class Config:
    GAMMA = 0.99
    NUM_EPISODES = 100
    ALPHA = 0.1
    EPSILON = 0.1
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01
    POLICY_IMPROVEMENT_EVERY = 10


config = Config()


# ========================= CCXT FIX (CRÍTICO) =========================
def reset_ccxt():
    """Reseta conexões internas do CCXT evitando travamentos no Python 3.12."""
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
    """Baixa os dados recriando a exchange para evitar erros nas execuções repetidas."""
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


# ========================= FUNÇÕES AUXILIARES =========================
def reward_function(history):
    portfolio_vals = history["portfolio_valuation"]
    if len(portfolio_vals) < 2:
        return 0
    ratio = portfolio_vals[-1] / portfolio_vals[-2]
    if ratio <= 0:
        return -1
    return np.log(ratio)


def discretize(obs, df=None):
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


def politica_fixa(observation):
    if random.random() < 0.3:
        return random.randint(0, 5)
    return 3


def politica_epsilon_greedy(estado, epsilon, Q, env):
    estado_discreto = discretize(estado)
    if random.random() < epsilon:
        return env.action_space.sample()
    q_values = [Q.get((estado_discreto, a), 0) for a in range(env.action_space.n)]
    max_q = max(q_values)
    best = [a for a, q in enumerate(q_values) if q == max_q]
    return random.choice(best)


def politica_gulosa(estado, Q, env):
    return politica_epsilon_greedy(estado, 0, Q, env)


# ========================= CARREGAMENTO DE DADOS =========================
def load_data():
    print("♻️ Resetando CCXT...")
    reset_ccxt()


    os.makedirs("data", exist_ok=True)
    data_path = "./data/binance-ETHUSDT-1d.pkl"


    # ⚠️ IMPORTANTE: só baixa SE o arquivo ainda não existir
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




# ========================= AGENTES =========================
class MonteCarloEveryVisit:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.returns_state = defaultdict(list)
        self.returns_sa = defaultdict(list)
        self.V = {}
        self.Q = {}
        self.episode_returns = []


    def run(self, env, num_ep, policy):
        print("=== Monte Carlo Every Visit ===")
        for ep in range(num_ep):
            traj = []
            obs, info = env.reset()
            done = trunc = False
            ep_ret = 0


            while not done and not trunc:
                s = discretize(obs)
                a = policy(obs)
                nxt, r, done, trunc, info = env.step(a)
                traj.append((s, r, a))
                obs = nxt
                ep_ret += r


            self.episode_returns.append(ep_ret)


            G = 0
            for t in reversed(range(len(traj))):
                s, r, a = traj[t]
                G = r + self.gamma * G
                self.returns_state[s].append(G)
                self.V[s] = np.mean(self.returns_state[s])
                self.returns_sa[(s,a)].append(G)
                self.Q[(s,a)] = np.mean(self.returns_sa[(s,a)])


        return self.V, self.Q, self.episode_returns


class MonteCarloFirstVisit:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.returns_state = defaultdict(list)
        self.returns_sa = defaultdict(list)
        self.V = {}
        self.Q = {}
        self.episode_returns = []


    def run(self, env, num_ep, policy):
        print("=== MC First Visit ===")
        for ep in range(num_ep):
            traj = []
            obs, info = env.reset()
            done = trunc = False
            ep_ret = 0


            while not done and not trunc:
                s = discretize(obs)
                a = policy(obs)
                nxt, r, done, trunc, info = env.step(a)
                traj.append((s, r, a))
                obs = nxt
                ep_ret += r


            self.episode_returns.append(ep_ret)


            visited_s = set()
            visited_sa = set()
            G = 0


            for t in reversed(range(len(traj))):
                s, r, a = traj[t]
                G = r + self.gamma * G


                if s not in visited_s:
                    visited_s.add(s)
                    self.returns_state[s].append(G)
                    self.V[s] = np.mean(self.returns_state[s])


                if (s,a) not in visited_sa:
                    visited_sa.add((s,a))
                    self.returns_sa[(s,a)].append(G)
                    self.Q[(s,a)] = np.mean(self.returns_sa[(s,a)])


        return self.V, self.Q, self.episode_returns


class MonteCarloPolicyImprovement:
    def __init__(self, gamma=0.99, improvement_every=10):
        self.gamma = gamma
        self.improvement_every = improvement_every
        self.returns_sa = defaultdict(list)
        self.Q = {}
        self.policy = {}
        self.episode_returns = []
        self.policy_changes_history = []


    def run(self, env, num_ep):
        print("=== MC Policy Improvement ===")


        def initial_policy(obs):
            return env.action_space.sample()


        cur_policy = initial_policy


        for ep in range(num_ep):
            traj = []
            obs, info = env.reset()
            done = trunc = False
            ep_ret = 0


            while not done and not trunc:
                s = discretize(obs)
                a = cur_policy(obs)
                nxt, r, done, trunc, info = env.step(a)
                traj.append((s, r, a))
                obs = nxt
                ep_ret += r


            self.episode_returns.append(ep_ret)


            visited_sa = set()
            G = 0
            for t in reversed(range(len(traj))):
                s, r, a = traj[t]
                G = r + self.gamma * G
                if (s,a) not in visited_sa:
                    visited_sa.add((s,a))
                    self.returns_sa[(s,a)].append(G)
                    self.Q[(s,a)] = np.mean(self.returns_sa[(s,a)])


            if (ep+1) % self.improvement_every == 0:
                changes = self._improve_policy(env)
                self.policy_changes_history.append(changes)


                def improved(obs):
                    s = discretize(obs)
                    return self.policy.get(s, env.action_space.sample())


                cur_policy = improved


        return self.Q, self.episode_returns, self.policy


    def _improve_policy(self, env):
        changes = 0
        for (s,a), q in self.Q.items():
            best = a
            best_q = q
            for ac in range(env.action_space.n):
                q2 = self.Q.get((s,ac), -np.inf)
                if q2 > best_q:
                    best = ac
                    best_q = q2
            if s not in self.policy or self.policy[s] != best:
                self.policy[s] = best
                changes += 1
        return changes


class SarsaAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, eps_decay=0.995, min_eps=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        self.Q = {}
        self.V = {}
        self.episode_returns = []


    def run(self, env, num_ep):
        print("=== SARSA ===")
        for ep in range(num_ep):
            obs, info = env.reset()
            s_disc = discretize(obs)
            a = self._choose(obs, env)
            done = False
            ep_ret = 0


            while not done:
                nxt, r, term, trunc, info = env.step(a)
                nxt_disc = discretize(nxt)
                done = term or trunc
                nxt_a = self._choose(nxt, env)


                q = self.Q.get((s_disc,a), 0)
                q_next = 0 if done else self.Q.get((nxt_disc,nxt_a), 0)
                self.Q[(s_disc,a)] = q + self.alpha*(r + self.gamma*q_next - q)


                self.V[s_disc] = max(self.Q.get((s_disc,x),0) for x in range(env.action_space.n))


                s_disc = nxt_disc
                a = nxt_a
                ep_ret += r


            self.episode_returns.append(ep_ret)
            self.epsilon = max(self.min_eps, self.epsilon * self.eps_decay)


        return self.V, self.Q, self.episode_returns


    def _choose(self, estado, env):
        return politica_epsilon_greedy(estado, self.epsilon, self.Q, env)


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, eps_decay=0.995, min_eps=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        self.Q = {}
        self.V = {}
        self.episode_returns = []

    def run(self, env, num_ep):
        print("=== Q-Learning ===")
        for ep in range(num_ep):
            obs, info = env.reset()
            s_disc = discretize(obs)
            done = False
            ep_ret = 0

            while not done:
                a = self._choose(obs, env)
                nxt, r, term, trunc, info = env.step(a)
                done = term or trunc
                nxt_disc = discretize(nxt)

                q = self.Q.get((s_disc, a), 0)
                max_q_next = max([self.Q.get((nxt_disc, a2), 0) for a2 in range(env.action_space.n)])
                self.Q[(s_disc, a)] = q + self.alpha * (r + self.gamma * max_q_next - q)
                self.V[s_disc] = max([self.Q.get((s_disc, a2), 0) for a2 in range(env.action_space.n)])

                s_disc = nxt_disc
                obs = nxt
                ep_ret += r

            self.episode_returns.append(ep_ret)
            self.epsilon = max(self.min_eps, self.epsilon * self.eps_decay)

        return self.V, self.Q, self.episode_returns

    def _choose(self, estado, env):
        return politica_epsilon_greedy(estado, self.epsilon, self.Q, env)

# ========================= TESTE FINAL PARA RENDER =========================
def test_sarsa_with_render(env, sarsa_agent):
    print("\n🎥 EXECUTANDO SARSA FINAL PARA RENDER...")

    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        s_disc = discretize(obs)

        # Ação gulosa (sem exploração)
        a = politica_gulosa(obs, sarsa_agent.Q, env)

        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        total_reward += r

    print(f"✅ Episódio renderizado - Retorno SARSA: {total_reward:.4f}")

    # SALVA LOG PARA RENDER
    env.save_for_render(dir="render_logs")


# ========================= MAIN =========================
def main(return_results=False):
    print("⚡ Modo rápido" if return_results else "🎯 Execução com gráficos")
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


    mc_ev = MonteCarloEveryVisit(config.GAMMA)
    V_ev, Q_ev, ret_ev = mc_ev.run(env, config.NUM_EPISODES, politica_fixa)
    results["MC Every Visit"] = {"V":V_ev,"Q":Q_ev,"returns":ret_ev}


    mc_fv = MonteCarloFirstVisit(config.GAMMA)
    V_fv, Q_fv, ret_fv = mc_fv.run(env, config.NUM_EPISODES, politica_fixa)
    results["MC First Visit"] = {"V":V_fv,"Q":Q_fv,"returns":ret_fv}


    mc_pi = MonteCarloPolicyImprovement(config.GAMMA, config.POLICY_IMPROVEMENT_EVERY)
    Q_pi, ret_pi, pol_pi = mc_pi.run(env, config.NUM_EPISODES)
    results["MC Policy Improvement"] = {"Q":Q_pi,"returns":ret_pi}


    sarsa = SarsaAgent(
        config.ALPHA,config.GAMMA,
        config.EPSILON,config.EPSILON_DECAY,
        config.MIN_EPSILON
    )
    V_sa, Q_sa, ret_sa = sarsa.run(env, config.NUM_EPISODES)
    results["SARSA"] = {"V":V_sa,"Q":Q_sa,"returns":ret_sa}
    
    if not return_results:
        test_sarsa_with_render(env, sarsa)

    qlearn = QLearningAgent(
        config.ALPHA, config.GAMMA,
        config.EPSILON, config.EPSILON_DECAY,
        config.MIN_EPSILON
    )
    V_ql, Q_ql, ret_ql = qlearn.run(env, config.NUM_EPISODES)
    results["Q-Learning"] = {"V":V_ql,"Q":Q_ql,"returns":ret_ql}

    if return_results:
        env.close()
        return results


    print(f"⏱ Tempo total: {time.time()-start:.2f}s")
    env.close()
    return results


# ========================= LOOP DE 10 EXECUÇÕES =========================
def run_multiple_executions(n_runs=10):
    print(f"\n🚀 Rodando {n_runs} execuções completas...\n")


    metrics = {k: [] for k in ["MC Every Visit","MC First Visit","MC Policy Improvement","SARSA","Q-Learning"]}


    for i in range(n_runs):
        print(f"\n============================")
        print(f"   EXECUÇÃO {i+1}/{n_runs}")
        print(f"============================\n")


        r = main(return_results=True)
        for algo in metrics.keys():
            m = np.mean(r[algo]["returns"][-5:])
            metrics[algo].append(m)
            print(f"  {algo}: {m:.4f}")

    print("\n📈 MÉDIAS FINAIS:\n")
    for algo, vals in metrics.items():
        print(f"{algo}: média={np.mean(vals):.4f}, desvio={np.std(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}")


    return metrics


# ========================= EXECUÇÃO =========================
if __name__ == "__main__":
    run_multiple_executions(10)