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



# ========================= CONFIGURAÇÕES =========================
class Config:
    GAMMA = 0.99
    NUM_EPISODES = 20
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
                state = np.array(obs, dtype=np.float32)
                a = self.act(state)
                next_obs, r, term, trunc, _ = env.step(a)
                done = term or trunc
                next_state = np.array(next_obs, dtype=np.float32)
                self.remember(state, a, r, next_state, done)
                self.learn()
                obs = next_obs
                ep_ret += r

            self.epsilon = max(self.min_eps, self.epsilon * self.eps_decay)
            self.episode_returns.append(ep_ret)

            if (ep + 1) % update_target_every == 0:
                self.update_target()

        return self.episode_returns

# ========================= TESTE FINAL PARA RENDER =========================
def test_sarsa_with_render(env, sarsa_agent):
    print("\n🎥 EXECUTANDO SARSA FINAL PARA RENDER...")

    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        s_disc = discretize(obs)
        a = politica_gulosa(obs, sarsa_agent.Q, env)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        total_reward += r

    print(f"✅ Episódio renderizado - Retorno SARSA: {total_reward:.4f}")
    env.save_for_render(dir="render_logs")


# ========================= PLOTAGEM AGREGADA =========================
def plot_aggregated_results(all_metrics, all_returns_history):
    """Plota as métricas agregadas de todas as execuções."""
    plt.figure(figsize=(16, 12))
    
    algoritmos = list(all_metrics.keys())
    
    # 1. Retornos por episódio (média das execuções)
    plt.subplot(2, 3, 1)
    for algo in algoritmos:
        returns_array = np.array(all_returns_history[algo])
        mean_returns = np.mean(returns_array, axis=0)
        std_returns = np.std(returns_array, axis=0)
        episodes = range(len(mean_returns))
        
        plt.plot(episodes, mean_returns, label=algo, alpha=0.8, linewidth=2)
        plt.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)
    
    plt.title('Comparação de Retornos por Episódio (Média)')
    plt.xlabel('Episódio')
    plt.ylabel('Retorno')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Boxplot dos retornos finais
    plt.subplot(2, 3, 2)
    data_to_plot = [all_metrics[algo] for algo in algoritmos]
    box = plt.boxplot(data_to_plot, tick_labels=algoritmos, patch_artist=True)  # <- Correção aqui
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
    plt.title(f'Distribuição dos Retornos (últimos 5 eps)\n{len(all_metrics[algoritmos[0]])} execuções')
    plt.ylabel('Retorno Médio')
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    
    # 3. Barras com média e desvio padrão
    plt.subplot(2, 3, 3)
    means = [np.mean(all_metrics[algo]) for algo in algoritmos]
    stds = [np.std(all_metrics[algo]) for algo in algoritmos]
    x_pos = np.arange(len(algoritmos))
    bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    plt.xticks(x_pos, algoritmos, rotation=30, ha='right')
    plt.title('Retorno Médio (últimos 5 episódios)')
    plt.ylabel('Retorno')
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, (m, s) in enumerate(zip(means, stds)):
        plt.text(i, m + s + 0.01, f'{m:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    # 4. Comparação Min/Max
    plt.subplot(2, 3, 4)
    mins = [np.min(all_metrics[algo]) for algo in algoritmos]
    maxs = [np.max(all_metrics[algo]) for algo in algoritmos]
    x_pos = np.arange(len(algoritmos))
    width = 0.35
    plt.bar(x_pos - width/2, mins, width, label='Min', alpha=0.7)
    plt.bar(x_pos + width/2, maxs, width, label='Max', alpha=0.7)
    plt.xticks(x_pos, algoritmos, rotation=30, ha='right')
    plt.title('Valores Mínimos e Máximos')
    plt.ylabel('Retorno')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Ranking por desempenho médio
    plt.subplot(2, 3, 5)
    sorted_algos = sorted(algoritmos, key=lambda x: np.mean(all_metrics[x]), reverse=True)
    sorted_means = [np.mean(all_metrics[algo]) for algo in sorted_algos]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_algos)))
    bars = plt.barh(sorted_algos, sorted_means, color=colors)
    plt.title('Ranking por Desempenho Médio')
    plt.xlabel('Retorno Médio')
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, (algo, val) in enumerate(zip(sorted_algos, sorted_means)):
        plt.text(val, i, f' {val:.4f}', va='center', fontsize=9)
    
    # 6. Comparação First Visit vs Every Visit
    plt.subplot(2, 3, 6)
    if 'MC First Visit' in all_returns_history and 'MC Every Visit' in all_returns_history:
        first_returns = np.array(all_returns_history['MC First Visit'])
        every_returns = np.array(all_returns_history['MC Every Visit'])
        
        first_mean = np.mean(first_returns, axis=0)
        every_mean = np.mean(every_returns, axis=0)
        first_std = np.std(first_returns, axis=0)
        every_std = np.std(every_returns, axis=0)
        
        episodes = range(len(first_mean))
        
        plt.plot(episodes, first_mean, label='First Visit', alpha=0.8, linewidth=2)
        plt.fill_between(episodes, first_mean - first_std, first_mean + first_std, alpha=0.2)
        
        plt.plot(episodes, every_mean, label='Every Visit', alpha=0.8, linewidth=2)
        plt.fill_between(episodes, every_mean - every_std, every_mean + every_std, alpha=0.2)
        
        plt.title('First Visit vs Every Visit (Média)')
        plt.xlabel('Episódio')
        plt.ylabel('Retorno')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Análise Agregada de Múltiplas Execuções', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


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
    
    state_dim = len(env.reset()[0])  # Usa o vetor original, não discretizado
    action_dim = env.action_space.n

    dqn = DQNAgent(state_dim, action_dim, config.GAMMA, 1e-3, 1.0, config.EPSILON_DECAY, config.MIN_EPSILON)
    ret_dqn = dqn.run(env, config.NUM_EPISODES)
    results["DQN"] = {"returns": ret_dqn}


    if return_results:
        env.close()
        return results

    print(f"⏱ Tempo total: {time.time()-start:.2f}s")
    env.close()
    return results


# ========================= LOOP DE 10 EXECUÇÕES =========================
def run_multiple_executions(n_runs=10):
    print(f"\n🚀 Rodando {n_runs} execuções completas...\n")

    metrics = {k: [] for k in ["MC Every Visit","MC First Visit","MC Policy Improvement","SARSA","Q-Learning","DQN"]}
    returns_history = {k: [] for k in ["MC Every Visit","MC First Visit","MC Policy Improvement","SARSA","Q-Learning","DQN"]}


    for i in range(n_runs):
        print(f"\n============================")
        print(f"   EXECUÇÃO {i+1}/{n_runs}")
        print(f"============================\n")

        r = main(return_results=True)
        for algo in metrics.keys():
            m = np.mean(r[algo]["returns"][-5:])
            metrics[algo].append(m)
            returns_history[algo].append(r[algo]["returns"])
            print(f"  {algo}: {m:.4f}")


    print("\n📈 MÉDIAS FINAIS:\n")
    for algo, vals in metrics.items():
        print(f"{algo}: média={np.mean(vals):.4f}, desvio={np.std(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    # Gera os gráficos agregados após todas as execuções
    print("\n📊 Gerando gráficos agregados...")
    plot_aggregated_results(metrics, returns_history)

    return metrics


# ========================= EXECUÇÃO =========================
if __name__ == "__main__":
    run_multiple_executions(10)