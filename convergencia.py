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
    NUM_EPISODES = 50000
    ALPHA = 0.1
    EPSILON = 0.1
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01
    POLICY_IMPROVEMENT_EVERY = 10


config = Config()

# ========================= CONVERGÊNCIA =========================
CONVERGENCE_WINDOW = 500     # tamanho da janela deslizante
CONVERGENCE_EPS = 0.001      # tolerância de variação
CONVERGENCE_PATIENCE = 3     # número de janelas estáveis necessárias



# ========================= CCXT FIX ========================
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
    reset_ccxt()


    os.makedirs("data", exist_ok=True)
    data_path = "./data/binance-ETHUSDT-1d.pkl"


    # só baixa SE o arquivo ainda não existir
    if not os.path.exists(data_path):
        print("⬇Download dos dados")
        safe_download()
    else:
        print("Indo direto")

    df = pd.read_pickle(data_path)


    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7).max()


    df.dropna(inplace=True)
    print(f"{len(df)} registros\n")
    return df


def detect_convergence(returns, window, eps, patience, state):
    """
    Detecta convergência por estabilidade estatística
    """
    if len(returns) < window * (patience + 1):
        return False, state
    
    converged = True
    base = np.mean(returns[-window:])

    for i in range(1, patience + 1):
        prev = np.mean(returns[-window*(i+1):-window*i])
        if abs(base - prev) > eps:
            converged = False
            break

    if converged and state["episode"] is None:
        state["episode"] = len(returns)

    return converged, state
def make_env(df):
    env = gym.make(
        "TradingEnv",
        name="ETHUSD",
        df=df,
        positions=[-1, 0, 0.25, 0.5, 0.75, 1],
        trading_fees=0.001/100,
        borrow_interest_rate=0.0003/100,
        reward_function=reward_function,
    )

    env.add_metric('Position Changes', lambda h: np.sum(np.diff(h['position']) != 0))
    env.add_metric('Episode Length', lambda h: len(h['position']))

    return env


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

        self.convergence = {
            "episode": None,
            "value": None,
            "detected": False
        }

        self.window = 500
        self.tolerance = 0.001
        self.extra_after_converge = 500
        self._stop_at = None

    def run(self, env, num_ep):
        print("=== SARSA ===")
        start = time.time()

        for ep in range(num_ep):
            obs, info = env.reset()
            s_disc = discretize(obs)
            a = self._choose(obs, env)

            done = False
            ep_ret = 0

            while not done:
                nxt, r, term, trunc, info = env.step(a)
                done = term or trunc

                nxt_disc = discretize(nxt)
                nxt_a = self._choose(nxt, env)

                q = self.Q.get((s_disc, a), 0)
                q_next = 0 if done else self.Q.get((nxt_disc, nxt_a), 0)

                self.Q[(s_disc, a)] = q + self.alpha * (r + self.gamma * q_next - q)

                self.V[s_disc] = max(
                    self.Q.get((s_disc, ac), 0)
                    for ac in range(env.action_space.n)
                )

                s_disc = nxt_disc
                a = nxt_a
                ep_ret += r

            self.episode_returns.append(ep_ret)
            self.epsilon = max(self.min_eps, self.epsilon * self.eps_decay)

            # ===== DETECÇÃO DE CONVERGÊNCIA =====
            if len(self.episode_returns) >= 2 * self.window:

                prev = np.mean(self.episode_returns[-2*self.window:-self.window])
                current = np.mean(self.episode_returns[-self.window:])

                if not self.convergence["detected"] and abs(current - prev) < self.tolerance:
                    self.convergence["detected"] = True
                    self.convergence["episode"] = ep
                    self.convergence["value"] = current
                    self._stop_at = ep + self.extra_after_converge

                    print(f"\n✅ SARSA convergiu no episódio {ep}")
                    print(f"   Retorno estabilizado: {current:.6f}")
                    print(f"   Rodando até o episódio {self._stop_at} para confirmação...\n")

                if self.convergence["detected"] and ep >= self._stop_at:
                    print(f"\n🛑 SARSA interrompido após estabilidade ({ep})")
                    break

        print(f"⏱ SARSA Finalizado em {time.time() - start:.2f}s\n")
        return self.V, self.Q, self.episode_returns

    def _choose(self, obs, env):
        return politica_epsilon_greedy(obs, self.epsilon, self.Q, env)



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

        self.convergence = {
            "episode": None,
            "value": None,
            "detected": False
        }

        self.window = 500
        self.tolerance = 0.001
        self.extra_after_converge = 500
        self._stop_at = None

    def run(self, env, num_ep):
        print("=== Q-Learning ===")
        start = time.time()

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
                max_q_next = max(
                    self.Q.get((nxt_disc, ac), 0)
                    for ac in range(env.action_space.n)
                )

                self.Q[(s_disc, a)] = q + self.alpha * (r + self.gamma * max_q_next - q)

                self.V[s_disc] = max(
                    self.Q.get((s_disc, ac), 0)
                    for ac in range(env.action_space.n)
                )

                obs = nxt
                s_disc = nxt_disc
                ep_ret += r

            self.episode_returns.append(ep_ret)
            self.epsilon = max(self.min_eps, self.epsilon * self.eps_decay)

            # ===== DETECÇÃO DE CONVERGÊNCIA =====
            if len(self.episode_returns) >= 2 * self.window:

                prev = np.mean(self.episode_returns[-2*self.window:-self.window])
                current = np.mean(self.episode_returns[-self.window:])

                if not self.convergence["detected"] and abs(current - prev) < self.tolerance:
                    self.convergence["detected"] = True
                    self.convergence["episode"] = ep
                    self.convergence["value"] = current
                    self._stop_at = ep + self.extra_after_converge

                    print(f"\n✅ Q-Learning convergiu no episódio {ep}")
                    print(f"   Retorno estabilizado: {current:.6f}")
                    print(f"   Rodando até o episódio {self._stop_at} para confirmação...\n")

                if self.convergence["detected"] and ep >= self._stop_at:
                    print(f"\n🛑 Q-Learning interrompido após estabilidade ({ep})")
                    break

        print(f"⏱ Q-Learning Finalizado em {time.time() - start:.2f}s\n")
        return self.V, self.Q, self.episode_returns

    def _choose(self, obs, env):
        return politica_epsilon_greedy(obs, self.epsilon, self.Q, env)


# ========================= TESTE FINAL PARA RENDER =========================
def test_sarsa_with_render(env, sarsa_agent):

    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        s_disc = discretize(obs)
        a = politica_gulosa(obs, sarsa_agent.Q, env)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        total_reward += r

    print(f"Retorno SARSA: {total_reward:.4f}")
    env.save_for_render(dir="render_logs")


# ========================= PLOTAGEM AGREGADA =========================
def plot_aggregated_results(all_metrics, all_returns_history):

    algoritmos = list(all_metrics.keys())

    # ========= 1. Retornos médios por episódio (ALINHADO PELO MENOR TAMANHO) =========
    plt.figure(figsize=(10, 5))

    for algo in algoritmos:

        histories = all_returns_history[algo]

        # Segurança extra
        histories = [h for h in histories if len(h) > 0]

        if len(histories) == 0:
            print(f"⚠ Nenhum histórico válido para {algo}")
            continue

        min_len = min(len(h) for h in histories)
        aligned = np.array([h[:min_len] for h in histories])

        mean_returns = np.mean(aligned, axis=0)
        std_returns = np.std(aligned, axis=0)

        episodes = range(len(mean_returns))

        plt.plot(episodes, mean_returns, label=algo, linewidth=2)
        plt.fill_between(
            episodes,
            mean_returns - std_returns,
            mean_returns + std_returns,
            alpha=0.2
        )

    plt.title("Retorno médio por episódio (execuções alinhadas)")
    plt.xlabel("Episódios")
    plt.ylabel("Retorno")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ========= 2. Boxplot =========
    plt.figure(figsize=(8, 5))
    data_to_plot = [all_metrics[a] for a in algoritmos]
    plt.boxplot(data_to_plot, tick_labels=algoritmos)
    plt.title("Distribuição dos retornos finais (near-convergência)")
    plt.ylabel("Retorno médio")
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.show()

    # ========= 3. Média e desvio =========
    means = [np.mean(all_metrics[a]) for a in algoritmos]
    stds = [np.std(all_metrics[a]) for a in algoritmos]
    x = np.arange(len(algoritmos))

    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, algoritmos, rotation=30)
    plt.title("Média e desvio dos retornos finais")
    plt.ylabel("Retorno médio")
    plt.grid(True)
    plt.show()

    # ========= 4. Mínimos e máximos =========
    mins = [np.min(all_metrics[a]) for a in algoritmos]
    maxs = [np.max(all_metrics[a]) for a in algoritmos]
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, mins, width, label="Min")
    plt.bar(x + width/2, maxs, width, label="Max")
    plt.xticks(x, algoritmos, rotation=30)
    plt.title("Valores mínimos e máximos")
    plt.ylabel("Retorno")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ========= 5. Ranking =========
    sorted_algos = sorted(algoritmos, key=lambda x: np.mean(all_metrics[x]), reverse=True)
    sorted_means = [np.mean(all_metrics[a]) for a in sorted_algos]

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_algos, sorted_means)
    plt.title("Ranking por desempenho médio")
    plt.xlabel("Retorno médio")
    plt.grid(True)
    plt.show()


def plot_convergence(returns_dict, convergence_points, window=200):

    plt.figure(figsize=(12,6))

    for algo, returns in returns_dict.items():
        r = np.array(returns)

        plt.plot(r, alpha=0.25, label=f"{algo} cru")

        rolling = pd.Series(r).rolling(window).mean()
        plt.plot(rolling, linewidth=2, label=f"{algo} média móvel")

        if convergence_points.get(algo) is not None:
            ep = convergence_points[algo]
            y = rolling.dropna().iloc[-1] if not rolling.dropna().empty else r[ep]
            plt.axvline(ep, linestyle="--")
            plt.text(ep + 50, y, f"{algo} convergiu\nEp {ep}", rotation=90)

    plt.title("Convergência observada via MÉDIA MÓVEL")
    plt.xlabel("Episódios")
    plt.ylabel("Retorno")
    plt.legend()
    plt.grid(True)
    plt.show()



# ========================= MAIN =========================
def main(return_results=False):

    print("\n============================")
    print("   NOVA EXECUÇÃO INICIADA")
    print("============================\n")

    start_global = time.time()

    # ======= CARREGA DADOS =======
    df = load_data()

    # ======= CRIA AMBIENTES SEPARADOS =======
    env_sarsa = make_env(df)
    env_q = make_env(df)

    results = {}

    # =============================
    #          SARSA
    # =============================
    print("\n▶ Iniciando treinamento SARSA...")
    start_sarsa = time.time()

    sarsa = SarsaAgent(
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        eps_decay=config.EPSILON_DECAY,
        min_eps=config.MIN_EPSILON
    )

    V_sa, Q_sa, ret_sa = sarsa.run(env_sarsa, config.NUM_EPISODES)

    elapsed_sarsa = time.time() - start_sarsa
    print(f"\n⏱ SARSA finalizado em {elapsed_sarsa:.2f} segundos\n")

    results["SARSA"] = {
        "V": V_sa,
        "Q": Q_sa,
        "returns": ret_sa,
        "convergence": sarsa.convergence,
        "time": elapsed_sarsa
    }

    # =============================
    #        Q-LEARNING
    # =============================
    print("\n▶ Iniciando treinamento Q-Learning...")
    start_q = time.time()

    qlearn = QLearningAgent(
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        eps_decay=config.EPSILON_DECAY,
        min_eps=config.MIN_EPSILON
    )

    V_ql, Q_ql, ret_ql = qlearn.run(env_q, config.NUM_EPISODES)

    elapsed_q = time.time() - start_q
    print(f"\n⏱ Q-Learning finalizado em {elapsed_q:.2f} segundos\n")

    results["Q-Learning"] = {
        "V": V_ql,
        "Q": Q_ql,
        "returns": ret_ql,
        "convergence": qlearn.convergence,
        "time": elapsed_q
    }

    # =============================
    #        RELATÓRIO FINAL
    # =============================
    print("\n=========== RELATÓRIO FINAL ===========")

    for algo, res in results.items():
        conv = res["convergence"]["episode"]

        if conv is not None:
            print(f"✅ {algo} convergiu no episódio {conv}")
            print(f"   Retorno estabilizado: {res['convergence']['value']:.6f}")
        else:
            print(f"⚠ {algo} não convergiu")

        print(f"⏱ Tempo: {res['time']:.2f} segundos\n")

    print("======================================")

    # =============================
    #         FINALIZA
    # =============================
    env_sarsa.close()
    env_q.close()

    elapsed_total = time.time() - start_global
    print(f"\n⏱️ Tempo TOTAL da execução: {elapsed_total:.2f} segundos\n")

    # ===== RETORNO PARA LOOP MÚLTIPLO =====
    if return_results:
        return results

    return results



# ========================= LOOP DE 10 EXECUÇÕES =========================
def run_multiple_executions(n_runs=3):

    print(f"\nRodando {n_runs} execuções completas\n")

    metrics = {k: [] for k in ["SARSA", "Q-Learning"]}
    returns_history = {k: [] for k in ["SARSA", "Q-Learning"]}
    convergence_points = {}

    for i in range(n_runs):

        print(f"\n============================")
        print(f"   EXECUÇÃO {i+1}/{n_runs}")
        print(f"============================\n")

        r = main(return_results=True)

        for algo in metrics.keys():

            cut = r[algo]["convergence"]["episode"]

            if cut is None:
                cut = len(r[algo]["returns"])

            window = 500

            if cut > window:
                m = np.mean(r[algo]["returns"][cut-window:cut])
            else:
                m = np.mean(r[algo]["returns"])

            metrics[algo].append(m)
            returns_history[algo].append(r[algo]["returns"])

            print(f"  {algo} média near-convergência: {m:.4f}")

            # guarda convergência da ÚLTIMA execução
            if i == n_runs - 1:
                convergence_points[algo] = r[algo]["convergence"]["episode"]

    # =========================
    #        RESULTADOS
    # =========================
    print("\nMÉDIAS FINAIS:\n")

    for algo, vals in metrics.items():
        print(f"{algo}: média={np.mean(vals):.4f}, "
              f"desvio={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, "
              f"max={np.max(vals):.4f}")

    # =========================
    #    GRÁFICOS FINAIS
    # =========================
    print("\nGerando gráficos finais...")

    # 1) Gráficos agregados
    plot_aggregated_results(metrics, returns_history)

    # 2) Gráfico da última convergência
    print("\nConvergência da ÚLTIMA execução:")

    plot_convergence(
        returns_dict={
            "SARSA": returns_history["SARSA"][-1],
            "Q-Learning": returns_history["Q-Learning"][-1]
        },
        convergence_points=convergence_points
    )

    return metrics


# ========================= EXECUÇÃO =========================
if __name__ == "__main__":
    run_multiple_executions(3)