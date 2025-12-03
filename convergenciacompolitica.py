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
from gym_trading_env.renderer import Renderer



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
CONVERGENCE_WINDOW = 500
CONVERGENCE_EPS = 0.001
CONVERGENCE_PATIENCE = 3


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


# ========================= EXTRAÇÃO DE POLÍTICA ÓTIMA =========================
def extract_optimal_policy(Q, env):
    """
    Extrai a política ótima a partir da função Q
    Retorna um dicionário {estado: melhor_ação}
    """
    policy = {}
    states = set(s for (s, a) in Q.keys())
    
    for state in states:
        q_values = [Q.get((state, a), -np.inf) for a in range(env.action_space.n)]
        best_action = np.argmax(q_values)
        policy[state] = best_action
    
    return policy


def analyze_policy(policy, action_names=None, episode_returns=None, episode_actions=None):
    """
    Analisa e exibe estatísticas sobre a política
    """
    print(f"  Total de estados na política: {len(policy)}")
    
    actions = list(policy.values())
    unique, counts = np.unique(actions, return_counts=True)

    print("  Distribuição de ações:")
    for action, count in zip(unique, counts):
        percentage = (count / len(actions)) * 100
        print(f"    Ação {action}: {count} estados ({percentage:.1f}%)")

    # ===== AÇÃO MÉDIA =====
    probs = counts / len(actions)
    weighted_avg_action = np.sum(unique * probs)
    print(f"\n Ação média ponderada: {weighted_avg_action:.4f}")

    # ===== ENTROPIA =====
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(unique)) if len(unique) > 1 else 1
    predictability = 1 - (entropy / max_entropy)
    print(f"Previsibilidade da política: {predictability:.2%}")
    print(f"Entropia: {entropy:.4f} / Máx: {max_entropy:.4f}")

    # ===== AÇÃO DOMINANTE =====
    dominant_idx = np.argmax(counts)
    dominant_action = unique[dominant_idx]
    dominant_pct = (counts[dominant_idx] / len(actions)) * 100

    print(f"👑 Ação dominante: {dominant_action} ({dominant_pct:.1f}% dos estados)")

    # ===== RETORNOS CONDICIONADOS =====
    if episode_returns is not None and episode_actions is not None:
        dominant_returns = []

        idx = 0
        for ep_return in episode_returns:
            if idx < len(episode_actions):
                if episode_actions[idx] == dominant_action:
                    dominant_returns.append(ep_return)
            idx += 1

        if dominant_returns:
            avg = np.mean(dominant_returns)
            std = np.std(dominant_returns)
            print(f"Retorno médio quando ação dominante ocorre: {avg:.4f} ± {std:.4f}")
            print(f"(Baseado em {len(dominant_returns)} episódios)")
        else:
            print("Nenhuma ocorrência válida da ação dominante nos episódios.")

    return dict(zip(unique, counts))



def compare_policies(policies_dict):
    print("\n========== COMPARAÇÃO DE POLÍTICAS ==========")

    # Estados totais
    all_states = set()
    for policy in policies_dict.values():
        all_states.update(policy.keys())

    # Estados em comum
    common_states = set.intersection(*(set(p.keys()) for p in policies_dict.values()))
    
    print(f"Estados em comum entre todas as políticas: {len(common_states)}")

    if not common_states:
        print("Nenhum estado em comum para comparação.")
        return

    agreements = 0

    for state in common_states:
        actions = [
            policies_dict[algo].get(state) 
            for algo in policies_dict 
            if state in policies_dict[algo]
        ]
        
        if len(set(actions)) == 1:
            agreements += 1

    agreement_rate = agreements / len(common_states) * 100

    print(f"Taxa de concordância: {agreement_rate:.1f}%")
    print(f"Concordam em {agreements}/{len(common_states)} estados")



# ========================= CARREGAMENTO DE DADOS =========================
def load_data():
    reset_ccxt()

    os.makedirs("data", exist_ok=True)
    data_path = "./data/binance-ETHUSDT-1d.pkl"

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
        self.episode_actions = []  # NOVO: rastrear ações

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
            ep_actions = [] #ações deste episódio

            while not done:
                ep_actions.append(a)  #registra ação
                
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
            self.episode_actions.extend(ep_actions)  #armazena todas as ações

            self.epsilon = max(self.min_eps, self.epsilon * self.eps_decay)

            if len(self.episode_returns) >= 2 * self.window:
                prev = np.mean(self.episode_returns[-2*self.window:-self.window])
                current = np.mean(self.episode_returns[-self.window:])

                if not self.convergence["detected"] and abs(current - prev) < self.tolerance:
                    self.convergence["detected"] = True
                    self.convergence["episode"] = ep
                    self.convergence["value"] = current
                    self._stop_at = ep + self.extra_after_converge

                    print(f"\nSARSA convergiu no episódio {ep}")
                    print(f"   Retorno estabilizado: {current:.6f}")
                    print(f"   Rodando até o episódio {self._stop_at} para confirmação...\n")

                if self.convergence["detected"] and ep >= self._stop_at:
                    print(f"\nSARSA interrompido após estabilidade ({ep})")
                    break

        print(f"SARSA Finalizado em {time.time() - start:.2f}s\n")
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
        self.episode_actions = []  #rastrear ações

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
            ep_actions = []  #ações deste episódio

            while not done:
                a = self._choose(obs, env)
                ep_actions.append(a)  #registra ação
                
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
            self.episode_actions.extend(ep_actions)  #armazena todas as ações

            self.epsilon = max(self.min_eps, self.epsilon * self.eps_decay)

            if len(self.episode_returns) >= 2 * self.window:
                prev = np.mean(self.episode_returns[-2*self.window:-self.window])
                current = np.mean(self.episode_returns[-self.window:])

                if not self.convergence["detected"] and abs(current - prev) < self.tolerance:
                    self.convergence["detected"] = True
                    self.convergence["episode"] = ep
                    self.convergence["value"] = current
                    self._stop_at = ep + self.extra_after_converge

                    print(f"\nQ-Learning convergiu no episódio {ep}")
                    print(f"Retorno estabilizado: {current:.6f}")
                    print(f"Rodando até o episódio {self._stop_at} para confirmação...\n")

                if self.convergence["detected"] and ep >= self._stop_at:
                    print(f"\nQ-Learning interrompido após estabilidade ({ep})")
                    break

        print(f"Q-Learning Finalizado em {time.time() - start:.2f}s\n")
        return self.V, self.Q, self.episode_returns

    def _choose(self, obs, env):
        return politica_epsilon_greedy(obs, self.epsilon, self.Q, env)


# ========================= PLOTAGEM =========================
def plot_aggregated_results(all_metrics, all_returns_history):
    algoritmos = list(all_metrics.keys())

    plt.figure(figsize=(10, 5))

    for algo in algoritmos:
        histories = all_returns_history[algo]
        histories = [h for h in histories if len(h) > 0]

        if len(histories) == 0:
            print(f"Nenhum histórico válido para {algo}")
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

    # Boxplot
    plt.figure(figsize=(8, 5))
    data_to_plot = [all_metrics[a] for a in algoritmos]
    plt.boxplot(data_to_plot, tick_labels=algoritmos)
    plt.title("Distribuição dos retornos finais (near-convergência)")
    plt.ylabel("Retorno médio")
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.show()


# ========================= MAIN =========================
def main(return_results=False):
    print("\n============================")
    print("   NOVA EXECUÇÃO INICIADA")
    print("============================\n")

    start_global = time.time()

    # =========================
    # DADOS
    # =========================
    df = load_data()

    # =========================
    # AMBIENTES
    # =========================
    env_sarsa = make_env(df)
    env_q = make_env(df)

    results = {}

    # =========================
    # SARSA
    # =========================
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

    policy_sarsa = extract_optimal_policy(Q_sa, env_sarsa)

    print(f"\n⏱ SARSA finalizado em {elapsed_sarsa:.2f} segundos\n")

    results["SARSA"] = {
        "V": V_sa,
        "Q": Q_sa,
        "returns": ret_sa,
        "convergence": sarsa.convergence,
        "time": elapsed_sarsa,
        "policy": policy_sarsa,
        "episode_actions": sarsa.episode_actions
    }

    # =========================
    # Q-LEARNING
    # =========================
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

    policy_q = extract_optimal_policy(Q_ql, env_q)

    print(f"\n⏱ Q-Learning finalizado em {elapsed_q:.2f} segundos\n")

    results["Q-Learning"] = {
        "V": V_ql,
        "Q": Q_ql,
        "returns": ret_ql,
        "convergence": qlearn.convergence,
        "time": elapsed_q,
        "policy": policy_q,
        "episode_actions": qlearn.episode_actions
    }

    # =========================
    # ANÁLISE DAS POLÍTICAS
    # =========================
    print("\n=========== ANÁLISE DAS POLÍTICAS ÓTIMAS ===========\n")

    print("📊 SARSA - Política Ótima:")
    analyze_policy(
        policy_sarsa,
        episode_returns=results["SARSA"]["returns"],
        episode_actions=results["SARSA"]["episode_actions"]
    )

    print("\n📊 Q-Learning - Política Ótima:")
    analyze_policy(
        policy_q,
        episode_returns=results["Q-Learning"]["returns"],
        episode_actions=results["Q-Learning"]["episode_actions"]
    )

    compare_policies({"SARSA": policy_sarsa, "Q-Learning": policy_q})

    # =========================
    # RELATÓRIO FINAL
    # =========================
    print("\n=========== RELATÓRIO FINAL ===========")

    for algo, res in results.items():
        conv = res["convergence"]["episode"]

        if conv is not None:
            print(f"✅ {algo} convergiu no episódio {conv}")
            print(f"   Retorno estabilizado: {res['convergence']['value']:.6f}")
        else:
            print(f"⚠ {algo} não convergiu")

        print(f"⏱ Tempo: {res['time']:.2f}s\n")

    print("======================================")

    # =========================
    # RENDERIZAÇÃO (AGORA SIM)
    # =========================
    
    print("\n💾 Salvando episódios para renderização...")

    env_sarsa.save_for_render(dir="render_logs/SARSA")
    env_q.save_for_render(dir="render_logs/QLEARNING")

    env_sarsa.close()
    env_q.close()

    # =========================
    # TEMPO TOTAL
    # =========================
    elapsed_total = time.time() - start_global
    print(f"\n⏱ Tempo TOTAL da execução: {elapsed_total:.2f} segundos\n")

    if return_results:
        return results

    return results



# ========================= LOOP DE MÚLTIPLAS EXECUÇÕES =========================
def run_multiple_executions(n_runs=3):
    print(f"\nRodando {n_runs} execuções completas\n")

    metrics = {k: [] for k in ["SARSA", "Q-Learning"]}
    returns_history = {k: [] for k in ["SARSA", "Q-Learning"]}
    policies_history = {k: [] for k in ["SARSA", "Q-Learning"]}
    actions_history = {k: [] for k in ["SARSA", "Q-Learning"]}

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
            policies_history[algo].append(r[algo]["policy"])
            actions_history[algo].append(r[algo]["episode_actions"])

            print(f"  {algo} média near-convergência: {m:.4f}")

    # =========================
    # RESULTADOS CONSOLIDADOS
    # =========================
    print("\n" + "="*60)
    print("RESULTADOS CONSOLIDADOS DAS EXECUÇÕES")
    print("="*60)

    print("\nMÉDIAS FINAIS:\n")
    for algo, vals in metrics.items():
        print(f"{algo}: média={np.mean(vals):.4f}, "
              f"desvio={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, "
              f"max={np.max(vals):.4f}")

    # =========================
    # ANÁLISE FINAL DAS POLÍTICAS
    # =========================
    print("\n" + "="*60)
    print("POLÍTICAS ÓTIMAS DA ÚLTIMA EXECUÇÃO")
    print("="*60 + "\n")

    for algo in ["SARSA", "Q-Learning"]:
        print(f"\n{algo} - Política Final:")

        last_policy = policies_history[algo][-1]
        last_returns = returns_history[algo][-1]
        last_actions = actions_history[algo][-1]

        analyze_policy(
            last_policy,
            episode_returns=last_returns,
            episode_actions=last_actions
        )

    compare_policies({
        "SARSA": policies_history["SARSA"][-1],
        "Q-Learning": policies_history["Q-Learning"][-1]
    })

    # =========================
    # GRÁFICOS
    # =========================
    print("\nGerando gráficos finais...")
    plot_aggregated_results(metrics, returns_history)

    return metrics, policies_history

if __name__ == "__main__":
    run_multiple_executions(3)

    # =========================
    # RENDERIZAÇÃO FINAL
    # =========================
    print("\nIniciando visualização com Renderer...\n")

    from gym_trading_env.renderer import Renderer
    import pandas as pd

    renderer = Renderer(render_logs_dir="render_logs")

    # MÉDIAS MÓVEIS
    renderer.add_line(
        name="SMA 10",
        function=lambda df: df["close"].rolling(10).mean(),
        line_options={"color": "blue", "width": 1}
    )

    renderer.add_line(
        name="SMA 20",
        function=lambda df: df["close"].rolling(20).mean(),
        line_options={"color": "red", "width": 1}
    )

    # MÉTRICAS
    renderer.add_metric(
        name="Annual Market Return",
        function=lambda df: f"{((df['close'].iloc[-1] / df['close'].iloc[0]) ** (1 / ((df.index[-1] - df.index[0]).days / 365)) - 1) * 100:.2f}%"
    )

    renderer.add_metric(
        name="Annual Portfolio Return",
        function=lambda df: f"{((df['portfolio_valuation'].iloc[-1] / df['portfolio_valuation'].iloc[0]) ** (1 / ((df.index[-1] - df.index[0]).days / 365)) - 1) * 100:.2f}%"
    )

    # ABRIR DASHBOARD
    renderer.run()
