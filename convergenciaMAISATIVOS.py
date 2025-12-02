import gymnasium as gym
import numpy as np
import gym_trading_env
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
from tabulate import tabulate


START_TIME = time.time()


# ========================= HISTÓRICO GLOBAL =========================
ALL_FINAL_RETURNS = defaultdict(list)
ALL_RETURNS_HISTORY = defaultdict(list)
ALL_FINAL_RETURNS_BY_ASSET = defaultdict(lambda: defaultdict(list))
ALL_RETURNS_HISTORY_BY_ASSET = defaultdict(lambda: defaultdict(list))
ALL_CONVERGENCE_POINTS = defaultdict(list)  # Novo: pontos de convergência


# ========================= CONFIGURAÇÕES =========================
class Config:
    GAMMA = 0.99
    NUM_EPISODES = 40000
    ALPHA = 0.1
    EPSILON = 0.1
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01
    
    # ===== PARÂMETROS DE CONVERGÊNCIA =====
    CONVERGENCE_WINDOW = 500
    CONVERGENCE_TOLERANCE = 0.001
    EXTRA_AFTER_CONVERGE = 500

config = Config()

# ========================= DEFINIÇÃO DE ATIVOS =========================
ASSETS = {
    # ================= CRIPTOMOEDAS =================
    "BTC/USDT": {"exchange": "binance", "source": "ccxt",  "name": "Bitcoin"},
    "BNB/USDT": {"exchange": "binance", "source": "ccxt",  "name": "Binance Coin"},
    "XRP/USDT": {"exchange": "binance", "source": "ccxt",  "name": "Ripple"}
}

# ========================= DOWNLOAD SÍNCRONO (FIX PYTHON 3.13) =========================
def download_data_sync(symbol, exchange_name="binance"):
    """Download síncrono usando ccxt diretamente (sem asyncio)"""
    
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    exchange = getattr(ccxt, exchange_name)({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    try:
        since = int(datetime.datetime(2022, 1, 1).timestamp() * 1000)
        until = int(datetime.datetime(2025, 11, 1).timestamp() * 1000)
        
        all_candles = []
        current_since = since
        max_attempts = 3
        
        print(f"Baixando dados", end=" ", flush=True)
        
        while current_since < until:
            attempt = 0
            success = False
            
            while attempt < max_attempts and not success:
                try:
                    candles = exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe='1d',
                        since=current_since,
                        limit=1000
                    )
                    
                    if not candles:
                        success = True
                        break
                    
                    all_candles.extend(candles)
                    current_since = candles[-1][0] + 86400000
                    
                    print(".", end="", flush=True)
                    success = True
                    time.sleep(0.3)
                    
                except ccxt.RateLimitExceeded:
                    print("", end="", flush=True)
                    time.sleep(2)
                    attempt += 1
                    
                except ccxt.NetworkError as e:
                    print("", end="", flush=True)
                    time.sleep(1)
                    attempt += 1
                    
                except DeprecationWarning:
                    success = True
                    
                except Exception as e:
                    if "deprecated" in str(e).lower() or "utcfromtimestamp" in str(e).lower():
                        success = True
                    else:
                        print(f"\nErro: {e}")
                        attempt += 1
            
            if not success:
                break
        
        print(" ✓")
        
        if len(all_candles) == 0:
            raise Exception("Nenhum dado retornado da API")
        
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        
        print(f"{len(df)} registros baixados")
        
        return df
        
    except Exception as e:
        raise Exception(f"Falha: {e}")
    finally:
        try:
            exchange.close()
        except:
            pass

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

# ========================= CARREGAMENTO DE DADOS =========================
import yfinance as yf

def download_yahoo(symbol):
    print(f"Yahoo Finance: {symbol}")

    df = yf.download(
        symbol,
        start="2022-01-01",
        end="2025-11-01",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        raise Exception("Yahoo retornou dados vazios")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = df.columns.str.lower()

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise Exception(f"Colunas inválidas no Yahoo: {df.columns.tolist()}")

    df = df[["open", "high", "low", "close", "volume"]]
    df.dropna(inplace=True)

    print(f"{len(df)} registros Yahoo")
    return df


def load_data(symbol):
    os.makedirs("data", exist_ok=True)
    
    meta = ASSETS[symbol]
    source = meta["source"]
    exchange = meta["exchange"]

    symbol_clean = symbol.replace("/", "").replace("^", "")
    data_path = f"./data/{exchange}-{symbol_clean}-1d.pkl"
    
    if not os.path.exists(data_path):
        print(f"Baixando {symbol}...")
        try:
            if source == "ccxt":
                df = download_data_sync(symbol, exchange)
            elif source == "yahoo":
                df = download_yahoo(symbol)
            else:
                raise Exception("Fonte desconhecida")
            
            df.to_pickle(data_path)
            print(f"Salvo: {data_path}")
            
        except Exception as e:
            print(f"Erro: {e}")
            return None
    else:
        print(f"{symbol} (cache)")
        try:
            df = pd.read_pickle(data_path)
        except Exception as e:
            print(f"Erro ao carregar: {e}")
            return None

    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7).max()

    df.dropna(inplace=True)
    return df


# ========================= AGENTES COM CONVERGÊNCIA =========================
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
        
        # ===== CONTROLE DE CONVERGÊNCIA =====
        self.convergence = {
            "episode": None,
            "value": None,
            "detected": False
        }
        
        self.window = config.CONVERGENCE_WINDOW
        self.tolerance = config.CONVERGENCE_TOLERANCE
        self.extra_after_converge = config.EXTRA_AFTER_CONVERGE
        self._stop_at = None

    def run(self, env, num_ep):
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
            
            # ===== MONITORAMENTO DE CONVERGÊNCIA =====
            if len(self.episode_returns) >= 2 * self.window:
                prev = np.mean(self.episode_returns[-2*self.window:-self.window])
                curr = np.mean(self.episode_returns[-self.window:])

                if not self.convergence["detected"] and abs(curr - prev) < self.tolerance:
                    self.convergence["detected"] = True
                    self.convergence["episode"] = ep
                    self.convergence["value"] = curr
                    self._stop_at = ep + self.extra_after_converge

                    print(f"\n✅ SARSA convergiu no episódio {ep}")
                    print(f"   Retorno estabilizado: {curr:.6f}")
                    print(f"   Rodará até o episódio {self._stop_at}\n")

                if self.convergence["detected"] and ep >= self._stop_at:
                    print(f"🛑 SARSA interrompido após estabilização (ep {ep})\n")
                    break

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
        
        # ===== CONTROLE DE CONVERGÊNCIA =====
        self.convergence = {
            "episode": None,
            "value": None,
            "detected": False
        }
        
        self.window = config.CONVERGENCE_WINDOW
        self.tolerance = config.CONVERGENCE_TOLERANCE
        self.extra_after_converge = config.EXTRA_AFTER_CONVERGE
        self._stop_at = None

    def run(self, env, num_ep):
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
            
            # ===== MONITORAMENTO DE CONVERGÊNCIA =====
            if len(self.episode_returns) >= 2 * self.window:
                prev = np.mean(self.episode_returns[-2*self.window:-self.window])
                curr = np.mean(self.episode_returns[-self.window:])

                if not self.convergence["detected"] and abs(curr - prev) < self.tolerance:
                    self.convergence["detected"] = True
                    self.convergence["episode"] = ep
                    self.convergence["value"] = curr
                    self._stop_at = ep + self.extra_after_converge

                    print(f"\n✅ Q-Learning convergiu no episódio {ep}")
                    print(f"   Retorno estabilizado: {curr:.6f}")
                    print(f"   Rodará até o episódio {self._stop_at}\n")

                if self.convergence["detected"] and ep >= self._stop_at:
                    print(f"🛑 Q-Learning interrompido após estabilidade (ep {ep})\n")
                    break

        return self.V, self.Q, self.episode_returns

    def _choose(self, estado, env):
        return politica_epsilon_greedy(estado, self.epsilon, self.Q, env)

# ========================= MAIN POR ATIVO =========================
def run_asset(symbol):

    print(f"\n{'='*60}")
    print(f"  {ASSETS[symbol]['name']} ({symbol})")
    print(f"{'='*60}")
    
    df = load_data(symbol)

    if df is None or len(df) < 100:
        print(f"Dados insuficientes")
        return None

    print(f"{len(df)} registros carregados")

    env = gym.make(
        "TradingEnv",
        name=symbol.replace("/", ""),
        df=df,
        positions=[-1,0,0.25,0.5,0.75,1],
        trading_fees=0.001/100,
        borrow_interest_rate=0.0003/100,
        reward_function=reward_function,
    )

    env.add_metric('Position Changes', lambda h: np.sum(np.diff(h['position'])!=0))
    env.add_metric('Episode Length', lambda h: len(h['position']))

    results = {}

    print(f"Treinando SARSA", end=" ", flush=True)
    start = time.time()
    sarsa = SarsaAgent(config.ALPHA, config.GAMMA, config.EPSILON, config.EPSILON_DECAY, config.MIN_EPSILON)
    V_sa, Q_sa, ret_sa = sarsa.run(env, config.NUM_EPISODES)
    results["SARSA"] = {
        "V": V_sa,
        "Q": Q_sa,
        "returns": ret_sa,
        "convergence": sarsa.convergence
    }
    print(f"✓ ({time.time()-start:.1f}s)")

    print(f"Treinando Q-Learning", end=" ", flush=True)
    start = time.time()
    qlearn = QLearningAgent(config.ALPHA, config.GAMMA, config.EPSILON, config.EPSILON_DECAY, config.MIN_EPSILON)
    V_ql, Q_ql, ret_ql = qlearn.run(env, config.NUM_EPISODES)
    results["Q-Learning"] = {
        "V": V_ql,
        "Q": Q_ql,
        "returns": ret_ql,
        "convergence": qlearn.convergence
    }
    print(f"✓ ({time.time()-start:.1f}s)")

    env.close()
    return results

# ========================= TABELAS =========================
def print_results_table(all_results, run_number):
    print(f"\n{'='*80}")
    print(f"RESULTADOS DA EXECUÇÃO {run_number}")
    print(f"{'='*80}\n")
    
    table_data = []
    headers = ["Ativo", "SARSA", "Q-Learning", "Melhor"]
    
    for symbol in ASSETS.keys():
        if symbol not in all_results or all_results[symbol] is None:
            continue
            
        row = [ASSETS[symbol]['name'][:12]]
        results_for_asset = all_results[symbol]
        values = {}
        
        for algo in ["SARSA", "Q-Learning"]:
            mean_return = np.mean(results_for_asset[algo]["returns"][-5:])
            values[algo] = mean_return
            row.append(f"{mean_return:>7.2f}")
        
        best_algo = max(values, key=values.get)
        row.append(best_algo[:8])
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def print_consolidated_table(consolidated_results):
    print(f"\n{'='*100}")
    print(f"  RESULTADOS CONSOLIDADOS")
    print(f"{'='*100}\n")
    
    table_data = []
    headers = ["Ativo", "Algoritmo", "Média", "Desvio", "Min", "Max", "Mediana"]
    
    for symbol in ASSETS.keys():
        if symbol not in consolidated_results:
            continue
            
        asset_name = ASSETS[symbol]['name'][:12]
        
        for algo in ["SARSA", "Q-Learning"]:
            if algo not in consolidated_results[symbol]:
                continue
                
            values = consolidated_results[symbol][algo]
            if len(values) == 0:
                continue
                
            row = [
                asset_name if algo == "SARSA" else "",
                algo[:10],
                f"{np.mean(values):>7.2f}",
                f"{np.std(values):>6.2f}",
                f"{np.min(values):>7.2f}",
                f"{np.max(values):>7.2f}",
                f"{np.median(values):>7.2f}"
            ]
            table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def print_convergence_table(convergence_data):
    """Nova função: Tabela de convergência"""
    print(f"\n{'='*100}")
    print(f"  ANÁLISE DE CONVERGÊNCIA")
    print(f"{'='*100}\n")
    
    table_data = []
    headers = ["Ativo", "Algoritmo", "Convergiu?", "Episódio", "Valor", "Tempo Médio"]
    
    for symbol in ASSETS.keys():
        if symbol not in convergence_data:
            continue
            
        asset_name = ASSETS[symbol]['name'][:12]
        
        for algo in ["SARSA", "Q-Learning"]:
            if algo not in convergence_data[symbol]:
                continue
                
            conv_list = convergence_data[symbol][algo]
            if len(conv_list) == 0:
                continue
            
            converged_runs = [c for c in conv_list if c["detected"]]
            conv_rate = len(converged_runs) / len(conv_list) * 100
            
            if converged_runs:
                avg_episode = np.mean([c["episode"] for c in converged_runs])
                avg_value = np.mean([c["value"] for c in converged_runs])
                row = [
                    asset_name if algo == "SARSA" else "",
                    algo[:10],
                    f"{conv_rate:.0f}%",
                    f"{avg_episode:.0f}",
                    f"{avg_value:.4f}",
                    f"Ep {avg_episode:.0f}"
                ]
            else:
                row = [
                    asset_name if algo == "SARSA" else "",
                    algo[:10],
                    "Não",
                    "-",
                    "-",
                    "-"
                ]
            
            table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
def print_global_summary(all_metrics):
    print("\nRESUMO GLOBAL POR ALGORITMO\n")

    headers = ["Algoritmo", "Média", "Desvio", "Mínimo", "Máximo", "Mediana", "Execuções"]
    table = []

    for algo, valores in all_metrics.items():
        table.append([
            algo,
            round(np.mean(valores), 4),
            round(np.std(valores), 4),
            round(np.min(valores), 4),
            round(np.max(valores), 4),
            round(np.median(valores), 4),
            len(valores)
        ])

    print(tabulate(table, headers=headers, tablefmt="grid"))

    
# ========================= PLOTAGEM AGREGADA =========================
def plot_convergence(returns_dict, convergence_points, asset_name=""):
    """Nova função: Gráfico de convergência"""
    plt.figure(figsize=(12, 6))

    for algo, returns in returns_dict.items():
        plt.plot(returns, label=algo, alpha=0.8, linewidth=1.5)

        # Linha vertical da convergência
        if convergence_points.get(algo):
            ep = convergence_points[algo]
            plt.axvline(ep, linestyle="--", alpha=0.7, linewidth=2)
            
            # Texto da convergência
            y_pos = np.mean(returns[-500:]) if len(returns) > 500 else np.mean(returns)
            plt.text(ep + 200, y_pos,
                     f"{algo}\nEp {ep}",
                     rotation=90, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Linha horizontal do patamar final
        if len(returns) > 500:
            plateau = np.mean(returns[-500:])
            plt.axhline(plateau, linestyle=":", alpha=0.5)

    title = f"Convergência dos Agentes - {asset_name}" if asset_name else "Convergência dos Agentes"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Episódios")
    plt.ylabel("Retorno")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_aggregated_results(all_metrics, all_returns_history):

    algoritmos = list(all_metrics.keys())

    # 1. Retornos médios por episódio
    plt.figure(figsize=(10, 5))

    for algo in algoritmos:

        histories = all_returns_history[algo]

        # Alinhar séries para mesmo tamanho
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

    plt.title("Retorno médio por episódio (pós-convergência alinhada)")
    plt.xlabel("Episódios")
    plt.ylabel("Retorno")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Boxplot
    plt.figure(figsize=(8, 5))
    data_to_plot = [all_metrics[a] for a in algoritmos]
    plt.boxplot(data_to_plot, tick_labels=algoritmos)
    plt.title("Distribuição dos retornos finais")
    plt.ylabel("Retorno médio final")
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.show()

    # 3. Média e desvio
    means = [np.mean(all_metrics[a]) for a in algoritmos]
    stds = [np.std(all_metrics[a]) for a in algoritmos]
    x = np.arange(len(algoritmos))

    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, algoritmos, rotation=30)
    plt.title("Média e desvio dos retornos")
    plt.ylabel("Retorno médio")
    plt.grid(True)
    plt.show()

    # 4. Mínimos e máximos
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

    # 5. Ranking
    sorted_algos = sorted(algoritmos, key=lambda x: np.mean(all_metrics[x]), reverse=True)
    sorted_means = [np.mean(all_metrics[a]) for a in sorted_algos]

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_algos, sorted_means)
    plt.title("Ranking por desempenho médio")
    plt.xlabel("Retorno médio")
    plt.grid(True)
    plt.show()

def plot_summary_by_asset(all_final_returns_by_asset, all_returns_history_by_asset, convergence_by_asset):

    for symbol, algos_data in all_final_returns_by_asset.items():

        asset_name = ASSETS[symbol]["name"]
        algoritmos = list(algos_data.keys())

        # ======== 1. Média dos retornos finais (barras) ========
        means = [np.mean(algos_data[a]) for a in algoritmos]
        stds = [np.std(algos_data[a]) for a in algoritmos]

        plt.figure(figsize=(8, 5))
        plt.bar(algoritmos, means, yerr=stds)
        plt.title(f"{asset_name} - Retorno médio final por algoritmo")
        plt.ylabel("Retorno médio")
        plt.grid(True)
        plt.show()

        # ======== 2. Curva média de aprendizado COM CONVERGÊNCIA ========
        if symbol in convergence_by_asset:
            conv_points = {}
            for algo in algoritmos:
                conv_list = convergence_by_asset[symbol].get(algo, [])
                converged = [c for c in conv_list if c["detected"]]
                if converged:
                    conv_points[algo] = int(np.mean([c["episode"] for c in converged]))
            
            # Pegar apenas uma execução representativa
            returns_dict = {}
            for algo in algoritmos:
                if all_returns_history_by_asset[symbol][algo]:
                    returns_dict[algo] = all_returns_history_by_asset[symbol][algo][0]
            
            if returns_dict:
                plot_convergence(returns_dict, conv_points, asset_name)

        # ======== 3. Boxplot de estabilidade ========
        plt.figure(figsize=(8, 5))
        data = [algos_data[a] for a in algoritmos]

        plt.boxplot(data, tick_labels=algoritmos)
        plt.title(f"{asset_name} - Estabilidade dos algoritmos")
        plt.ylabel("Retornos finais")
        plt.grid(True)
        plt.show()
        
        
# ========================= LOOP DE MÚLTIPLAS EXECUÇÕES =========================
def run_multiple_executions(n_runs=1):
    print(f"\n{'#'*80}")
    print(f" INICIANDO {n_runs} EXECUÇÕES PARA {len(ASSETS)} ATIVOS")
    print(f"{'#'*80}")
    
    consolidated_results = {
        symbol: {algo: [] for algo in ["SARSA", "Q-Learning"]} 
        for symbol in ASSETS.keys()
    }
    
    # Nova estrutura para convergência
    convergence_by_asset = {
        symbol: {algo: [] for algo in ["SARSA", "Q-Learning"]}
        for symbol in ASSETS.keys()
    }
    
    for run in range(1, n_runs + 1):
        print(f"\n{'='*60}")
        print(f"  EXECUÇÃO {run}/{n_runs}")
        print(f"{'='*60}")
        
        run_results = {}
        
        for symbol in ASSETS.keys():
            try:
                results = run_asset(symbol)
                
                if results is None:
                    continue
                
                run_results[symbol] = results
                
                for algo in ["SARSA", "Q-Learning"]:
                    mean_return = np.mean(results[algo]["returns"][-5:])

                    # Métricas para gráfico
                    ALL_FINAL_RETURNS[algo].append(mean_return)
                    ALL_RETURNS_HISTORY[algo].append(results[algo]["returns"])

                    ALL_FINAL_RETURNS_BY_ASSET[symbol][algo].append(mean_return)
                    ALL_RETURNS_HISTORY_BY_ASSET[symbol][algo].append(results[algo]["returns"])

                    # Tabela consolidada
                    consolidated_results[symbol][algo].append(mean_return)
                    
                    # ===== NOVO: Armazenar dados de convergência =====
                    convergence_by_asset[symbol][algo].append(results[algo]["convergence"])
                    
            except KeyboardInterrupt:
                print_consolidated_table(consolidated_results)
                print_convergence_table(convergence_by_asset)
                return consolidated_results
            except Exception as e:
                print(f"Erro: {e}")
                continue
        
        print_results_table(run_results, run)
    
    # Tabelas finais
    print_consolidated_table(consolidated_results)
    print_convergence_table(convergence_by_asset)
    
    # Salvar CSV com dados de convergência
    os.makedirs("results", exist_ok=True)
    rows = []
    for symbol in ASSETS.keys():
        if symbol not in consolidated_results:
            continue
        for algo in ["SARSA", "Q-Learning"]:
            values = consolidated_results[symbol][algo]
            if len(values) == 0:
                continue
            
            # Dados de convergência
            conv_list = convergence_by_asset[symbol][algo]
            converged_runs = [c for c in conv_list if c["detected"]]
            conv_rate = len(converged_runs) / len(conv_list) * 100 if conv_list else 0
            avg_conv_ep = np.mean([c["episode"] for c in converged_runs]) if converged_runs else None
            
            rows.append({
                "Asset": symbol,
                "Asset_Name": ASSETS[symbol]['name'],
                "Algorithm": algo,
                "Mean": np.mean(values),
                "Std": np.std(values),
                "Min": np.min(values),
                "Max": np.max(values),
                "Median": np.median(values),
                "N_Runs": len(values),
                "Convergence_Rate": conv_rate,
                "Avg_Convergence_Episode": avg_conv_ep
            })
    
    df = pd.DataFrame(rows)
    csv_path = "./results/consolidated_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")
    
    return consolidated_results, convergence_by_asset

# ========================= EXECUÇÃO =========================
if __name__ == "__main__":
    try:
        consolidated_results, convergence_by_asset = run_multiple_executions(n_runs=1)
        
        # Gráficos agregados
        plot_aggregated_results(ALL_FINAL_RETURNS, ALL_RETURNS_HISTORY)
        
        # Gráficos por ativo com convergência
        plot_summary_by_asset(ALL_FINAL_RETURNS_BY_ASSET, ALL_RETURNS_HISTORY_BY_ASSET, convergence_by_asset)
        
        # Resumo global
        print_global_summary(ALL_FINAL_RETURNS)
        
        print("\n✅ Execução concluída!")

    except KeyboardInterrupt:
        print("\n⚠️ Programa interrompido pelo usuário")

    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()

    finally:
        END_TIME = time.time()
        elapsed = END_TIME - START_TIME

        print(f"\n{'='*60}")
        print("TEMPO TOTAL DE EXECUÇÃO")
        print(f"{'='*60}")
        print(f"Segundos: {elapsed:.2f}")
        print(f"Minutos: {elapsed/60:.2f}")
        
        if elapsed >= 3600:
            print(f"Horas: {elapsed/3600:.2f}")