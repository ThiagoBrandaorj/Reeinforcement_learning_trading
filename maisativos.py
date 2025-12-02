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
ALL_FINAL_RETURNS = defaultdict(list)       # retorno médio final
ALL_RETURNS_HISTORY = defaultdict(list)     # vetor de retornos por episódio
ALL_FINAL_RETURNS_BY_ASSET = defaultdict(lambda: defaultdict(list))
ALL_RETURNS_HISTORY_BY_ASSET = defaultdict(lambda: defaultdict(list))


# ========================= CONFIGURAÇÕES =========================
class Config:
    GAMMA = 0.99
    NUM_EPISODES = 50000
    ALPHA = 0.1
    EPSILON = 0.1
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01

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
    
    # Suprimir warnings do datetime
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    exchange = getattr(ccxt, exchange_name)({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    try:
        # Buscar dados históricos
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
                        success = True  # Chegou ao fim
                        break
                    
                    all_candles.extend(candles)
                    current_since = candles[-1][0] + 86400000  # +1 dia
                    
                    print(".", end="", flush=True)
                    success = True
                    time.sleep(0.3)  # Rate limit
                    
                except ccxt.RateLimitExceeded:
                    print("", end="", flush=True)
                    time.sleep(2)
                    attempt += 1
                    
                except ccxt.NetworkError as e:
                    print("", end="", flush=True)
                    time.sleep(1)
                    attempt += 1
                    
                except DeprecationWarning:
                    # Ignorar warnings de deprecação
                    success = True
                    
                except Exception as e:
                    if "deprecated" in str(e).lower() or "utcfromtimestamp" in str(e).lower():
                        # Ignorar erros relacionados a warnings
                        success = True
                    else:
                        print(f"\nErro: {e}")
                        attempt += 1
            
            if not success:
                break
        
        print(" ✓")
        
        if len(all_candles) == 0:
            raise Exception("Nenhum dado retornado da API")
        
        # Converter para DataFrame
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

    # ✅ ACHATAR COLUNAS MULTIINDEX
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ✅ NORMALIZAR NOMES
    df.columns = df.columns.str.lower()

    # ✅ VERIFICAR COLUNAS
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise Exception(f"Colunas inválidas no Yahoo: {df.columns.tolist()}")

    # ✅ PADRONIZAR
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

    # features padronizadas
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7).max()

    df.dropna(inplace=True)
    return df


# ========================= AGENTES =========================
class MonteCarloFirstVisit:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.returns_state = defaultdict(list)
        self.returns_sa = defaultdict(list)
        self.V = {}
        self.Q = {}
        self.episode_returns = []

    def run(self, env, num_ep, policy):
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

    #print(f"Treinando MC First Visit", end=" ", flush=True)
    #start = time.time()
    #mc_fv = MonteCarloFirstVisit(config.GAMMA)
    #V_fv, Q_fv, ret_fv = mc_fv.run(env, config.NUM_EPISODES, politica_fixa)
    #results["MC First Visit"] = {"V":V_fv,"Q":Q_fv,"returns":ret_fv}
    #print(f"✓ ({time.time()-start:.1f}s)")

    print(f"Treinando SARSA", end=" ", flush=True)
    start = time.time()
    sarsa = SarsaAgent(config.ALPHA, config.GAMMA, config.EPSILON, config.EPSILON_DECAY, config.MIN_EPSILON)
    V_sa, Q_sa, ret_sa = sarsa.run(env, config.NUM_EPISODES)
    results["SARSA"] = {"V":V_sa,"Q":Q_sa,"returns":ret_sa}
    print(f"✓ ({time.time()-start:.1f}s)")

    print(f"Treinando Q-Learning", end=" ", flush=True)
    start = time.time()
    qlearn = QLearningAgent(config.ALPHA, config.GAMMA, config.EPSILON, config.EPSILON_DECAY, config.MIN_EPSILON)
    V_ql, Q_ql, ret_ql = qlearn.run(env, config.NUM_EPISODES)
    results["Q-Learning"] = {"V":V_ql,"Q":Q_ql,"returns":ret_ql}
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
                asset_name if algo == "MC First Visit" else "",
                algo[:10],
                f"{np.mean(values):>7.2f}",
                f"{np.std(values):>6.2f}",
                f"{np.min(values):>7.2f}",
                f"{np.max(values):>7.2f}",
                f"{np.median(values):>7.2f}"
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
def plot_aggregated_results(all_metrics, all_returns_history):

    algoritmos = list(all_metrics.keys())

    # 1. Retornos médios por episódio
    plt.figure(figsize=(10, 5))
    for algo in algoritmos:
        returns_array = np.array(all_returns_history[algo])
        mean_returns = np.mean(returns_array, axis=0)
        std_returns = np.std(returns_array, axis=0)
        episodes = range(len(mean_returns))

        plt.plot(episodes, mean_returns, label=algo, linewidth=2)
        plt.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)

    plt.title("Retorno médio por episódio")
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

def plot_summary_by_asset(all_final_returns_by_asset, all_returns_history_by_asset):

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

        # ======== 2. Curva média de aprendizado ========
        plt.figure(figsize=(9, 5))

        for algo in algoritmos:
            returns_array = np.array(all_returns_history_by_asset[symbol][algo])
            mean_curve = np.mean(returns_array, axis=0)

            plt.plot(mean_curve, label=algo, linewidth=2)

        plt.title(f"{asset_name} - Curva média de aprendizagem")
        plt.xlabel("Episódios")
        plt.ylabel("Retorno")
        plt.legend()
        plt.grid(True)
        plt.show()

        # ======== 3. Boxplot de estabilidade ========
        plt.figure(figsize=(8, 5))
        data = [algos_data[a] for a in algoritmos]

        plt.boxplot(data, tick_labels=algoritmos)
        plt.title(f"{asset_name} - Estabilidade dos algoritmos")
        plt.ylabel("Retornos finais")
        plt.grid(True)
        plt.show()

def plot_algorithms_by_asset(symbol, results):
    asset_name = ASSETS[symbol]["name"]
    algoritmos = list(results.keys())

    plt.figure(figsize=(12, 6))

    # Curvas de aprendizado
    for algo in algoritmos:
        plt.plot(results[algo]["returns"], label=algo, linewidth=2)

    plt.title(f"{asset_name} - Aprendizado por Episódio")
    plt.xlabel("Episódios")
    plt.ylabel("Retorno")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Distribuição de retorno por algoritmo
    plt.figure(figsize=(8, 5))
    data = [results[a]["returns"] for a in algoritmos]

    plt.boxplot(data, tick_labels=algoritmos)
    plt.title(f"{asset_name} - Estabilidade dos Algoritmos")
    plt.ylabel("Retorno")
    plt.grid(True)
    plt.show()

    # Desempenho médio com desvio
    means = [np.mean(results[a]["returns"]) for a in algoritmos]
    stds = [np.std(results[a]["returns"]) for a in algoritmos]

    x = np.arange(len(algoritmos))
    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds)
    plt.xticks(x, algoritmos)
    plt.title(f"{asset_name} - Retorno Médio por Algoritmo")
    plt.ylabel("Retorno médio")
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
    
    for run in range(1, n_runs + 1):
        print(f"  EXECUÇÃO {run}/{n_runs}")
        
        run_results = {}
        
        for symbol in ASSETS.keys():

            try:
                results = run_asset(symbol)

                
                if results is None:
                    continue
                
                run_results[symbol] = results
                
                for algo in ["SARSA", "Q-Learning"]:
                    mean_return = np.mean(results[algo]["returns"][-5:])

                    # métricas para gráfico
                    ALL_FINAL_RETURNS[algo].append(mean_return)
                    ALL_RETURNS_HISTORY[algo].append(results[algo]["returns"])

                    ALL_FINAL_RETURNS_BY_ASSET[symbol][algo].append(mean_return)
                    ALL_RETURNS_HISTORY_BY_ASSET[symbol][algo].append(results[algo]["returns"])


                    # tabela consolidada
                    consolidated_results[symbol][algo].append(mean_return)

                    
            except KeyboardInterrupt:
                print_consolidated_table(consolidated_results)
                return consolidated_results
            except Exception as e:
                print(f"Erro: {e}")
                continue
        
        print_results_table(run_results, run)
    
    print_consolidated_table(consolidated_results)
    
    # Salvar CSV
    os.makedirs("results", exist_ok=True)
    rows = []
    for symbol in ASSETS.keys():
        if symbol not in consolidated_results:
            continue
        for algo in ["SARSA", "Q-Learning"]:
            values = consolidated_results[symbol][algo]
            if len(values) == 0:
                continue
            rows.append({
                "Asset": symbol,
                "Asset_Name": ASSETS[symbol]['name'],
                "Algorithm": algo,
                "Mean": np.mean(values),
                "Std": np.std(values),
                "Min": np.min(values),
                "Max": np.max(values),
                "Median": np.median(values),
                "N_Runs": len(values)
            })
    
    df = pd.DataFrame(rows)
    csv_path = "./results/consolidated_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")
    
    return consolidated_results

# ========================= EXECUÇÃO =========================
if __name__ == "__main__":
    try:
        consolidated_results = run_multiple_executions(n_runs=1)
        plot_aggregated_results(ALL_FINAL_RETURNS, ALL_RETURNS_HISTORY)
        plot_summary_by_asset(ALL_FINAL_RETURNS_BY_ASSET, ALL_RETURNS_HISTORY_BY_ASSET)
        print_global_summary(ALL_FINAL_RETURNS)
        print("\nExecução concluída!")

    except KeyboardInterrupt:
        print("\nPrograma interrompido")

    except Exception as e:
        print(f"\nErro durante execução: {e}")
        import traceback
        traceback.print_exc()

    finally:
        END_TIME = time.time()
        elapsed = END_TIME - START_TIME

        print("\nTEMPO TOTAL DE EXECUÇÃO:")
        print(f"Segundos: {elapsed:.2f}")

        if elapsed < 60:
            print(f"Minutos: {elapsed/60:.2f}")
        else:
            print(f"Minutos: {elapsed/60:.2f}")
            print(f"Horas: {elapsed/3600:.2f}")
