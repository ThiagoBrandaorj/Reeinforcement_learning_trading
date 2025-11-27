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

# ========================= CONFIGURAÇÕES GERAIS =========================
class Config:
    GAMMA = 0.99
    NUM_EPISODES = 25
    ALPHA = 0.1
    EPSILON = 0.1
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01
    POLICY_IMPROVEMENT_EVERY = 10

config = Config()
start_time = time.time()

# ========================= FUNÇÕES AUXILIARES =========================
def reward_function(history):
    """Função de recompensa melhorada com tratamento de edge cases"""
    portfolio_vals = history["portfolio_valuation"]
    if len(portfolio_vals) < 2:
        return 0
    ratio = portfolio_vals[-1] / portfolio_vals[-2]
    if ratio <= 0:
        return -1  # Penalidade forte para valores negativos
    return np.log(ratio)

def discretize(obs, bins=10):
    """Discretização mais eficiente usando bins"""
    fc, fo, fh, fl, fv, cp, lp = obs
    
    # Discretiza features contínuas em bins
    fc_disc = np.digitize(float(fc), np.linspace(-0.1, 0.1, bins))
    fo_disc = np.digitize(float(fo), np.linspace(0.9, 1.1, bins))
    fh_disc = np.digitize(float(fh), np.linspace(0.9, 1.1, bins))
    fl_disc = np.digitize(float(fl), np.linspace(0.9, 1.1, bins))
    fv_disc = np.digitize(float(fv), np.linspace(0, 1, bins))
    
    return (fc_disc, fo_disc, fh_disc, fl_disc, fv_disc, int(cp), int(lp))

def politica_fixa(observation):
    """Política fixa melhorada"""
    if random.random() < 0.3:
        return random.randint(0, 5)
    return 3

def politica_epsilon_greedy(estado, epsilon, Q, env):
    """Política epsilon-greedy otimizada"""
    estado_discreto = discretize(estado)
    
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        q_values = [Q.get((estado_discreto, a), 0) for a in range(env.action_space.n)]
        max_q = max(q_values)
        # Em caso de empate, escolhe aleatoriamente entre as melhores ações
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)
    
def politica_gulosa(estado, Q, env):
    """Política gulosa pura (sem exploração)"""
    return politica_epsilon_greedy(estado, 0, Q, env)

# ========================= CARREGAMENTO DE DADOS =========================
def load_data():
    """Carrega e prepara os dados de forma eficiente"""
    os.makedirs("data", exist_ok=True)
    
    # Download apenas se necessário
    data_path = "./data/binance-BNBUSDT-1d.pkl"
    download(
        exchange_names=["binance"],
        symbols=["BNB/USDT"],
        timeframe="1d",
        dir="data",
        since=datetime.datetime(year=2020, month=1, day=1),
        until = datetime.datetime(year=2025,month=11,day=1),
    )
    
    df = pd.read_pickle(data_path)
    
    # Cria features de forma vetorizada
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    
    if 'volume' in df.columns:
        df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
    else:
        df["feature_volume"] = df["close"] / df["close"].rolling(7*24).max()
    
    df.dropna(inplace=True)
    return df

# ========================= ALGORITMOS DE APRENDIZADO =========================
class MonteCarloEveryVisit:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.returns_state = defaultdict(list)
        self.returns_sa = defaultdict(list)
        self.V = {}
        self.Q = {}
        self.episode_returns = []
        
    def run(self, env, num_episodes, policy):
        print("=== MONTE CARLO EVERY VISIT ===")
        
        for ep in range(num_episodes):
            trajectory = []
            obs, info = env.reset()
            done = truncate = False
            episode_return = 0
            
            # Coleta da trajetória
            while not done and not truncate:
                state = discretize(obs)
                action = policy(obs)
                next_obs, reward, done, truncate, info = env.step(action)
                trajectory.append((state, reward, action))
                obs = next_obs
                episode_return += reward
            
            self.episode_returns.append(episode_return)
            
            # Processamento da trajetória
            G = 0
            for t in reversed(range(len(trajectory))):
                s, r, a = trajectory[t]
                G = r + self.gamma * G
                
                self.returns_state[s].append(G)
                self.V[s] = np.mean(self.returns_state[s])
                
                self.returns_sa[(s, a)].append(G)
                self.Q[(s, a)] = np.mean(self.returns_sa[(s, a)])
            
            if (ep + 1) % 10 == 0:
                print(f"Episódio {ep + 1}/{num_episodes}, Retorno: {episode_return:.4f}")
        
        return self.V, self.Q, self.episode_returns
    
class MonteCarloFirstVisit:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.returns_state = defaultdict(list)
        self.returns_sa = defaultdict(list)
        self.V = {}
        self.Q = {}
        self.episode_returns = []
        self.V_medio_por_episodio = []
        self.Q_medio_por_episodio = []
        
    def run(self, env, num_episodes, policy):
        print("=== MONTE CARLO FIRST VISIT ===")
        
        for ep in range(num_episodes):
            trajectory = []
            obs, info = env.reset()
            done = truncate = False
            episode_return = 0
            
            # Coleta da trajetória
            while not done and not truncate:
                state = discretize(obs)
                action = policy(obs)
                next_obs, reward, done, truncate, info = env.step(action)
                trajectory.append((state, reward, action))
                obs = next_obs
                episode_return += reward
            
            self.episode_returns.append(episode_return)
            
            # Processamento da trajetória - FIRST VISIT
            visited_states = set()
            visited_state_actions = set()
            G = 0
            
            for t in reversed(range(len(trajectory))):
                s, r, a = trajectory[t]
                G = r + self.gamma * G
                
                # FIRST VISIT: só atualiza na primeira visita
                if s not in visited_states:
                    visited_states.add(s)
                    self.returns_state[s].append(G)
                    self.V[s] = np.mean(self.returns_state[s])
                
                if (s, a) not in visited_state_actions:
                    visited_state_actions.add((s, a))
                    self.returns_sa[(s, a)].append(G)
                    self.Q[(s, a)] = np.mean(self.returns_sa[(s, a)])
            
            # Estatísticas por episódio
            media_V = np.mean(list(self.V.values())) if self.V else 0
            self.V_medio_por_episodio.append(media_V)
            media_Q = np.mean(list(self.Q.values())) if self.Q else 0
            self.Q_medio_por_episodio.append(media_Q)
            
            if (ep + 1) % 10 == 0:
                print(f"Episódio {ep + 1}/{num_episodes}, Retorno: {episode_return:.4f}, "
                      f"V médio: {media_V:.4f}, Q médio: {media_Q:.4f}")
        
        return self.V, self.Q, self.episode_returns
    
class MonteCarloPolicyImprovement:
    def __init__(self, gamma=0.99, improvement_every=10):
        self.gamma = gamma
        self.improvement_every = improvement_every
        self.returns_sa = defaultdict(list)
        self.Q = {}
        self.policy = {}  # Política armazenada por estado
        self.episode_returns = []
        self.policy_changes_history = []
        
    def run(self, env, num_episodes):
        print("=== MONTE CARLO POLICY IMPROVEMENT ===")
        
        # Política inicial aleatória
        def initial_policy(observation):
            return env.action_space.sample()
        
        current_policy = initial_policy
        
        for ep in range(num_episodes):
            trajectory = []
            obs, info = env.reset()
            done = truncate = False
            episode_return = 0
            
            # Coleta da trajetória com política atual
            while not done and not truncate:
                state = discretize(obs)
                action = current_policy(obs)
                next_obs, reward, done, truncate, info = env.step(action)
                trajectory.append((state, reward, action))
                obs = next_obs
                episode_return += reward
            
            self.episode_returns.append(episode_return)
            
            # Evaluation: Atualiza Q-values (First Visit)
            visited_state_actions = set()
            G = 0
            
            for t in reversed(range(len(trajectory))):
                s, r, a = trajectory[t]
                G = r + self.gamma * G
                
                if (s, a) not in visited_state_actions:
                    visited_state_actions.add((s, a))
                    self.returns_sa[(s, a)].append(G)
                    self.Q[(s, a)] = np.mean(self.returns_sa[(s, a)])
            
            # Improvement: Melhora a política periodicamente
            if (ep + 1) % self.improvement_every == 0:
                policy_changes = self._improve_policy(env)
                self.policy_changes_history.append(policy_changes)
                
                # Atualiza a função de política
                def improved_policy(observation):
                    state = discretize(observation)
                    if state in self.policy:
                        return self.policy[state]
                    return env.action_space.sample()  # Fallback
                
                current_policy = improved_policy
                print(f"Episódio {ep + 1}: Policy Improvement - {policy_changes} mudanças de política")
            
            if (ep + 1) % 10 == 0:
                print(f"Episódio {ep + 1}/{num_episodes}, Retorno: {episode_return:.4f}")
        
        return self.Q, self.episode_returns, self.policy
    
    def _improve_policy(self, env):
        """Melhora a política para ser greedy em relação a Q"""
        policy_changes = 0
        
        for state_action, q_value in self.Q.items():
            state, action = state_action
            
            # Encontra a melhor ação para este estado
            best_action = action
            best_q = q_value
            
            for a in range(env.action_space.n):
                current_q = self.Q.get((state, a), -np.inf)
                if current_q > best_q:
                    best_q = current_q
                    best_action = a
            
            # Atualiza política se mudou
            if state not in self.policy or self.policy[state] != best_action:
                self.policy[state] = best_action
                policy_changes += 1
        
        return policy_changes

class SarsaAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = {}
        self.V = {}
        self.episode_returns = []
        
    def run(self, env, num_episodes):
        print("=== SARSA ===")
        
        for episode in range(num_episodes):
            estado, info = env.reset()
            estado_discreto = discretize(estado)
            acao = self._choose_action(estado, env)
            
            done = False
            episode_return = 0
            step_count = 0
            
            while not done:
                proximo_estado, recompensa, terminated, truncated, info = env.step(acao)
                proximo_estado_discreto = discretize(proximo_estado)
                done = terminated or truncated
                
                proxima_acao = self._choose_action(proximo_estado, env)
                
                # Atualização SARSA
                estado_acao_atual = (estado_discreto, acao)
                q_atual = self.Q.get(estado_acao_atual, 0)
                
                if done:
                    q_proximo = 0
                else:
                    q_proximo = self.Q.get((proximo_estado_discreto, proxima_acao), 0)
                
                self.Q[estado_acao_atual] = q_atual + self.alpha * (
                    recompensa + self.gamma * q_proximo - q_atual
                )
                
                # Atualiza V(s)
                self.V[estado_discreto] = max([
                    self.Q.get((estado_discreto, a), 0) 
                    for a in range(env.action_space.n)
                ])
                
                # Transição
                estado = proximo_estado
                estado_discreto = proximo_estado_discreto
                acao = proxima_acao
                
                episode_return += recompensa
                step_count += 1
            
            self.episode_returns.append(episode_return)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 10 == 0:
                print(f"Episódio {episode + 1}/{num_episodes}, "
                      f"Retorno: {episode_return:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return self.V, self.Q, self.episode_returns
    
    def _choose_action(self, estado, env):
        return politica_epsilon_greedy(estado, self.epsilon, self.Q, env)
    
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = {}
        self.V = {}
        self.episode_returns = []
        
    def run(self, env, num_episodes):
        print("=== Q-LEARNING ===")
        
        for episode in range(num_episodes):
            estado, info = env.reset()
            estado_discreto = discretize(estado)
            
            done = False
            episode_return = 0
            
            while not done:
                acao = self._choose_action(estado, env)
                proximo_estado, recompensa, terminated, truncated, info = env.step(acao)
                proximo_estado_discreto = discretize(proximo_estado)
                done = terminated or truncated
                
                # Atualização Q-Learning
                estado_acao_atual = (estado_discreto, acao)
                q_atual = self.Q.get(estado_acao_atual, 0)
                
                # Encontra o máximo Q para o próximo estado
                max_q_proximo = max([
                    self.Q.get((proximo_estado_discreto, a), 0) 
                    for a in range(env.action_space.n)
                ])
                
                self.Q[estado_acao_atual] = q_atual + self.alpha * (
                    recompensa + self.gamma * max_q_proximo - q_atual
                )
                
                # Atualiza V(s)
                self.V[estado_discreto] = max([
                    self.Q.get((estado_discreto, a), 0) 
                    for a in range(env.action_space.n)
                ])
                
                # Transição
                estado = proximo_estado
                estado_discreto = proximo_estado_discreto
                
                episode_return += recompensa
            
            self.episode_returns.append(episode_return)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 10 == 0:
                print(f"Episódio {episode + 1}/{num_episodes}, "
                      f"Retorno: {episode_return:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return self.V, self.Q, self.episode_returns
    
    def _choose_action(self, estado, env):
        return politica_epsilon_greedy(estado, self.epsilon, self.Q, env)

class DQNAgent:
    def __init__(self, alpha=0.001, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = {}
        self.V = {}
        self.episode_returns = []
        
    def run(self, env, num_episodes):
        print("=== DQN (SIMPLIFICADO) ===")
        
        for episode in range(num_episodes):
            estado, info = env.reset()
            estado_discreto = discretize(estado)
            
            done = False
            episode_return = 0
            
            while not done:
                acao = self._choose_action(estado, env)
                proximo_estado, recompensa, terminated, truncated, info = env.step(acao)
                proximo_estado_discreto = discretize(proximo_estado)
                done = terminated or truncated
                
                # Atualização DQN (versão simplificada com tabela Q)
                estado_acao_atual = (estado_discreto, acao)
                q_atual = self.Q.get(estado_acao_atual, 0)
                
                # Encontra o máximo Q para o próximo estado
                max_q_proximo = max([
                    self.Q.get((proximo_estado_discreto, a), 0) 
                    for a in range(env.action_space.n)
                ])
                
                # Target DQN
                target = recompensa + self.gamma * max_q_proximo
                
                self.Q[estado_acao_atual] = q_atual + self.alpha * (target - q_atual)
                
                # Atualiza V(s)
                self.V[estado_discreto] = max([
                    self.Q.get((estado_discreto, a), 0) 
                    for a in range(env.action_space.n)
                ])
                
                # Transição
                estado = proximo_estado
                estado_discreto = proximo_estado_discreto
                
                episode_return += recompensa
            
            self.episode_returns.append(episode_return)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 10 == 0:
                print(f"Episódio {episode + 1}/{num_episodes}, "
                      f"Retorno: {episode_return:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return self.V, self.Q, self.episode_returns
    
    def _choose_action(self, estado, env):
        return politica_epsilon_greedy(estado, self.epsilon, self.Q, env)            

# ========================= EXECUÇÃO PRINCIPAL =========================
def main():
    # Carrega dados
    df = load_data()
    
    # Cria ambiente
    env = gym.make(
        "TradingEnv",
        name="BNBUSD",
        df=df,
        positions=[-1, 0, 0.25, 0.5, 0.75, 1],
        trading_fees=0.001/100,
        borrow_interest_rate=0.0003/100,
        reward_function=reward_function,
    )
    
    env.add_metric('Position Changes', lambda history: np.sum(np.diff(history['position']) != 0))
    env.add_metric("Episode Length", lambda history: len(history["position"]))
    
    # Executa algoritmos
    results = {}
    
    # Monte Carlo Every Visit
    print("\n" + "="*50)
    mc_every_agent = MonteCarloEveryVisit(gamma=config.GAMMA)
    V_mc_every, Q_mc_every, returns_mc_every = mc_every_agent.run(env, config.NUM_EPISODES, politica_fixa)
    results['MC Every Visit'] = {'V': V_mc_every, 'Q': Q_mc_every, 'returns': returns_mc_every}
    
    # Monte Carlo First Visit
    print("\n" + "="*50)
    mc_first_agent = MonteCarloFirstVisit(gamma=config.GAMMA)
    V_mc_first, Q_mc_first, returns_mc_first = mc_first_agent.run(env, config.NUM_EPISODES, politica_fixa)
    results['MC First Visit'] = {'V': V_mc_first, 'Q': Q_mc_first, 'returns': returns_mc_first}
    
    # Monte Carlo Policy Improvement
    print("\n" + "="*50)
    mc_policy_agent = MonteCarloPolicyImprovement(
        gamma=config.GAMMA, 
        improvement_every=config.POLICY_IMPROVEMENT_EVERY
    )
    Q_mc_policy, returns_mc_policy, learned_policy = mc_policy_agent.run(env, config.NUM_EPISODES)
    results['MC Policy Improvement'] = {'Q': Q_mc_policy, 'returns': returns_mc_policy}
    
    # SARSA
    print("\n" + "="*50)
    sarsa_agent = SarsaAgent(
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        min_epsilon=config.MIN_EPSILON
    )
    V_sarsa, Q_sarsa, returns_sarsa = sarsa_agent.run(env, config.NUM_EPISODES)
    results['SARSA'] = {'V': V_sarsa, 'Q': Q_sarsa, 'returns': returns_sarsa}
    
    # Q-Learning (NOVO)
    print("\n" + "="*50)
    qlearning_agent = QLearningAgent(
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        min_epsilon=config.MIN_EPSILON
    )
    V_ql, Q_ql, returns_ql = qlearning_agent.run(env, config.NUM_EPISODES)
    results['Q-Learning'] = {'V': V_ql, 'Q': Q_ql, 'returns': returns_ql}
    
    # DQN (NOVO)
    print("\n" + "="*50)
    dqn_agent = DQNAgent(
        alpha=0.001,  # Alpha menor para DQN
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        min_epsilon=config.MIN_EPSILON
    )
    V_dqn, Q_dqn, returns_dqn = dqn_agent.run(env, config.NUM_EPISODES)
    results['DQN'] = {'V': V_dqn, 'Q': Q_dqn, 'returns': returns_dqn}
    
    # Teste final com Policy Improvement
    print("\n=== TESTE FINAL COM POLICY IMPROVEMENT ===")
    test_policy_improvement(env, learned_policy)
    
    # Visualizações
    plot_results(results, mc_policy_agent)
    
    end_time = time.time()
    print(f"\nTempo total de execução: {end_time - start_time:.2f} segundos")
    
    env.close()

def test_policy_improvement(env, policy):
    """Testa a política aprendida com Policy Improvement"""
    estado, info = env.reset()
    done = False
    retorno_total = 0
    passos = 0
    
    while not done:
        # Usa a política aprendida
        state_disc = discretize(estado)
        if state_disc in policy:
            acao = policy[state_disc]
        else:
            acao = env.action_space.sample()  # Fallback
        
        proximo_estado, recompensa, terminated, truncated, info = env.step(acao)
        done = terminated or truncated
        
        estado = proximo_estado
        retorno_total += recompensa
        passos += 1
    
    print(f"Retorno com Policy Improvement: {retorno_total:.4f}")
    print(f"Passos: {passos}")
    print(f"Tamanho da política aprendida: {len(policy)} estados")
    env.render()

def plot_results(results, mc_policy_agent=None):
    """Plota resultados comparativos"""
    plt.figure(figsize=(16, 12))
    
    # Plot de retornos por episódio
    plt.subplot(2, 3, 1)
    for algo, data in results.items():
        plt.plot(data['returns'], label=algo, alpha=0.7, linewidth=2)
    plt.title('Comparação de Retornos por Episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Retorno')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot de valores V(s) - para algoritmos que têm V
    plt.subplot(2, 3, 2)
    for algo, data in results.items():
        if 'V' in data and data['V']:
            values = list(data['V'].values())[:50]  # Limita para visualização
            plt.scatter(range(len(values)), values, label=algo, alpha=0.6)
    plt.title('Valores de V(s)')
    plt.xlabel('Estados (amostra)')
    plt.ylabel('V(s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot de valores Q(s,a)
    plt.subplot(2, 3, 3)
    for algo, data in results.items():
        if data['Q']:
            q_values = list(data['Q'].values())[:50]  # Limita para visualização
            plt.scatter(range(len(q_values)), q_values, label=algo, alpha=0.6)
    plt.title('Valores de Q(s,a)')
    plt.xlabel('Pares Estado-Ação (amostra)')
    plt.ylabel('Q(s,a)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Estatísticas finais
    plt.subplot(2, 3, 4)
    algoritmos = list(results.keys())
    retornos_finais = [np.mean(data['returns'][-5:]) for data in results.values()]  # Últimos 5 episódios
    
    bars = plt.bar(algoritmos, retornos_finais, alpha=0.7)
    plt.title('Retorno Médio (últimos 5 episódios)')
    plt.ylabel('Retorno Médio')
    plt.xticks(rotation=45)
    
    # Adiciona valores nas barras
    for bar, valor in zip(bars, retornos_finais):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{valor:.4f}', ha='center', va='bottom', rotation=90)
    
    # Policy Changes (se disponível)
    if mc_policy_agent and hasattr(mc_policy_agent, 'policy_changes_history'):
        plt.subplot(2, 3, 5)
        episodes = [i * config.POLICY_IMPROVEMENT_EVERY for i in range(len(mc_policy_agent.policy_changes_history))]
        plt.plot(episodes, mc_policy_agent.policy_changes_history, 'o-', linewidth=2)
        plt.title('Mudanças de Política no Policy Improvement')
        plt.xlabel('Episódio')
        plt.ylabel('Número de Mudanças')
        plt.grid(True, alpha=0.3)
    
    # Comparação First Visit vs Every Visit
    plt.subplot(2, 3, 6)
    if 'MC First Visit' in results and 'MC Every Visit' in results:
        first_visit_returns = results['MC First Visit']['returns']
        every_visit_returns = results['MC Every Visit']['returns']
        
        plt.plot(first_visit_returns, label='First Visit', alpha=0.7)
        plt.plot(every_visit_returns, label='Every Visit', alpha=0.7)
        plt.title('First Visit vs Every Visit')
        plt.xlabel('Episódio')
        plt.ylabel('Retorno')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()