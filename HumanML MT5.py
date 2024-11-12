import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import time
import os

# =========================
# Configuration Parameters
# =========================
LOGIN = 312128713
PASSWORD = "Sexo247420@"
SERVER = "XMGlobal-MT5 7"
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_D1
DAYS = 600
LOT_SIZE = 0.1
MAGIC_NUMBER = 234000
AUTOMATION_INTERVAL = 3600  # in seconds (1 hour)
PROFIT_TARGET = 0.01  # Profit target to close the trade
MANUAL_TRADES_FILE = "manual_trades.csv"  # Path to your manual trades data

# =========================
# MetaTrader 5 Connection
# =========================
def connect_to_mt5(login, password, server):
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return False

    authorized = mt5.login(login=login, password=password, server=server)
    if authorized:
        print("Connected to MetaTrader 5")
        return True
    else:
        print("Failed to connect to MetaTrader 5")
        mt5.shutdown()
        return False

# =========================
# Data Fetching Functions
# =========================
def fetch_historical_data(symbol, timeframe, days):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, days)
    if rates is None or len(rates) == 0:
        raise Exception("Failed to retrieve rates")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'close']]

def load_manual_trades(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Manual trades file not found: {file_path}")
    manual_trades = pd.read_csv(file_path)
    if 'action' not in manual_trades.columns:
        raise ValueError("Manual trades CSV must contain an 'action' column.")
    return manual_trades

# =========================
# Custom Gym Environment
# =========================
class TradingEnv(gym.Env):
    """Custom Trading Environment for OpenAI Gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, manual_trades):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.manual_trades = manual_trades
        self.current_step = 0
        self.end_step = len(df) - 1
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [Close, MA10, MA20, MA50, MA100]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for no position
        self.entry_price = 0

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        return self._next_observation()

    def _next_observation(self):
        # Example features: Close price, MA10, MA20, MA50, MA100
        window = self.df.iloc[self.current_step]
        ma10 = self.df['close'].rolling(window=10).mean().iloc[self.current_step]
        ma20 = self.df['close'].rolling(window=20).mean().iloc[self.current_step]
        ma50 = self.df['close'].rolling(window=50).mean().iloc[self.current_step]
        ma100 = self.df['close'].rolling(window=100).mean().iloc[self.current_step]
        
        obs = np.array([
            window['close'],
            ma10 if not np.isnan(ma10) else 0,
            ma20 if not np.isnan(ma20) else 0,
            ma50 if not np.isnan(ma50) else 0,
            ma100 if not np.isnan(ma100) else 0
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0

        # Execute action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                # print(f"Buy at {current_price}")
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                # print(f"Sell at {current_price}")
        elif action == 0:  # Hold
            pass

        # Calculate reward based on manual trades
        if self.current_step < len(self.manual_trades):
            manual_action = self.manual_trades['action'].iloc[self.current_step]
            if action == manual_action:
                reward += 1  # Positive reward for matching manual trade
            else:
                reward -= 1  # Negative reward for not matching

        # Update balance based on position
        if self.position != 0:
            profit = (current_price - self.entry_price) * self.position * LOT_SIZE * 100  # Adjust profit calculation as needed
            self.balance += profit
            reward += profit  # Reward can be the profit

            # Reset position
            self.position = 0

        self.current_step += 1
        done = self.current_step >= self.end_step

        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        profit = self.balance - self.initial_balance
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Profit: {profit:.2f}")

# =========================
# Reinforcement Learning Agent
# =========================
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class Agent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQNAgent(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state).detach().clone()
            target_f[action] = target
            output = self.model(state)
            loss = self.criterion(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =========================
# Training Function
# =========================
def train_agent(env, agent, episodes=1000, batch_size=32):
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for time_step in range(len(env.df)):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.replay(batch_size)
        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")
    print("Training completed.")

# =========================
# Main Function
# =========================
def main():
    if not connect_to_mt5(LOGIN, PASSWORD, SERVER):
        return

    try:
        # Fetch historical data
        data = fetch_historical_data(SYMBOL, TIMEFRAME, DAYS)
        print(f"Fetched {len(data)} days of historical data for {SYMBOL}.")

        # Load manual trades
        manual_trades = load_manual_trades(MANUAL_TRADES_FILE)
        if len(manual_trades) != len(data):
            print(f"Warning: Number of manual trades ({len(manual_trades)}) does not match number of data points ({len(data)}).")
            min_length = min(len(manual_trades), len(data))
            manual_trades = manual_trades.iloc[:min_length].reset_index(drop=True)
            data = data.iloc[:min_length].reset_index(drop=True)
            print(f"Truncated data to {min_length} entries.")

        # Initialize environment
        env = TradingEnv(data, manual_trades)
        print("Initialized trading environment.")

        # Initialize agent
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = Agent(state_size, action_size)
        print("Initialized DQN agent.")

        # Train agent
        train_agent(env, agent, episodes=1000, batch_size=32)

        # Optionally, save the trained model
        model_path = "dqn_trading_agent.pth"
        torch.save(agent.model.state_dict(), model_path)
        print(f"Trained model saved to {model_path}.")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        mt5.shutdown()
        print("Disconnected from MetaTrader 5.")

if __name__ == "__main__":
    main()
