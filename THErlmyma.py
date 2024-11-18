# Filename: btc_rl_trading_bot.py

import os
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
import datetime
import random
from collections import deque
import pickle

# =========================
# Configuration Parameters
# =========================

# Replace with your MT5 login credentials or set them as environment variables
LOGIN = os.getenv('MT5_LOGIN') or 'YOUR_LOGIN'
PASSWORD = os.getenv('MT5_PASSWORD') or 'YOUR_PASSWORD'
SERVER = os.getenv('MT5_SERVER') or 'YOUR_SERVER'

SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1  # 1-minute timeframe
DATA_POINTS = 1000            # Number of data points to fetch
LOT_SIZE = 0.1
MAGIC_NUMBER = 234000
CHECKPOINT_DIR = 'checkpoints'  # Directory to save checkpoints
CHECKPOINT_INTERVAL = 10        # Save a checkpoint every N episodes

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure the checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================
# MetaTrader 5 Connection
# =========================

def connect_to_mt5(login, password, server):
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return False

    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        print("Connected to MetaTrader 5")
        return True
    else:
        print("Failed to connect to MetaTrader 5")
        mt5.shutdown()
        return False

# =========================
# Data Fetching Function
# =========================

def fetch_historical_data(symbol, timeframe, data_points):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, data_points)
    if rates is None or len(rates) == 0:
        raise Exception("Failed to retrieve rates")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# =========================
# Custom Gym Environment
# =========================

class TradingEnv(gym.Env):
    """Custom Trading Environment for OpenAI Gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.end_step = len(df) - 1

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: [Open, High, Low, Close, Volume]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.balance = 10000
        self.initial_balance = self.balance
        self.position = 0  # 1 for long, -1 for short, 0 for no position
        self.entry_price = 0

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.loc[self.current_step, ['open', 'high', 'low', 'close', 'tick_volume']].values
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0

        # Execute action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
        elif action == 0:  # Hold
            pass

        # Update balance based on position
        if self.position != 0:
            price_change = current_price - self.entry_price
            profit = price_change * self.position * LOT_SIZE * 100
            self.balance += profit
            reward = profit  # Reward is the profit

            # Close position
            self.position = 0
            self.entry_price = 0

        self.current_step += 1
        done = self.current_step >= self.end_step

        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        profit = self.balance - self.initial_balance
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Profit: {profit:.2f}")

# =========================
# Deep Q-Network Model
# =========================

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # Increased hidden layer size
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# =========================
# Agent Class
# =========================

class Agent:
    def __init__(self, state_size, action_size, learning_rate=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Replay memory
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQNAgent(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action based on the current state."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size=64):
        """Train the agent with a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).unsqueeze(1).to(device)
        
        # Compute target Q-values
        with torch.no_grad():
            target_q = rewards + (self.gamma * torch.max(self.model(next_states), dim=1, keepdim=True)[0] * (1 - dones))
        
        # Compute current Q-values
        current_q = self.model(states).gather(1, actions)
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decrease exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename, episode):
        """Save the model and optimizer state."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory,
        }
        temp_filename = filename + '.tmp'
        torch.save(checkpoint, temp_filename)
        os.replace(temp_filename, filename)
        print(f"Checkpoint saved at episode {episode}")

    def load(self, filename):
        """Load the model and optimizer state."""
        if os.path.isfile(filename):
            print(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = checkpoint['memory']
            episode = checkpoint['episode'] + 1
            print(f"Loaded checkpoint '{filename}' (episode {episode})")
            return episode
        else:
            print(f"No checkpoint found at '{filename}'")
            return 1  # Start from episode 1

# =========================
# Training Function
# =========================

def train_agent(env, agent, episodes=1000):
    start_episode = agent.load(os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth.tar'))

    for e in range(start_episode, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Perform replay and training
            agent.replay()

        print(f"Episode {e}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

        # Save the model at intervals
        if e % CHECKPOINT_INTERVAL == 0:
            checkpoint_filename = os.path.join(CHECKPOINT_DIR, f'checkpoint_{e}.pth.tar')
            agent.save(checkpoint_filename, e)
            # Also save the latest checkpoint
            latest_checkpoint = os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth.tar')
            agent.save(latest_checkpoint, e)
            # Optionally, implement cleanup of old checkpoints

    print("Training completed.")

# =========================
# Main Function
# =========================

def main():
    if not connect_to_mt5(LOGIN, PASSWORD, SERVER):
        return

    try:
        # Fetch historical data
        data = fetch_historical_data(SYMBOL, TIMEFRAME, DATA_POINTS)
        print(f"Fetched {len(data)} data points for {SYMBOL}.")

        # Initialize environment
        env = TradingEnv(data)
        print("Initialized trading environment.")

        # Initialize agent
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = Agent(state_size, action_size)
        print("Initialized DQN agent.")

        # Train agent
        train_agent(env, agent, episodes=1000)

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        mt5.shutdown()
        print("Disconnected from MetaTrader 5.")

if __name__ == "__main__":
    main()
