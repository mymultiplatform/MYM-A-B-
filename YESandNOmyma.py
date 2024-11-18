# Filename: btc_rl_trading_bot_lstm.py

import os
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

# =========================
# Configuration Parameters
# =========================

SYMBOL = "BTCUSD"
DATA_FILE = 'historical_data.csv'  # CSV file containing historical data
LOT_SIZE = 1.0                    # Lot size for simulation
CHECKPOINT_DIR = 'checkpoints'    # Directory to save checkpoints
CHECKPOINT_INTERVAL = 10          # Save a checkpoint every N episodes

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
# Data Fetching and Preprocessing
# =========================

def fetch_historical_data(file_path):
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    return df

def add_indicators(df):
    # Simple Moving Averages
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    # Relative Strength Index (RSI)
    df['rsi'] = compute_rsi(df['close'], window=14)
    # Fill NaN values
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-6)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =========================
# Custom Gym Environment
# =========================

class TradingEnv(gym.Env):
    """Custom Trading Environment for OpenAI Gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=10):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.current_step = self.window_size - 1
        self.end_step = len(df) - 1

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell, 3 = Close Position
        self.action_space = spaces.Discrete(4)

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.df.shape[1]),
            dtype=np.float32
        )

        # Initialize account information
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = None  # Can be 'long', 'short', or None
        self.position_price = 0.0

    def reset(self):
        self.current_step = self.window_size - 1
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = None
        self.position_price = 0.0
        return self._next_observation()

    def _next_observation(self):
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1
        obs = self.df.iloc[start:end].values  # shape: (window_size, num_features)
        return obs.astype(np.float32)

    def step(self, action):
        done = False

        current_price = self.df.loc[self.current_step, 'close']

        # Store previous equity
        previous_equity = self.equity

        # Execute action
        if action == 1:  # Buy
            if self.position is None:
                self.position = 'long'
                self.position_price = current_price
        elif action == 2:  # Sell
            if self.position is None:
                self.position = 'short'
                self.position_price = current_price
        elif action == 3:  # Close Position
            if self.position == 'long':
                profit = (current_price - self.position_price) * LOT_SIZE
                self.balance += profit
                self.position = None
                self.position_price = 0.0
            elif self.position == 'short':
                profit = (self.position_price - current_price) * LOT_SIZE
                self.balance += profit
                self.position = None
                self.position_price = 0.0

        # Update equity
        if self.position == 'long':
            unrealized_pnl = (current_price - self.position_price) * LOT_SIZE
        elif self.position == 'short':
            unrealized_pnl = (self.position_price - current_price) * LOT_SIZE
        else:
            unrealized_pnl = 0.0

        self.equity = self.balance + unrealized_pnl

        # Calculate reward as change in equity
        reward = self.equity - previous_equity

        self.current_step += 1
        if self.current_step >= self.end_step:
            done = True
            # Close any open positions
            if self.position == 'long':
                profit = (current_price - self.position_price) * LOT_SIZE
                self.balance += profit
            elif self.position == 'short':
                profit = (self.position_price - current_price) * LOT_SIZE
                self.balance += profit
            self.position = None
            self.position_price = 0.0
            self.equity = self.balance

        obs = self._next_observation()
        info = {}

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        profit = self.equity - self.initial_balance
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Equity: {self.equity:.2f}, Profit: {profit:.2f}")

# =========================
# Deep Q-Network with LSTM Model
# =========================

class DQNAgent(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=128, num_layers=1):
        super(DQNAgent, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, sequence_length, input_size)
        out, hidden = self.lstm(x, hidden)
        # Take the output from the last time step
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out, hidden

# =========================
# Agent Class
# =========================

class Agent:
    def __init__(self, state_size, action_size, hidden_size=128, num_layers=1, learning_rate=1e-4,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size  # (window_size, num_features)
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.memory = deque(maxlen=100000)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQNAgent(state_size[1], action_size, hidden_size, num_layers).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.target_model = DQNAgent(state_size[1], action_size, hidden_size, num_layers).to(device)
        self.update_target_model()  # Initialize target model
        self.step_count = 0  # Counter for steps

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action based on the current state."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if not np.isfinite(state).all():
            print("Warning: State contains invalid values. Choosing random action.")
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # shape: (1, window_size, num_features)
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        hidden = (h0, c0)
        with torch.no_grad():
            q_values, _ = self.model(state, hidden)
        return torch.argmax(q_values[0]).item()

    def replay(self, batch_size=64):
        """Train the agent with a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        try:
            minibatch = random.sample(self.memory, batch_size)

            # Prepare batches
            states = np.array([t[0] for t in minibatch])  # shape: (batch_size, window_size, num_features)
            actions = np.array([t[1] for t in minibatch])
            rewards = np.array([t[2] for t in minibatch])
            next_states = np.array([t[3] for t in minibatch])
            dones = np.array([t[4] for t in minibatch])

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            # Initialize hidden states
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            hidden = (h0, c0)

            # Compute target Q-values using target network
            with torch.no_grad():
                target_q_values, _ = self.target_model(next_states, hidden)
                max_target_q_values = target_q_values.max(1, keepdim=True)[0]
                target = rewards + (self.gamma * max_target_q_values * (1 - dones))

            # Compute current Q-values
            current_q_values, _ = self.model(states, hidden)
            current_q_values = current_q_values.gather(1, actions)

            # Compute loss
            loss = self.criterion(current_q_values, target)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Soft update of target network
            self.soft_update(tau=0.01)

        except Exception as ex:
            print(f"An exception occurred during replay: {ex}")
            import traceback
            traceback.print_exc()

    def soft_update(self, tau=0.01):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename, episode):
        """Save the model and optimizer state."""
        try:
            checkpoint = {
                'episode': episode,
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
            }
            temp_filename = filename + '.tmp'
            torch.save(checkpoint, temp_filename)
            os.replace(temp_filename, filename)
            print(f"Checkpoint saved at episode {episode}")
        except Exception as ex:
            print(f"An exception occurred while saving checkpoint: {ex}")

    def load(self, filename):
        """Load the model and optimizer state."""
        if os.path.isfile(filename):
            try:
                print(f"Loading checkpoint '{filename}'")
                checkpoint = torch.load(filename, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                self.step_count = checkpoint['step_count']
                episode = checkpoint['episode'] + 1
                print(f"Loaded checkpoint '{filename}' (starting from episode {episode})")
                return episode
            except Exception as ex:
                print(f"An exception occurred while loading checkpoint: {ex}")
                return 1  # Start from episode 1 if loading fails
        else:
            print(f"No checkpoint found at '{filename}'")
            return 1  # Start from episode 1

# =========================
# Training Function
# =========================

def train_agent(env, agent, episodes=1000):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    start_episode = agent.load(os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth.tar'))

    for e in range(start_episode, episodes + 1):
        try:
            state = env.reset()
            total_reward = 0
            done = False
            step = 0

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step += 1

                # Perform replay and training
                agent.replay()

            print(f"Episode {e}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

            # Log metrics to TensorBoard
            writer.add_scalar('Total Reward', total_reward, e)
            writer.add_scalar('Epsilon', agent.epsilon, e)

            # Save the model at intervals
            if e % CHECKPOINT_INTERVAL == 0:
                checkpoint_filename = os.path.join(CHECKPOINT_DIR, f'checkpoint_{e}.pth.tar')
                agent.save(checkpoint_filename, e)
                latest_checkpoint = os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth.tar')
                agent.save(latest_checkpoint, e)

        except Exception as ex:
            print(f"An exception occurred during episode {e}: {ex}")
            import traceback
            traceback.print_exc()
            continue  # Proceed to the next episode

    writer.close()
    print("Training completed.")

# =========================
# Main Function
# =========================

def main():
    # Load data from CSV or other source
    data = fetch_historical_data(DATA_FILE)
    data = add_indicators(data)
    print(f"Loaded {len(data)} data points.")

    # Initialize environment with window_size
    window_size = 10
    env = TradingEnv(data, window_size=window_size)
    print("Initialized trading environment.")

    # Initialize agent
    state_size = env.observation_space.shape  # Now a tuple (window_size, num_features)
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)
    print("Initialized DQN agent.")

    # Train agent
    train_agent(env, agent, episodes=1000)

if __name__ == "__main__":
    main()
