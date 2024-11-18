# Filename: btc_rl_trading_bot_mt5.py

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
LOT_SIZE = 0.01               # Adjusted lot size for risk management
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

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell, 3 = Close Position
        self.action_space = spaces.Discrete(4)

        # Observation space includes:
        # [Open, High, Low, Close, Volume, Bid, Ask, Spread, Balance, Equity, Margin, Free Margin, Positions]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # Initialize account information
        account_info = mt5.account_info()
        if account_info is None:
            raise Exception("Failed to get account info")
        self.initial_balance = account_info.balance
        self.balance = self.initial_balance
        self.equity = account_info.equity
        self.margin = account_info.margin
        self.free_margin = account_info.margin_free
        self.positions = []
        self.spread = 0.0

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.margin = 0.0
        self.free_margin = self.balance
        self.positions = []
        return self._next_observation()

    def _next_observation(self):
        # Get market data
        obs = self.df.loc[self.current_step, ['open', 'high', 'low', 'close', 'tick_volume']].values.astype(np.float32)

        # Get latest bid and ask prices
        symbol_info_tick = mt5.symbol_info_tick(SYMBOL)
        if symbol_info_tick is None:
            raise Exception(f"Failed to get symbol info tick for {SYMBOL}")
        bid = symbol_info_tick.bid
        ask = symbol_info_tick.ask
        self.spread = ask - bid

        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            raise Exception("Failed to get account info")
        self.balance = account_info.balance
        self.equity = account_info.equity
        self.margin = account_info.margin
        self.free_margin = account_info.margin_free

        # Get open positions
        positions = mt5.positions_get(symbol=SYMBOL)
        num_positions = len(positions) if positions else 0

        additional_obs = np.array([bid, ask, self.spread, self.balance, self.equity, self.margin, self.free_margin, num_positions], dtype=np.float32)

        full_obs = np.concatenate((obs, additional_obs))

        return full_obs

    def step(self, action):
        done = False
        reward = 0

        # Execute action
        if action == 1:  # Buy
            self._execute_order(mt5.ORDER_TYPE_BUY)
        elif action == 2:  # Sell
            self._execute_order(mt5.ORDER_TYPE_SELL)
        elif action == 3:  # Close Position
            self._close_all_positions()
        else:  # Hold
            pass

        # Update account info
        account_info = mt5.account_info()
        if account_info is None:
            raise Exception("Failed to get account info")
        self.balance = account_info.balance
        self.equity = account_info.equity

        # Calculate reward (change in equity)
        profit = self.equity - self.initial_balance
        reward = profit

        # Implement a penalty to discourage holding positions for too long
        if self.current_step % 10 == 0 and self._has_open_position():
            reward -= 1  # Penalty

        self.current_step += 1
        if self.current_step >= self.end_step:
            done = True
            self._close_all_positions()
            # Final reward calculation
            profit = self.equity - self.initial_balance
            reward = profit

        obs = self._next_observation()
        return obs, reward, done, {}

    def _execute_order(self, order_type):
        symbol = SYMBOL
        lot = LOT_SIZE
        deviation = 10

        price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

        # Check if we already have a position in the opposite direction
        if self._has_open_position():
            positions = mt5.positions_get(symbol=symbol)
            for pos in positions:
                if pos.type != order_type:
                    self._close_position(pos.ticket)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": MAGIC_NUMBER,
            "comment": "Reinforcement Learning Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {result.retcode}")
        else:
            print(f"Order placed: {result}")

    def _close_position(self, ticket):
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            print(f"No position with ticket {ticket}")
            return
        pos = position[0]
        symbol = pos.symbol
        lot = pos.volume
        deviation = 10
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": deviation,
            "magic": MAGIC_NUMBER,
            "comment": "Reinforcement Learning Trade Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(close_request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close position {ticket}: {result.retcode}")
        else:
            print(f"Position {ticket} closed")

    def _close_all_positions(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None or len(positions) == 0:
            return
        for pos in positions:
            self._close_position(pos.ticket)

    def _has_open_position(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        return positions is not None and len(positions) > 0

    def render(self, mode='human', close=False):
        profit = self.equity - self.initial_balance
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Equity: {self.equity:.2f}, Profit: {profit:.2f}")

# =========================
# Deep Q-Network Model
# =========================

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# =========================
# Agent Class
# =========================

class Agent:
    def __init__(self, state_size, action_size, learning_rate=1e-4, gamma=0.99, epsilon=0.5, epsilon_min=0.05, epsilon_decay=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQNAgent(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.target_model = DQNAgent(state_size, action_size).to(device)
        self.update_target_model()  # Initialize target model
        self.update_target_steps = 1000  # Update target model every N steps
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
        
        # Compute target Q-values using target network
        with torch.no_grad():
            target_q = rewards + (self.gamma * self.target_model(next_states).max(1, keepdim=True)[0] * (1 - dones))
        
        # Compute current Q-values
        current_q = self.model(states).gather(1, actions)
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.update_target_model()

    def save(self, filename, episode):
        """Save the model and optimizer state."""
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

    def load(self, filename):
        """Load the model and optimizer state."""
        if os.path.isfile(filename):
            print(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
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

            # Optionally render the environment
            # env.render()

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
