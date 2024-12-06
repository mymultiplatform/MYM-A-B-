# Filename: btc_rl_trading_bot_mt5_lstm.py

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

LOGIN = os.getenv('MT5_LOGIN') or 'YOUR_LOGIN'
PASSWORD = os.getenv('MT5_PASSWORD') or 'YOUR_PASSWORD'
SERVER = os.getenv('MT5_SERVER') or 'YOUR_SERVER'

SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
DATA_POINTS = 2000           # More data for LSTM context
LOT_SIZE = 0.01
MAGIC_NUMBER = 234000
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_INTERVAL = 50     # Save checkpoints every 50 episodes
LOOKBACK = 60                # LSTM lookback window

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================
# MT5 Connection
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
# Data Fetching
# =========================

def fetch_historical_data(symbol, timeframe, data_points):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, data_points)
    if rates is None or len(rates) == 0:
        raise Exception("Failed to retrieve rates")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# =========================
# Custom Trading Environment
# =========================
class TradingEnv(gym.Env):
    """
    This environment currently allows only discrete actions: 0 = Hold, 1 = Buy, 2 = Sell, 3 = Close All.

    To give the bot full autonomy and encourage exploration:
    - Consider adding actions to modify or set TP/SL levels dynamically.
    - Introduce continuous action spaces to choose exact levels of TP/SL or partial close volumes.
    - Remove hard-coded profit targets/loss limits; let the agent decide when to exit based solely on reward signals.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, lookback=60):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.end_step = len(df) - 1
        self.lookback = lookback

        # Current actions: Discrete
        # For more flexibility, consider:
        # - Expanding the action space to continuous actions for SL/TP
        # - Adding actions to increment/decrement TP/SL
        # - Allowing partial closes of positions
        self.action_space = spaces.Discrete(4)

        # Observation includes both LSTM portion (OHLCV data) and static features (account and market info).
        obs_shape = (5 * self.lookback + 8,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        self.initial_balance = mt5.account_info().balance if mt5.account_info() else 10000.0
        self.equity = self.initial_balance
        self.last_equity = self.equity

    def reset(self):
        self.current_step = self.lookback
        self._close_all_positions()
        self.last_equity = self._get_equity()
        return self._get_observation()

    def _get_observation(self):
        start_idx = self.current_step - self.lookback
        end_idx = self.current_step
        ohlcv = self.df.loc[start_idx:end_idx-1, ['open','high','low','close','tick_volume']].values
        ohlcv_flat = ohlcv.flatten()

        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            bid, ask = (self.df.loc[self.current_step, 'close'], self.df.loc[self.current_step, 'close'])
        else:
            bid, ask = tick.bid, tick.ask
        spread = ask - bid

        account_info = mt5.account_info()
        if account_info:
            balance = account_info.balance
            equity = account_info.equity
            margin = account_info.margin
            free_margin = account_info.margin_free
        else:
            balance = self.initial_balance
            equity = self.equity
            margin = 0
            free_margin = balance

        positions = mt5.positions_get(symbol=SYMBOL)
        num_positions = len(positions) if positions else 0

        static_features = np.array([bid, ask, spread, balance, equity, margin, free_margin, num_positions], dtype=np.float32)
        obs = np.concatenate((ohlcv_flat, static_features))
        return obs

    def _get_equity(self):
        info = mt5.account_info()
        if info:
            return info.equity
        else:
            return self.equity  # fallback

    def step(self, action):
        done = False
        reward = 0

        if action == 1:  # Buy
            self._execute_order(mt5.ORDER_TYPE_BUY)
        elif action == 2:  # Sell
            self._execute_order(mt5.ORDER_TYPE_SELL)
        elif action == 3:  # Close All
            self._close_all_positions()
        # Action 0 = Hold does nothing

        self.current_step += 1
        if self.current_step >= self.end_step:
            done = True
            self._close_all_positions()

        # Reward: difference in equity
        # To encourage exploration, consider:
        # - Using longer horizons (final equity at end of episode)
        # - Adding penalties for inactivity or large drawdowns
        current_equity = self._get_equity()
        reward = current_equity - self.last_equity
        self.last_equity = current_equity

        obs = self._get_observation()
        return obs, reward, done, {}

    def _execute_order(self, order_type):
        self._close_opposite_positions(order_type)
        symbol = SYMBOL
        deviation = 10
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": LOT_SIZE,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": MAGIC_NUMBER,
            "comment": "RL Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {result.retcode}")

    def _close_position(self, ticket):
        pos = mt5.positions_get(ticket=ticket)
        if pos is None or len(pos) == 0:
            return
        pos = pos[0]
        symbol = pos.symbol
        lot = pos.volume
        deviation = 10
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return
        price = tick.bid if order_type == mt5.ORDER_TYPE_BUY else tick.ask

        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": deviation,
            "magic": MAGIC_NUMBER,
            "comment": "RL Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        mt5.order_send(close_request)

    def _close_all_positions(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions:
            for pos in positions:
                self._close_position(pos.ticket)

    def _close_opposite_positions(self, order_type):
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions:
            for pos in positions:
                if order_type == mt5.ORDER_TYPE_BUY and pos.type == mt5.POSITION_TYPE_SELL:
                    self._close_position(pos.ticket)
                elif order_type == mt5.ORDER_TYPE_SELL and pos.type == mt5.POSITION_TYPE_BUY:
                    self._close_position(pos.ticket)

    def render(self, mode='human', close=False):
        profit = self.last_equity - self.initial_balance
        print(f"Step: {self.current_step}, Equity: {self.last_equity:.2f}, Profit: {profit:.2f}")

# =========================
# DQN with LSTM Model
# =========================
class DQN_LSTM(nn.Module):
    def __init__(self, lookback=60, num_features=5, static_features=8, action_size=4):
        super(DQN_LSTM, self).__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.static_features = static_features
        self.action_size = action_size

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128 + static_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_size)

    def forward(self, obs):
        batch_size = obs.size(0)
        lstm_part = obs[:, :self.lookback*self.num_features]
        static_part = obs[:, self.lookback*self.num_features:]
        lstm_part = lstm_part.view(batch_size, self.lookback, self.num_features)
        lstm_out, _ = self.lstm(lstm_part)
        lstm_out = lstm_out[:, -1, :]

        x = torch.cat([lstm_out, static_part], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# =========================
# Agent
# =========================
class Agent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = DQN_LSTM(lookback=LOOKBACK, num_features=5, static_features=8, action_size=action_dim).to(device)
        self.target_model = DQN_LSTM(lookback=LOOKBACK, num_features=5, static_features=8, action_size=action_dim).to(device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.update_target_steps = 1000
        self.step_count = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy exploration ensures the agent tries random strategies at first.
        # To further encourage exploration, consider parameter noise or other exploration strategies.
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(s)
            return torch.argmax(q_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(device)
        actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(device)
        dones = torch.FloatTensor([m[4] for m in minibatch]).unsqueeze(1).to(device)

        with torch.no_grad():
            target_q = self.target_model(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.model(states).gather(1, actions)
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.update_target_model()

    def save(self, filename, episode):
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at episode {episode}")

    def load(self, filename):
        if os.path.isfile(filename):
            print(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            episode = checkpoint['episode'] + 1
            print(f"Loaded checkpoint from episode {episode}")
            return episode
        else:
            return 1

# =========================
# Training Loop
# =========================
def train_agent(env, agent, episodes=500):
    start_episode = agent.load(os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth.tar'))

    for e in range(start_episode, episodes+1):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

        print(f"Episode {e}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        if e % CHECKPOINT_INTERVAL == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, f'checkpoint_{e}.pth.tar'), e)
            agent.save(os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth.tar'), e)

    print("Training completed.")

# =========================
# Main
# =========================
def main():
    if not connect_to_mt5(LOGIN, PASSWORD, SERVER):
        return

    try:
        data = fetch_historical_data(SYMBOL, TIMEFRAME, DATA_POINTS)
        env = TradingEnv(data, lookback=LOOKBACK)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = Agent(state_dim, action_dim)

        train_agent(env, agent, episodes=500)

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        mt5.shutdown()
        print("Disconnected from MetaTrader 5.")

if __name__ == "__main__":
    main()

# =========================
# Additional Suggestions (Not Implemented)
# =========================
# To give the bot full autonomy and promote exploration:
# - Introduce continuous action spaces (e.g., using PPO, SAC) to allow the agent to output TP/SL adjustments directly.
# - Modify the environment to accept actions that set or adjust TP/SL levels dynamically.
# - Remove any hard-coded profit/loss constraints from the environment and rely solely on reward signals over time.
# - Shape the reward to encourage exploration, e.g., by giving a higher final reward for achieving stable, profitable outcomes.
# - Increase exploration initially and decay it over time so the agent tries various strategies before converging to better ones.
