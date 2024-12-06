import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import datetime
import time

# ============================================
# Configuration
# ============================================
LOGIN = os.getenv('MT5_LOGIN') or 'YOUR_LOGIN'
PASSWORD = os.getenv('MT5_PASSWORD') or 'YOUR_PASSWORD'
SERVER = os.getenv('MT5_SERVER') or 'YOUR_SERVER'

SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
DATA_POINTS = 2000
LOT_SIZE = 0.01
MAGIC_NUMBER = 234000
CHECKPOINT_DIR = 'checkpoints_sac'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOOKBACK = 60
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# Additional Suggestions and Final Revision Notes
# ============================================
# The following TODO items and comments are derived from the final revision and previous suggestions:
#
# TODO: Persistent MT5 Connection
# - Implement a more robust connection function with retries and error handling.
#
# TODO: Symbol Verification
# - Verify the symbol is available and visible in MT5 before trading.
#
# TODO: Data Preprocessing and Indicators
# - Consider adding technical indicators (MA, RSI, etc.) and normalization to observations.
# - Implement standardized or normalized input features for stability.
#
# TODO: Curiosity-Driven Exploration
# - Add forward and inverse dynamics models to produce intrinsic rewards.
# - Intrinsic reward = prediction error of forward model. Encourages exploration.
# - Useful if environment has sparse external rewards or large state spaces.
#
# TODO: PPO or A2C Integration
# - Consider using policy gradient methods like PPO or A2C instead of (or in addition to) SAC.
# - These methods produce a stochastic policy and can improve exploration.
# - PPO may offer better stability and sample efficiency.
#
# TODO: Advanced RL Features
# - Entropy tuning (target entropy) for SAC or PPO to balance exploration/exploitation.
# - Prioritized replay buffer for more efficient training.
# - TensorBoard logging for metrics and debugging.
# - Early stopping and validation environment for model selection.
# - Hyperparameter optimization with tools like Optuna.
#
# TODO: Risk Management and Reward Shaping
# - Incorporate transaction costs (spreads, commissions) into reward.
# - Add risk-adjusted metrics (e.g. Sharpe ratio) or max drawdown constraints.
# - Introduce trailing stops and dynamic position sizing.
#
# TODO: Scalability and Evaluation
# - Parallel training environments.
# - Ensemble agents for more robust policies.
# - Backtesting and paper trading modes to evaluate performance before live deployment.

# The code below remains as in the baseline, with added TODO comments for future improvements.

# ============================================
# MT5 Connection
# ============================================
def connect_to_mt5(login, password, server):
    # TODO: Add retry logic and enhanced error messages, handle initialization failure gracefully
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return False

    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        print("Connected to MetaTrader 5")
        # TODO: verify_symbol(SYMBOL) here if implemented
        return True
    else:
        print("Failed to connect to MetaTrader 5")
        mt5.shutdown()
        return False

# ============================================
# Data Fetching
# ============================================
def fetch_historical_data(symbol, timeframe, data_points):
    # TODO: Consider verifying symbol and handling no data returned more gracefully.
    # TODO: Add indicator computation or normalization here or inside the environment.
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, data_points)
    if rates is None or len(rates) == 0:
        raise Exception("Failed to retrieve rates")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# ============================================
# Trading Environment with Continuous Actions
# ============================================
class TradingEnv(gym.Env):
    """
    A continuous-action trading environment integrated with MT5.

    Observation:
      - last LOOKBACK OHLCV data plus static features (bid, ask, spread, balance, equity, margin, free_margin, num_positions)
    Action:
      - action = [pos_scale, tp_offset, sl_offset]
        * pos_scale ∈ [-1,1]: scale of position (negative=short, positive=long)
        * tp_offset, sl_offset ∈ [-1,1]: scaled offsets for TP/SL
    Reward:
      - Current simplistic: change in equity from last step. 
        TODO: Add transaction costs, risk-adjusted returns, or intrinsic rewards for exploration.

    Further TODO:
    - Add curiosity-driven exploration: forward model prediction error as intrinsic reward.
    - Try PPO or A2C for a stochastic policy and potentially better exploration.
    - Incorporate technical indicators, normalization, or other advanced features.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df, lookback=60):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.end_step = len(df) - 1
        self.lookback = lookback

        # Continuous action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # TODO: Add indicators or normalized features. Possibly use sklearn's StandardScaler.
        obs_shape = (5 * self.lookback + 8,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        self.initial_balance = mt5.account_info().balance if mt5.account_info() else 10000.0
        self.equity = self.initial_balance
        self.last_equity = self.equity

        # TP/SL range (TODO: make dynamic or environment-dependent)
        self.tp_range = 1000.0
        self.sl_range = 1000.0

        # TODO: Implement max drawdown or risk constraints for early stopping if desired.

    def reset(self):
        self._close_all_positions()
        self.current_step = self.lookback
        self.last_equity = self._get_equity()
        return self._get_observation()

    def _get_observation(self):
        start_idx = self.current_step - self.lookback
        end_idx = self.current_step
        ohlcv = self.df.loc[start_idx:end_idx-1, ['open','high','low','close','tick_volume']].values
        
        # TODO: Add indicators like moving averages or RSI here if desired.
        # TODO: Normalize OHLCV if using a scaler.

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
            return self.equity

    def step(self, action):
        # TODO: Incorporate transaction cost, or add intrinsic reward for curiosity-driven exploration.
        # reward = (change in equity) + λ * (intrinsic_reward) - transaction_cost
        pos_scale, tp_offset, sl_offset = action
        self._apply_action(pos_scale, tp_offset, sl_offset)

        self.current_step += 1
        done = self.current_step >= self.end_step

        current_equity = self._get_equity()
        reward = current_equity - self.last_equity
        self.last_equity = current_equity

        # TODO: If implementing max drawdown: 
        # if self.equity < self.initial_balance * (1 - self.max_drawdown):
        #     done = True
        #     reward -= penalty

        obs = self._get_observation()
        return obs, reward, done, {}

    def _apply_action(self, pos_scale, tp_offset, sl_offset):
        # TODO: Dynamic position sizing based on risk
        positions = mt5.positions_get(symbol=SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return
        price = tick.ask
        direction = 1 if pos_scale >= 0 else -1

        desired_volume = abs(pos_scale) * LOT_SIZE

        # Calculate TP/SL
        if direction > 0:  # Long
            tp_price = price + (tp_offset * self.tp_range * mt5.symbol_info(SYMBOL).point)
            sl_price = price - (sl_offset * self.sl_range * mt5.symbol_info(SYMBOL).point)
        else:  # Short
            tp_price = tick.bid - (tp_offset * self.tp_range * mt5.symbol_info(SYMBOL).point)
            sl_price = tick.bid + (sl_offset * self.sl_range * mt5.symbol_info(SYMBOL).point)

        if not positions:
            if desired_volume > 0:
                self._open_position(direction, desired_volume, tp_price, sl_price)
        else:
            pos = positions[0]
            current_dir = 1 if pos.type == mt5.POSITION_TYPE_BUY else -1
            if desired_volume == 0:
                self._close_position(pos.ticket)
            else:
                if current_dir != direction:
                    self._close_position(pos.ticket)
                    self._open_position(direction, desired_volume, tp_price, sl_price)
                else:
                    self._modify_position(pos.ticket, tp_price, sl_price)

    def _open_position(self, direction, volume, tp, sl):
        symbol = SYMBOL
        deviation = 10
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return
        order_type = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": MAGIC_NUMBER,
            "comment": "RL Open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        # TODO: Enhanced error handling and logging (check result, mt5.last_error())

    def _close_position(self, ticket):
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
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

        result = mt5.order_send(close_request)
        # TODO: Error handling on close

    def _modify_position(self, ticket, tp, sl):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
            "symbol": SYMBOL,
            "magic": MAGIC_NUMBER,
            "comment": "Modify SL/TP",
        }
        result = mt5.order_send(request)
        # TODO: Check and handle errors

    def _close_all_positions(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions:
            for pos in positions:
                self._close_position(pos.ticket)

    def render(self, mode='human', close=False):
        profit = self.last_equity - self.initial_balance
        print(f"Step: {self.current_step}, Equity: {self.last_equity:.2f}, Profit: {profit:.2f}")

# ============================================
# SAC Implementation
# ============================================
# TODO: Advanced Architectures:
#  - Consider Residual MLP, LSTMs, or LayerNorm for stability.
# TODO: Gradient Clipping:
#  - Prevent exploding gradients with torch.nn.utils.clip_grad_norm_.
# TODO: Entropy Temperature Tuning:
#  - Adaptively adjust alpha (entropy coefficient).
# TODO: Prioritized Replay:
#  - Use a prioritized replay buffer to sample more informative transitions.

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GaussianPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        # Adjust log_prob for Tanh transformation
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        return action, log_prob.sum(-1, keepdim=True)


class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 alpha_min=0.01, alpha_decay=0.999, buffer_size=100000, batch_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size

        self.policy = GaussianPolicy(state_dim, action_dim).to(device)
        self.q1 = MLP(state_dim+action_dim, 1).to(device)
        self.q2 = MLP(state_dim+action_dim, 1).to(device)
        self.q1_target = MLP(state_dim+action_dim, 1).to(device)
        self.q2_target = MLP(state_dim+action_dim, 1).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.memory = deque(maxlen=buffer_size)

        # TODO: Implement adaptive alpha tuning with a target entropy.
        # self.target_entropy = -action_dim
        # self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # TODO: Apply gradient clipping if needed.

    def select_action(self, state, eval_mode=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if eval_mode:
                # Deterministic action at evaluation
                mean, log_std = self.policy.forward(state_t)
                z = mean
                action = torch.tanh(z)
            else:
                action, _ = self.policy.sample(state_t)
        return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        # TODO: For prioritized replay, store transitions with priority
        self.memory.append((state, action, reward, next_state, float(done)))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.FloatTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Compute target Q
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_states_t)
            q1_next = self.q1_target(torch.cat([next_states_t, next_action], dim=-1))
            q2_next = self.q2_target(torch.cat([next_states_t, next_action], dim=-1))
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = rewards_t + (1 - dones_t) * self.gamma * q_next

        # Critic loss
        q1_pred = self.q1(torch.cat([states_t, actions_t], dim=-1))
        q2_pred = self.q2(torch.cat([states_t, actions_t], dim=-1))
        q1_loss = ((q1_pred - target_q)**2).mean()
        q2_loss = ((q2_pred - target_q)**2).mean()

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Actor update
        new_action, log_prob = self.policy.sample(states_t)
        q1_new = self.q1(torch.cat([states_t, new_action], dim=-1))
        q2_new = self.q2(torch.cat([states_t, new_action], dim=-1))
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # Soft updates for critics
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Decay alpha
        if self.alpha > self.alpha_min:
            self.alpha *= self.alpha_decay

        # TODO: Implement adaptive alpha tuning if desired.

    def save(self, filename):
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'alpha': self.alpha
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.q1.load_state_dict(checkpoint['q1_state_dict'])
            self.q2.load_state_dict(checkpoint['q2_state_dict'])
            self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
            self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
            self.alpha = checkpoint['alpha']
            print(f"Loaded checkpoint from {filename}")

# ============================================
# Training Loop
# ============================================
def train_sac(env, agent, episodes=500, steps_per_episode=200):
    # TODO: Implement TensorBoard logging for metrics (reward, losses, etc.)
    # TODO: Implement separate validation environment and early stopping
    # TODO: Possibly integrate curiosity-driven intrinsic rewards here
    # TODO: Hyperparameter tuning with a library like Optuna
    best_reward = -np.inf
    for e in range(1, episodes+1):
        state = env.reset()
        episode_reward = 0
        for _ in range(steps_per_episode):
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            episode_reward += reward
            if done:
                break
        
        print(f"Episode {e}: Reward={episode_reward:.2f}, Alpha={agent.alpha:.4f}")
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(CHECKPOINT_DIR, 'best_sac.pth'))

        # Save latest model periodically
        if e % 50 == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, f'checkpoint_{e}.pth'))

# ============================================
# Main
# ============================================
def main():
    if not connect_to_mt5(LOGIN, PASSWORD, SERVER):
        return

    try:
        data = fetch_historical_data(SYMBOL, TIMEFRAME, DATA_POINTS)
        env = TradingEnv(data, lookback=LOOKBACK)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SACAgent(state_dim, action_dim)

        # TODO: Consider pretraining the model on historical data in a supervised manner if desired.
        # e.g., training a forward model or embeddings before RL training.

        # TODO: Implement validation environment or backtest environment
        # TODO: Try adding curiosity-driven exploration models (forward/inverse dynamics)
        # TODO: Compare performance under PPO or A2C for better exploration if SAC doesn’t suffice.

        train_sac(env, agent, episodes=500)

        # TODO: Evaluate or paper trade the agent after training
        # TODO: Monitor metrics using TensorBoard or another visualization tool

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        mt5.shutdown()
        print("Disconnected from MetaTrader 5.")

if __name__ == "__main__":
    main()
