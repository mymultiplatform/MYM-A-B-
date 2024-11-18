# Filename: btc_rl_trading_bot_lstm.py

import os
import pandas as pd
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import logging
from sklearn.preprocessing import StandardScaler
import ta
from torch.utils.tensorboard import SummaryWriter

# =========================
# Configuration Parameters
# =========================

SYMBOL = "BTCUSD"
DATA_FILE = 'historical_data.csv'  # CSV file containing historical data
LOT_SIZE = 1.0                    # Lot size for simulation
CHECKPOINT_DIR = 'checkpoints'    # Directory to save checkpoints
CHECKPOINT_INTERVAL = 10          # Save a checkpoint every N episodes
LOG_FILE = 'trading_bot.log'      # Log file name

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure the checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(filename=LOG_FILE,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('TradingBot')

# =========================
# Data Fetching and Preprocessing
# =========================

def fetch_historical_data(file_path: str) -> pd.DataFrame:
    """
    Load historical trading data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing historical data.

    Returns:
        pd.DataFrame: DataFrame with historical trading data.
    """
    try:
        df = pd.read_csv(file_path)
        required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        df['time'] = pd.to_datetime(df['time'])
        logger.info(f"Loaded {len(df)} data points from {file_path}.")
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        raise

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    Parameters:
        series (pd.Series): Series of closing prices.
        window (int): The number of periods to use for RSI calculation.

    Returns:
        pd.Series: RSI values.
    """
    try:
        rsi = ta.momentum.RSIIndicator(close=series, window=window).rsi()
        return rsi
    except Exception as e:
        logger.error(f"Error computing RSI: {e}")
        raise

def compute_macd(series: pd.Series) -> pd.Series:
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given price series.

    Parameters:
        series (pd.Series): Series of closing prices.

    Returns:
        pd.Series: MACD difference values.
    """
    try:
        macd = ta.trend.MACD(close=series)
        return macd.macd_diff()
    except Exception as e:
        logger.error(f"Error computing MACD: {e}")
        raise

def compute_bollinger_bands(series: pd.Series, window: int = 20, window_dev: int = 2) -> tuple:
    """
    Calculate Bollinger Bands for a given price series.

    Parameters:
        series (pd.Series): Series of closing prices.
        window (int): The number of periods to use for Bollinger Bands calculation.
        window_dev (int): The number of standard deviations for the upper and lower bands.

    Returns:
        tuple: Upper and lower Bollinger Bands.
    """
    try:
        bollinger = ta.volatility.BollingerBands(close=series, window=window, window_dev=window_dev)
        upper = bollinger.bollinger_hband()
        lower = bollinger.bollinger_lband()
        return upper, lower
    except Exception as e:
        logger.error(f"Error computing Bollinger Bands: {e}")
        raise

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with historical trading data.

    Returns:
        pd.DataFrame: DataFrame enriched with technical indicators.
    """
    try:
        df['rsi'] = compute_rsi(df['close'])
        df['macd'] = compute_macd(df['close'])
        df['bollinger_upper'], df['bollinger_lower'] = compute_bollinger_bands(df['close'])
        df['ma5'] = ta.trend.SMAIndicator(close=df['close'], window=5).sma_indicator()
        df['ma10'] = ta.trend.SMAIndicator(close=df['close'], window=10).sma_indicator()
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        logger.info("Added technical indicators to the DataFrame.")
        return df
    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess the DataFrame by adding indicators and scaling features.

    Parameters:
        df (pd.DataFrame): Raw DataFrame with historical trading data.

    Returns:
        tuple: Preprocessed DataFrame and the scaler used for normalization.
    """
    try:
        df = add_indicators(df)
        scaler = StandardScaler()
        feature_columns = ['open', 'high', 'low', 'close', 'tick_volume', 
                           'ma5', 'ma10', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        # Drop non-feature columns
        df = df.drop(columns=['time'])
        logger.info("Preprocessed data with scaling.")
        return df, scaler
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

# =========================
# Custom Gym Environment
# =========================

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for OpenAI Gym.

    Attributes:
        df (pd.DataFrame): Historical trading data with technical indicators.
        window_size (int): Number of past data points to include in each observation.
        current_step (int): Current step in the environment.
        end_step (int): Final step in the environment.
        action_space (gym.spaces.Discrete): Action space (Hold, Buy, Sell, Close Position).
        observation_space (gym.spaces.Box): Observation space with window_size x num_features.
        balance (float): Current account balance.
        equity (float): Current account equity (balance + unrealized PnL).
        position (str or None): Current position ('long', 'short', or None).
        position_price (float): Price at which the current position was opened.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, window_size: int = 10):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.current_step = self.window_size - 1
        self.end_step = len(df) - 1

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell, 3 = Close Position
        self.action_space = spaces.Discrete(4)

        # Observation space: window_size x num_features
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

        # Risk management parameters
        self.max_drawdown = 0.2  # 20% drawdown
        self.peak_equity = self.equity

        # Transaction costs
        self.transaction_cost_pct = 0.001  # 0.1% per trade

    def reset(self) -> np.ndarray:
        """
        Reset the environment to the initial state.

        Returns:
            np.ndarray: Initial observation.
        """
        self.current_step = self.window_size - 1
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = None
        self.position_price = 0.0
        self.peak_equity = self.equity
        return self._next_observation()

    def _next_observation(self) -> np.ndarray:
        """
        Get the next observation.

        Returns:
            np.ndarray: Observation for the current step.
        """
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1
        obs = self.df.iloc[start:end].values  # shape: (window_size, num_features)
        return obs.astype(np.float32)

    def step(self, action: int) -> tuple:
        """
        Execute one time step within the environment.

        Parameters:
            action (int): Action to take.

        Returns:
            tuple: (observation, reward, done, info)
        """
        done = False
        current_price = self.df.loc[self.current_step, 'close']
        previous_equity = self.equity

        # Execute action with position management
        if action == 1:  # Buy
            if self.position is None:
                self.position = 'long'
                self.position_price = current_price
                self.balance -= current_price * LOT_SIZE * self.transaction_cost_pct
                logger.info(f"Opened LONG position at {current_price}")
        elif action == 2:  # Sell
            if self.position is None:
                self.position = 'short'
                self.position_price = current_price
                self.balance -= current_price * LOT_SIZE * self.transaction_cost_pct
                logger.info(f"Opened SHORT position at {current_price}")
        elif action == 3:  # Close Position
            if self.position == 'long':
                profit = (current_price - self.position_price) * LOT_SIZE
                self.balance += profit
                self.balance -= current_price * LOT_SIZE * self.transaction_cost_pct
                logger.info(f"Closed LONG position at {current_price} with profit {profit}")
                self.position = None
                self.position_price = 0.0
            elif self.position == 'short':
                profit = (self.position_price - current_price) * LOT_SIZE
                self.balance += profit
                self.balance -= current_price * LOT_SIZE * self.transaction_cost_pct
                logger.info(f"Closed SHORT position at {current_price} with profit {profit}")
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

        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Calculate drawdown
        drawdown = (self.peak_equity - self.equity) / self.peak_equity

        # Calculate reward as change in equity minus transaction costs and penalties
        reward = self.equity - previous_equity

        # Apply risk penalty for drawdown
        if drawdown > self.max_drawdown:
            reward -= 100  # Large penalty for exceeding max drawdown
            done = True
            logger.warning(f"Max drawdown exceeded: {drawdown*100:.2f}%")

        # Increment step after processing to avoid index out-of-bounds
        self.current_step += 1
        if self.current_step >= self.end_step:
            done = True
            # Close any open positions at the end
            if self.position == 'long':
                profit = (current_price - self.position_price) * LOT_SIZE
                self.balance += profit
                self.balance -= current_price * LOT_SIZE * self.transaction_cost_pct
                logger.info(f"Closed LONG position at end of data with profit {profit}")
            elif self.position == 'short':
                profit = (self.position_price - current_price) * LOT_SIZE
                self.balance += profit
                self.balance -= current_price * LOT_SIZE * self.transaction_cost_pct
                logger.info(f"Closed SHORT position at end of data with profit {profit}")
            self.position = None
            self.position_price = 0.0
            self.equity = self.balance
            reward = self.equity - previous_equity

        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}

        return obs, reward, done, info

    def render(self, mode: str = 'human', close: bool = False):
        """
        Render the environment state.

        Parameters:
            mode (str): Render mode.
            close (bool): Whether to close the rendering.
        """
        profit = self.equity - self.initial_balance
        logger.info(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Equity: {self.equity:.2f}, Profit: {profit:.2f}")

# =========================
# Deep Q-Network with LSTM Model
# =========================

class DQNAgentModel(nn.Module):
    """
    Deep Q-Network model with LSTM layers to capture temporal dependencies.

    Attributes:
        lstm (nn.LSTM): LSTM layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, input_size: int, action_size: int, hidden_size: int = 128, num_layers: int = 1):
        super(DQNAgentModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor, hidden: tuple = None) -> tuple:
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            hidden (tuple): Hidden state for LSTM.

        Returns:
            tuple: Q-values and updated hidden state.
        """
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]  # Take the output from the last time step
        out = torch.relu(self.fc1(out))
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out, hidden

# =========================
# Agent Class
# =========================

class Agent:
    """
    Reinforcement Learning Agent using Deep Q-Network with LSTM.

    Attributes:
        state_size (tuple): Shape of the state space.
        action_size (int): Number of possible actions.
        hidden_size (int): Hidden size for LSTM.
        num_layers (int): Number of LSTM layers.
        memory (deque): Experience replay memory.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for exploration.
        model (DQNAgentModel): Q-Network.
        target_model (DQNAgentModel): Target Q-Network.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        step_count (int): Counter for steps to manage target network updates.
    """
    def __init__(self, state_size: tuple, action_size: int, hidden_size: int = 128, num_layers: int = 1,
                 learning_rate: float = 1e-4, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995):
        self.state_size = state_size  # (window_size, num_features)
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.memory = deque(maxlen=100000)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQNAgentModel(state_size[1], action_size, hidden_size, num_layers).to(device)
        self.target_model = DQNAgentModel(state_size[1], action_size, hidden_size, num_layers).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.step_count = 0  # Counter for steps

    def update_target_model(self):
        """
        Update the target network with the weights from the main network.
        """
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        logger.info("Updated target model.")

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Store experience in replay memory.

        Parameters:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """
        Choose an action based on the current state.

        Parameters:
            state (np.ndarray): Current state.

        Returns:
            int: Action to take.
        """
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            logger.debug(f"Random action taken: {action}")
            return action
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # shape: (1, window_size, num_features)
        with torch.no_grad():
            q_values, _ = self.model(state)
        action = torch.argmax(q_values[0]).item()
        logger.debug(f"Predicted action taken: {action}")
        return action

    def replay(self, batch_size: int = 64):
        """
        Train the agent with a batch of experiences.

        Parameters:
            batch_size (int): Number of experiences to sample for training.
        """
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

            # Compute target Q-values using target network
            with torch.no_grad():
                target_q_values, _ = self.target_model(next_states)
                max_target_q_values = target_q_values.max(1, keepdim=True)[0]
                target = rewards + (self.gamma * max_target_q_values * (1 - dones))

            # Compute current Q-values
            current_q_values, _ = self.model(states)
            current_q_values = current_q_values.gather(1, actions)

            # Compute loss
            loss = self.criterion(current_q_values, target)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Decrease exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Soft update of target network
            self.soft_update(tau=0.01)

            self.step_count += 1

        except Exception as ex:
            logger.error(f"Exception during replay: {ex}")
            import traceback
            traceback.print_exc()

    def soft_update(self, tau: float = 0.01):
        """
        Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters:
            tau (float): Interpolation parameter.
        """
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        logger.debug("Performed soft update of target model.")

    def save(self, filename: str, episode: int):
        """
        Save the model and optimizer state.

        Parameters:
            filename (str): Path to save the checkpoint.
            episode (int): Current episode number.
        """
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
            logger.info(f"Checkpoint saved at episode {episode}")
        except Exception as ex:
            logger.error(f"Exception during saving checkpoint: {ex}")

    def load(self, filename: str) -> int:
        """
        Load the model and optimizer state.

        Parameters:
            filename (str): Path to the checkpoint file.

        Returns:
            int: Episode number to start from.
        """
        if os.path.isfile(filename):
            try:
                logger.info(f"Loading checkpoint '{filename}'")
                checkpoint = torch.load(filename, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                self.step_count = checkpoint['step_count']
                episode = checkpoint['episode'] + 1
                logger.info(f"Loaded checkpoint '{filename}' (starting from episode {episode})")
                return episode
            except Exception as ex:
                logger.error(f"Exception during loading checkpoint: {ex}")
                return 1  # Start from episode 1 if loading fails
        else:
            logger.info(f"No checkpoint found at '{filename}'. Starting from episode 1.")
            return 1  # Start from episode 1

# =========================
# Training Function
# =========================

def train_agent(env: TradingEnv, agent: Agent, episodes: int = 1000, batch_size: int = 64):
    """
    Train the RL agent in the given environment.

    Parameters:
        env (TradingEnv): The trading environment.
        agent (Agent): The RL agent.
        episodes (int): Number of training episodes.
        batch_size (int): Batch size for experience replay.
    """
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
                agent.replay(batch_size)

            logger.info(f"Episode {e}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.4f}")
            print(f"Episode {e}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.4f}")

            # Log metrics to TensorBoard
            writer.add_scalar('Total Reward', total_reward, e)
            writer.add_scalar('Epsilon', agent.epsilon, e)

            # Save the model at intervals
            if e % CHECKPOINT_INTERVAL == 0:
                checkpoint_filename = os.path.join(CHECKPOINT_DIR, f'checkpoint_{e}.pth.tar')
                agent.save(checkpoint_filename, e)
                latest_checkpoint = os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth.tar')
                agent.save(latest_checkpoint, e)

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
            print("Training interrupted by user.")
            agent.save(os.path.join(CHECKPOINT_DIR, 'checkpoint_interrupted.pth.tar'), e)
            break
        except Exception as ex:
            logger.error(f"Exception during training at episode {e}: {ex}")
            import traceback
            traceback.print_exc()
            continue  # Proceed to the next episode

    writer.close()
    logger.info("Training completed.")
    print("Training completed.")

# =========================
# Main Function
# =========================

def main():
    """
    Main function to load data, initialize environment and agent, and start training.
    """
    try:
        # Load and preprocess data
        data = fetch_historical_data(DATA_FILE)
        data, scaler = preprocess_data(data)

        # Initialize environment with window_size
        window_size = 10
        env = TradingEnv(data, window_size=window_size)
        logger.info("Initialized trading environment.")
        print("Initialized trading environment.")

        # Initialize agent
        state_size = env.observation_space.shape  # (window_size, num_features)
        action_size = env.action_space.n
        agent = Agent(state_size, action_size)
        logger.info("Initialized DQN agent.")
        print("Initialized DQN agent.")

        # Train agent
        train_agent(env, agent, episodes=1000, batch_size=64)

    except Exception as e:
        logger.error(f"Exception in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Shutting down.")
        print("Shutting down.")

if __name__ == "__main__":
    main()
