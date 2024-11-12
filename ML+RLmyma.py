import os
import MetaTrader5 as mt5
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time
import yfinance as yf
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MetaTrader5 and login using environment variables
login = int(os.getenv('MT5_LOGIN'))
password = os.getenv('MT5_PASSWORD')
server = os.getenv('MT5_SERVER')

if not mt5.initialize():
    raise Exception("MetaTrader5 initialization failed")

if mt5.login(login=login, password=password, server=server):
    print("Connected to MetaTrader 5 for data retrieval")
else:
    raise Exception("Failed to connect to MetaTrader 5")

# RL Neural Network model for decision making
class TradeRLModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TradeRLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to calculate the RSI
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.abs(np.where(delta < 0, delta, 0))
    
    avg_gain = np.mean(gain[-period:]) if len(gain) >= period else np.mean(gain)
    avg_loss = np.mean(loss[-period:]) if len(loss) >= period else np.mean(loss)
    
    if avg_loss == 0:
        return 100  # Maximum RSI
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to fetch historical BTC price data
def fetch_historical_btc_data(period="100d", interval="1m"):
    try:
        btc = yf.Ticker("BTC-USD")
        btc_data = btc.history(period=period, interval=interval)
        if btc_data.empty:
            raise Exception("Fetched BTC data is empty.")
        return btc_data
    except Exception as e:
        raise Exception(f"Failed to fetch historical BTC data: {e}")

# Function to calculate rupture price
def calculate_rupture(balance, entry_price, spread_cost, lot_size, maintenance_margin, leverage):
    try:
        # Input validations
        if balance <= 0:
            raise ValueError("Balance must be greater than 0.")
        if spread_cost < 0:
            raise ValueError("Spread cost cannot be negative.")
        if lot_size <= 0:
            raise ValueError("Lot size must be greater than 0.")
        if not (0 <= maintenance_margin < 100):
            raise ValueError("Maintenance margin must be between 0 and 100 percent.")

        # Calculate initial margin and required maintenance margin
        initial_margin = balance / leverage
        required_margin = initial_margin * (maintenance_margin / 100)

        # Calculate maximum allowable loss before hitting maintenance margin
        max_loss = balance - required_margin - spread_cost
        if max_loss <= 0:
            raise ValueError("Balance is too low to cover the spread cost and maintenance margin.")

        # Calculate price movement considering leverage and lot size
        price_movement = max_loss / (leverage * lot_size)

        # Calculate rupture price
        rupture_price = entry_price - price_movement
        rupture_price = max(rupture_price, 0.0)  # Ensure rupture price is not negative

        # Calculate percentage drop
        price_drop_percentage = (price_movement / entry_price) * 100.0

        return rupture_price, price_drop_percentage
    except ValueError as ve:
        raise ValueError(f"Input Error: {ve}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

# Function to interpolate between two colors channel-wise
def interpolate_color(color_start, color_end, factor: float):
    factor = max(0.0, min(1.0, factor))  # Clamp factor between 0 and 1
    start_rgb = tuple(int(color_start[i:i+2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(color_end[i:i+2], 16) for i in (1, 3, 5))
    interp_rgb = tuple(int(s + (e - s) * factor) for s, e in zip(start_rgb, end_rgb))
    return f'#{interp_rgb[0]:02x}{interp_rgb[1]:02x}{interp_rgb[2]:02x}'

# Function to calculate thermal levels for visual representation
def calculate_thermal_levels(entry_price, rupture_price, balance, leverage, lot_size, spread_cost):
    # Define the fixed step size
    step_size = 10.0
    price_levels = []
    current_price = entry_price
    while current_price >= rupture_price:
        price_levels.append(current_price)
        current_price -= step_size
    if not price_levels or price_levels[-1] > rupture_price:
        price_levels.append(rupture_price)
    if price_levels[-1] > 0.0:
        price_levels.append(0.0)

    # Calculate corresponding balances for each price level
    loss_per_dollar = leverage * lot_size
    thermal_levels = []
    for price in price_levels:
        price_drop = entry_price - price
        current_balance = balance - (price_drop * loss_per_dollar)
        current_balance = max(current_balance, 0.0)  # Avoid negative balance
        thermal_levels.append((price, current_balance))
    
    return thermal_levels

# Environment for the RL agent interacting with MT5
class MT5Environment:
    def __init__(self, historical_data, max_trades_per_episode=10):
        self.historical_data = historical_data.reset_index()
        self.current_step = 0
        self.max_steps = len(self.historical_data) - 1
        self.max_trades = max_trades_per_episode
        self.current_trades = 0
        self.account_info = mt5.account_info()
        if self.account_info is None:
            raise Exception("Failed to retrieve account information")
        self.balance = float(self.account_info.balance)
        self.initial_balance = self.balance
        self.leverage = float(self.account_info.leverage) if hasattr(self.account_info, 'leverage') else 100.0  # Default leverage
        self.lot_size = 0.1  # Example lot size
        self.spread_cost = 0.0  # To be updated based on symbol's spread
        self.maintenance_margin = 50.0  # Example maintenance margin percentage
    
    def get_state(self):
        """Get current state information for the RL agent, including market data."""
        if self.current_step >= self.max_steps:
            self.current_step = 0  # Reset to start if end is reached

        # Get current candle data
        current_candle = self.historical_data.iloc[self.current_step]
        close_prices = self.historical_data['Close'].values[:self.current_step + 1]
        
        # Calculate RSI
        rsi = calculate_rsi(close_prices)
        
        # Calculate price returns
        returns = np.diff(close_prices) / close_prices[:-1]
        mean_return = np.mean(returns) if len(returns) > 0 else 0.0
        volatility = np.std(returns) if len(returns) > 0 else 0.0

        # Latest bid/ask prices for BTCUSD from current candle
        bid = current_candle['Close']  # Simplification: using close price as bid
        ask = current_candle['Close']  # Simplification: using close price as ask

        # Fetch live BTC price (using historical close as live price)
        live_price = current_candle['Close']

        # Calculate rupture price and price drop percentage
        rupture_price, price_drop_percentage = calculate_rupture(
            balance=self.balance,
            entry_price=live_price,
            spread_cost=self.spread_cost,
            lot_size=self.lot_size,
            maintenance_margin=self.maintenance_margin,
            leverage=self.leverage
        )

        # Calculate thermal levels
        thermal_levels = calculate_thermal_levels(
            entry_price=live_price,
            rupture_price=rupture_price,
            balance=self.balance,
            leverage=self.leverage,
            lot_size=self.lot_size,
            spread_cost=self.spread_cost
        )

        # For simplicity, include only the current balance at the first thermal level
        thermal_balance = thermal_levels[0][1] if thermal_levels else self.balance

        # State: [balance, number of trades, equity, last close price, mean close price, bid, ask, RSI, mean return, volatility, rupture_price, price_drop_percentage, thermal_balance]
        equity = self.balance  # Simplification: assuming equity equals balance
        open_trades = self.current_trades  # Simplification: tracking trades count

        state = [
            self.balance,
            open_trades,
            equity,
            live_price,  # Most recent close price
            np.mean(close_prices),  # Average of close prices up to current step
            bid,
            ask,
            rsi,
            mean_return,
            volatility,
            rupture_price,
            price_drop_percentage,
            thermal_balance
        ]
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension

    def step(self, action):
        """Execute a step in the environment based on action (buy/sell/no action)."""
        if self.current_trades >= self.max_trades:
            action = 2  # Force neutral action if max trades reached

        if action == 0:
            self.buy()
            self.current_trades += 1
        elif action == 1:
            self.sell()
            self.current_trades += 1
        else:
            pass  # Neutral (no action)
        
        # Update balance after the action
        self.balance = float(mt5.account_info().balance)
        reward = self.get_reward()
        self.current_step += 1
        next_state = self.get_state()
        
        done = self.current_trades >= self.max_trades or self.current_step >= self.max_steps
        return next_state, reward, done

    def buy(self):
        """Execute a buy order."""
        symbol = "BTCUSD"
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"Failed to get tick data for {symbol}")
            return
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.lot_size,
            "type": mt5.ORDER_TYPE_BUY,  # Corrected constant
            "price": tick.ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "RL Buy Order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Buy order failed: {result.retcode}")

    def sell(self):
        """Execute a sell order."""        
        symbol = "BTCUSD"
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"Failed to get tick data for {symbol}")
            return
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.lot_size,
            "type": mt5.ORDER_TYPE_SELL,  # Corrected constant
            "price": tick.bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "RL Sell Order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Sell order failed: {result.retcode}")

    def get_reward(self):
        """Calculate reward based on balance change and rupture."""
        # Reward is primarily based on balance change
        reward = self.balance - self.initial_balance
        
        # Additional punishment based on price drop percentage
        # The deeper the drop, the more negative the reward
        try:
            _, price_drop_percentage = calculate_rupture(
                balance=self.initial_balance,
                entry_price=self.historical_data.iloc[self.current_step]['Close'],
                spread_cost=self.spread_cost,
                lot_size=self.lot_size,
                maintenance_margin=self.maintenance_margin,
                leverage=self.leverage
            )
            # Penalize the agent proportionally to the price drop
            penalty = -price_drop_percentage
            reward += penalty
        except Exception as e:
            print(f"Error calculating penalty: {e}")
        
        return reward

    def reset(self):
        """Reset the environment to its initial state."""        
        self.balance = self.initial_balance
        self.current_trades = 0
        self.current_step = 0
        return self.get_state()

# Agent interacting with the environment
class TradeAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = TradeRLModel(state_size, 128, action_size).to(device)  # Moved model to device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)  # Use deque for efficient memory management
        self.gamma = 0.95
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # Best action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough memories to replay

        minibatch = random.sample(self.memory, batch_size)
        states = torch.cat([m[0] for m in minibatch]).to(device)
        actions = torch.tensor([m[1] for m in minibatch], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([m[2] for m in minibatch], dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.cat([m[3] for m in minibatch]).to(device)
        dones = torch.tensor([m[4] for m in minibatch], dtype=torch.float32, device=device).unsqueeze(1)

        # Current Q-values
        current_q = self.model(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss
        loss = self.criterion(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training function
def train_agent():
    # Fetch historical data once before training
    historical_data = fetch_historical_btc_data(period="100d", interval="1m")
    env = MT5Environment(historical_data=historical_data, max_trades_per_episode=5)  # Start with fewer trades
    state_size = 13  # Updated state size to 13
    action_size = 3  # Buy, Sell, Neutral
    agent = TradeAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)
            
            if done:
                break  # Exit the loop if max trades reached or end of data
        
        # Log rewards and epsilon to TensorBoard
        writer.add_scalar('Reward/Episode', total_reward, episode)
        writer.add_scalar('Epsilon/Episode', agent.epsilon, episode)

        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}/{episodes} - Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    writer.close()
    print("Training completed.")

if __name__ == "__main__":
    try:
        train_agent()
    finally:
        # Shutdown MetaTrader5 connection gracefully
        mt5.shutdown()
