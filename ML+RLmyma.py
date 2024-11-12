import MetaTrader5 as mt5
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time
import yfinance as yf
from decimal import Decimal, getcontext, ROUND_HALF_UP

# Set decimal precision
getcontext().prec = 10
getcontext().rounding = ROUND_HALF_UP

# Initialize MetaTrader5 and login
login = 312128713
password = "Sexo247420@"
server = "XMGlobal-MT5 7"

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
    gain = np.where(delta > 0, delta, 0).mean()
    loss = np.abs(np.where(delta < 0, delta, 0)).mean()
    if loss == 0:
        return 100  # Return maximum RSI if there's no loss
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to fetch live BTC price with retry mechanism
def fetch_btc_price():
    retries = 3
    for attempt in range(retries):
        try:
            btc = yf.Ticker("BTC-USD")
            btc_data = btc.history(period="1d")
            live_price = Decimal(btc_data["Close"].iloc[-1])
            return live_price
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                raise Exception(f"Failed to fetch BTC price: {e}")

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
        required_margin = initial_margin * (maintenance_margin / Decimal(100))

        # Calculate maximum allowable loss before hitting maintenance margin
        max_loss = balance - required_margin - spread_cost
        if max_loss <= 0:
            raise ValueError("Balance is too low to cover the spread cost and maintenance margin.")

        # Calculate price movement considering leverage and lot size
        price_movement = max_loss / (leverage * lot_size)

        # Calculate rupture price
        rupture_price = entry_price - price_movement
        rupture_price = max(rupture_price, Decimal(0))  # Ensure rupture price is not negative

        # Calculate percentage drop
        price_drop_percentage = (price_movement / entry_price) * Decimal(100)
        
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
    step_size = Decimal('10')
    price_levels = []
    current_price = entry_price
    while current_price >= rupture_price:
        price_levels.append(current_price)
        current_price -= step_size
    if price_levels[-1] > rupture_price:
        price_levels.append(rupture_price)
    if price_levels[-1] > Decimal(0):
        price_levels.append(Decimal(0))

    # Calculate corresponding balances for each price level
    loss_per_dollar = leverage * lot_size
    thermal_levels = []
    for price in price_levels:
        price_drop = entry_price - price
        current_balance = balance - (price_drop * loss_per_dollar)
        current_balance = max(current_balance, Decimal(0))  # Avoid negative balance
        thermal_levels.append((price, current_balance))
    
    return thermal_levels

# Environment for the RL agent interacting with MT5
class MT5Environment:
    def __init__(self):
        self.account_info = mt5.account_info()
        if self.account_info is None:
            raise Exception("Failed to retrieve account information")
        self.balance = Decimal(self.account_info.balance)
        self.initial_balance = self.balance
        self.leverage = Decimal(self.account_info.leverage) if hasattr(self.account_info, 'leverage') else Decimal('100')  # Default leverage
        self.lot_size = Decimal('0.1')  # Example lot size
        self.spread_cost = Decimal('0.0')  # To be updated based on symbol's spread
        self.maintenance_margin = Decimal('50')  # Example maintenance margin percentage

    def get_state(self):
        """Get current state information for the RL agent, including market data."""
        # Retrieve account info
        account_info = mt5.account_info()
        self.balance = Decimal(account_info.balance)
        equity = Decimal(account_info.equity)
        open_trades = len(mt5.positions_get())

        # Historical market data for BTCUSD (last 100 candles on M1 timeframe)
        symbol = "BTCUSD"
        if not mt5.symbol_select(symbol, True):
            raise Exception(f"Failed to select symbol {symbol}")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        if rates is None or len(rates) == 0:
            raise Exception("Failed to retrieve market data")

        close_prices = [rate['close'] for rate in rates]
        close_prices_decimal = [Decimal(str(price)) for price in close_prices]
        close_prices_np = np.array(close_prices)

        # Calculate price returns
        returns = np.diff(close_prices_np) / close_prices_np[:-1]
        mean_return = np.mean(returns) if len(returns) > 0 else 0
        volatility = np.std(returns) if len(returns) > 0 else 0

        # Latest bid/ask prices for BTCUSD
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise Exception(f"Failed to get tick data for {symbol}")
        bid, ask = Decimal(str(tick.bid)), Decimal(str(tick.ask))

        # Calculate RSI
        rsi = calculate_rsi(close_prices_np)

        # Fetch live BTC price
        live_price = fetch_btc_price()

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

        # For simplicity, we'll include only basic thermal level information in the state
        # You can expand this based on your specific requirements
        thermal_balance = thermal_levels[0][1] if thermal_levels else self.balance

        # State: [balance, number of trades, equity, last close price, mean close price, bid, ask, RSI, mean return, volatility, rupture_price, price_drop_percentage, thermal_balance]
        state = [
            float(self.balance),
            open_trades,
            float(equity),
            float(close_prices[-1]),  # Most recent close price
            float(np.mean(close_prices_np)),  # Average of last 100 close prices
            float(bid),
            float(ask),
            float(rsi),
            float(mean_return),
            float(volatility),
            float(rupture_price),
            float(price_drop_percentage),
            float(thermal_balance)
        ]
        return torch.tensor(state, dtype=torch.float32)

    def step(self, action):
        """Execute a step in the environment based on action (buy/sell/no action)."""
        if action == 0:
            self.buy()
        elif action == 1:
            self.sell()
        else:
            pass  # Neutral (no action)
        
        # Update balance after the action
        self.balance = Decimal(mt5.account_info().balance)
        reward = self.get_reward()
        next_state = self.get_state()
        
        return next_state, reward

    def buy(self):
        """Execute a buy order."""
        symbol = "BTCUSD"
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(self.lot_size),
            "type": mt5.ORDER_TYPE_BUY,  # Corrected constant
            "price": float(mt5.symbol_info_tick(symbol).ask),
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
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(self.lot_size),
            "type": mt5.ORDER_TYPE_SELL,  # Corrected constant
            "price": float(mt5.symbol_info_tick(symbol).bid),
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
        reward = float(self.balance - self.initial_balance)
        
        # Additional punishment based on price drop percentage
        # The deeper the drop, the more negative the reward
        try:
            _, price_drop_percentage = calculate_rupture(
                balance=self.initial_balance,
                entry_price=fetch_btc_price(),
                spread_cost=self.spread_cost,
                lot_size=self.lot_size,
                maintenance_margin=self.maintenance_margin,
                leverage=self.leverage
            )
            # Penalize the agent proportionally to the price drop
            penalty = -float(price_drop_percentage)
            reward += penalty
        except Exception as e:
            print(f"Error calculating penalty: {e}")
        
        return reward

    def reset(self):
        """Reset the environment to its initial state."""        
        self.balance = self.initial_balance
        return self.get_state()

# Agent interacting with the environment
class TradeAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = TradeRLModel(state_size, 128, action_size)  # Increased hidden size for complexity
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

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
        for state, action, reward, next_state in minibatch:
            target = reward
            with torch.no_grad():
                target += self.gamma * torch.max(self.model(next_state)).item()
            
            # Ensure the Q-values are handled correctly for the chosen action
            current_q_values = self.model(state)  # Get Q-values for all actions
            current_q = current_q_values[0, action]  # Extract the Q-value for the specific action
            
            # Calculate the loss between the predicted Q-value and the target
            loss = self.criterion(current_q, torch.tensor(target))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent():
    env = MT5Environment()
    agent = TradeAgent(state_size=13, action_size=3)  # Updated state size to 13
    episodes = 1000
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        state = state.unsqueeze(0)  # Add batch dimension

        for t in range(200):  # Limit the steps per episode
            action = agent.act(state)
            next_state, reward = env.step(action)
            next_state = next_state.unsqueeze(0)
            agent.remember(state, action, reward, next_state)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        print(f"Episode {episode + 1}/{episodes} - Balance: {env.balance}, Epsilon: {agent.epsilon}")

if __name__ == "__main__":
    train_agent()
