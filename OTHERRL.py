import tkinter as tk
from tkinter import messagebox
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import threading
import time
import gym
from gym import spaces
import os

# ------------------------------ Custom Gym Environment ------------------------------ #

class MT5TradingEnv(gym.Env):
    """A custom trading environment for MetaTrader 5."""
    metadata = {'render.modes': ['human']}

    def __init__(self, symbol="BTCUSD", timeframe=mt5.TIMEFRAME_H1, window_size=60):
        super(MT5TradingEnv, self).__init__()

        self.symbol = symbol
        self.timeframe = timeframe
        self.window_size = window_size

        # Define action and observation space
        # Actions: 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)

        # Observations: window_size number of past prices and indicators
        # Features: Close, MA, RSI, MACD, Bollinger Bands, Volume
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, 7), dtype=np.float32
        )

        # Initialize state
        self.state = None
        self.current_step = 0
        self.position = 0  # -1 for short, 0 for flat, 1 for long
        self.entry_price = 0
        self.balance = 10000  # Starting balance
        self.equity = self.balance
        self.max_steps = 1000  # Max steps per episode

        # Load historical data
        self.data = self.fetch_data()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.processed_data = self.preprocess_data()

    def fetch_data(self):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 2000)
        if rates is None or len(rates) == 0:
            raise ValueError("No data fetched from MT5")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        return df

    def preprocess_data(self):
        df = self.data.copy()
        # Feature Engineering: Calculate technical indicators
        df['ma'] = df['close'].rolling(window=15).mean()
        df['rsi'] = self.compute_rsi(df['close'], window=14)
        df['macd'] = self.compute_macd(df['close'])
        df['bollinger_upper'], df['bollinger_lower'] = self.compute_bollinger_bands(df['close'])
        df.fillna(method='backfill', inplace=True)

        # Scaling
        scaled_features = self.scaler.fit_transform(df[['close', 'ma', 'rsi', 'macd',
                                                        'bollinger_upper', 'bollinger_lower', 'tick_volume']])
        return scaled_features

    def compute_rsi(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        gain_avg = gain.rolling(window=window).mean()
        loss_avg = loss.rolling(window=window).mean()
        rs = gain_avg / loss_avg
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, series, slow=26, fast=12, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - signal_line
        return macd_histogram

    def compute_bollinger_bands(self, series, window=20, num_std_dev=2):
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        return upper_band, lower_band

    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0
        self.equity = self.balance
        self.done = False
        return self.processed_data[self.current_step - self.window_size:self.current_step]

    def step(self, action):
        done = False

        current_price = self.processed_data[self.current_step, 0]  # Scaled close price
        unscaled_price = self.data['close'].iloc[self.current_step]  # Actual close price

        # Reward calculation
        reward = 0

        # Update position and calculate reward
        if action == 2:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = unscaled_price
            elif self.position == -1:
                profit = self.entry_price - unscaled_price
                self.equity += profit
                reward = profit
                self.position = 1
                self.entry_price = unscaled_price
        elif action == 0:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = unscaled_price
            elif self.position == 1:
                profit = unscaled_price - self.entry_price
                self.equity += profit
                reward = profit
                self.position = -1
                self.entry_price = unscaled_price
        else:  # Hold
            if self.position == 1:
                profit = unscaled_price - self.entry_price
                reward = profit
            elif self.position == -1:
                profit = self.entry_price - unscaled_price
                reward = profit

        # Transaction cost
        transaction_cost = 0.0001 * unscaled_price  # Example: 0.01% of price per trade
        if action != 1:
            reward -= transaction_cost

        # Risk management: Apply penalties for large drawdowns
        if self.equity < self.balance * 0.9:
            reward -= 10  # Penalty for drawdown exceeding 10%

        self.current_step += 1

        if self.current_step >= len(self.processed_data) - 1 or self.current_step >= self.max_steps:
            done = True

        next_state = self.processed_data[self.current_step - self.window_size:self.current_step]

        return next_state, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Equity: {self.equity}")

# ------------------------------ RL Trading Bot Integration ------------------------------ #

def main():
    global root, connect_button, message_label, status_label

    root = tk.Tk()
    root.title("MYM-A RL Trading Bot")
    root.geometry("600x400")

    # Create frames
    login_frame = tk.Frame(root, width=300, height=400)
    login_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    status_frame = tk.Frame(root, width=300, height=400)
    status_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Login UI
    login_title = tk.Label(login_frame, text="🏧MYM-A", font=("Helvetica", 20))
    login_title.pack(pady=20)

    login_label = tk.Label(login_frame, text="Login:", font=("Helvetica", 14))
    login_label.pack(pady=5)
    login_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    login_entry.pack(pady=5)

    password_label = tk.Label(login_frame, text="Password:", font=("Helvetica", 14))
    password_label.pack(pady=5)
    password_entry = tk.Entry(login_frame, show="*", font=("Helvetica", 14))
    password_entry.pack(pady=5)

    server_label = tk.Label(login_frame, text="Server:", font=("Helvetica", 14))
    server_label.pack(pady=5)
    server_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    server_entry.pack(pady=5)

    connect_button = tk.Button(
        login_frame,
        text="Connect",
        font=("Helvetica", 14),
        command=lambda: connect_to_mt5(
            login_entry.get(), password_entry.get(), server_entry.get()
        )
    )
    connect_button.pack(pady=20)

    # Status Display
    status_label = tk.Label(status_frame, text="Not Connected", font=("Helvetica", 14), fg="red")
    status_label.pack(pady=20)

    message_label = tk.Label(status_frame, text="", font=("Helvetica", 14), fg="green")
    message_label.pack(pady=10)

    root.mainloop()

def connect_to_mt5(login, password, server):
    if not mt5.initialize():
        messagebox.showerror("Error", "initialize() failed")
        mt5.shutdown()
        update_status("Initialization Failed", "red")
        return

    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        print("Connected to MetaTrader 5")
        update_status("Connected to MT5", "green")
        message_label.config(text="Starting RL Automation...", fg="blue")
        start_automation()
    else:
        messagebox.showerror("Error", "Failed to connect to MetaTrader 5")
        update_status("Connection Failed", "red")

def update_status(text, color):
    status_label.config(text=text, fg=color)

def update_message(text, color):
    message_label.config(text=text, fg=color)

def start_automation():
    def automation_loop():
        try:
            env = MT5TradingEnv(symbol="BTCUSD", timeframe=mt5.TIMEFRAME_H1, window_size=60)
            env = DummyVecEnv([lambda: env])  # Wrap the environment

            model = PPO("MlpPolicy", env, verbose=1)
            print("Starting training...")
            model.learn(total_timesteps=20000)  # Increased training time
            model.save("ppo_mt5_trading_model")
            print("Initial training completed.")

            while True:
                obs = env.reset()
                done = False
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)

                    # Execute the action using MT5 API
                    if action == 2:
                        place_trade(mt5.ORDER_TYPE_BUY)
                        root.after(0, update_message, "Action: Buy", "green")
                    elif action == 0:
                        place_trade(mt5.ORDER_TYPE_SELL)
                        root.after(0, update_message, "Action: Sell", "red")
                    else:
                        root.after(0, update_message, "Action: Hold", "blue")

                    # Sleep to simulate real-time trading (adjust as needed)
                    time.sleep(1)

                # Periodically retrain the model with new data
                print("Retraining the model with new data...")
                model.learn(total_timesteps=5000)
                model.save("ppo_mt5_trading_model")
                print("Retraining completed.")
        except Exception as e:
            print(f"Automation error: {e}")
            message_label.config(text=f"Automation Error: {e}", fg="red")

    threading.Thread(target=automation_loop, daemon=True).start()

def place_trade(order_type):
    symbol = "BTCUSD"
    lot_size = calculate_lot_size(symbol)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        messagebox.showerror("Error", f"Failed to get tick data for {symbol}")
        return

    if order_type == mt5.ORDER_TYPE_BUY:
        price = tick.ask
        sl = price - dynamic_stop_loss(symbol)
        tp = price + dynamic_take_profit(symbol)
    elif order_type == mt5.ORDER_TYPE_SELL:
        price = tick.bid
        sl = price + dynamic_stop_loss(symbol)
        tp = price - dynamic_take_profit(symbol)
    else:
        return  # No action for Hold

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "RL Automated Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Trade successfully placed: {result.order}")
        message_label.config(text=f"Trade placed: {result.order}", fg="green")
    else:
        messagebox.showerror("Trade Error", f"Failed to place trade: {result.retcode}")
        message_label.config(text=f"Trade Error: {result.retcode}", fg="red")

def calculate_lot_size(symbol):
    account_info = mt5.account_info()
    if account_info is None:
        messagebox.showerror("Error", "Failed to retrieve account info")
        return 0.01  # Default small lot size

    balance = account_info.balance
    risk_percentage = 0.01  # Risk 1% of balance per trade
    risk_amount = balance * risk_percentage

    # Get symbol properties
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0.01  # Default small lot size

    # Calculate stop loss in points
    stop_loss = dynamic_stop_loss(symbol)
    tick_size = symbol_info.point

    if tick_size == 0:
        return 0.01  # Default small lot size

    stop_loss_points = stop_loss / tick_size
    lot_size = risk_amount / (stop_loss_points * symbol_info.trade_contract_size * tick_size)

    # Ensure lot_size meets broker's minimum requirements
    min_lot = symbol_info.volume_min
    lot_size = max(lot_size, min_lot)
    lot_size = round(lot_size, 2)  # Round to 2 decimal places

    return lot_size

def dynamic_stop_loss(symbol):
    # Calculate dynamic stop loss based on ATR
    atr = calculate_atr(symbol)
    return atr * 2  # Example: Stop loss at 2 times ATR

def dynamic_take_profit(symbol):
    # Calculate dynamic take profit based on ATR
    atr = calculate_atr(symbol)
    return atr * 4  # Example: Take profit at 4 times ATR

def calculate_atr(symbol, period=14):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period+1)
    df = pd.DataFrame(rates)
    df['tr'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
    atr = df['tr'].rolling(window=period).mean().iloc[-1]
    return atr

# ------------------------------ Run the Application ------------------------------ #

if __name__ == "__main__":
    main()
