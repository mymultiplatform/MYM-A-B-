import tkinter as tk
import MetaTrader5 as mt5
import threading
import time
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx  # For neural network visualization

# ✅ Custom Trading Environment with Multiple Assets
class TradingEnv(gym.Env):
    """Custom Environment for trading multiple assets in MetaTrader 5."""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(TradingEnv, self).__init__()
        
        self.symbols = ["ETHUSD", "BTCUSD", "BTCJPY", "DOGEUSD"]  # Updated symbols
        self.num_assets = len(self.symbols)
        
        self.balance = 10000
        self.prices = np.ones(self.num_assets) * 50000  # Initialize prices
        self.positions = np.zeros(self.num_assets)  # Position holdings
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=((self.num_assets * 2) + 1,), dtype=np.float32
        )

        # Connect to MetaTrader 5
        if not mt5.initialize():
            raise RuntimeError("Failed to connect to MetaTrader 5")
        print("Connected to MetaTrader 5")

    def step(self, action):
        """Perform trade action: Buy, Sell, or Hold"""
        reward = 0
        terminated = False  # Episode termination (e.g., balance <= 0)
        truncated = False  # Episode truncation (not used here)

        for i, symbol in enumerate(self.symbols):
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print(f"Error: No tick data for {symbol}")
                continue  # Skip this symbol if no tick data

            price = tick.ask if action == 0 else tick.bid

            if action == 0:  # Buy
                self.positions[i] += 1
                self.balance -= price
            elif action == 1:  # Sell
                if self.positions[i] > 0:
                    self.positions[i] -= 1
                    self.balance += price

        obs = np.array(np.concatenate((self.prices, self.positions, [self.balance])), dtype=np.float32)
        return obs, reward, terminated, truncated, {}  # Return 5 values

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        self.balance = 10000
        self.prices = np.ones(self.num_assets) * 50000
        self.positions = np.zeros(self.num_assets)

        obs = np.array(np.concatenate((self.prices, self.positions, [self.balance])), dtype=np.float32)
        info = {}  # Add an empty info dictionary
        return obs, info  # Return both observation and info

    def render(self, mode="human"):
        """Print current trading status"""
        print(f"Balance: {self.balance}, Positions: {self.positions}")

# ✅ Train the Model
def train_model():
    """Train the RL model"""
    env = DummyVecEnv([lambda: Monitor(TradingEnv())])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model, env

# ✅ Trade Execution Loop
def trade_loop(model, env):
    """Live trading loop"""
    while True:
        obs = env.reset()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action[0])
        print(f"Step result: {obs}, {reward}")
        time.sleep(1)

# ✅ Tkinter UI with Neural Network Visualization
class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MT5 PPO Trading Bot")
        self.root.geometry("600x400")

        tk.Label(root, text="Login:").pack()
        self.login_entry = tk.Entry(root)
        self.login_entry.pack()
        self.login_entry.insert(0, "312128713")

        tk.Label(root, text="Password:").pack()
        self.password_entry = tk.Entry(root, show="*")
        self.password_entry.pack()
        self.password_entry.insert(0, "Sexo247420@")

        tk.Label(root, text="Server:").pack()
        self.server_entry = tk.Entry(root)
        self.server_entry.pack()
        self.server_entry.insert(0, "XMGlobal-MT5 7")

        self.connect_button = tk.Button(root, text="Connect", command=self.connect)
        self.connect_button.pack()

        self.trade_button = tk.Button(root, text="Run Trading Bot", command=self.run_trading)
        self.trade_button.pack()

        self.plot_area = tk.Frame(root)
        self.plot_area.pack()

        # ✅ Initialize the environment and model
        self.env = DummyVecEnv([lambda: TradingEnv()])
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def connect(self):
        """Connect to MetaTrader 5"""
        login = self.login_entry.get()
        password = self.password_entry.get()
        server = self.server_entry.get()

        if mt5.initialize():
            print("Connected to MetaTrader 5")
            tk.Label(self.root, text="Connected", fg="green").pack()
        else:
            print("Failed to connect to MetaTrader 5")
            tk.Label(self.root, text="Failed to Connect", fg="red").pack()

    def run_trading(self):
        """Start the trading bot"""
        self.trade_thread = threading.Thread(target=self.trade_loop, daemon=True)
        self.trade_thread.start()

    def trade_loop(self):
        """Continuous trading loop with model learning"""
        while True:
            action, _ = self.model.predict(np.zeros((1, (self.env.envs[0].num_assets * 2) + 1)))
            obs, reward, terminated, truncated, _ = self.env.envs[0].step(action[0])
            print(f"Step result: {obs}, {reward}")
            self.model.learn(total_timesteps=1)  # Online learning
            self.visualize_nn()
            time.sleep(5)

    def visualize_nn(self):
        """Visualize the Neural Network structure"""
        fig, ax = plt.subplots(figsize=(5, 3))
        G = nx.DiGraph()

        layers = [(self.env.envs[0].num_assets * 2) + 1, 10, 3]  # Example: Input, Hidden, Output
        positions = {}
        node_id = 0
        y_offset = 0.5

        for layer_idx, num_nodes in enumerate(layers):
            x_positions = np.linspace(-1, 1, num_nodes)
            for x in x_positions:
                G.add_node(node_id, layer=layer_idx)
                positions[node_id] = (x, -layer_idx * y_offset)
                node_id += 1

        for src in range(layers[0]):  
            for dst in range(layers[0], layers[0] + layers[1]):  
                G.add_edge(src, dst)
        for src in range(layers[0], layers[0] + layers[1]):  
            for dst in range(layers[0] + layers[1], node_id):  
                G.add_edge(src, dst)

        nx.draw(G, positions, with_labels=False, node_size=300, node_color="blue", edge_color="gray", ax=ax)
        ax.set_title("Neural Network Structure")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()
