import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # Input layer
        self.fc2 = nn.Linear(128, 128)         # Hidden layer
        self.fc3 = nn.Linear(128, action_size) # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size          # Number of features in the state
        self.action_size = action_size        # Number of possible actions
        self.memory = deque(maxlen=10000)     # Replay memory
        self.gamma = 0.99                     # Discount factor
        self.epsilon = 1.0                    # Exploration rate
        self.epsilon_min = 0.01               # Minimum exploration rate
        self.epsilon_decay = 0.995            # Exploration decay rate
        self.learning_rate = 1e-4             # Learning rate
        self.batch_size = 64                  # Batch size for training
        self.model = DQN(state_size, action_size).to(device)  # Neural network model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()         # Loss function
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action based on the current state."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()
    
    def replay(self):
        """Train the agent with a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).unsqueeze(1).to(device)
        
        # Compute target Q-values
        with torch.no_grad():
            target_q_values = rewards + self.gamma * torch.max(self.model(next_states), dim=1, keepdim=True)[0] * (1 - dones)
        
        # Compute current Q-values
        current_q_values = self.model(states).gather(1, actions)
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decrease exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def initialize_mt5(login, password, server):    
    """Initialize and connect to MT5."""
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
        return False

def get_state(symbol, timeframe, n):
    """Fetch the latest state (market data)."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        print("Failed to retrieve rates")
        return None
    close_prices = [rate[4] for rate in rates]  # Close prices
    state = np.array(close_prices)
    return state

def place_order(symbol, order_type):
    """Place a market order."""
    lot_size = 0.1
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Reinforcement Learning trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("Order placed successfully")
        return result
    else:
        print(f"Failed to place order: {result.retcode}")
        return None

def close_order(ticket, order_type):
    """Close an open position."""
    symbol = 'BTCUSD'
    lot_size = 0.1
    price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
    close_type = mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Reinforcement Learning trade close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("Position closed successfully")
        return True
    else:
        print(f"Failed to close position: {result.retcode}")
        return False

def get_closed_profit(ticket):
    """Retrieve the profit from a closed position."""
    deals = mt5.history_deals_get(ticket=ticket)
    if deals is None or len(deals) == 0:
        print("No deal history found")
        return 0
    profit = sum(deal.profit for deal in deals)
    return profit

def main():
    # Replace with your MT5 login credentials
    login = '312128713'
    password = 'Sexo247420@'
    server = 'XMGlobal-MT5 7'

    if not initialize_mt5(login, password, server):
        return

    symbol = 'BTCUSD'
    timeframe = mt5.TIMEFRAME_M1  # 1-minute timeframe

    state_size = 60  # Number of previous time steps to include in the state
    action_size = 3  # Possible actions: hold, buy, sell
    agent = Agent(state_size, action_size)

    position = None          # Current open position
    position_type = None     # Type of the current position (buy/sell)

    while True:
        # Get the current state
        state = get_state(symbol, timeframe, state_size)
        if state is None:
            time.sleep(60)  # Wait before retrying
            continue

        # Normalize the state
        state = (state - np.mean(state)) / np.std(state)

        # Agent chooses an action
        action = agent.act(state)

        reward = 0
        done = False

        if action == 1:  # Buy
            if position is None:
                result = place_order(symbol, mt5.ORDER_TYPE_BUY)
                if result:
                    position = result.order
                    position_type = mt5.ORDER_TYPE_BUY
            elif position_type == mt5.ORDER_TYPE_SELL:
                # Close sell position
                success = close_order(position, position_type)
                if success:
                    # Get profit from the closed position
                    reward = get_closed_profit(position)
                    position = None
                    position_type = None
                    done = True
        elif action == 2:  # Sell
            if position is None:
                result = place_order(symbol, mt5.ORDER_TYPE_SELL)
                if result:
                    position = result.order
                    position_type = mt5.ORDER_TYPE_SELL
            elif position_type == mt5.ORDER_TYPE_BUY:
                # Close buy position
                success = close_order(position, position_type)
                if success:
                    # Get profit from the closed position
                    reward = get_closed_profit(position)
                    position = None
                    position_type = None
                    done = True
        else:  # Hold
            reward = 0
            done = False

        # Wait for the next time step
        time.sleep(60)

        # Get the next state
        next_state = get_state(symbol, timeframe, state_size)
        if next_state is None:
            continue

        # Normalize the next state
        next_state = (next_state - np.mean(next_state)) / np.std(next_state)

        # Update reward if position is still open
        if position is not None:
            positions = mt5.positions_get(ticket=position)
            if positions:
                profit = positions[0].profit
                reward = profit
            else:
                position = None
                position_type = None

        # Store the experience
        agent.remember(state, action, reward, next_state, done)

        # Train the agent
        agent.replay()

        # Save the model periodically (optional)
        # torch.save(agent.model.state_dict(), 'dqn_model.pth')

        # Update the state
        state = next_state

if __name__ == "__main__":
    main()
