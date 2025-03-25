import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class TradingPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(TradingPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads for different trading parameters
        self.lot_size_head = nn.Linear(hidden_dim, 1)
        self.direction_head = nn.Linear(hidden_dim, 1)
        self.stop_loss_head = nn.Linear(hidden_dim, 1)
        self.take_profit_head = nn.Linear(hidden_dim, 1)
        self.hold_time_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Get trading parameters with appropriate activations
        lot_size = torch.sigmoid(self.lot_size_head(x))  # 0-1 range
        direction = torch.sigmoid(self.direction_head(x))  # 0-1 (sell-buy)
        stop_loss = F.softplus(self.stop_loss_head(x))  # >0
        take_profit = F.softplus(self.take_profit_head(x))  # >0
        hold_time = F.softplus(self.hold_time_head(x))  # >0
        
        return {
            'lot_size': lot_size,
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'hold_time': hold_time
        }

class TradingValueNetwork(nn.Module):
    """Critic network to estimate state value"""
    def __init__(self, input_dim, hidden_dim=128):
        super(TradingValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPOAgent:
    def __init__(self, state_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.policy = TradingPolicyNetwork(state_dim).to(device)
        self.value_net = TradingValueNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value_net.parameters()), lr=3e-4)
        
        # RL parameters
        self.gamma = 0.99
        self.clip_param = 0.2
        self.ppo_epochs = 4
        self.batch_size = 64
        
        # Memory buffers
        self.memory = []
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_params = self.policy(state)
            value = self.value_net(state)
            
        # Sample actions from distributions
        lot_size = action_params['lot_size']
        direction = torch.distributions.Bernoulli(action_params['direction'])
        stop_loss = Normal(action_params['stop_loss'], action_params['stop_loss']*0.1)
        take_profit = Normal(action_params['take_profit'], action_params['take_profit']*0.1)
        
        # Sample actions
        dir_action = direction.sample()
        sl_action = stop_loss.sample()
        tp_action = take_profit.sample()
        
        # Store for training
        self.states.append(state)
        self.actions.append(torch.cat([lot_size, dir_action, sl_action, tp_action], dim=-1))
        self.values.append(value)
        
        return {
            'lot_size': lot_size.item(),
            'direction': 'buy' if dir_action.item() > 0.5 else 'sell',
            'stop_loss': sl_action.item(),
            'take_profit': tp_action.item()
        }
        
    def update(self):
        # Convert to tensors
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        returns = torch.zeros_like(torch.cat(self.values))
        
        # Calculate discounted returns
        R = 0
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + self.gamma * R * self.masks[i]
            returns[i] = R
            
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            for state, action, old_value, return_, old_log_prob in self._get_batches(states, actions, returns):
                # Calculate advantages
                advantage = return_ - old_value
                
                # Get new action probs and value
                action_params = self.policy(state)
                value = self.value_net(state)
                
                # Calculate new log probs
                lot_size = action_params['lot_size']
                direction = torch.distributions.Bernoulli(action_params['direction'])
                stop_loss = Normal(action_params['stop_loss'], action_params['stop_loss']*0.1)
                take_profit = Normal(action_params['take_profit'], action_params['take_profit']*0.1)
                
                new_log_prob = direction.log_prob(action[:,1]) + \
                              stop_loss.log_prob(action[:,2]) + \
                              take_profit.log_prob(action[:,3])
                
                # Policy loss
                ratio = (new_log_prob - old_log_prob).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(value, return_)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.optimizer.step()
                
        # Clear memory
        self._clear_memory()
        
    def _get_batches(self, states, actions, returns):
        batch_size = min(self.batch_size, len(states))
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), batch_size):
            idx = indices[start:start+batch_size]
            yield states[idx], actions[idx], self.values[idx], returns[idx], self.log_probs[idx]
            
    def _clear_memory(self):
        self.memory = []
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

# Integration with MT5 system
class MT5RLTradingSystem:
    def __init__(self):
        # State dimensions: [balance, equity, margin, symbols_data...]
        self.agent = PPOAgent(state_dim=10 + len(SYMBOLS)*5) 
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "XAUUSD"]
        
    def get_state(self):
        """Create state vector from MT5 data"""
        account = mt5.account_info()
        state = [
            account.balance,
            account.equity,
            account.margin,
            account.margin_free,
            account.margin_level
        ]
        
        # Add market data for each symbol
        for symbol in self.symbols:
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            state += [
                tick.bid,
                tick.ask,
                (tick.ask - tick.bid),  # spread
                info.volume,
                info.volume_real
            ]
            
        return np.array(state, dtype=np.float32)
    
    def execute_trade(self, action):
        """Execute trade based on PyTorch agent's action"""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": action['symbol'],
            "volume": action['lot_size'],
            "type": mt5.ORDER_TYPE_BUY if action['direction'] == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(action['symbol']).ask if action['direction'] == 'buy' else mt5.symbol_info_tick(action['symbol']).bid,
            "sl": action['stop_loss'],
            "tp": action['take_profit'],
            "deviation": 10,
            "magic": 234000,
            "comment": "PyTorch RL Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result
    
    def calculate_reward(self, trade_result):
        """Calculate reward based on punish/reward hierarchy"""
        reward = 0
        
        if trade_result['hit_stop_loss']:
            reward -= 0.20 * trade_result['loss_amount']
            
        if trade_result['balance_reduction']:
            reward -= 0.35 * abs(trade_result['balance_change'])
            
        if trade_result['hit_take_profit']:
            reward += 0.21 * trade_result['profit_amount']
            
        if trade_result['balance_growth']:
            reward += 0.36 * trade_result['balance_change']
            
        if trade_result['profit_magnitude'] > 0:
            reward += 0.37 * trade_result['profit_magnitude']
            
        return reward
    
    def train_loop(self):
        while True:
            state = self.get_state()
            action = self.agent.get_action(state)
            action['symbol'] = self.select_symbol()  # Implement symbol selection logic
            
            trade_result = self.execute_trade(action)
            
            # Monitor trade and get outcome
            trade_outcome = self.monitor_trade(trade_result.order)
            reward = self.calculate_reward(trade_outcome)
            
            # Store experience
            self.agent.rewards.append(reward)
            self.agent.masks.append(1 - trade_outcome['done'])
            
            # Update agent
            if len(self.agent.memory) > self.agent.batch_size:
                self.agent.update()
                
            time.sleep(1)  # Prevent excessive trading
