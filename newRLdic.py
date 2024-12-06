import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class TradingEnvironment:
    def __init__(self, price_data, initial_balance=10000, lookback=60):
        self.price_data = price_data
        self.initial_balance = initial_balance
        self.lookback = lookback
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0   # +1 for long, -1 for short, 0 for flat
        self.current_step = self.lookback
        self.done = False
        self.equity = self.balance
        self.last_equity = self.balance
        self.entry_price = 0.0
        return self._get_state()

    def _get_state(self):
        # State: last 'lookback' closing prices + current position
        state_prices = self.price_data[self.current_step - self.lookback:self.current_step]
        # Append current position to state
        state = np.append(state_prices, [self.position])
        return state

    def step(self, action):
        # Actions: 0 = hold, 1 = buy, 2 = sell
        current_price = self.price_data[self.current_step]

        # Execute action
        if action == 1:  # Buy
            if self.position <= 0:
                # Realize P/L if short
                if self.position < 0:
                    self.balance += (self.entry_price - current_price) * abs(self.position)
                # Go long
                self.position = 1
                self.entry_price = current_price

        elif action == 2:  # Sell
            if self.position >= 0:
                # Realize P/L if long
                if self.position > 0:
                    self.balance += (current_price - self.entry_price) * abs(self.position)
                # Go short
                self.position = -1
                self.entry_price = current_price

        # Compute unrealized P/L
        unrealized_pnl = 0
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) * abs(self.position)
        elif self.position < 0:
            unrealized_pnl = (self.entry_price - current_price) * abs(self.position)

        self.equity = self.balance + unrealized_pnl

        # Advance to next time step
        self.current_step += 1
        if self.current_step >= len(self.price_data):
            self.done = True

        next_state = self._get_state()
        reward = self.equity - self.last_equity
        self.last_equity = self.equity

        return next_state, reward, self.done, {}

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

def build_q_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model

if __name__ == "__main__":
    # Example price data: replace with actual historical data
    price_data = np.random.rand(10000)

    env = TradingEnvironment(price_data=price_data, initial_balance=10000, lookback=60)
    state_dim = env.reset().shape[0]
    action_dim = 3  # hold, buy, sell

    main_model = build_q_model(state_dim, action_dim)
    target_model = build_q_model(state_dim, action_dim)
    target_model.set_weights(main_model.get_weights())

    buffer = ReplayBuffer(10000)

    episodes = 10
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    update_target_freq = 1000
    step_count = 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            step_count += 1
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(action_dim)
            else:
                q_values = main_model.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(q_values[0])

            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # Training step
            if buffer.size() > batch_size:
                states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = buffer.sample(batch_size)

                next_q = target_model.predict(next_states_batch, verbose=0)
                max_next_q = np.max(next_q, axis=1)
                target = main_model.predict(states_batch, verbose=0)

                for i in range(batch_size):
                    target[i, actions_batch[i]] = rewards_batch[i] + (0 if done_batch[i] else gamma * max_next_q[i])

                main_model.fit(states_batch, target, epochs=1, verbose=0)

            # Update target model periodically
            if step_count % update_target_freq == 0:
                target_model.set_weights(main_model.get_weights())

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode: {ep}, Reward: {episode_reward}, Epsilon: {epsilon:.4f}")
