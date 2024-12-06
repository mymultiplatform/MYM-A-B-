class Agent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.99, 
                 temperature=1.0, temperature_min=0.01, temperature_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        # Remove epsilon-related parameters if not using epsilon-greedy
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay

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
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(s)[0].cpu().numpy()

        # Compute softmax probabilities
        # Avoid numerical instability by subtracting max Q-value before exp
        max_q = np.max(q_values)
        exp_q = np.exp((q_values - max_q) / self.temperature)
        probs = exp_q / np.sum(exp_q)

        # Sample an action from the probability distribution
        action = np.random.choice(self.action_dim, p=probs)
        return action

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

        # Decay the temperature for less exploration over time
        if self.temperature > self.temperature_min:
            self.temperature *= self.temperature_decay

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.update_target_model()
