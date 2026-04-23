import random
import numpy as np
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_light_env import TrafficLightEnv
import torch
import torch.nn as nn
import torch.optim as optim
from agents.baselines import evaluate


# =========================
# 1. Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.stack(states)),       # handle numpy arrays safely
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.stack(next_states)),  # handle numpy arrays safely
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# =========================
# 2. DQN Network
# =========================
class DQNNetwork(nn.Module):
    def __init__(self, state_size=12, action_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 3. DQN Agent
# =========================
class DQNAgent:
    def __init__(self):
        self.state_size = 12
        self.action_size = 4

        self.gamma = 0.99
        self.lr = 0.001
        self.batch_size = 32

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNNetwork().to(self.device)
        self.target_net = DQNNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer()

        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q(s,a)
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # target Q
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # تحديث target network
        self.step_count += 1
        if self.step_count % 200 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =========================
# 4. Training Loop
# =========================
def train_dqn(episodes=3000):
    env = TrafficLightEnv(max_steps=100)
    agent = DQNAgent()

    rewards_log = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards_log.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_log[-100:])
            print(f"Episode {ep+1} | Avg Reward: {avg:.2f} | Epsilon: {agent.epsilon:.3f}")

    return agent


# =========================
# 5. Evaluate Trained Model
# =========================
class DQNWrapperAgent:
    def __init__(self, trained_agent):
        self.agent = trained_agent

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            q_values = self.agent.policy_net(state)
        return torch.argmax(q_values).item()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    trained_agent = train_dqn()

    print("\nEvaluating trained DQN...\n")
    eval_agent = DQNWrapperAgent(trained_agent)

    results = evaluate(eval_agent, env_kwargs={"max_steps": 100}, n_episodes=100)

    print("\nDQN Results:")
    print(f"Avg Reward: {results['avg_reward']:.2f}")