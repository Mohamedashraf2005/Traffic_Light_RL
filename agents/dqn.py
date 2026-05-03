import random
import numpy as np
from collections import deque
import sys
import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure paths are correct for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.traffic_light_env import TrafficLightEnv
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
            torch.FloatTensor(np.stack(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# =========================
# 2. DQN Network (Upgraded to 128 Neurons)
# =========================
class DQNNetwork(nn.Module):
    def __init__(self, state_size=17, action_size=4):
        super().__init__()
        # Increased capacity to 128 neurons for complex state learning[cite: 1]
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# 3. DQN Agent
# =========================
class DQNAgent:
    def __init__(self):
        self.state_size = 17 
        self.action_size = 4
        self.gamma = 0.99
        self.lr = 0.001
        self.batch_size = 128  # Optimized based on previous run[cite: 1]
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
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states, actions = states.to(self.device), actions.to(self.device)
        rewards, next_states, dones = rewards.to(self.device), next_states.to(self.device), dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        # Stabilize targets every 200 steps[cite: 1, 2]
        if self.step_count % 200 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# =========================
# 4. Training Loop (Fixed at 2000 Episodes)
# =========================
def train_dqn(episodes=2000): # Stopped at 2000 for comparison[cite: 1]
    env = TrafficLightEnv(max_steps=100)
    agent = DQNAgent()
    
    # New run name for the updated architecture
    wandb.init(
        project="traffic-light-rl", 
        name="dqn_batch128_layers128 ", 
        config={
            "batch_size": agent.batch_size,
            "hidden_layers": 128,
            "episodes": episodes,
            "lr": agent.lr
        }
    )

    rewards_log = []
    for ep in range(episodes):
        state = env.reset()
        total_reward, episode_loss = 0, []
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.update()
            if loss: episode_loss.append(loss)
            state, total_reward = next_state, total_reward + reward
            if done: break
            
        agent.decay_epsilon()
        rewards_log.append(total_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        
        wandb.log({
            "episode": ep + 1,
            "total_reward": total_reward,
            "avg_loss": avg_loss,
            "epsilon": agent.epsilon
        })

        if (ep + 1) % 100 == 0:
            avg_rew = np.mean(rewards_log[-100:])
            print(f"Ep {ep+1} | Avg Reward: {avg_rew:.2f} | Loss: {avg_loss:.4f}")

    wandb.finish()
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
            return torch.argmax(self.agent.policy_net(state)).item()

if __name__ == "__main__":
    trained_agent = train_dqn(episodes=2000)
    results = evaluate(DQNWrapperAgent(trained_agent), env_kwargs={"max_steps": 100}, n_episodes=100)
    print(f"\nFinal Evaluation Avg Reward: {results['avg_reward']:.2f}")