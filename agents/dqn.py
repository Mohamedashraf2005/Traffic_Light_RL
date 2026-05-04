import random
import numpy as np
from collections import deque
import sys
import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

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
# 2. DQN Network
# =========================
class DQNNetwork(nn.Module):
    def __init__(self, state_size=17, action_size=4):
        super().__init__()

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

        # Hyperparameters
        self.gamma = 0.99
        self.lr = 0.001
        self.batch_size = 128

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net = DQNNetwork(self.state_size, self.action_size).to(self.device)

        # Copy policy weights to target
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer & Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Replay Buffer
        self.memory = ReplayBuffer()

        # Step counter for target updates
        self.step_count = 0

    # -------------------------
    # Action Selection
    # -------------------------
    def select_action(self, state):
        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state)

        return torch.argmax(q_values).item()

    # -------------------------
    # Training Update
    # -------------------------
    def update(self):
        # Wait until enough samples
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss
        loss = self.criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network every 200 steps
        self.step_count += 1
        if self.step_count % 200 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # -------------------------
    # Epsilon Decay
    # -------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # =========================
    # 4. Save Model
    # =========================
    def save_model(self, path="checkpoints/trained_dqn_model.pth"):
        # Create models folder if not exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

        print(f"Model saved successfully at: {path}")

    # =========================
    # 5. Load Model
    # =========================
    def load_model(self, path="checkpoints/trained_dqn_model.pth"):
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epsilon = checkpoint['epsilon']

        print(f"Model loaded successfully from: {path}")


# =========================
# 6. Training Function
# =========================
def train_dqn(episodes=2000):
    env = TrafficLightEnv(max_steps=100)
    agent = DQNAgent()

    # Initialize Weights & Biases
    wandb.init(
        project="traffic-light-rl",
        name="dqn_batch128_layers128",
        config={
            "batch_size": agent.batch_size,
            "hidden_layers": 128,
            "episodes": episodes,
            "learning_rate": agent.lr,
            "gamma": agent.gamma,
            "epsilon_decay": agent.epsilon_decay
        }
    )

    rewards_log = []

    for ep in range(episodes):
        state = env.reset()

        total_reward = 0
        episode_losses = []

        while True:
            # Select action
            action = agent.select_action(state)

            # Environment step
            next_state, reward, done, _ = env.step(action)

            # Store transition
            agent.memory.push(state, action, reward, next_state, done)

            # Train
            loss = agent.update()

            if loss is not None:
                episode_losses.append(loss)

            # Move state
            state = next_state
            total_reward += reward

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Logging
        rewards_log.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0

        wandb.log({
            "episode": ep + 1,
            "total_reward": total_reward,
            "avg_loss": avg_loss,
            "epsilon": agent.epsilon
        })

        # Console progress
        if (ep + 1) % 100 == 0:
            avg_reward_last_100 = np.mean(rewards_log[-100:])
            print(
                f"Episode {ep + 1}/{episodes} | "
                f"Avg Reward (Last 100): {avg_reward_last_100:.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    wandb.finish()

    return agent


# =========================
# 7. Evaluation Wrapper
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
# 8. Main Execution
# =========================
if __name__ == "__main__":

    # -------------------------
    # Train Agent
    # -------------------------
    trained_agent = train_dqn(episodes=2000)

    # -------------------------
    # Save Trained Model
    # -------------------------
    trained_agent.save_model("checkpoints/trained_dqn_model.pth")

    # -------------------------
    # Evaluate Agent
    # -------------------------
    results = evaluate(
        DQNWrapperAgent(trained_agent),
        env_kwargs={"max_steps": 100},
        n_episodes=100
    )

    print(f"\nFinal Evaluation Avg Reward: {results['avg_reward']:.2f}")

    # =========================
    # Example: Reload Model Later
    # =========================
    # loaded_agent = DQNAgent()
    # loaded_agent.load_model("models/trained_dqn_model.pth")