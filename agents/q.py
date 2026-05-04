import numpy as np
import wandb
import random
import pickle
import os

from baselines import DiscretizedWrapper, evaluate
from env.traffic_light_env import TrafficLightEnv


class QLearningAgent:
    def __init__(self, action_size, alpha=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.q_table = {}
        self.best_reward = -float("inf")

    def get_qs(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key]

    def select_action(self, state):
        state_key = tuple(state)
        q_values = self.get_qs(state_key)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state):
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        q_current = self.get_qs(state_key)
        q_next = self.get_qs(next_state_key)

        next_max = np.max(q_next)

        q_current[action] = q_current[action] + self.alpha * (
            reward + self.gamma * next_max - q_current[action]
        )

    def save(self, filepath="q_learning_model.pkl"):
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath="q_learning_model.pkl"):
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.q_table = data["q_table"]
                self.epsilon = data["epsilon"]
                self.alpha = data["alpha"]
                self.gamma = data["gamma"]
            print(f"Model loaded from {filepath}")
        else:
            print("No saved model found.")


    def train_agent(self, env, episodes=3000, save_path="best_model.pkl"):
        
        wandb.init(project="traffic-light-qlearning", name="q-learning-run")

        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)

                self.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.save(save_path)

            wandb.log({
                "episode": ep,
                "total_reward": total_reward,
                "epsilon": self.epsilon,
                "q_table_size": len(self.q_table)
            })

            if ep % 100 == 0:
                print(f"Episode {ep}: Reward = {total_reward}, Epsilon = {self.epsilon:.3f}")

        wandb.finish()


if __name__ == "__main__":
    
    env = DiscretizedWrapper(TrafficLightEnv())

    agent = QLearningAgent(
        action_size=4,
        alpha=0.01,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.999
    )

    agent.train_agent(env, episodes=5000)

    agent.save("final_model.pkl")

    print("\n--- Training Finished ---")

