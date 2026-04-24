import numpy as np
import wandb
import random
from baselines import DiscretizedWrapper, evaluate
def update(self, state, action, reward, next_state):
    state_key = tuple(state)
    next_state_key = tuple(next_state)

    if next_state_key not in self.q_table:
        self.q_table[next_state_key] = np.zeros(self.action_size)

    old_value = self.q_table[state_key][action]
    next_max = np.max(self.q_table[next_state_key])
    
    new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
    self.q_table[state_key][action] = new_value

def train_agent(agent, env, episodes=3000):
    wandb.init(project="traffic-light-qlearning", name="initial-run")

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        wandb.log({
            "episode": ep,
            "total_reward": total_reward,
            "epsilon": agent.epsilon,
            "q_table_size": len(agent.q_table)
        })
        
        if ep % 100 == 0:
            print(f"Episode {ep}: Reward = {total_reward}, Epsilon = {agent.epsilon:.2f}")

    wandb.finish()
