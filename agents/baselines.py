import random
import numpy as np
from env.traffic_light_env import TrafficLightEnv, DIRECTION_NUMBERS


# ======================================================================= #
#  1. DISCRETIZED WRAPPER  (for Q-Learning)                                #
# ======================================================================= #

# Thresholds that define the bucket boundaries for car counts.
# Tune these to match the scale of your intersection.
BUCKET_THRESHOLDS = [0, 5, 15]   # ≤0 → Empty(0), ≤5 → Light(1), ≤15 → Medium(2), >15 → Heavy(3)

BUCKET_LABELS = {0: "Empty", 1: "Light", 2: "Medium", 3: "Heavy"}


def _discretize(count: int) -> int:
    """Map a raw car count to a discrete bucket index."""
    if count <= BUCKET_THRESHOLDS[0]:
        return 0   # Empty
    elif count <= BUCKET_THRESHOLDS[1]:
        return 1   # Light
    elif count <= BUCKET_THRESHOLDS[2]:
        return 2   # Medium
    else:
        return 3   # Heavy


class DiscretizedWrapper:
    """
    Wraps TrafficLightEnv and converts the raw 12-integer state into
    12 discrete bucket indices — suitable for a Q-table.

    Raw state  : [0, 12, 3, 0, 7, 1, ...]   ← exact car counts (for DQN)
    Bucketed   : [0,  3, 1, 0, 2, 1, ...]   ← category indices  (for Q-Learning)

    All other behaviour (rewards, done, info) is passed through unchanged.
    """

    def __init__(self, env: TrafficLightEnv):
        self.env        = env
        self.action_size = env.action_size
        self.state_size  = env.state_size   # still 12 — same length, different values

    def reset(self) -> list:
        raw_state = self.env.reset()
        return [_discretize(c) for c in raw_state]

    def step(self, action: int) -> tuple:
        raw_state, reward, done, info = self.env.step(action)
        bucketed_state = [_discretize(c) for c in raw_state]
        info["raw_state"]      = raw_state        # keep raw counts for debugging
        info["bucketed_state"] = bucketed_state
        return bucketed_state, reward, done, info

    # Convenience passthrough
    def get_signal_timers(self):
        return self.env.get_signal_timers()

    def render(self):
        self.env.render()


# ======================================================================= #
#  2. RANDOM AGENT                                                          #
# ======================================================================= #

class RandomAgent:
    """
    Baseline: picks a uniformly random action at every step.
    Represents the lower-bound performance — any trained agent should beat this.
    """

    def __init__(self, action_size: int = 4):
        self.action_size = action_size

    def select_action(self, state=None) -> int:   # state ignored
        return random.randint(0, self.action_size - 1)


# ======================================================================= #
#  3. FIXED-TIMER AGENT                                                     #
# ======================================================================= #

class FixedTimerAgent:
    """
    Baseline: holds the current action for `hold_steps` steps, then
    moves to the next direction in a fixed round-robin cycle.

    Mirrors the fixed-cycle logic of simulation.py's repeat() function.
    """

    def __init__(self, action_size: int = 4, hold_steps: int = 15):
        self.action_size = action_size
        self.hold_steps  = hold_steps
        self._counter    = 0
        self._action     = 0   # starts on 'right'

    def select_action(self, state=None) -> int:   # state ignored
        if self._counter >= self.hold_steps:
            self._action  = (self._action + 1) % self.action_size
            self._counter = 0
        self._counter += 1
        return self._action

    def reset(self):
        """Call at the start of each episode to restart the cycle."""
        self._counter = 0
        self._action  = 0


# ======================================================================= #
#  4. EVALUATION LOOP                                                       #
# ======================================================================= #

def evaluate(agent, env_kwargs: dict = None, n_episodes: int = 100,
             use_wrapper: bool = False, seed: int = 0) -> dict:
    """
    Run `n_episodes` episodes with the given agent and collect rewards.

    Parameters
    ----------
    agent        : Any agent with a select_action(state) method.
    env_kwargs   : kwargs forwarded to TrafficLightEnv (e.g. max_steps).
    n_episodes   : Number of episodes to evaluate.
    use_wrapper  : If True, wraps env with DiscretizedWrapper (for Q-Learning).
    seed         : Base RNG seed (each episode gets seed+episode_index).

    Returns
    -------
    dict with keys:
        episode_rewards  — list of total rewards, one per episode
        avg_reward       — mean across all episodes
        std_reward       — standard deviation
        avg_crossed      — mean total vehicles crossed per episode
        avg_waiting      — mean final waiting cars per episode
    """
    if env_kwargs is None:
        env_kwargs = {}

    episode_rewards = []
    episode_crossed = []
    episode_waiting = []

    for ep in range(n_episodes):
        # Fresh env each episode with a unique seed for reproducibility
        env = TrafficLightEnv(seed=seed + ep, **env_kwargs)
        if use_wrapper:
            env = DiscretizedWrapper(env)

        # Reset agent counter if it supports it (FixedTimerAgent)
        if hasattr(agent, "reset"):
            agent.reset()

        state      = env.reset()
        total_reward = 0.0
        last_info    = {}

        while True:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            last_info     = info
            if done:
                break

        episode_rewards.append(total_reward)
        episode_crossed.append(sum(last_info.get("total_crossed", [0])))
        episode_waiting.append(last_info.get("total_waiting", 0))

    return {
        "episode_rewards": episode_rewards,
        "avg_reward":      float(np.mean(episode_rewards)),
        "std_reward":      float(np.std(episode_rewards)),
        "avg_crossed":     float(np.mean(episode_crossed)),
        "avg_waiting":     float(np.mean(episode_waiting)),
    }


# ======================================================================= #
#  5. MAIN — run both baselines and print comparison report                 #
# ======================================================================= #

if __name__ == "__main__":

    ENV_KWARGS   = {"max_steps": 100}
    N_EPISODES   = 100
    ACTION_SIZE  = 4

    print("=" * 58)
    print("   Baseline Evaluation  —  100 episodes each")
    print("=" * 58)

    # ── Random Agent ────────────────────────────────────────────────── #
    print("\n[1/2]  Running Random Agent ...")
    random_agent  = RandomAgent(action_size=ACTION_SIZE)
    random_results = evaluate(random_agent, env_kwargs=ENV_KWARGS,
                               n_episodes=N_EPISODES, seed=0)

    # ── Fixed-Timer Agent ────────────────────────────────────────────── #
    print("[2/2]  Running Fixed-Timer Agent ...")
    fixed_agent   = FixedTimerAgent(action_size=ACTION_SIZE, hold_steps=15)
    fixed_results  = evaluate(fixed_agent, env_kwargs=ENV_KWARGS,
                               n_episodes=N_EPISODES, seed=0)

    # ── Report ───────────────────────────────────────────────────────── #
    print("\n" + "=" * 58)
    print(f"{'Metric':<28} {'Random':>12} {'FixedTimer':>12}")
    print("-" * 58)
    print(f"{'Avg Episode Reward':<28} {random_results['avg_reward']:>12.1f} {fixed_results['avg_reward']:>12.1f}")
    print(f"{'Std Episode Reward':<28} {random_results['std_reward']:>12.1f} {fixed_results['std_reward']:>12.1f}")
    print(f"{'Avg Vehicles Crossed':<28} {random_results['avg_crossed']:>12.1f} {fixed_results['avg_crossed']:>12.1f}")
    print(f"{'Avg Final Waiting Cars':<28} {random_results['avg_waiting']:>12.1f} {fixed_results['avg_waiting']:>12.1f}")
    print("=" * 58)

    print("\nSave these numbers as your baseline benchmark.")
    print("Any trained RL agent (DQN / Q-Learning) must beat:")
    best = max(random_results['avg_reward'], fixed_results['avg_reward'])
    print(f"  Target avg reward > {best:.1f}\n")

    # ── Wrapper demo ─────────────────────────────────────────────────── #
    print("─" * 58)
    print("  Wrapper Demo (Q-Learning bucketed state)")
    print("─" * 58)
    raw_env     = TrafficLightEnv(max_steps=5, seed=7)
    wrapped_env = DiscretizedWrapper(raw_env)
    state = wrapped_env.reset()
    print(f"  Bucketed initial state : {state}")
    for _ in range(3):
        action = random.randint(0, 3)
        state, reward, done, info = wrapped_env.step(action)
        print(f"  Action: {DIRECTION_NUMBERS[action]:<6}  "
              f"Raw: {info['raw_state']}  →  "
              f"Bucketed: {info['bucketed_state']}")
        if done:
            break
