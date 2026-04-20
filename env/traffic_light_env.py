import numpy as np
import random



DEFAULT_GREEN  = {0: 10, 1: 10, 2: 10, 3: 10}
DEFAULT_RED    = 150
DEFAULT_YELLOW = 5

# Matches simulation.py: directionNumbers
DIRECTION_NUMBERS = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
DIRECTION_INDICES = {'right': 0, 'down': 1, 'left': 2, 'up': 3}

# Matches simulation.py: vehicleTypes + speeds
VEHICLE_TYPES  = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
VEHICLE_SPEEDS = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5}

# How many cars depart per step per vehicle type (proportional to speed)
_BASE_DEPART   = 3.0
_SPEED_BASE    = VEHICLE_SPEEDS['car']
DEPART_BY_TYPE = {
    vtype: max(1, round(_BASE_DEPART * (spd / _SPEED_BASE)))
    for vtype, spd in VEHICLE_SPEEDS.items()
}
# → {'car': 3, 'bus': 2, 'truck': 2, 'bike': 3}

# Arrival distribution — matches simulation.py dist = [25, 50, 75, 100]
ARRIVAL_PROBS = [0.25, 0.25, 0.25, 0.25]   # equal per direction

# Number of lanes per direction (simulation uses lanes 1 & 2; lane 0 exists)
NUM_LANES = 3


class TrafficLightEnv:
    """
    RL environment whose internal model mirrors simulation.py's logic
    so it can be wired to the Pygame UI with minimal glue code.

    State  (12 integers):
        Flattened lane car-counts:
        [right_l0, right_l1, right_l2,
          down_l0,  down_l1,  down_l2,
          left_l0,  left_l1,  left_l2,
            up_l0,    up_l1,    up_l2]

    Actions (4):
        0 → right gets green
        1 → down  gets green
        2 → left  gets green
        3 → up    gets green
        (mirrors currentGreen in simulation.py)

    Reward:
        + crossed cars this step      (throughput — we want to maximise)
        − total waiting cars          (congestion — we want to minimise)
        − switching_penalty           (if action changed from last step)
        − yellow_penalty * yellow_steps consumed  (time lost in yellow)
    """

    def __init__(
        self,
        max_steps: int = 200,
        arrival_lambda: float = 1.5,
        switching_penalty: float = 5.0,
        yellow_penalty: float = 1.0,
        max_cars_per_lane: int = 50,
        default_green: dict = None,
        default_yellow: int = DEFAULT_YELLOW,
        seed: int = None,
    ):
        self.max_steps         = max_steps
        self.arrival_lambda    = arrival_lambda
        self.switching_penalty = switching_penalty
        self.yellow_penalty    = yellow_penalty
        self.max_cars_per_lane = max_cars_per_lane
        self.default_green     = default_green if default_green else dict(DEFAULT_GREEN)
        self.default_yellow    = default_yellow

        self.rng = np.random.default_rng(seed)

        # Initialised in reset()
        self.cars: list          = [[0]*NUM_LANES for _ in range(4)]
        self.crossed: list       = [0, 0, 0, 0]
        self.current_green: int  = 0
        self.current_yellow: int = 0
        self.current_step: int   = 0
        self.done: bool          = False
        self.signals: list       = []

    # ------------------------------------------------------------------ #
    #  reset                                                               #
    # ------------------------------------------------------------------ #
    def reset(self) -> list:
        """
        Reset episode. Random initial cars in lanes 1 & 2 only —
        matches simulation.py which spawns only in lanes 1 & 2.

        Returns
        -------
        state : list[int]  length-12 flat state vector
        """
        self.cars = [
            [0,
             int(self.rng.integers(1, 8)),
             int(self.rng.integers(1, 8))]
            for _ in range(4)
        ]
        self.crossed        = [0, 0, 0, 0]
        self.current_step   = 0
        self.current_green  = int(self.rng.integers(0, 4))
        self.current_yellow = 0
        self.done           = False

        # Signal timers — mirrors simulation.py initialize()
        self.signals = []
        for i in range(4):
            if i == 0:
                s = {'red': 0,           'yellow': DEFAULT_YELLOW, 'green': self.default_green[0]}
            else:
                s = {'red': DEFAULT_RED, 'yellow': DEFAULT_YELLOW, 'green': self.default_green[i]}
            self.signals.append(s)

        return self._get_state()

    # ------------------------------------------------------------------ #
    #  step                                                                #
    # ------------------------------------------------------------------ #
    def step(self, action: int) -> tuple:
        """
        One decision step. The agent picks which direction gets green next.

        Parameters
        ----------
        action : int  0=right, 1=down, 2=left, 3=up
                      (mirrors currentGreen in simulation.py)

        Returns
        -------
        next_state : list[int]  flat 12-element state vector
        reward     : float
        done       : bool
        info       : dict  — contains 'currentGreen', 'signals', etc.
                             for direct use by simulation.py
        """
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")
        if action not in range(4):
            raise ValueError(f"Invalid action {action}. Must be 0-3.")

        # ── 1. Switching penalty ──────────────────────────────────────── #
        switched       = (action != self.current_green)
        switch_penalty = -self.switching_penalty if switched else 0.0

        # ── 2. Yellow phase — no departures, time cost ───────────────── #
        # Mirrors simulation.py: currentYellow=1 during transition,
        # stop coords reset, then currentYellow=0
        yellow_cost = 0.0
        if switched:
            self.current_yellow = 1
            yellow_cost = -self.yellow_penalty * self.default_yellow
            # Reset of stop coords happens in simulation.py per-vehicle;
            # here we just consume the time penalty
            self.current_yellow = 0

        # ── 3. Update current green (mirrors currentGreen = nextGreen) ── #
        self.current_green = action

        # ── 4. Traffic flow — depart from ALL lanes of green direction ── #
        # Speed-weighted average: car=40%, bus=20%, truck=20%, bike=20%
        # matches simulation.py vehicle_type distribution (randint 0-3)
        avg_depart = round(
            DEPART_BY_TYPE['car']   * 0.4 +
            DEPART_BY_TYPE['bus']   * 0.2 +
            DEPART_BY_TYPE['truck'] * 0.2 +
            DEPART_BY_TYPE['bike']  * 0.2
        )
        step_crossed = 0
        for lane in range(NUM_LANES):
            if self.cars[action][lane] == 0:
                continue
            departed = min(self.cars[action][lane], avg_depart)
            self.cars[action][lane] -= departed
            step_crossed            += departed

        self.crossed[action] += step_crossed

        # ── 5. Stochastic arrivals — Poisson, lanes 1 & 2 only ────────── #
        # Matches simulation.py: lane_number = random.randint(1, 2)
        arrivals_log = []
        for d in range(4):
            dir_arrivals = []
            for lane in [1, 2]:
                new_cars = int(self.rng.poisson(lam=self.arrival_lambda))
                self.cars[d][lane] = min(
                    self.cars[d][lane] + new_cars,
                    self.max_cars_per_lane,
                )
                dir_arrivals.append(new_cars)
            arrivals_log.append(dir_arrivals)

        # ── 6. Update signal timers (mirrors updateValues()) ──────────── #
        for i in range(4):
            if i == self.current_green:
                self.signals[i]['green'] = max(0, self.signals[i]['green'] - 1)
            else:
                self.signals[i]['red']   = max(0, self.signals[i]['red']   - 1)

        # Reset exhausted green timer (mirrors repeat())
        if self.signals[self.current_green]['green'] == 0:
            self.signals[self.current_green]['green']  = self.default_green[self.current_green]
            self.signals[self.current_green]['yellow'] = DEFAULT_YELLOW
            self.signals[self.current_green]['red']    = DEFAULT_RED

        # ── 7. Reward ─────────────────────────────────────────────────── #
        total_waiting = sum(
            self.cars[d][l] for d in range(4) for l in range(NUM_LANES)
        )
        reward = (
              float(step_crossed)    # + throughput reward
            - float(total_waiting)   # − congestion penalty
            + switch_penalty         # − phase-switch cost
            + yellow_cost            # − yellow time cost
        )

        # ── 8. Termination ────────────────────────────────────────────── #
        self.current_step += 1
        self.done = self.current_step >= self.max_steps

        # ── 9. Info dict — UI bridge ───────────────────────────────────── #
        # Variable names mirror simulation.py globals exactly so the
        # integration glue code can do:  currentGreen = info['currentGreen']
        info = {
            # ← simulation.py globals
            "currentGreen":   self.current_green,
            "currentYellow":  self.current_yellow,
            "nextGreen":      (self.current_green + 1) % 4,
            "signals":        [dict(s) for s in self.signals],

            # ← RL diagnostics
            "step":           self.current_step,
            "direction":      DIRECTION_NUMBERS[self.current_green],
            "switched":       switched,
            "switch_penalty": switch_penalty,
            "yellow_cost":    yellow_cost,
            "step_crossed":   step_crossed,
            "total_crossed":  list(self.crossed),
            "arrivals":       arrivals_log,
            "total_waiting":  total_waiting,
            "cars_per_lane":  {
                DIRECTION_NUMBERS[d]: list(self.cars[d]) for d in range(4)
            },
        }

        return self._get_state(), reward, self.done, info

    # ------------------------------------------------------------------ #
    #  get_signal_timers  — UI integration hook                            #
    # ------------------------------------------------------------------ #
    def get_signal_timers(self) -> list:
        """
        Returns signal timers in the same format as simulation.py's
        TrafficSignal objects.

        Usage in simulation.py:
            timers = env.get_signal_timers()
            for i in range(noOfSignals):
                signals[i].green  = timers[i]['green']
                signals[i].yellow = timers[i]['yellow']
                signals[i].red    = timers[i]['red']
        """
        return [dict(s) for s in self.signals]

    # ------------------------------------------------------------------ #
    #  render  — text fallback when Pygame is not running                  #
    # ------------------------------------------------------------------ #
    def render(self) -> None:
        """ASCII intersection view."""
        bar_max = 16
        print("\n" + "═" * 54)
        print(f"  Step {self.current_step:>3}/{self.max_steps}  |  "
              f"Green: {DIRECTION_NUMBERS[self.current_green].upper()}")
        print("─" * 54)
        for d in range(4):
            icon  = "🟢" if d == self.current_green else "🔴"
            dname = DIRECTION_NUMBERS[d]
            lanes = self.cars[d]
            total = sum(lanes)
            filled = int((total / (self.max_cars_per_lane * NUM_LANES)) * bar_max)
            bar    = "█" * filled + "░" * (bar_max - filled)
            lane_str = f"[{lanes[0]},{lanes[1]},{lanes[2]}]"
            print(f"  {icon} {dname:<6} lanes{lane_str:<12} [{bar}] {total:>3} total")
        total_all = sum(self.cars[d][l] for d in range(4) for l in range(NUM_LANES))
        print(f"  Total waiting : {total_all}")
        print(f"  Total crossed : { {DIRECTION_NUMBERS[i]: v for i,v in enumerate(self.crossed)} }")
        print("═" * 54 + "\n")

    # ------------------------------------------------------------------ #
    #  helpers                                                             #
    # ------------------------------------------------------------------ #
    def _get_state(self) -> list:
        """
        Flat 12-element state vector.
        Order: right_l0, right_l1, right_l2, down_l0 ... up_l2
        Matches vehicles[direction][lane] structure in simulation.py.
        """
        return [self.cars[d][l] for d in range(4) for l in range(NUM_LANES)]

    @property
    def state_size(self) -> int:
        """12 = 4 directions × 3 lanes."""
        return 4 * NUM_LANES

    @property
    def action_size(self) -> int:
        """4 = one action per direction (mirrors currentGreen in simulation.py)."""
        return 4

    def __repr__(self) -> str:
        return (
            f"TrafficLightEnv("
            f"step={self.current_step}/{self.max_steps}, "
            f"green={DIRECTION_NUMBERS[self.current_green]}, "
            f"cars={self.cars})"
        )


# ====================================================================== #
#  Smoke-test / demo                                                       #
# ====================================================================== #
if __name__ == "__main__":
    print("=" * 55)
    print("   Traffic Light Env — UI-Compatible Demo Run")
    print("=" * 55)

    env = TrafficLightEnv(max_steps=12, seed=42)
    state = env.reset()

    print(f"\nState size  : {env.state_size}  (4 directions x 3 lanes)")
    print(f"Action size : {env.action_size}  (0=right, 1=down, 2=left, 3=up)")
    print(f"Initial state vector : {state}")
    env.render()

    total_reward = 0.0
    for step_n in range(1, env.max_steps + 1):
        action = random.randint(0, 3)   # random agent — swap with trained model

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(
            f"Step {step_n:>2}  |  "
            f"Action: {DIRECTION_NUMBERS[action]:<6}  |  "
            f"Reward: {reward:>8.1f}  |  "
            f"Crossed: {info['step_crossed']:>2}  |  "
            f"Waiting: {info['total_waiting']:>3}"
            + ("  ⚠ SWITCH" if info["switched"] else "")
        )

        if done:
            env.render()
            print(f"Episode done  —  cumulative reward : {total_reward:.1f}")
            print(f"Total crossed : { {DIRECTION_NUMBERS[i]: v for i,v in enumerate(info['total_crossed'])} }")
            break
