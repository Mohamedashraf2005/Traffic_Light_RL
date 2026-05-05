
import sys
import os
import numpy as np
import pickle
import torch
import pygame

# ---------------------------------------------------------------------------
# Path setup – add the simulation package so "import simulation" works
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "simulation"))

from agents.baselines import _discretize   # bucketing helper used by Q-agent
import simulation

# ---------------------------------------------------------------------------
# Simulation bootstrap
# ---------------------------------------------------------------------------
pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((simulation.WIDTH, simulation.HEIGHT))
pygame.display.set_caption("Traffic Light RL")
clock = pygame.time.Clock()

simulation.init()
simulation.start_simulation()

# ---------------------------------------------------------------------------
# Timing constants  (all in frames at 60 fps)
# ---------------------------------------------------------------------------
FPS            = 60
YELLOW_FRAMES  = 2 * FPS   # 2-second yellow phase
MIN_GREEN_FRAMES = 5 * FPS  # agent cannot switch more often than every 5 s

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
app_state   = "MENU"   # "MENU" | "RUNNING"
algorithm   = None
model       = None
model_type  = None

# Yellow-phase bookkeeping
yellow_timer    = 0          # counts down from YELLOW_FRAMES to 0
pending_action  = None       # the action we will commit after yellow

# Green-phase bookkeeping
green_frames = 0             # how many frames the current green has been held

# ---------------------------------------------------------------------------
# DQN network definition (must match the architecture used during training)
# ---------------------------------------------------------------------------
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, state_size: int = 17, action_size: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(choice: str) -> None:
    """Load either the Q-learning or DQN checkpoint into global *model*."""
    global model, model_type

    base_path = os.path.join(os.path.dirname(__file__), "checkpoints")

    # ── Q-Learning ──────────────────────────────────────────────────────────
    if choice == "Q-Learning":
        path = os.path.join(base_path, "final_model.pkl")
        with open(path, "rb") as f:
            full_data = pickle.load(f)
        # The checkpoint is a dict {"q_table": ..., "epsilon": ..., ...}
        model = full_data["q_table"]
        model_type = "qlearning"
        print("✅ Q-Learning model loaded")

    # ── DQN ─────────────────────────────────────────────────────────────────
    elif choice == "DQN":
        path = os.path.join(base_path, "trained_dqn_model.pkl")

        # The checkpoint was saved with pickle as a dict containing
        # 'policy_net_state_dict', NOT as a bare nn.Module.
        # torch.load with weights_only=True cannot handle that – use pickle.
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        net = DQNNetwork(state_size=17, action_size=4)
        net.load_state_dict(checkpoint["policy_net_state_dict"])
        net.eval()

        model = net
        model_type = "dqn"
        print("✅ DQN model loaded")


# ---------------------------------------------------------------------------
# State reader
# ---------------------------------------------------------------------------
def get_current_state() -> np.ndarray:
    """
    Build a 17-element state vector that matches the TrafficLightEnv format:
        [queue_norm(12)] + [green_onehot(4)] + [time_progress(1)]

    FIX: lane-2 vehicles are now counted from real vehicle objects (was 0).
    FIX: time_progress is computed from green_frames / expected_green_frames
         so it actually changes over time (was hard-coded to 0.5).
    """
    directions = ["right", "down", "left", "up"]
    MAX_CARS   = 50.0
    queue_norm = []

    for d in directions:
        waiting = [v for v in simulation.vehicles[d] if not v.crossed]
        for lane_id in range(3):          # lanes 0, 1, 2
            cnt = sum(1 for v in waiting if v.lane == lane_id)
            queue_norm.append(min(cnt / MAX_CARS, 1.0))

    green_onehot          = np.zeros(4, dtype=np.float32)
    green_onehot[simulation.currentGreen] = 1.0

    # A rough progress indicator that varies between 0 and 1 as the green
    # phase ages (expected ~10 s = 600 frames at 60 fps)
    expected_green_frames = 10 * FPS
    time_progress = np.array(
        [min(green_frames / expected_green_frames, 1.0)], dtype=np.float32
    )

    return np.concatenate(
        [np.array(queue_norm, dtype=np.float32), green_onehot, time_progress]
    )


# ---------------------------------------------------------------------------
# Fallback: pick the direction with most waiting vehicles
# ---------------------------------------------------------------------------
def _busiest_direction() -> int:
    directions = ["right", "down", "left", "up"]
    counts = [
        sum(1 for v in simulation.vehicles[d] if not v.crossed)
        for d in directions
    ]
    return int(np.argmax(counts))


# ---------------------------------------------------------------------------
# Action selector
# ---------------------------------------------------------------------------
def get_action(state: np.ndarray) -> int:
    """
    Query the loaded model for the best action.

    FIX (Q-Learning): when the state key is absent from the Q-table the old
    code did np.argmax(np.zeros(4)) → always 0 → light stuck on direction 0.
    Now falls back to _busiest_direction() so unseen states make a sensible
    greedy choice.
    """
    global model, model_type

    if model_type == "qlearning":
        state_key = tuple(state)
        q_values  = model.get(state_key, None)
        if q_values is None or np.all(q_values == 0):
            return _busiest_direction()
        return int(np.argmax(q_values))

    elif model_type == "dqn":
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        return int(torch.argmax(q_values).item())

    # No model loaded – use busy-direction heuristic
    return _busiest_direction()


# ---------------------------------------------------------------------------
# Action application  (FIX: the old version never committed the new green)
# ---------------------------------------------------------------------------
def request_action(action: int) -> None:
    """
    Called by the main loop when the agent decides to switch (or stay).

    - If *action* == currentGreen: do nothing (hold the green).
    - If *action* != currentGreen: start a yellow phase; the actual switch to
      the new green happens after YELLOW_FRAMES frames.

    The global *yellow_timer* and *pending_action* drive the transition.
    """
    global yellow_timer, pending_action

    if action == simulation.currentGreen:
        return  # agent chose to stay – nothing to do

    # Begin yellow phase
    simulation.currentYellow = True
    yellow_timer   = YELLOW_FRAMES
    pending_action = action


# ---------------------------------------------------------------------------
# Menu UI
# ---------------------------------------------------------------------------
font_menu = pygame.font.SysFont("segoeui", 36, bold=True)

btn_w, btn_h = 260, 70
qlearning_btn = pygame.Rect(
    simulation.WIDTH  // 2 - btn_w - 30,
    simulation.HEIGHT // 2 - btn_h // 2,
    btn_w, btn_h,
)
dqn_btn = pygame.Rect(
    simulation.WIDTH  // 2 + 30,
    simulation.HEIGHT // 2 - btn_h // 2,
    btn_w, btn_h,
)


def draw_menu_overlay(surface: pygame.Surface) -> None:
    overlay = pygame.Surface((simulation.WIDTH, simulation.HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    title_part1 = font_menu.render("Traffic Light Control by ", True, (255, 255, 255)) 
    title_part2 = font_menu.render("RL Agent", True, (224, 207, 16))                

    surface.blit(title_part1, title_part1.get_rect(center=(simulation.WIDTH // 2 - 55,
                                               simulation.HEIGHT // 2 - 120)))
    surface.blit(title_part2, title_part2.get_rect(center=(simulation.WIDTH // 2 + 220,
                                               simulation.HEIGHT // 2 - 120)))
    
    
    title = font_menu.render("Choose Algorithm", True, (255, 255, 255))
    
    surface.blit(title, title.get_rect(center=(simulation.WIDTH // 2,
                                               simulation.HEIGHT // 2 - 80)))
    
    pygame.draw.rect(surface, (0, 150, 200),  qlearning_btn, border_radius=10)
    
    pygame.draw.rect(surface, (200, 100, 0),  dqn_btn,       border_radius=10)

    surface.blit(
        font_menu.render("Q-Learning", True, (255, 255, 255)),
        font_menu.render("Q-Learning", True, (255, 255, 255))
              .get_rect(center=qlearning_btn.center),
    )
    surface.blit(
        font_menu.render("DQN", True, (255, 255, 255)),
        font_menu.render("DQN", True, (255, 255, 255))
              .get_rect(center=dqn_btn.center),
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:

    # ── Event handling ───────────────────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if app_state == "MENU":
                if qlearning_btn.collidepoint(mx, my):
                    algorithm = "Q-Learning"
                    load_model(algorithm)
                    app_state = "RUNNING"
                    green_frames = 0
                    print("▶ Q-Learning started")
                elif dqn_btn.collidepoint(mx, my):
                    algorithm = "DQN"
                    load_model(algorithm)
                    app_state = "RUNNING"
                    green_frames = 0
                    print("▶ DQN started")

        if event.type == simulation.SPAWN_VEHICLE_EVENT:
            simulation.handle_spawn_event()

    # ── Menu rendering ───────────────────────────────────────────────────────
    if app_state == "MENU":
        simulation.render_frame(screen, moving=False)
        draw_menu_overlay(screen)

    # ── RL running ───────────────────────────────────────────────────────────
    elif app_state == "RUNNING":

        # --- Yellow phase countdown -----------------------------------------
        if simulation.currentYellow:
            yellow_timer -= 1
            if yellow_timer <= 0:
                # Yellow expired → commit the switch
                simulation.currentGreen  = pending_action
                simulation.currentYellow = False
                pending_action           = None
                green_frames             = 0
                print(f"🟢 Green → direction {simulation.currentGreen} "
                      f"({['right','down','left','up'][simulation.currentGreen]})")

        # --- Green phase: let the agent decide ------------------------------
        else:
            green_frames += 1

            # Only query the agent once the minimum green time has elapsed
            if green_frames >= MIN_GREEN_FRAMES:
                raw_state = get_current_state()

                if model_type == "qlearning":
                    state = [_discretize(c) for c in raw_state]
                else:
                    state = raw_state

                action = get_action(state)
                request_action(action)   # FIX: was apply_action() which broke

        simulation.render_frame(screen, moving=True)

    pygame.display.flip()
    clock.tick(FPS)
