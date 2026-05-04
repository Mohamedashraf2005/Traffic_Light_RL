import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "simulation"))
import simulation
import pygame
import torch
import pickle
import numpy as np
from simulation import get_current_state, apply_action

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((simulation.WIDTH, simulation.HEIGHT))
pygame.display.set_caption("SIMULATION")
clock = pygame.time.Clock()

# Start the signal thread
simulation.init() 
simulation.start_simulation()

app_state = "MENU"   
algorithm = None

#define menu buttons
font_menu = pygame.font.SysFont("segoeui", 36, bold=True)
btn_w, btn_h = 260, 70
qlearning_btn = pygame.Rect(simulation.WIDTH//2 - btn_w - 30, simulation.HEIGHT//2 - btn_h//2, btn_w, btn_h)
dqn_btn       = pygame.Rect(simulation.WIDTH//2 + 30,         simulation.HEIGHT//2 - btn_h//2, btn_w, btn_h)



MODEL_PATHS = {
    "Q-Learning": "checkpoints/final_model.pkl",
    "DQN": "checkpoints/trained_dqn_model.pth"
}

q_agent = None
dqn_model = None


def load_models():
    global q_agent, dqn_model

    # ===== Q-LEARNING =====
    with open(MODEL_PATHS["Q-Learning"], "rb") as f:
        q_agent = pickle.load(f)

    # ===== DQN =====
    dqn_model = torch.load(MODEL_PATHS["DQN"], map_location="cpu")
    dqn_model.eval()
def draw_menu_overlay(surface):
    # Dark semi-transparent overlay
    overlay = pygame.Surface((simulation.WIDTH, simulation.HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    # Title
    title = font_menu.render("Choose Algorithm", True, (255, 255, 255))
    surface.blit(title, title.get_rect(center=(simulation.WIDTH//2, simulation.HEIGHT//2 - 80)))

    # Buttons
    pygame.draw.rect(surface, (0, 150, 200), qlearning_btn, border_radius=10)
    pygame.draw.rect(surface, (200, 100, 0), dqn_btn,       border_radius=10)

    ql_text = font_menu.render("Q-Learning", True, (255, 255, 255))
    dq_text = font_menu.render("DQN",        True, (255, 255, 255))
    surface.blit(ql_text, ql_text.get_rect(center=qlearning_btn.center))
    surface.blit(dq_text, dq_text.get_rect(center=dqn_btn.center))

# MASTER LOOP
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            print(f"📍 Click: ({mx}, {my})")

            #button click → switch state
            if app_state == "MENU":
                if qlearning_btn.collidepoint(mx, my):
                    algorithm = "Q-Learning"
                    app_state = "RUNNING"
                    print("▶ Starting Q-Learning simulation")
                elif dqn_btn.collidepoint(mx, my):
                    algorithm = "DQN"
                    app_state = "RUNNING"
                    print("▶ Starting DQN simulation")

        # Spawn vehicles regardless of state
        if event.type == simulation.SPAWN_VEHICLE_EVENT:
            simulation.handle_spawn_event()
            total = sum(len(simulation.vehicles[d]) for d in simulation.vehicles)
            print(f"🚗 Spawn event fired — total vehicles: {total}")

    # ---- Render ----
    if app_state == "MENU":
        simulation.render_frame(screen, moving=False)  # frozen background
        draw_menu_overlay(screen)                       # overlay + buttons

    elif app_state == "RUNNING":
    
    # =========================
    # 0. LOAD MODELS ON FIRST RUN
    # =========================
        if q_agent is None and dqn_model is None:
            load_models()
            print("✅ Models loaded")

        # =========================
        # 1.  GET STATE
        # =========================
        state = get_current_state()

        # =========================
        # 2. GET ACTION
        # =========================
        if algorithm == "Q-Learning":

            state_key = tuple(state)
            q_values = q_agent.get(state_key, np.zeros(4))
            action = int(np.argmax(q_values))

        elif algorithm == "DQN":

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                q_values = dqn_model(state_tensor)
                action = int(torch.argmax(q_values).item())

        else:
            action = 0  # fallback

        # =========================
        # 3.  APPLY ACTION
        # =========================
        apply_action(action)

        # =========================
        # 4.  RENDER FRAME
        # =========================
        simulation.render_frame(screen, moving=True)

        pygame.display.flip()
        clock.tick(60)
