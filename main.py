import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "simulation"))

import simulation
import pygame
import numpy as np
import pickle
import torch


pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((simulation.WIDTH, simulation.HEIGHT))
pygame.display.set_caption("SIMULATION")
clock = pygame.time.Clock()

simulation.init()
simulation.start_simulation()

app_state = "MENU"
algorithm = None
model = None
model_type = None


# =========================
# 📦 LOAD MODELS
# =========================
def load_model(choice):
    global model, model_type

    base_path = os.path.join(os.path.dirname(__file__), "checkpoints")

    # -------------------------
    # Q-LEARNING MODEL
    # -------------------------
    if choice == "Q-Learning":
        path = os.path.join(base_path, "final_model.pkl")

        with open(path, "rb") as f:
            model = pickle.load(f)

        model_type = "qlearning"
        print("✅ Q-Learning model loaded")

    # -------------------------
    # DQN MODEL
    # -------------------------
    elif choice == "DQN":
        path = os.path.join(base_path, "trained_dqn_model.pkl")

        model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()

        model_type = "dqn"
        print("✅ DQN model loaded")


# =========================
# 🧠 ACTION SELECTOR
# =========================
def get_action(state):
    global model, model_type

    if model_type == "qlearning":
        state_key = tuple(state)
        q_values = model.get(state_key, np.zeros(4))
        return int(np.argmax(q_values))

    elif model_type == "dqn":
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        return int(torch.argmax(q_values).item())

    return 0


# =========================
# MENU BUTTONS
# =========================
font_menu = pygame.font.SysFont("segoeui", 36, bold=True)

btn_w, btn_h = 260, 70
qlearning_btn = pygame.Rect(simulation.WIDTH//2 - btn_w - 30,
                            simulation.HEIGHT//2 - btn_h//2,
                            btn_w, btn_h)

dqn_btn = pygame.Rect(simulation.WIDTH//2 + 30,
                      simulation.HEIGHT//2 - btn_h//2,
                      btn_w, btn_h)


def draw_menu_overlay(surface):
    overlay = pygame.Surface((simulation.WIDTH, simulation.HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    title = font_menu.render("Choose Algorithm", True, (255, 255, 255))
    surface.blit(title, title.get_rect(center=(simulation.WIDTH//2, simulation.HEIGHT//2 - 80)))

    pygame.draw.rect(surface, (0, 150, 200), qlearning_btn, border_radius=10)
    pygame.draw.rect(surface, (200, 100, 0), dqn_btn, border_radius=10)

    ql_text = font_menu.render("Q-Learning", True, (255, 255, 255))
    dq_text = font_menu.render("DQN", True, (255, 255, 255))

    surface.blit(ql_text, ql_text.get_rect(center=qlearning_btn.center))
    surface.blit(dq_text, dq_text.get_rect(center=dqn_btn.center))


# =========================
# MAIN LOOP
# =========================
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # -------------------------
        # BUTTON CLICK
        # -------------------------
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()

            if app_state == "MENU":

                if qlearning_btn.collidepoint(mx, my):
                    algorithm = "Q-Learning"
                    load_model(algorithm)
                    app_state = "RUNNING"
                    print("▶ Q-Learning started")

                elif dqn_btn.collidepoint(mx, my):
                    algorithm = "DQN"
                    load_model(algorithm)
                    app_state = "RUNNING"
                    print("▶ DQN started")

        # -------------------------
        # SPAWN VEHICLES
        # -------------------------
        if event.type == simulation.SPAWN_VEHICLE_EVENT:
            simulation.handle_spawn_event()

    # =========================
    # 🎮 MENU STATE
    # =========================
    if app_state == "MENU":
        simulation.render_frame(screen, moving=False)
        draw_menu_overlay(screen)

    # =========================
    # 🚦 RL RUNNING STATE
    # =========================
    elif app_state == "RUNNING":

        state = simulation.get_current_state()
        
        action = get_action(state)

        simulation.apply_action(action)

        simulation.render_frame(screen, moving=True)

    pygame.display.flip()
    clock.tick(60)