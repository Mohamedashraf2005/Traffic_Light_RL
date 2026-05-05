import pygame
import sys
import os
import random
import time
import threading
import numpy as np

# PATHS
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# SCREEN & CLOCK
WIDTH, HEIGHT = 1400, 800

# screen and clock are passed in from main.py


# ANALYTICS VARIABLES
throughput = 0
total_wait_time = 0.0
mock_q_value_confidence = 0.85 # Placeholder variable to plug your model into
active_algorithm = "None"


# LOAD & SCALE ASSETS
SIGNAL_SIZE = (25, 65)

# These are loaded later by init() after pygame.display is ready
background = None
redSignal = None
yellowSignal = None
greenSignal = None

def init():
    """Call from main.py AFTER pygame.display.set_mode() to load all assets."""
    global background, redSignal, yellowSignal, greenSignal

    background = pygame.image.load(
        os.path.join(PROJECT_ROOT, "images", "intersection.png")
    ).convert()

    redSignal = pygame.image.load(
        os.path.join(PROJECT_ROOT, "images", "signals", "red.png")
    ).convert_alpha()
    redSignal = pygame.transform.smoothscale(redSignal, SIGNAL_SIZE)

    yellowSignal = pygame.image.load(
        os.path.join(PROJECT_ROOT, "images", "signals", "yellow.png")
    ).convert_alpha()
    yellowSignal = pygame.transform.smoothscale(yellowSignal, SIGNAL_SIZE)

    greenSignal = pygame.image.load(
        os.path.join(PROJECT_ROOT, "images", "signals", "green.png")
    ).convert_alpha()
    greenSignal = pygame.transform.smoothscale(greenSignal, SIGNAL_SIZE)

    pygame.time.set_timer(SPAWN_VEHICLE_EVENT, 1200)

# VEHICLE CONFIG
speeds = {
    'car': 4.6,
    'taxi': 4.8,
    'bus': 3.6,
    'truck': 3.4,
    'moto': 6.4,
    'ambulance': 7.0
}

sizes = {
    'car': (100, 45),
    'taxi': (100, 45),
    'bus': (150, 55),
    'truck': (170, 60),
    'moto': (90, 50),
    'ambulance': (150, 70)
}

vehicleTypes = ['car', 'bus', 'truck', 'moto', 'taxi', 'ambulance']

directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

x = {
    'right': [0, 0], 
    'down': [540, 610], 
    'left': [1400, 1400], 
    'up': [700, 790] 
}    
y = {
    'right': [410, 475], 
    'down': [0, 0], 
    'left': [270, 330], 
    'up': [800, 800]
}

stopLines = {  
    'right': 470,   
    'down': 200,
    'left': 912,
    'up': 620
}

vehicles = {'right': [], 'down': [], 'left': [], 'up': []}


# SIGNAL SYSTEM & TIMERS
class TrafficSignal:
    def __init__(self, green, yellow):
        self.green = green
        self.yellow = yellow

signals = [
    TrafficSignal(5, 2),
    TrafficSignal(5, 2),
    TrafficSignal(5, 2),
    TrafficSignal(5, 2)
]

currentGreen = 0
currentYellow = False
time_left = 0  # This variable will be updated by the signal thread and read by the timer display function

def updateSignals():
    global currentGreen, currentYellow, time_left
    while True:
        currentYellow = False
        for i in range(signals[currentGreen].green, 0, -1):
            time_left = i
            time.sleep(1)

        currentYellow = True
        for i in range(signals[currentGreen].yellow, 0, -1):
            time_left = i
            time.sleep(1)

        currentGreen = (currentGreen + 1) % 4

def get_time_left(idx):
    if idx == currentGreen:
        return time_left
    else:
        diff = (idx - currentGreen) % 4
        if currentYellow:
            current_phase_remaining = time_left
        else:
            current_phase_remaining = time_left + signals[currentGreen].yellow
        
        full_phase = signals[0].green + signals[0].yellow
        wait_time = current_phase_remaining + (diff - 1) * full_phase
        return wait_time


# VEHICLE CLASS
class Vehicle:
    def __init__(self, lane, vehicleClass, direction_number, direction):
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.direction = direction
        self.direction_number = direction_number
        self.speed = speeds[vehicleClass]
        
        self.wait_time = 0.0 # <-- Tracks time spent waiting at a red light / in traffic

        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = False

        path = os.path.join(PROJECT_ROOT, "images", "vehicles", vehicleClass + ".png")
        self.image = pygame.image.load(path).convert_alpha()
        self.image = pygame.transform.smoothscale(self.image, sizes[vehicleClass])

        rotations = {
            'right': 0,
            'down': -90,
            'left': 180,
            'up': 90
        }
        self.image = pygame.transform.rotate(self.image, rotations[direction])
        vehicles[direction].append(self)

    def is_front_clear(self):
        safe_distance = 40  
        
        for other in vehicles[self.direction]:
            if other is self or other.lane != self.lane:
                continue
            
            if self.direction == 'right':
                if self.x <= other.x and other.x - (self.x + self.image.get_width()) < safe_distance:
                    return False
            elif self.direction == 'left':
                if self.x >= other.x and self.x - (other.x + other.image.get_width()) < safe_distance:
                    return False
            elif self.direction == 'down':
                if self.y <= other.y and other.y - (self.y + self.image.get_height()) < safe_distance:
                    return False
            elif self.direction == 'up':
                if self.y >= other.y and self.y - (other.y + other.image.get_height()) < safe_distance:
                    return False
                    
        return True

    def move(self):
        # Accumulate wait time if the car cannot move
        if not self.is_front_clear():
            self.wait_time += 1 / 60.0
            return

        if not self.crossed:
            if self.direction_number != currentGreen or currentYellow:
                if self.direction == 'right' and self.x + self.image.get_width() >= stopLines['right']:
                    self.wait_time += 1 / 60.0
                    return
                if self.direction == 'left' and self.x <= stopLines['left']:
                    self.wait_time += 1 / 60.0
                    return
                if self.direction == 'down' and self.y + self.image.get_height() >= stopLines['down']:
                    self.wait_time += 1 / 60.0
                    return
                if self.direction == 'up' and self.y <= stopLines['up']:
                    self.wait_time += 1 / 60.0
                    return

        # Regular movement
        if self.direction == 'right':
            if self.x > stopLines['right']:
                self.crossed = True
            self.x += self.speed
        elif self.direction == 'left':
            if self.x < stopLines['left']:
                self.crossed = True
            self.x -= self.speed
        elif self.direction == 'down':
            if self.y > stopLines['down']:
                self.crossed = True
            self.y += self.speed
        elif self.direction == 'up':
            if self.y < stopLines['up']:
                self.crossed = True
            self.y -= self.speed


# GENERATE VEHICLES EVENT
SPAWN_VEHICLE_EVENT = pygame.USEREVENT + 1

# SIGNAL & TIMER POSITIONS 
signalCoods = [
    (488, 102),   # Signal above left
    (873, 100),   # Signal above right
    (877, 528),   # Signal below right
    (480, 528)    # Signal below left
]

# The get_current_state function compiles the current state of the environment into a 17-dimensional vector for the RL agent.
def get_current_state():
    # 1.match traffic_light_env.py (0:right, 1:down, 2:left, 3:up)
    directions = ['right', 'down', 'left', 'up']
    MAX_CARS_PER_LANE = 50.0

    # 2. Queue Lengths (12 dimensions) - Reconstructing 3 lanes per direction
    queue_norm = []
    for d in directions:
        # Count cars explicitly by their assigned Pygame lanes
        lane0_count = sum(1 for v in vehicles[d] if not v.crossed and v.lane == 0)
        lane1_count = sum(1 for v in vehicles[d] if not v.crossed and v.lane == 1)
        lane2_count = sum((1 for v in vehicles[d] if ((not v.crossed) and (v.lane == 2))))
        
        # Normalize between 0.0 and 1.0 to prevent neural network blowout
        queue_norm.extend([
            min(lane0_count / MAX_CARS_PER_LANE, 1.0),
            min(lane1_count / MAX_CARS_PER_LANE, 1.0),
            min(lane2_count / MAX_CARS_PER_LANE, 1.0)
        ])

    # 3. One-hot current green (4 dimensions)
    green_onehot = np.zeros(4, dtype=np.float32)
    green_onehot[currentGreen] = 1.0

    # 4. Time progress (1 dimension)
    # The training env uses current_step / max_steps. 
    # Since the UI runs indefinitely, passing a static neutral value or mock progress is required.
    time_progress = np.array([pygame.time.get_ticks() / 10000.0])

    # FINAL 17-D VECTOR (12 + 4 + 1)
    state = np.concatenate([
        np.array(queue_norm, dtype=np.float32),
        green_onehot,
        time_progress
    ])

    return state


def apply_action(action):
    global currentGreen, currentYellow

    if currentGreen != action:
        currentYellow = True

        currentGreen = action

        currentYellow = False

    return True


# UI DASHBOARD RENDERER
def draw_analytics_dashboard(surface):
    dash_x, dash_y = 20, 20
    dash_h = 300  
    dash_w = 280

    # Background overlay with rounded corners
    rounded_overlay = pygame.Surface((dash_w, dash_h), pygame.SRCALPHA)
    pygame.draw.rect(rounded_overlay, (20, 22, 28, 230), rounded_overlay.get_rect(), border_radius=12)
    pygame.draw.rect(rounded_overlay, (80, 85, 100, 255), rounded_overlay.get_rect(), 2, border_radius=12)
    surface.blit(rounded_overlay, (dash_x, dash_y))

    font_title = pygame.font.SysFont("segoeui", 22, bold=True)
    font_main = pygame.font.SysFont("segoeui", 16)
    font_bold = pygame.font.SysFont("segoeui", 16, bold=True)
    font_small = pygame.font.SysFont("segoeui", 14)

    title = font_title.render("Live Analytics", True, (240, 240, 240))
    surface.blit(title, (dash_x + 20, dash_y + 15))

    current_y = dash_y + 55
    spacing = 25

    # 0: right (origin LEFT), 1: down (origin TOP), 2: left (origin RIGHT), 3: up (origin BOTTOM)
    ui_phases = ['LEFT', 'UP', 'RIGHT', 'DOWN']
    active_dir = ui_phases[currentGreen]
    
    if currentYellow:
        phase_text = font_bold.render(f"PHASE: {active_dir} (YELLOW)", True, (255, 200, 50))
    else:
        phase_text = font_bold.render(f"PHASE: {active_dir} (GREEN)", True, (50, 255, 100))
    surface.blit(phase_text, (dash_x + 20, current_y))
    current_y += spacing

    # 2. Simulation Time
    elapsed_seconds = pygame.time.get_ticks() // 1000
    mins, secs = divmod(elapsed_seconds, 60)
    time_text = font_main.render(f"Simulation Time: {mins:02d}:{secs:02d}", True, (200, 200, 200))
    surface.blit(time_text, (dash_x + 20, current_y))
    current_y += spacing

    # 3. Total Throughput
    tp_text = font_main.render(f"Throughput: {throughput} vehicles", True, (200, 200, 200))
    surface.blit(tp_text, (dash_x + 20, current_y))
    current_y += spacing

    # 4. Max Wait Time
    max_wait = 0.0
    for d in vehicles:
        for v in vehicles[d]:
            if not v.crossed and v.wait_time > max_wait:
                max_wait = v.wait_time
    
    wait_color = (255, 100, 100) if max_wait > 30 else (200, 200, 200)
    wait_text = font_main.render(f"Max Wait Time: {max_wait:.1f} s", True, wait_color)
    surface.blit(wait_text, (dash_x + 20, current_y))
    
    current_y += spacing + 10 # Extra gap before queues

    # 5. Queue Lengths
    q_title = font_main.render("Queue Lengths:", True, (200, 200, 200))
    surface.blit(q_title, (dash_x + 20, current_y))
    current_y += 30
    
    # Map Visual Label to Internal Array Key
    queue_mapping = [
        ('UP', 'down'),     # Cars moving down originate at the top
        ('DOWN', 'up'),    # Cars moving up originate at the bottom
        ('LEFT', 'right'),   # Cars moving right originate at the left
        ('RIGHT', 'left')    # Cars moving left originate at the right
    ]
    
    max_bar_width = 100
    
    for i, (ui_label, internal_dir) in enumerate(queue_mapping):
        q_len = sum(1 for v in vehicles[internal_dir] if not v.crossed)
        
        # Draw UI Label
        lbl = font_small.render(ui_label, True, (150, 150, 150))
        surface.blit(lbl, (dash_x + 20, current_y + i * 22))
        
        # Bar background
        bar_y = current_y + 5 + i * 22
        pygame.draw.rect(surface, (50, 50, 60), (dash_x + 80, bar_y, max_bar_width, 10), border_radius=3)
        
        # Active Bar
        bar_w = min(q_len * 10, max_bar_width) 
        if bar_w > 0:
            r = min(255, 50 + (q_len * 20))
            g = max(50, 200 - (q_len * 15))
            bar_color = (r, g, 50)
            pygame.draw.rect(surface, bar_color, (dash_x + 80, bar_y, bar_w, 10), border_radius=3)
        
        # Count Text
        cnt = font_small.render(str(q_len), True, (240, 240, 240))
        surface.blit(cnt, (dash_x + 190, current_y + i * 22))

def start_simulation():
    """Call this once from main.py to kick off the signal thread."""
    # signal_thread = threading.Thread(target=updateSignals, daemon=True)
    # signal_thread.start()
    pass


# CALLABLE RENDER FUNCTIONS
def handle_spawn_event():
    """Called by main.py when SPAWN_VEHICLE_EVENT fires."""
    vtype = random.choice(vehicleTypes)
    lane = random.randint(0, 1)
    direction_number = random.randint(0, 3)
    direction_str = directionNumbers[direction_number]

    safe_to_spawn = True
    for v in vehicles[direction_str]:
        if v.lane == lane:
            if direction_str == 'right' and v.x < 180: safe_to_spawn = False
            elif direction_str == 'left' and v.x > WIDTH - 180: safe_to_spawn = False
            elif direction_str == 'down' and v.y < 180: safe_to_spawn = False
            elif direction_str == 'up' and v.y > HEIGHT - 180: safe_to_spawn = False

    if safe_to_spawn:
        Vehicle(lane, vtype, direction_number, direction_str)



def render_frame(screen, moving=True):
    """
    Called every frame by main.py.
    moving=False → draw background + frozen cars (used during MENU state)
    moving=True  → full simulation with car movement (used during RUNNING state)
    """
    global throughput, total_wait_time

    # 1. Draw background
    bg_rect = background.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(background, bg_rect)

    # 2. Draw signals
    for i in range(4):
        if i == currentGreen:
            if currentYellow:
                screen.blit(yellowSignal, signalCoods[i])
            else:
                screen.blit(greenSignal, signalCoods[i])
        else:
            screen.blit(redSignal, signalCoods[i])

    # 3. Draw and optionally move vehicles
    for direction in vehicles:
        for vehicle in vehicles[direction][:]:
            screen.blit(vehicle.image, (vehicle.x, vehicle.y))

            if moving:
                vehicle.move()

                # Remove vehicles that have exited the screen
                if (vehicle.x > WIDTH + 200 or vehicle.x < -200 or
                        vehicle.y > HEIGHT + 200 or vehicle.y < -200):
                    throughput += 1
                    total_wait_time += vehicle.wait_time
                    vehicles[direction].remove(vehicle)

    draw_analytics_dashboard(screen)