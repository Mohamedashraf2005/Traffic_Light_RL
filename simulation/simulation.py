import pygame
import sys
import os
import random
import time
import threading

# =====================
# PATHS
# =====================
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# =====================
# SCREEN & CLOCK
# =====================
WIDTH, HEIGHT = 1400, 800
pygame.init()
pygame.font.init()  # 🔠 تفعيل الخطوط للعداد
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SIMULATION")

clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 30, bold=False)
# =====================
# LOAD & SCALE ASSETS
# =====================
background = pygame.image.load(
    os.path.join(PROJECT_ROOT, "images", "intersection.png")
).convert()

SIGNAL_SIZE = (25, 65) 

redSignal = pygame.image.load(os.path.join(PROJECT_ROOT, "images", "signals", "red.png")).convert_alpha()
redSignal = pygame.transform.smoothscale(redSignal, SIGNAL_SIZE)

yellowSignal = pygame.image.load(os.path.join(PROJECT_ROOT, "images", "signals", "yellow.png")).convert_alpha()
yellowSignal = pygame.transform.smoothscale(yellowSignal, SIGNAL_SIZE)

greenSignal = pygame.image.load(os.path.join(PROJECT_ROOT, "images", "signals", "green.png")).convert_alpha()
greenSignal = pygame.transform.smoothscale(greenSignal, SIGNAL_SIZE)

# =====================
# VEHICLE CONFIG
# =====================
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

# =====================
# SIGNAL SYSTEM & TIMERS
# =====================
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
time_left = 0  # متغير الثواني

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

threading.Thread(target=updateSignals, daemon=True).start()

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

# =====================
# VEHICLE CLASS
# =====================
class Vehicle:
    def __init__(self, lane, vehicleClass, direction_number, direction):
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.direction = direction
        self.direction_number = direction_number
        self.speed = speeds[vehicleClass]

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
        if not self.is_front_clear():
            return

        if not self.crossed:
            if self.direction_number != currentGreen or currentYellow:
                if self.direction == 'right' and self.x + self.image.get_width() >= stopLines['right']:
                    return
                if self.direction == 'left' and self.x <= stopLines['left']:
                    return
                if self.direction == 'down' and self.y + self.image.get_height() >= stopLines['down']:
                    return
                if self.direction == 'up' and self.y <= stopLines['up']:
                    return

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

# =====================
# GENERATE VEHICLES EVENT
# =====================
SPAWN_VEHICLE_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWN_VEHICLE_EVENT, 1200) 

# =====================
# SIGNAL & TIMER POSITIONS 
# =====================
signalCoods = [
    (488, 102),   # إشارة فوق شمال
    (873, 100),   # إشارة فوق يمين
    (877, 528),   # إشارة تحت يمين
    (480, 528)    # إشارة تحت شمال
]

# إحداثيات العدادات متظبطة عشان تبقى جمب إحداثياتك بالظبط
timerCoods = [
    (443, 109),   # عداد فوق شمال
    (903, 107),   # عداد فوق يمين
    (907, 535),   # عداد تحت يمين
    (435, 535)    # عداد تحت شمال  # عداد تحت شمال
]

# =====================
# MAIN LOOP
# =====================
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            print(f"📍 إحداثيات الضغطة: ({mouse_x}, {mouse_y})")

        if event.type == SPAWN_VEHICLE_EVENT:
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

    bg_rect = background.get_rect(center=(WIDTH//2, HEIGHT//2))
    screen.blit(background, bg_rect)

    # 🚦 رسم الإشارات والعدادات
    for i in range(4):
        if i == currentGreen:
            if currentYellow:
                screen.blit(yellowSignal, signalCoods[i])
                text_color = (255, 255, 0)
            else:
                screen.blit(greenSignal, signalCoods[i])
                text_color = (0, 255, 0)
        else:
            screen.blit(redSignal, signalCoods[i])
            text_color = (255, 0, 0)

        # رسم العداد
        timer_rect = pygame.Rect(timerCoods[i][0], timerCoods[i][1], 40, 50)
        pygame.draw.rect(screen, (20, 20, 20), timer_rect) 
        pygame.draw.rect(screen, (100, 100, 100), timer_rect, 2) 
        
        current_timer_val = get_time_left(i)
        timer_text = font.render(str(current_timer_val), True, text_color)
        text_rect = timer_text.get_rect(center=timer_rect.center)
        screen.blit(timer_text, text_rect)

    for direction in vehicles:
        for vehicle in vehicles[direction][:]:
            screen.blit(vehicle.image, (vehicle.x, vehicle.y))
            vehicle.move()
            
            if vehicle.x > WIDTH + 200 or vehicle.x < -200 or vehicle.y > HEIGHT + 200 or vehicle.y < -200:
                vehicles[direction].remove(vehicle)

    pygame.display.update()
    clock.tick(60)