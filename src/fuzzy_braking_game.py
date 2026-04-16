import pygame
import numpy as np
import skfuzzy as fuzz
import sys

pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fuzzy Braking System")
font = pygame.font.SysFont("Arial", 16)
title_font = pygame.font.SysFont("Arial", 24, bold=True)

WHITE = (255, 255, 255)
RED = (200, 50, 50)
BLUE = (50, 50, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARKGRAY = (150, 150, 150)
GREEN = (50, 200, 50)

# --- Fuzzy Memberships ---
rel_speed = np.arange(-150, 151, 1)
distance = np.arange(0, 201, 1)
brake_force = np.arange(0, 2501, 1)

rel_speed_neg = fuzz.trimf(rel_speed, [-150, -50, 0])
rel_speed_zero = fuzz.trimf(rel_speed, [-50, 0, 50])
rel_speed_pos = fuzz.trimf(rel_speed, [0, 50, 150])

distance_close = fuzz.trimf(distance, [0, 0, 50])
distance_medium = fuzz.trimf(distance, [50, 150, 250])
distance_far = fuzz.trimf(distance, [150, 300, 300])

brake_light = fuzz.trimf(brake_force, [0, 0, 1000])
brake_moderate = fuzz.trimf(brake_force, [500, 1250, 2000])
brake_hard = fuzz.trimf(brake_force, [1500, 2500, 2500])

def compute_brake_force(rel_speed_val, distance_val):
    rs_neg = fuzz.interp_membership(rel_speed, rel_speed_neg, rel_speed_val)
    rs_zero = fuzz.interp_membership(rel_speed, rel_speed_zero, rel_speed_val)
    rs_pos = fuzz.interp_membership(rel_speed, rel_speed_pos, rel_speed_val)

    d_close = fuzz.interp_membership(distance, distance_close, distance_val)
    d_med = fuzz.interp_membership(distance, distance_medium, distance_val)
    d_far = fuzz.interp_membership(distance, distance_far, distance_val)

    print(f"RS: neg={rs_neg}, zero={rs_zero}, pos={rs_pos}")
    print(f"DIST: close={d_close}, med={d_med}, far={d_far}")

    rule1 = np.fmin(d_close, rs_pos)
    rule2 = np.fmin(d_med, rs_pos)
    rule3 = np.fmin(d_far, rs_pos)
    rule4 = np.fmin(d_close, rs_zero)
    rule5 = np.fmin(d_far, rs_neg)

    brake_activation_hard = np.fmin(rule1, brake_hard)
    brake_activation_moderate = np.fmax(np.fmin(rule2, brake_moderate), np.fmin(rule4, brake_moderate))
    brake_activation_light = np.fmax(np.fmin(rule3, brake_light), np.fmin(rule5, brake_light))

    # Aggregate all output membership functions
    aggregated = np.fmax(brake_activation_hard,
                  np.fmax(brake_activation_moderate,
                          brake_activation_light))

    # Ensure we avoid empty aggregation
    if np.all(aggregated == 0):
        print(f"[WARNING] Empty aggregated MF for rel_speed={rel_speed_val}, distance={distance_val}")
        return 0.0
    
    return fuzz.defuzz(brake_force, aggregated, 'centroid')

# --- Simulation Class ---
class BrakingSim:
    def __init__(self, p0, v0, pB0, vB, m=100):
        self.p, self.v = p0, v0
        self.pB, self.vB = pB0, vB
        self.m = m
        self.dt = 0.05
        self.scale = 4
        self.running = True
        self.paused = False
        self.history = {'time': [], 'v': [], 'vB': [], 'dist': []}
        self.t = 0

    def step(self):
        rel_speed_val = self.v - self.vB
        distance_val = max(self.pB - self.p, 0)
        fb = compute_brake_force(rel_speed_val, distance_val)
        a = fb / self.m

        self.v -= a * self.dt
        self.p += self.v * self.dt
        self.pB += self.vB * self.dt

        # Store history
        self.t += self.dt
        self.history['time'].append(self.t)
        self.history['v'].append(self.v)
        self.history['vB'].append(self.vB)
        self.history['dist'].append(distance_val)

        return self.v, self.vB, self.p, self.pB, distance_val, fb

    def is_finished(self):
        return self.pB - self.p < 1 or self.p * self.scale > WIDTH or self.pB * self.scale > WIDTH

# --- Buttons ---
def draw_button(rect, text, hover=False, color=GRAY):
    pygame.draw.rect(screen, DARKGRAY if hover else color, rect)
    label = font.render(text, True, BLACK)
    screen.blit(label, (rect.x + 10, rect.y + 10))

# --- Graph ---
def draw_graph(history):
    graph_rect = pygame.Rect(600, 10, 380, 150)
    pygame.draw.rect(screen, (245, 245, 245), graph_rect)
    if len(history['time']) < 2:
        return

    max_points = 150
    t_norm = history['time'][-max_points:]
    v = history['v'][-max_points:]
    vB = history['vB'][-max_points:]
    d = history['dist'][-max_points:]

    scale_x = graph_rect.width / len(t_norm)
    scale_y = 1.2

    def normalize(data, offset=0):
        return [graph_rect.bottom - int(y / scale_y) - offset for y in data]

    v_pts = list(zip([graph_rect.x + int(i * scale_x) for i in range(len(v))], normalize(v)))
    vB_pts = list(zip([graph_rect.x + int(i * scale_x) for i in range(len(vB))], normalize(vB, 10)))
    d_pts = list(zip([graph_rect.x + int(i * scale_x) for i in range(len(d))], normalize(d, 20)))

    if len(v_pts) >= 2:
        pygame.draw.lines(screen, BLUE, False, v_pts, 2)
        pygame.draw.lines(screen, RED, False, vB_pts, 2)
        pygame.draw.lines(screen, GREEN, False, d_pts, 2)

    labels = ["Ego v", "Lead v", "Distance"]
    for i, text in enumerate(labels):
        label = font.render(text, True, [BLUE, RED, GREEN][i])
        screen.blit(label, (graph_rect.x + 5 + i * 80, graph_rect.y + 5))

# --- Run Simulation ---
def run_simulation(p0, v0, pB0, vB):
    clock = pygame.time.Clock()
    sim = BrakingSim(p0, v0, pB0, vB)

    replay_btn = pygame.Rect(400, 500, 100, 40)
    back_btn = pygame.Rect(520, 500, 120, 40)
    ended = False

    while True:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                sim.paused = not sim.paused
            if event.type == pygame.MOUSEBUTTONDOWN and ended:
                if replay_btn.collidepoint(event.pos):
                    return 'replay'
                if back_btn.collidepoint(event.pos):
                    return 'menu'

        if not sim.paused and not ended:
            v, vB, p, pB, dist, fb = sim.step()

        # Draw road and cars
        pygame.draw.line(screen, BLACK, (0, HEIGHT//2 + 20), (WIDTH, HEIGHT//2 + 20), 2)
        pygame.draw.rect(screen, BLUE, (int(sim.p * sim.scale), HEIGHT//2, 40, 20))
        pygame.draw.rect(screen, RED, (int(sim.pB * sim.scale), HEIGHT//2, 40, 20))

        # Display info
        info = [
            f"Ego Speed: {sim.v:.1f} ft/s",
            f"Lead Speed: {sim.vB:.1f} ft/s",
            f"Distance: {max(sim.pB - sim.p, 0):.1f} ft",
        ]
        for i, text in enumerate(info):
            label = font.render(text, True, BLACK)
            screen.blit(label, (10, 10 + i * 20))

        draw_graph(sim.history)

        if sim.is_finished():
            ended = True
            draw_button(replay_btn, "Replay", replay_btn.collidepoint(pygame.mouse.get_pos()))
            draw_button(back_btn, "Back to Menu", back_btn.collidepoint(pygame.mouse.get_pos()))

        pygame.display.flip()
        clock.tick(60)

# --- Menu ---
def main_menu():
    buttons = [
        (pygame.Rect(100, 200, 250, 40), "Scenario 1: Ego 45 ft/s", (0, 45, 45, 50)),
        (pygame.Rect(100, 260, 250, 40), "Scenario 2: Ego 75 ft/s", (0, 75, 45, 50)),
        (pygame.Rect(100, 320, 250, 40), "Scenario 3: Ego 100 ft/s", (0, 100, 45, 50)),
    ]

    while True:
        screen.fill(WHITE)
        title = title_font.render("Select a Braking Scenario", True, BLACK)
        screen.blit(title, (100, 100))

        for rect, text, _ in buttons:
            hover = rect.collidepoint(pygame.mouse.get_pos())
            draw_button(rect, text, hover)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for rect, _, params in buttons:
                    if rect.collidepoint(event.pos):
                        result = run_simulation(*params)
                        while result == 'replay':
                            result = run_simulation(*params)

        pygame.display.flip()
        pygame.time.Clock().tick(30)

# 🚀 Launch Menu
main_menu()