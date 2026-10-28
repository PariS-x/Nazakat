import pygame
import math
import numpy as np
from dynamic_rover import DynamicRover
from mars_loader import MarsTerrain

# -----------------------------
# CONFIG
# -----------------------------
WIDTH = 1200
HEIGHT = 700
FPS = 60

SUN_DIR = np.array([0.6, 0.6])   # light coming from top-right
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)

TOP_VIEW_WIDTH = 800
SIDE_VIEW_WIDTH = 400

SCALE = 3  # meters → pixels (reduced for stability)
DRAW_RADIUS = 120  # number of terrain cells around rover to draw

terrain_overlay_mode = False

dust_particles = []
# -----------------------------
# INIT
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mars Rover Simulator")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 16)

# -----------------------------
# LOAD TERRAIN + ROVER
# -----------------------------
terrain = MarsTerrain("mars_terrain/megt00n000hb.img")

min_h = np.min(terrain.elevation)
max_h = np.max(terrain.elevation)

rover = DynamicRover(
    mass=200,
    wheel_base=1.2,
    track_width=1.0,
    body_height=0.8
)

rover.set_terrain_profile(terrain.elevation, dx=10)

# -----------------------------
# DRAW FUNCTIONS
# -----------------------------

def draw_top_view():

    camera_x = rover.x
    camera_y = rover.y

    center_i = int(camera_y / rover.dx)
    center_j = int(camera_x / rover.dx)

    # Draw limited region around rover
    for i in range(center_i - DRAW_RADIUS, center_i + DRAW_RADIUS):
        for j in range(center_j - DRAW_RADIUS, center_j + DRAW_RADIUS):

            if i < 0 or j < 0 or \
               i >= terrain.elevation.shape[0] or \
               j >= terrain.elevation.shape[1]:
                continue

            height = terrain.elevation[i, j]

            # Normalize elevation
            norm = (height - min_h) / (max_h - min_h)

            # --- Compute gradient ---
            dz_dx = 0
            dz_dy = 0

            if 1 <= i < terrain.elevation.shape[0]-1 and \
            1 <= j < terrain.elevation.shape[1]-1:

                dz_dx = (terrain.elevation[i, j+1] - terrain.elevation[i, j-1]) / (2 * rover.dx)
                dz_dy = (terrain.elevation[i+1, j] - terrain.elevation[i-1, j]) / (2 * rover.dx)

            gradient = np.array([-dz_dx, -dz_dy])
            grad_mag = np.linalg.norm(gradient)

            if grad_mag != 0:
                gradient = gradient / grad_mag

            # --- Light intensity ---
            light = np.dot(gradient, SUN_DIR)
            light = max(0, light)

            base_shade = 50 + norm * 100
            shade = base_shade * (0.6 + 0.8 * light)

            shade = int(max(30, min(200, shade)))

            if terrain_overlay_mode:

                slope_mag = math.sqrt(dz_dx**2 + dz_dy**2)

                # Scale slope to 0–1 range (tune factor as needed)
                slope_norm = min(1.0, slope_mag * 5)

                # Red = steep/difficult
                # Green = flat/safe
                red = int(255 * slope_norm)
                green = int(255 * (1 - slope_norm))

                color = (red, green, 0)

            else:
                color = (
                    shade,
                    int(shade * 0.45),
                    int(shade * 0.25)
                )


            world_x = j * rover.dx
            world_y = i * rover.dx

            screen_x = (world_x - camera_x) * SCALE + TOP_VIEW_WIDTH // 2
            screen_y = (world_y - camera_y) * SCALE + HEIGHT // 2

            cell_size = int(SCALE * rover.dx)

            pygame.draw.rect(
                screen,
                color,
                (
                    int(screen_x),
                    int(screen_y),
                    cell_size,
                    cell_size
                )
            )

    # --- Rover always centered ---
    rover_x = TOP_VIEW_WIDTH // 2
    rover_y = HEIGHT // 2

    rover_surface = pygame.Surface((70, 50), pygame.SRCALPHA)

    # Body
    pygame.draw.rect(rover_surface, (230,230,240), (15,15,40,20))

    # 6 wheels (3 per side)
    wheel_positions = [
        (10, 10), (10, 25), (10, 40),   # left side
        (60, 10), (60, 25), (60, 40)    # right side
    ]

    for i, (wx, wy) in enumerate(wheel_positions):

        load = rover.wheel_normals[i]

        # Normalize load visually
        load_ratio = min(1.0, load / (rover.mass * 3.71 / 6))

        # Color changes with load
        color = (
            int(80 + 150 * load_ratio),
            int(30 + 100 * load_ratio),
            30
        )

        pygame.draw.circle(rover_surface, color, (wx, wy), 6)

    rotated = pygame.transform.rotate(rover_surface, -math.degrees(rover.yaw))
    rect = rotated.get_rect(center=(rover_x, rover_y))
    screen.blit(rotated, rect)

    if rover.slip_ratio > 0.2:
        color = (255,50,50)


def draw_side_view():

    pygame.draw.rect(screen, (15, 15, 30),
                     (TOP_VIEW_WIDTH, 0, SIDE_VIEW_WIDTH, HEIGHT))

    ground_y = HEIGHT // 2 + 80

    pygame.draw.line(
        screen,
        (150, 100, 50),
        (TOP_VIEW_WIDTH, ground_y),
        (WIDTH, ground_y),
        3
    )

    base_x = TOP_VIEW_WIDTH + SIDE_VIEW_WIDTH // 2
    base_y = ground_y - 40

    rover_surface = pygame.Surface((350, 200), pygame.SRCALPHA)

    center_x = 175
    center_y = 100

    body_width = 220
    body_height = 50

    # --- Draw body ---
    body_rect = pygame.Rect(
        center_x - body_width//2,
        center_y - body_height//2,
        body_width,
        body_height
    )
    pygame.draw.rect(rover_surface, (200, 200, 220), body_rect)

    # --- Draw 3 wheels in horizontal line ---
    wheel_radius = 20
    wheel_spacing = 90

    for i in range(3):

        wx = center_x - wheel_spacing + i * wheel_spacing
        wy = center_y + body_height//2 + wheel_radius - 5

        load = rover.wheel_normals[i]  # just use left side for visual
        load_ratio = min(1.0, load / (rover.mass * 3.71 / 6))

        color = (
            int(80 + 150 * load_ratio),
            40,
            30
        )

        pygame.draw.circle(rover_surface, color,
                           (int(wx), int(wy)), wheel_radius)

    # --- Rotate entire rover by pitch ---
    rotated = pygame.transform.rotate(
        rover_surface,
        -math.degrees(rover.slope_pitch)
    )

    rect = rotated.get_rect(center=(base_x, base_y))

    screen.blit(rotated, rect)


def draw_telemetry():

    panel_width = 300
    panel_height = 220

    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((20, 20, 20, 180))  # translucent dark

    screen.blit(panel, (10, 10))

    N_left, N_right = rover.compute_normal_forces()

    data = [
        "=== SYSTEM STATUS ===",
        f"Velocity: {rover.velocity:.2f} m/s",
        f"Battery: {rover.battery_remaining_wh:.1f} Wh",
        "",
        "=== TERRAIN ===",
        f"Pitch: {math.degrees(rover.slope_pitch):.2f} deg",
        f"Roll: {math.degrees(rover.roll):.2f} deg",
        f"Friction (μ): {rover.get_local_friction():.2f}",
        "",
        "=== LOAD ===",
        f"N Left: {N_left:.1f} N",
        f"N Right: {N_right:.1f} N",
        "",
        f"Slip Ratio: {rover.slip_ratio:.2f}"
    ]

    for i, line in enumerate(data):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (25, 20 + i * 18))

def draw_speedometer():

    center = (160, HEIGHT - 120)
    radius = 70

    pygame.draw.circle(screen, (40,40,40), center, radius)
    pygame.draw.circle(screen, (150,150,150), center, radius, 3)

    max_speed = 8
    speed_ratio = min(1, abs(rover.velocity)/max_speed)

    angle = -math.pi/2 + speed_ratio * math.pi

    needle_x = center[0] + radius * 0.8 * math.cos(angle)
    needle_y = center[1] + radius * 0.8 * math.sin(angle)

    pygame.draw.line(screen, (255,0,0), center, (needle_x, needle_y), 3)

    label = font.render("Speed", True, (255,255,255))
    screen.blit(label, (center[0]-30, center[1]+radius+5))

def update_dust():

    # --- Spawn per-wheel dust ---
    if rover.slip_ratio > 0.2 and abs(rover.velocity) > 0.5:

        wheel_offsets = [
            (-20,-15), (-20,0), (-20,15),
            (20,-15), (20,0), (20,15)
        ]

        for i in range(6):

            # Only spawn dust from wheels that have contact
            if rover.wheel_normals[i] > 0:

                wx, wy = wheel_offsets[i]

                angle = rover.yaw + math.pi
                speed = np.random.uniform(1.0, 3.0)

                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed

                dust_particles.append({
                    "x": TOP_VIEW_WIDTH//2 + wx,
                    "y": HEIGHT//2 + wy,
                    "vx": vx,
                    "vy": vy,
                    "life": 25
                })

    # --- Update particles ---
    for p in dust_particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1

    dust_particles[:] = [p for p in dust_particles if p["life"] > 0]

def draw_dust():
    for p in dust_particles:
        alpha = int(255 * (p["life"]/30))
        surf = pygame.Surface((6,6), pygame.SRCALPHA)
        pygame.draw.circle(surf, (200,150,100,alpha), (3,3), 3)
        screen.blit(surf, (p["x"], p["y"]))

def draw_battery_bar():

    bar_x = WIDTH - 250
    bar_y = 30
    bar_width = 200
    bar_height = 25

    pygame.draw.rect(screen, (100,100,100),
                     (bar_x, bar_y, bar_width, bar_height), 2)

    ratio = rover.battery_remaining_wh / rover.battery_capacity_wh
    fill_width = int(bar_width * ratio)

    color = (0,200,0)
    if ratio < 0.3:
        color = (200,200,0)
    if ratio < 0.1:
        color = (200,0,0)

    pygame.draw.rect(screen, color,
                     (bar_x, bar_y, fill_width, bar_height))

    label = font.render("Battery", True, (255,255,255))
    screen.blit(label, (bar_x, bar_y - 20))

# MAIN LOOP

running = True
dt = 1 / FPS

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                terrain_overlay_mode = not terrain_overlay_mode

    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        rover.set_drive_force(250)
    elif keys[pygame.K_DOWN]:
        rover.set_drive_force(-250)
    else:
        rover.set_drive_force(0)

    base_speed = 2.0

    forward = 0
    turn = 0

    if keys[pygame.K_UP]:
        forward += 1
    if keys[pygame.K_DOWN]:
        forward -= 1
    if keys[pygame.K_LEFT]:
        turn -= 1
    if keys[pygame.K_RIGHT]:
        turn += 1


    # Differential mixing
    rover.left_wheel_speed  = base_speed * (forward - turn)
    rover.right_wheel_speed = base_speed * (forward + turn)

    rover.step(dt)

    update_dust()

    screen.fill((0, 0, 0))

    draw_top_view()
    draw_dust()
    draw_side_view()
    draw_telemetry()
    draw_speedometer()
    draw_battery_bar()

    pygame.display.flip()

pygame.quit()