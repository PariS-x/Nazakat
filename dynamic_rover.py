import math

MARS_GRAVITY = 3.71


class DynamicRover:
    def __init__(self,
                 mass,
                 wheel_base,
                 track_width,
                 body_height):

        self.mass = mass
        self.wheel_base = wheel_base
        self.track_width = track_width
        self.body_height = body_height
        self.terrain = None

        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.wheel_radius = 0.25  # meters

        self.battery_capacity_wh = 2000
        self.battery_remaining_wh = 2000
        self.sensor_power_draw = 0

        self.slip_ratio = 0.0
        
        self.wheel_normals = [0]*6
        self.wheel_traction_limits = [0]*6

        self.max_drive_force = 0.0
        self.max_speed = 12.0   # optional, future use

        self.roll = 0.0
        self.roll_velocity = 0.0
        self.time = 0.0
        self.motor_power = 100

        self.slope_roll = 0.0
        self.slope_pitch = 0.0

        self.roll_tipping = False
        self.roll_finished = False

        # --- Longitudinal motion ---
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0  # heading angle (radians)
        self.velocity = 0.0

        self.drive_force = 0.0
        self.friction_coeff = 0.7   # Mars regolith approx
        self.max_speed = 8.0   # m/s

        # dimensions
        self.h = body_height / 4
        self.b = track_width

        self.motor_power = 500  # Watts (realistic small rover scale)
        self.max_drive_force = 900
        self.turn_radius = None

        self.crashed = False
        self.slide_friction_coeff = 0.9   # metal scraping regolith

    # --------------------------------------------------
    # Moment of inertia (rectangular body approx)
    # --------------------------------------------------

    def compute_inertia_roll(self):
        return (1/12) * self.mass * (self.body_height**2 + self.track_width**2)

    # --------------------------------------------------
    # Set terrain
    # --------------------------------------------------
    def set_drive_force(self, force):
        self.drive_force = force

    def set_turn_radius(self, radius):
        self.turn_radius = radius

    def set_terrain_profile(self, elevation_profile, dx):
        self.terrain = elevation_profile
        self.dx = dx
        self.min_h = elevation_profile.min()
        self.max_h = elevation_profile.max()

    def update_longitudinal_dynamics(self, dt):

        # If rover has crashed, apply sliding friction only
        if self.crashed:

            N = self.mass * MARS_GRAVITY
            F_friction = self.slide_friction_coeff * N

            if abs(self.velocity) < 0.2:
                self.velocity = 0
                return

            direction = -1 if self.velocity > 0 else 1
            a = (direction * F_friction) / self.mass

            self.velocity += a * dt

            # Prevent overshoot
            if (self.velocity > 0 and direction > 0) or \
            (self.velocity < 0 and direction < 0):
                self.velocity = 0

            self.x += self.velocity * math.cos(self.yaw) * dt
            self.y += self.velocity * math.sin(self.yaw) * dt
            return
        
        # Gravity along slope
        F_gravity = self.mass * MARS_GRAVITY * math.sin(self.slope_pitch)

        # Normal force
        N = self.mass * MARS_GRAVITY * math.cos(self.slope_pitch)

        C_rr = 0.02   # rolling resistance coefficient (Mars regolith)
        F_rr = C_rr * N * (-1 if self.velocity > 0 else 1 if self.velocity < 0 else 0)
        # Max traction
        # --- Terrain-dependent friction ---
        local_mu = self.get_local_friction()

        # --- 6 wheel contact model ---

        N_left, N_right = self.compute_normal_forces()

        # Split each side across 3 wheels
        N_left_wheel  = max(0, N_left)  / 3
        N_right_wheel = max(0, N_right) / 3

        # Friction limit per wheel
        F_left_wheel  = local_mu * N_left_wheel
        F_right_wheel = local_mu * N_right_wheel

        # Total traction capacity
        F_max_total = 3 * F_left_wheel + 3 * F_right_wheel


        # Store per-wheel data for visualization
        self.wheel_normals = [
            N_left_wheel, N_left_wheel, N_left_wheel,
            N_right_wheel, N_right_wheel, N_right_wheel
        ]

        self.wheel_traction_limits = [
            F_left_wheel, F_left_wheel, F_left_wheel,
            F_right_wheel, F_right_wheel, F_right_wheel
        ]

        if self.turn_radius and self.turn_radius != 0:
            # crude approximation of lateral load effect
            lateral_factor = min(0.3, abs(self.velocity) / 10)
            F_max_total *= (1 - lateral_factor)

        # -----------------------------
        # Proper motor model
        # -----------------------------

        v = abs(self.velocity)
        direction = math.copysign(1, self.drive_force) if self.drive_force != 0 else 0

        v_transition = self.motor_power / self.max_drive_force

        if v < v_transition:
            motor_force = self.max_drive_force * direction
        else:
            motor_force = (self.motor_power / v) * direction

        # -----------------------------
        # Gear-limited max speed
        # -----------------------------
        if v >= self.max_speed:
            motor_force = 0

        power_limited_force = motor_force

        requested_force = motor_force

        if F_max_total > 0:
            slip_ratio = max(0.0, abs(requested_force) / F_max_total - 1)
        else:
            slip_ratio = 1.0

        self.slip_ratio = slip_ratio

        applied_drive = max(-F_max_total, min(requested_force, F_max_total))

        self.slip_ratio = slip_ratio


        rho = 0.02      # Mars air density kg/m^3
        Cd = 1.6
        A = 2.5   # rover body + mast

        F_drag = 0.5 * rho * Cd * A * self.velocity**2
        F_drag *= -1 if self.velocity > 0 else 1

        # Net force
        F_net = applied_drive - F_gravity + F_rr + F_drag
        # Acceleration
        a = F_net / self.mass

        # Integrate
        self.velocity += a * dt
        self.x += self.velocity * math.cos(self.yaw) * dt
        self.y += self.velocity * math.sin(self.yaw) * dt

        # -----------------------------------
        # Energy consumption
        # -----------------------------------

        motor_power_used = abs(applied_drive * self.velocity)
        total_power = motor_power_used + self.sensor_power_draw  # Watts

        energy_used_wh = (total_power * dt) / 3600  # convert to Wh
        self.battery_remaining_wh -= energy_used_wh

        if self.battery_remaining_wh <= 0:
            print("Battery depleted")
            self.drive_force = 0
            self.velocity = 0

    def set_slope(self, pitch_deg, roll_deg):
        self.slope_pitch = math.radians(pitch_deg)
        self.slope_roll = math.radians(roll_deg)


    def compute_terrain_gradients(self):

        if self.terrain is None:
            return 0, 0

        # Convert meters → floating indices
        i_f = self.y / self.dx
        j_f = self.x / self.dx

        i = int(i_f)
        j = int(j_f)

        if i < 1 or j < 1 or \
        i >= self.terrain.shape[0]-2 or \
        j >= self.terrain.shape[1]-2:
            return 0, 0

        # Bilinear interpolation neighbors
        z00 = self.terrain[i, j]
        z10 = self.terrain[i+1, j]
        z01 = self.terrain[i, j+1]
        z11 = self.terrain[i+1, j+1]

        di = i_f - i
        dj = j_f - j

        # Interpolated height
        z = (
            z00 * (1-di)*(1-dj) +
            z10 * di*(1-dj) +
            z01 * (1-di)*dj +
            z11 * di*dj
        )

        # Finite difference using interpolated neighbors
        dz_dx = (self.terrain[i, j+1] - self.terrain[i, j-1]) / (2*self.dx)
        dz_dy = (self.terrain[i+1, j] - self.terrain[i-1, j]) / (2*self.dx)

        forward_slope = dz_dx * math.cos(self.yaw) + dz_dy * math.sin(self.yaw)
        lateral_slope = -dz_dx * math.sin(self.yaw) + dz_dy * math.cos(self.yaw)

        pitch = math.atan(forward_slope)
        roll = math.atan(lateral_slope)

        return pitch, roll

    # --------------------------------------------------
    # Normal force calculation
    # --------------------------------------------------

    def compute_normal_forces(self):

        mg = self.mass * MARS_GRAVITY
        theta = self.slope_roll

        a_lat = 0

        if self.turn_radius and self.turn_radius != 0:
            a_lat = (self.velocity ** 2) / self.turn_radius

        lateral_force = self.mass * a_lat

        terrain_transfer = mg * self.h * math.tan(theta) / self.b
        dynamic_transfer = lateral_force * self.h / self.b

        N_left = (mg / 2) - terrain_transfer - dynamic_transfer
        N_right = mg - N_left

        return N_left, N_right

    # --------------------------------------------------
    # Simulation step
    # --------------------------------------------------

    def step(self, dt):
        
        # Longitudinal motion
        self.update_longitudinal_dynamics(dt)
        self.update_differential_kinematics(dt)

        self.time += dt
        t = self.time

        # update dynamic terrain
        self.slope_pitch, self.slope_roll = self.compute_terrain_gradients()

        if not self.roll_tipping and not self.roll_finished:

            N_left, N_right = self.compute_normal_forces()

            if N_left <= 0 or N_right <= 0:
                print("Wheel lift-off detected → dynamic rollover begins")
                self.roll_tipping = True
            else:
                suspension_stiffness = 8.0
                self.roll += suspension_stiffness * (self.slope_roll - self.roll) * dt
                self.roll_velocity = 0

        if self.roll_tipping and not self.roll_finished:
            self.update_roll_dynamics(dt)

        N_left, N_right = self.compute_normal_forces()

        return {
            "time": round(self.time, 2),
            "position": (round(self.x, 2), round(self.y, 2)),
            "velocity": round(self.velocity, 2),
            "roll_deg": round(math.degrees(self.roll), 2),
            "pitch_deg": round(math.degrees(self.slope_pitch), 2)
        }
    # --------------------------------------------------
    # Roll dynamics after lift-off
    # --------------------------------------------------

    def update_roll_dynamics(self, dt):

        SIDE_ANGLE = math.pi / 2
        I = self.compute_inertia_roll()

        # COM relative to pivot
        y = self.track_width / 2
        z = self.body_height / 2

        # Rotated COM horizontal distance
        y_rot = y * math.cos(self.roll) - z * math.sin(self.roll)

        k_roll = 2000  # Nm/rad
        torque = self.mass * MARS_GRAVITY * y_rot - k_roll * self.roll

        damping = 20
        alpha = (torque - damping * self.roll_velocity) / I

        self.roll_velocity += alpha * dt
        self.roll += self.roll_velocity * dt
        self.roll = max(-math.pi/2, min(math.pi/2, self.roll))

        # -----------------------------
        # Irreversible rollover logic
        # -----------------------------

        critical_angle = math.atan(self.b / self.h)

        if abs(self.roll) > critical_angle:

            if (self.roll > 0 and self.roll_velocity > 0) or \
            (self.roll < 0 and self.roll_velocity < 0):

                print("Irreversible rollover committed")

                self.roll = math.copysign(SIDE_ANGLE, self.roll)
                self.roll_velocity = 0
                self.roll_tipping = False
                self.roll_finished = True

                self.crashed = True
                self.drive_force = 0   # no more powered motion
                return

        # -----------------------------
        # Ground impact hard stop
        # -----------------------------
        if abs(self.roll) >= SIDE_ANGLE:
            self.roll = math.copysign(SIDE_ANGLE, self.roll)
            self.roll_velocity = 0
            self.roll_tipping = False
            self.roll_finished = True

    def get_local_friction(self):

        if self.terrain is None:
            return self.friction_coeff

        i = int(self.y / self.dx)
        j = int(self.x / self.dx)

        if i < 0 or j < 0 or \
        i >= self.terrain.shape[0] or \
        j >= self.terrain.shape[1]:
            return self.friction_coeff

        height = self.terrain[i, j]

        # Normalize terrain height
        norm = (height - self.min_h) / (self.max_h - self.min_h)

        # Map elevation to friction
        mu_min = 0.3
        mu_max = 0.8

        return mu_min + norm * (mu_max - mu_min)
    
    def update_differential_kinematics(self, dt):

        # Linear velocity
        v = self.velocity

        # Angular velocity
        yaw_rate = (self.right_wheel_speed - self.left_wheel_speed) / self.track_width

        self.yaw += yaw_rate * dt

        self.x += self.velocity * math.cos(self.yaw) * dt
        self.y += self.velocity * math.sin(self.yaw) * dt