from dynamic_rover import DynamicRover
from mars_loader import MarsTerrain
from rover_components import *
from simulation_engine import SimulationEngine
from controller import ConstantThrottleController

builder = RoverBuilder()

rover_config = (
    builder
    .add_chassis(Chassis("Aluminum", 120, width=1.0, length=1.2, height=0.8))
    .add_wheels(WheelSystem(4, wheel_mass=10, friction_coeff=0.7))
    .add_motor(Motor(power=500, max_force=300))
    .add_battery(Battery(capacity_wh=2000, mass=40))
    .add_sensor(SensorPackage("NavCam", mass=5, power_draw=20))
    .build()
)
rover = DynamicRover(
    mass=rover_config["mass"],
    wheel_base=rover_config["wheel_base"],
    track_width=rover_config["track_width"],
    body_height=rover_config["body_height"]
)

rover.motor_power = rover_config["motor_power"]
rover.friction_coeff = rover_config["friction_coeff"]
rover.max_drive_force = rover_config["max_drive_force"]

rover.battery_capacity_wh = rover_config["battery_capacity_wh"]
rover.sensor_power_draw = rover_config["sensor_power_draw"]
rover.battery_remaining_wh = rover_config["battery_capacity_wh"]

terrain = MarsTerrain("mars_terrain/megt00n000hb.img")
controller = ConstantThrottleController(force=150)

profile = terrain.extract_profile()
# --- Rescale terrain for rover-scale simulation ---
profile = profile / 50     # reduce vertical exaggeration
dx = 10                    # treat each pixel as 10 meters instead of 463
dt = 0.01
sim_time = 0.0

engine = SimulationEngine(rover, terrain, controller)

while sim_time < 20:
    state = engine.step(dt)
    print(state)
    sim_time += dt
