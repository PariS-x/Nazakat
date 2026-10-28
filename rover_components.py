from dataclasses import dataclass

# ----------------------------
# Materials
# ----------------------------

@dataclass
class Chassis:
    material: str
    base_mass: float
    width: float
    length: float
    height: float


@dataclass
class WheelSystem:
    wheel_count: int
    wheel_mass: float
    friction_coeff: float


@dataclass
class Motor:
    power: float        # Watts
    max_force: float    # Newtons


@dataclass
class Battery:
    capacity_wh: float  # Watt-hours
    mass: float


@dataclass
class SensorPackage:
    name: str
    mass: float
    power_draw: float   # Watts

class RoverBuilder:

    def __init__(self):
        self.chassis = None
        self.wheels = None
        self.motor = None
        self.battery = None
        self.sensors = []

    def add_chassis(self, chassis: Chassis):
        self.chassis = chassis
        return self

    def add_wheels(self, wheels: WheelSystem):
        self.wheels = wheels
        return self

    def add_motor(self, motor: Motor):
        self.motor = motor
        return self

    def add_battery(self, battery: Battery):
        self.battery = battery
        return self

    def add_sensor(self, sensor: SensorPackage):
        self.sensors.append(sensor)
        return self

    def build(self):
        total_mass = (
            self.chassis.base_mass +
            self.wheels.wheel_count * self.wheels.wheel_mass +
            self.battery.mass +
            sum(s.mass for s in self.sensors)
        )

        total_sensor_power = sum(s.power_draw for s in self.sensors)

        return {
            "mass": total_mass,
            "wheel_base": self.chassis.length,
            "track_width": self.chassis.width,
            "body_height": self.chassis.height,
            "motor_power": self.motor.power,
            "max_drive_force": self.motor.max_force,
            "battery_capacity_wh": self.battery.capacity_wh,
            "sensor_power_draw": total_sensor_power,
            "friction_coeff": self.wheels.friction_coeff
        }