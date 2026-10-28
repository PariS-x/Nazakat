class SimulationEngine:

    def __init__(self, rover, terrain, controller=None):
        self.rover = rover
        self.terrain = terrain
        self.time = 0.0
        self.controller = controller

    def step(self, dt):

        if self.controller:
            drive = self.controller.compute_drive_force(self.rover, dt)
            self.rover.set_drive_force(drive)

        state = self.rover.step(dt)
        self.time += dt

        return {
            "time": self.time,
            "battery": self.rover.battery_remaining_wh,
            "velocity": self.rover.velocity,
            "position": (self.rover.x, self.rover.y),
            "roll": self.rover.roll
        }