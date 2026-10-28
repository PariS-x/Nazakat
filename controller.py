class BaseController:
    def compute_drive_force(self, rover, dt):
        return 0.0


class ConstantThrottleController(BaseController):
    def __init__(self, force):
        self.force = force

    def compute_drive_force(self, rover, dt):
        return self.force


class TimedThrottleController(BaseController):
    def __init__(self, force, duration):
        self.force = force
        self.duration = duration

    def compute_drive_force(self, rover, dt):
        if rover.time < self.duration:
            return self.force
        return 0.0