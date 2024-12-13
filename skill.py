import numpy as np

class Skill:
    def __init__(self, name: str, initial_value: int = 0):
        self.name = name
        self.value = initial_value 

    def gain_experience(self, base_gain: float, worker_count: int = 1):
        adjusted_gain = base_gain
        skill_gain = np.random.poisson(adjusted_gain)
        self.value = min(100, max(0, self.value + skill_gain))

    def __repr__(self):
        return f"{self.name.capitalize()} Skill (Level: {self.value:.2f})"