from resource_types import ResourceType
import numpy as np
class Resource:
    def __init__(self, 
                 resource_type, 
                 initial_amount, 
                 base_consumption_rate, 
                 recycling_efficiency,
                 consumption_variation: float = 0.2):
        self.ResourceType = resource_type
        self.amount = initial_amount
        self.base_consumption_rate = base_consumption_rate
        self.consumption_variation = consumption_variation
        self.recycling_efficiency = recycling_efficiency

    def consume(self, num_people: int, time: int):
        total_consumption = self.base_consumption_rate * num_people * time

        total_consumption *= np.random.gamma(1.0, self.consumption_variation)

        recycled_amount = total_consumption * self.recycling_efficiency
        self.amount -= total_consumption - recycled_amount  # Deduct consumed amount (after recycling)
        self.amount = max(0, self.amount)
        return total_consumption - recycled_amount 
    
    def is_empty(self) -> bool:
        return self.amount <= 0
