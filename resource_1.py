from resource_types import ResourceType
import numpy as np
class Resource:
    def __init__(self, 
                 resource_type, 
                 initial_amount, 
                 base_consumption_rate, 
                 recycling_efficiency,
                 capacity: float,
                 consumption_variation: float = 0.2):
        self.ResourceType = resource_type
        self.amount = initial_amount
        self.base_consumption_rate = base_consumption_rate
        self.consumption_variation = consumption_variation
        self.recycling_efficiency = recycling_efficiency
        self.capacity = capacity

    def consume(self, num_people: int, time: int):
        total_consumption = self.base_consumption_rate * num_people * time
        
        #Remove this maybe
        #consumption_variation = np.random.gamma(1.0, self.consumption_variation)
        #total_consumption *= consumption_variatioSn


        actual_consumption = min(total_consumption, self.amount)
        recycled_amount = actual_consumption * self.recycling_efficiency
        self.amount -= actual_consumption - recycled_amount  # Deduct consumed amount (after recycling)
        return actual_consumption

    def add(self, amount: float) -> float:
        if amount < 0:
            raise ValueError("Cannot add a negative amount to the resource.")

        available_space = self.capacity - self.amount
        actual_added = min(amount, available_space)
        self.amount += actual_added

        return actual_added
    
    def is_full(self) -> bool:
        return self.amount >= self.capacity
    
    def is_empty(self) -> bool:
        return self.amount <= 0
