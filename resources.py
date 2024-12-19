from resource_types import ResourceType
import numpy as np
from scipy.stats import beta

class Resource:
    def __init__(self, 
                 resource_type, 
                 initial_amount, 
                 base_consumption_rate, 
                 recycling_efficiency,
                 consumption_variation: float = 0.1,
                 alpha=2, beta=5): 
        self.ResourceType = resource_type
        self.amount = initial_amount
        self.base_consumption_rate = base_consumption_rate
        self.consumption_variation = consumption_variation
        self.recycling_efficiency = recycling_efficiency
        self.alpha = alpha
        self.beta = beta

    def consume(self, num_people: int, time: int):
        total_consumption = self.base_consumption_rate * num_people * time
        u1 = np.random.uniform(0, 1)
        u2 = 1 - u1
        beta_factor_1 = beta.ppf(u1, self.alpha, self.beta)
        beta_factor_2 = beta.ppf(u2, self.alpha, self.beta)
        beta_scaled_1 = 0.8 + beta_factor_1 * (1.5 - 0.8)
        beta_scaled_2 = 0.8 + beta_factor_2 * (1.5 - 0.8)
        total_consumption *= (beta_scaled_1 + beta_scaled_2) / 2
        recycled_amount = total_consumption * self.recycling_efficiency
        self.amount -= total_consumption - recycled_amount
        defecint = self.amount
        self.amount = max(0, self.amount)
        return total_consumption - recycled_amount, defecint
    
    def is_empty(self) -> bool:
        return self.amount <= 0
