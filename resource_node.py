import numpy as np
from scipy.stats import beta
from resource_types import ResourceType

class ResourceNode:
    def __init__(
        self, 
        resource_type: ResourceType, 
        base_extraction_rate: float = 10.0,
        capacity: float = 10000.0,
        extraction_variance: float = 0.1,
        alpha: float = 2,  
        beta: float = 5
    ):
        self.resource_type = resource_type
        self.base_extraction_rate = base_extraction_rate
        self.current_capacity = capacity  
        self.extraction_variance = extraction_variance
        self.alpha = alpha  
        self.beta = beta  

    def extract_resources(self, num_workers: int, avg_skill_score: int) -> float:
        extraction = (
            self.base_extraction_rate * 
            num_workers * (0.5 + (avg_skill_score / 100))
        )
        
        u1 = np.random.uniform(0, 1)
        u2 = 1 - u1  

        beta_factor_1 = beta.ppf(u1, self.alpha, self.beta)
        beta_factor_2 = beta.ppf(u2, self.alpha, self.beta)

        beta_scaled_1 = 0.8 + beta_factor_1 * (1.5 - 0.8)
        beta_scaled_2 = 0.8 + beta_factor_2 * (1.5 - 0.8)

        extraction_1 = extraction * beta_scaled_1
        extraction_2 = extraction * beta_scaled_2

        average_extraction = (extraction_1 + extraction_2) / 2
        
        extraction_final = min(average_extraction, self.current_capacity)
        self.current_capacity -= extraction_final
        
        return max(0.0, extraction_final)

    def __repr__(self):
        return (
            f"ResourceNode(type={self.resource_type}, "
            f"current_capacity={self.current_capacity:.2f})"
        )
