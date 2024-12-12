import random
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
from resource_types import ResourceType

class ResourceNode:
    def __init__(
        self, 
        resource_type: ResourceType, 
        base_extraction_rate: float = 10.0,
        capacity: float = 10000.0,
        difficulty_factor: float = 1.0, 
        extraction_variance: float = 0.1,
        equipment_reliability: float = 0.9
    ):
        self.resource_type = resource_type
        self.base_extraction_rate = base_extraction_rate
        self.current_capacity = capacity  
        self.difficulty_factor = difficulty_factor
        self.extraction_variance = extraction_variance
        self.equipment_reliability = equipment_reliability

    def extract_resources(self, num_workers: int, avg_skill_score: int) -> float:
        extraction = (
            self.base_extraction_rate * 
            num_workers * (0.5 + (avg_skill_score / 100))
        )
        #extraction *= np.random.gamma(1.0, self.extraction_variance)
        extraction = min(extraction, self.current_capacity)
        self.current_capacity -= extraction
        return max(0.0, extraction)

    def __repr__(self):
        return (
            f"ResourceNode(type={self.resource_type}, "
            f"current_capacity={self.current_capacity:.2f}/"
        )