from typing import List, Optional
from resource_types import ResourceType
from skill import Skill

class Colonist:
    def __init__(
        self, 
        id: int, 
        age: int,
        water_score: int,
        food_score: int,
        oxygen_score: int
    ):
        self.id = id
        self.age = age
        self.current_job = None

        self.water_skill = Skill(
            'water', 
            initial_value=water_score
        )

        self.food_skill = Skill(
            'food', 
            initial_value=food_score
        )

        self.oxygen_skill = Skill(
            'oxygen', 
            initial_value=oxygen_score
        )
        # Initialize skills
        self.water_skill = Skill('water', initial_value=water_score)
        self.food_skill = Skill('food', initial_value=food_score)
        self.oxygen_skill = Skill('oxygen', initial_value=oxygen_score)

        # Create skill mappings
        self.skills = {
            ResourceType.WATER: self.water_skill.value,
            ResourceType.FOOD: self.food_skill.value,
            ResourceType.OXYGEN: self.oxygen_skill.value
        }

        self.skill_map = {
            ResourceType.WATER: self.water_skill,
            ResourceType.FOOD: self.food_skill,
            ResourceType.OXYGEN: self.oxygen_skill
        }
        self.determine_highest_score = self.determine_highest_score()
    
    def determine_highest_score(self) -> ResourceType:
        skills = {
            ResourceType.WATER: self.water_skill.value,
            ResourceType.FOOD: self.food_skill.value,
            ResourceType.OXYGEN: self.oxygen_skill.value
        }
        return max(skills, key=skills.get)

    def get_skill_for_resource(self, resource_type: ResourceType) -> int:
        skill_map = {
            ResourceType.WATER: self.water_skill,
            ResourceType.FOOD: self.food_skill,
            ResourceType.OXYGEN: self.oxygen_skill
        }
        return skill_map.get(resource_type, Skill('default')).value

    def gain_skill(self, resource_type: ResourceType, worker_count: int):
        skill_map = {
            ResourceType.WATER: self.water_skill,
            ResourceType.FOOD: self.food_skill,
            ResourceType.OXYGEN: self.oxygen_skill
        }
        skill = skill_map.get(resource_type)
        if skill:
            skill.gain_experience(1.0, worker_count)

    def change_job(self, new_job, resource_type: ResourceType):
        self.current_job = new_job

    def age_up(self) -> int:
        if self.age < 3:
            self.age += 1
            if self.age == 2:
                self.water_skill.value //= 2
                self.food_skill.value //= 2
                self.oxygen_skill.value //= 2
        return self.age

    def __repr__(self):
        return (
            f"Colonist {self.id} (Age: {self.age}) - "
            f"Water: {self.water_skill}, "
            f"Food: {self.food_skill}, "
            f"Oxygen: {self.oxygen_skill}\n"
        )