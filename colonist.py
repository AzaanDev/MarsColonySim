from resource_types import ResourceType
from skill import Skill

class Colonist:
    def __init__(
        self, 
        id: int, 
        gender: int,        # 0 is male, 1 is female
        age_bracket: int = 0,   # Age Brackets: 0 = [0, 18), 1 = [18, 60), 2 = 60+ 
        age: int = 0,           # 18 years == 216 months/ticks, 60 years == 720 months/ticks
        water_score: int = 0,
        food_score: int = 0,
        oxygen_score: int = 0,
        pregnant: bool = False,
        reactive_behavior: float = 0,
        resilience: float = 0.5, 
    ):
        self.id = id
        self.age = age 
        # Age Brackets: 0 = [0, 18), 1 = [18, 60), 2 = 60+ 
        # 18 years == 216 months/ticks, 60 years == 720 months/ticks
        self.gender =  gender
        self.age_bracket = age_bracket
        self.pregnant = pregnant

        # Initialize skills
        self.water_skill = Skill('water', initial_value=water_score)
        self.food_skill = Skill('food', initial_value=food_score)
        self.oxygen_skill = Skill('oxygen', initial_value=oxygen_score)

        # Behaviors 
        self.reactive_behavior = reactive_behavior
        self.resilience = resilience

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
    
    def determine_highest_score(self) -> ResourceType:
        return max(self.skills, key=self.skills.get)

    def get_skill_for_resource(self, resource_type: ResourceType) -> int:
        return self.skill_map.get(resource_type, Skill('default')).value

    def gain_skill(self, resource_type: ResourceType, worker_count: int):
        skill = self.skill_map.get(resource_type)
        if skill:
            skill.gain_experience(1.0, worker_count)

    def __repr__(self):
        return (
            f"Colonist {self.id} (Age: {self.age}) - "
            f"Water: {self.water_skill}, "
            f"Food: {self.food_skill}, "
            f"Oxygen: {self.oxygen_skill}\n"
        )