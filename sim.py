import numpy as np
import heapq
from typing import Dict, List
from resource_types import ResourceType
from resource_node import ResourceNode
from resource import Resource
from colonist import Colonist

# Gompertz distribution parameters
beta = 0.0001  # Scale parameter
gamma = 0.2    # Shape parameter
# Survival function of the Gompertz model
# Function to calculate the expected time until death from a given age in years
def expected_time_until_death(age_in_years, beta, gamma):
    return (1 / (beta * gamma)) * (np.exp(gamma * age_in_years) - 1)

class Colony:
    def __init__(self, initial_population: int = 50, months_of_reserve: int = 6):
        self.current_tick = 0 # Each tick is a month
        self.simulation_duration = 1200  # 100 years (1200 months)

        # Population management
        self.population = initial_population
        self.unique_id = initial_population
        self.child_population = 0
        self.adult_population = initial_population
        self.elder_population = 0
        self.months_survived = 0

        # Rates 
        self.birth_rate = 1
        self.death_rate = 0.1 # Affected by the amount of resources 

        # Colonist management
        self.Children = {} 
        self.Adults = {
            i: Colonist(
                id=i,
                age_bracket=1,
                age=np.random.randint(216, 720),  # Random age in months for adults (18 to 60 years)
                gender=0 if i < initial_population // 2 else 1,   # 0 for male, 1 for female (50/50 chance)
                water_score=np.random.randint(50, 91),
                food_score=np.random.randint(50, 91),
                oxygen_score=np.random.randint(50, 91),
            )
            for i in range(initial_population)
        }
        self.Elders = {}    

        # Helper Structures FOR VALID ADULTS FOR MATING ONLY, Slightly more memory for faster execution instead of searching
        self.males = []
        self.females = []
        self.pregnant_females = []

        # Populate males, females lists by iterating through adults
        for colonist in self.Adults.values():
            if colonist.gender == 0:
                self.males.append(colonist.id)  # Append colonist's ID
            elif colonist.gender == 1:
                self.females.append(colonist.id) 


        # Inventory management
        self.inventory: Dict[ResourceType, Resource] = {
            ResourceType.WATER: Resource(
                resource_type=ResourceType.WATER,
                initial_amount=initial_population * months_of_reserve,
                base_consumption_rate=1.0,
                recycling_efficiency=0.2,
            ),
            ResourceType.FOOD: Resource(
                resource_type=ResourceType.FOOD,
                initial_amount=initial_population * months_of_reserve,
                base_consumption_rate=1.0,
                recycling_efficiency=0.0,
            ),
            ResourceType.OXYGEN: Resource(
                resource_type=ResourceType.OXYGEN,
                initial_amount=initial_population * months_of_reserve,
                base_consumption_rate=1.0,
                recycling_efficiency=0.2,
            )
        }

        # Heaps to track the time for age transitions
        # Min-heap: (birth_tick, adult_id)
        self.pregnancy_queue: List[tuple] = []  

        # Min-heap: (transition_tick, colonist_id)
        self.children_queue: List[tuple] = []  # Childern to Adult
        self.adults_queue: List[tuple] = []    # Adult to Elder
        self.elders_queue: List[tuple] = []    # Elder to Death

        # Resource nodes
        self.water_node = ResourceNode(ResourceType.WATER, base_extraction_rate=12, capacity=5000)
        self.food_node = ResourceNode(ResourceType.FOOD, base_extraction_rate=10, capacity=float('inf'))
        self.oxygen_node = ResourceNode(ResourceType.OXYGEN, base_extraction_rate=8, capacity=6000)

        self.resource_nodes = [self.water_node, self.food_node, self.oxygen_node]

        # Worker distribution
        self.worker_distribution: Dict[ResourceNode, List[int]] = {
            self.water_node: [],
            self.food_node: [],
            self.oxygen_node: []
        }

        self.initialize_age_transitions()
        self.initialize_jobs(17, 17, 16)


    def simulate(self):
        print("Starting Tick-Based Colony Simulation")
        print("-" * 40)

        while self.current_tick < self.simulation_duration:
            self.print_colony_stats()
            

            # Assign Work - Every Tick

            # Monthly processes
            # Update Resources Production - Consumption  
            self.update_inventory()

            # New Births
            self.process_births()
            self.process_pregnancy()

            # Age transitions
            self.process_age_transitions()

            # Deaths
            self.process_deaths()

            # Change Behaviors and death rates for the next tick

            # Increase Death Rate Based on Resource Depletion: If food is low, increase the death rate.


            self.current_tick += 1

        print("\nSimulation Completed")

# Resource Functions

    def update_inventory(self):
        self.resource_consumption()
        self.resource_production()

    def resource_consumption(self):
        for resource_type, resource in self.inventory.items():
            resource.consume(self.population, 1)  

    def resource_production(self):
        # Temporary lists to store changes
        workers_to_reassign = []
        nodes_to_remove = []

        for resource_node, workers in self.worker_distribution.items():
            avg_skill_score = self.calculate_average_skill(resource_node.resource_type, workers)
            production = resource_node.extract_resources(len(workers), avg_skill_score)
            self.inventory[resource_node.resource_type].amount += production

            for worker_id in workers:  
                worker = self.Adults.get(worker_id)
                if worker:
                    worker.gain_skill(resource_node.resource_type, len(workers))

            # Check if the resource node is depleted and mark it for removal
            if resource_node.current_capacity <= 0:
                nodes_to_remove.append(resource_node)

        # Remove depleted nodes
        for node in nodes_to_remove:
            self.remove_resource_node(node)

    def remove_resource_node(self, resource_node: ResourceNode):
        # Reallocate workers before removing the node
        if resource_node in self.worker_distribution:
            self.reallocate_workers(resource_node)

        # Remove the node from the worker distribution dictionary
        if resource_node in self.worker_distribution:
            del self.worker_distribution[resource_node]

        if resource_node in self.resource_nodes:
            self.resource_nodes.remove(resource_node)

        print(f"Resource node {resource_node} removed from the system.")

    def reallocate_workers(self, depleted_node: ResourceNode):
        workers_to_reallocate = self.worker_distribution.pop(depleted_node, [])
        print(f"Reallocating {len(workers_to_reallocate)} workers from {depleted_node}.")
    
        for worker_id in workers_to_reallocate:
            # Try assigning the worker to another resource node
            for target_node in self.worker_distribution.keys():
                if target_node.current_capacity > 0:  # Only consider active nodes
                    colonist = self.Adults.get(worker_id)
                    if colonist:
                        self.assign_worker_to_resource_node(colonist, target_node)
                        print(f"Assigned worker {worker_id} to {target_node}.")
                    break
            else:
                print(f"No available nodes to reassign worker {worker_id}. Worker remains idle.")

# Age, birth, and death functions

    def initialize_age_transitions(self):
        for adult in self.Adults.values():
            # Calculate the remaining months until the adult turns 60 years old
            remaining_ticks = 720 - adult.age  
            transition_tick = self.current_tick + remaining_ticks
            heapq.heappush(self.adults_queue, (transition_tick, adult.id))

    def process_births(self):
        if not self.females or not self.males:
            return  # No males or females to pair

        # Probability of selecting a valid pair, ie adult non-pregnant female and an adult male
        rate = 2 * (len(self.females) / self.population) * (len(self.males)/ self.population) * self.birth_rate
        if np.random.uniform() > rate:
            return  

        male_id = np.random.choice(self.males)
        female_id = np.random.choice(self.females)
        male = self.Adults[male_id]
        female = self.Adults[female_id]
        female.pregnant = True
        self.pregnant_females.append(female.id)  
        self.females.remove(female.id)
        colonist = self.Adults[female_id] 
        self.remove_worker_from_resource_node(colonist)
        birth_tick = self.current_tick + 9 
        heapq.heappush(self.pregnancy_queue, (birth_tick, female.id))  

    def process_pregnancy(self):
        while self.pregnancy_queue and self.pregnancy_queue[0][0] <= self.current_tick:
            birth_tick, colonist_id = heapq.heappop(self.pregnancy_queue)
            self.pregnant_females.remove(colonist_id)  
            colonist = self.Adults[colonist_id] 
            self.females.append(colonist_id)
            colonist.pregnant = False
            child_id = self.unique_id
            self.unique_id += 1
            self.population += 1
            self.child_population += 1
            child = Colonist( id=child_id, gender=np.random.randint(0, 1), )
            self.Children[child_id] = child  
            transition_tick = self.current_tick + 216
            heapq.heappush(self.adults_queue, (transition_tick, child_id))


    def process_age_transitions(self):
        # Childern To Adults
        while self.children_queue and self.children_queue[0][0] <= self.current_tick:
            transition_tick, colonist_id = heapq.heappop(self.children_queue)
            child = self.Children.pop(colonist_id)  # Remove from children
            # update age and age_bracket to adults
            child.age = 216
            child.age_bracket = 1
            self.child_population -= 1
            self.adult_population += 1
            self.Adults[colonist_id] = child  

            if child.gender == 0:
               self.males.append(colonist_id)
            else:
                self.females.append(colonist_id)

            # Schedule the transition to elder
            remaining_ticks = 720 - child.age  
            transition_tick = self.current_tick + remaining_ticks
            heapq.heappush(self.adults_queue, (transition_tick, colonist_id))

        # Adults to elders
        while self.adults_queue and self.adults_queue[0][0] <= self.current_tick:
            transition_tick, colonist_id = heapq.heappop(self.adults_queue)
            adult = self.Adults.pop(colonist_id)  
            adult.age = 720  
            adult.age_bracket = 2  
            self.adult_population -= 1
            self.elder_population += 1
            self.Elders[colonist_id] = adult  
            if adult.gender == 0:
                self.males.remove(colonist_id) 
            elif adult.gender == 1:
                self.females.remove(colonist_id) 

            self.remove_worker_from_resource_node(adult)


            # Schedule the transition to death using Gompertz distribution 
            age_in_years = adult.age / 12  # Convert age in months to years
            additional_years = expected_time_until_death(age_in_years, beta, gamma)

            life_expectancy_ticks = int(additional_years * 12)  # Convert years to months
            transition_tick = self.current_tick + (life_expectancy_ticks - adult.age)
            heapq.heappush(self.elders_queue, (transition_tick, colonist_id))

        # Elders Death
        while self.elders_queue and self.elders_queue[0][0] <= self.current_tick:
            transition_tick, colonist_id = heapq.heappop(self.elders_queue)
            elder = self.Elders.pop(colonist_id)  
            elder.age = transition_tick - self.current_tick  
            self.elder_population -= 1
            self.population -= 1

    def process_deaths(self):


# Job Functions

    def initialize_jobs(self, water_workers: int, food_workers: int, oxygen_workers: int):
        colonists = list(self.Adults.values())
        
        # Assign water workers
        for i in range(water_workers):
            colonist = colonists[i]
            colonist.current_job = ResourceType.WATER
            self.assign_worker_to_resource_node(colonist, self.water_node)

        # Assign food workers
        for i in range(food_workers):
            colonist = colonists[water_workers + i]
            colonist.current_job = ResourceType.FOOD
            self.assign_worker_to_resource_node(colonist, self.food_node)

        # Assign oxygen workers
        for i in range(oxygen_workers):
            colonist = colonists[water_workers + food_workers + i]
            colonist.current_job = ResourceType.OXYGEN
            self.assign_worker_to_resource_node(colonist, self.oxygen_node)

    def assign_worker_to_resource_node(self, colonist: Colonist, resource_node: ResourceNode):
        if resource_node not in self.worker_distribution:
            self.worker_distribution[resource_node] = []
        self.worker_distribution[resource_node].append(colonist.id)

    def remove_worker_from_resource_node(self, colonist: Colonist):
        for resource_node, colonists in self.worker_distribution.items():
            if colonist.id in colonists:
                colonists.remove(colonist.id)
                break

# Helper Functions

    def calculate_average_skill(self, resource_type: ResourceType, workers: list[int]) -> float:
        if not workers:
            return 0.0  
    
        skill_scores = []
        for worker_id in workers:
            colonist = self.Adults.get(worker_id)
            if not colonist:
                continue
        
        # Determine the skill score based on resource type
        if resource_type == ResourceType.WATER:
            skill_scores.append(colonist.water_skill.value)
        elif resource_type == ResourceType.FOOD:
            skill_scores.append(colonist.food_skill.value)
        elif resource_type == ResourceType.OXYGEN:
            skill_scores.append(colonist.oxygen_skill.value)

        # Calculate and return the average
        return sum(skill_scores) / len(skill_scores) if skill_scores else 0.0

    def print_colony_stats(self):
        year = self.current_tick // 12
        month = self.current_tick % 12 + 1  

        print(f"Year: {year}, Month: {month}")
        print(f"Population: {self.population}")
        print(f"Children: {self.child_population}, Adults: {self.adult_population}, Elders: {self.elder_population}")
    
        for resource_type, resource in self.inventory.items():
            print(f"{resource_type.name}: {resource.amount:.2f}")
    
        for node in self.resource_nodes:
            print(f"Node {node.resource_type}: Capacity: {node.current_capacity:.2f}")

        print("-" * 40)


def main():
    colony = Colony(initial_population=50)
    colony.simulate()

if __name__ == "__main__":
    main()

