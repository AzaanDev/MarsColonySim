import numpy as np
from scipy import stats
from scipy.stats import expon
import heapq
import matplotlib.pyplot as plt
from typing import List, Dict
from resource_types import ResourceType
from resource_node import ResourceNode
from resource_1 import Resource
from colonist import Colonist

def inverse_exponential(u, rate):
    scale = 1 / rate  
    return expon.ppf(u, scale=scale)

class Event:
    PRIORITY_MAP = {
        "consume": 1,
        "produce": 2,
        "death": 3, 
        "risk": 4,
        "birth": 5,
        "transition": 6,
        "resume": 7,
        "interaction": 8, #TODO
        "resource_node_discovery": 9, 
    }

    def __init__(self, time, event_type, details=None):
        self.time = time
        self.event_type = event_type
        self.details = details
        self.residual_time = None
        self.priority = self.PRIORITY_MAP.get(event_type, float('inf'))

    def calculate_residual_time(self, current_time):
        self.residual_time = max(0, self.time - current_time)

    def reschedule(self, current_time):
        if self.residual_time is not None:
            self.time = current_time + self.residual_time
            self.residual_time = None

    def __lt__(self, other):
        if self.time == other.time:
            return self.priority < other.priority
        return self.time < other.time

    def __repr__(self):
        return f"Event(Type: {self.event_type}, Details: {self.details})"

class Colony:
    def __init__(self, initial_population: int = 50, days_of_reserve: int = 30):
        self.resource_nodes: List[ResourceNode]
        self.population = initial_population
        self.unique_id = initial_population
        self.child_population = 0
        self.adult_population = initial_population
        self.elder_population = 0
        self.days_survived = 0
        self.time = 0
        self.last_consumption_time: Dict[ResourceType, float] = {
            ResourceType.WATER: 0.0,
            ResourceType.FOOD: 0.0,
            ResourceType.OXYGEN: 0.0
        }
        self.simulation_duration = 1000000

        #Rates
        self.child_birth_rate = 5
        self.adult_birth_rate = 20
        self.elder_birth_rate = 200
        self.elder_death_rate = 50
        self.discovery_rate = 100
        self.resource_production_rate = 10
        self.gamma_shape = 5  
        self.gamma_scale = 1

        # Event queue
        self.event_queue: List[Event] = []

        # Other queues
        self.children_queue: List[Event] = []  # Queue for each child to adult
        self.adults_queue: List[Event] = []    # Queue for each adult to elder
        self.elders_queue: List[Event] = []    # Queue for each elder to death

        # Inventory management
        self.inventory: Dict[ResourceType, Resource] = {
            ResourceType.WATER: Resource(
                resource_type=ResourceType.WATER,
                initial_amount=initial_population * days_of_reserve * 2.0,
                base_consumption_rate=1.0,
                recycling_efficiency=0.8,
                capacity=10000.0
            ),
            ResourceType.FOOD: Resource(
                resource_type=ResourceType.FOOD,
                initial_amount=initial_population * days_of_reserve * 2.0,
                base_consumption_rate=1.0,
                recycling_efficiency=0.0,
                capacity=10000.0
            ),
            ResourceType.OXYGEN: Resource(
                resource_type=ResourceType.OXYGEN,
                initial_amount=initial_population * days_of_reserve * 2.0,
                base_consumption_rate=1.0,
                recycling_efficiency=0.8,
                capacity=10000
            )
        }

        # 3 Inital Production Nodes
        self.water_node = ResourceNode(ResourceType.WATER, base_extraction_rate=12, capacity=5000)
        self.food_node = ResourceNode(ResourceType.FOOD, base_extraction_rate=10, capacity=float('inf')) # Farm
        self.oxygen_node = ResourceNode(ResourceType.OXYGEN, base_extraction_rate=8, capacity=6000)

        self.resource_nodes = [self.water_node, self.food_node, self.oxygen_node]

        # Worker distribution dictionary - mapping resource node to colonist IDs
        self.worker_distribution: Dict[ResourceNode, List[int]] = {
            self.water_node: [],
            self.food_node: [],
            self.oxygen_node: []
        }


        self.Adults = {i: Colonist(i, 1, np.random.randint(50, 91), np.random.randint(50, 91), np.random.randint(50, 91)) for i in range(initial_population)}
        self.Children = {} 
        self.Elders = {}    
        self.initialize_jobs(17, 17, 16)
        self.initialize_events()
   
    def initialize_jobs(self, water_workers: int, food_workers: int, oxygen_workers: int):
        colonists_iter = iter(self.Adults.values())  # Create an iterator over the colonists

        # Assign water workers
        for i in range(water_workers):
            colonist = next(colonists_iter)  # Get the next colonist from the iterator
            colonist.current_job = ResourceType.WATER
            self.assign_worker_to_resource_node(colonist, self.water_node)

        # Assign food workers
        for i in range(food_workers):
            colonist = next(colonists_iter)
            colonist.current_job = ResourceType.FOOD
            self.assign_worker_to_resource_node(colonist, self.food_node)

        # Assign oxygen workers
        for i in range(oxygen_workers):
            colonist = next(colonists_iter)
            colonist.current_job = ResourceType.OXYGEN
            self.assign_worker_to_resource_node(colonist, self.oxygen_node)

    def assign_worker_to_resource_node(self, colonist: Colonist, resource_node: ResourceNode):
        """Assigns a colonist to a resource node and updates the worker distribution."""
        if resource_node not in self.worker_distribution:
            self.worker_distribution[resource_node] = []
        self.worker_distribution[resource_node].append(colonist.id)

    # Super Slow
    def remove_worker_from_resource_node(self, colonist: Colonist):
        """Removes a colonist from a resource node and updates the worker distribution."""
        for resource_node, colonists in self.worker_distribution.items():
            if colonist.id in colonists:  # Directly compare the colonist's ID
                colonists.remove(colonist.id)  # Remove the ID from the list
                break

    def initialize_events(self):
        # Assume only adults in the beginning
        birth_time = self.time + np.random.exponential(self.child_birth_rate)  
        birth_event = Event(birth_time, "birth")
        heapq.heappush(self.event_queue, birth_event)

        for adult in self.Adults.values():
            # Create an event that transitions the adult to elder at a certain time
            transition_time = self.time + np.random.exponential(self.elder_birth_rate)
            transition_event = Event(transition_time, "transition", adult)
            heapq.heappush(self.adults_queue, transition_event)

        for resource_type, resource in self.inventory.items():
            event_time = self.time + np.random.gamma(self.gamma_shape, self.gamma_scale)
            heapq.heappush(self.event_queue, Event(event_time, "consume", resource_type))
        
        death_check_time = self.time + 10  # Every 10 time units
        heapq.heappush(self.event_queue, Event(death_check_time, "death_check"))

        for resource_node in self.worker_distribution.keys():
            production_time = self.time + np.random.exponential(self.resource_production_rate)  
            produce_event = Event(production_time, "produce", resource_node)
            heapq.heappush(self.event_queue, produce_event)

        discovery_time = self.time + np.random.exponential(self.discovery_rate) 
        discovery_event = Event(discovery_time, "resource_node_discovery", None)
        heapq.heappush(self.event_queue, discovery_event)

     

    def simulate(self):
        print("Starting Colony Simulation")
        print("-" * 40)
        self.schedule_risk_events(interval=100, lambda_rate=2)
        #Need to add stopping conditions ie population == 0 or population growth
        while self.time < self.simulation_duration:
            self.print_colony_stats()
            # Check the top of all event queues
            queues = [
                ("event_queue", self.event_queue),
                ("children_queue", self.children_queue),
                ("adults_queue", self.adults_queue),
                ("elders_queue", self.elders_queue)
            ]

            next_event_queue, next_event_list = min(queues, key=lambda q: q[1][0].time if q[1] else float('inf'))
            if not next_event_list:
                continue
            #print(next_event_queue, next_event_list)
            event = heapq.heappop(next_event_list)
            #print(event)
            self.time = event.time
            if event.event_type == "resume":
                self.handle_resume()
            else:
                self.handle_event(event)

        print("\nSimulation Completed")

    def handle_event(self, event):
        #print(event)
        self.time = event.time
        if event.event_type == "produce":
            print("produce")
            self.handle_production(event.details)
        elif event.event_type == "consume":
            print("consume")
            self.handle_consumption(event.details)
        elif event.event_type == "storage_check":
            print("storage check")
            self.handle_storage()
        elif event.event_type == "transition":
            print("transition")
            self.handle_transitions(event.details)
        elif event.event_type == "birth":
            print("birth")
            self.handle_birth()
        elif event.event_type == "risk":
            print("risk")
            self.handle_risk_event()
            if int(self.time) % 100 == 0:  # Every 100 time units
                self.schedule_risk_events(interval=100, lambda_rate=0.5)
        elif event.event_type == "death":
            print("death")
            self.handle_death(event.details)
        elif event.event_type == "death_check":
            print("death check")
            self.schedule_death_events()
            next_check_time = self.time + 10  # Reschedule the next death check
            heapq.heappush(self.event_queue, Event(next_check_time, "death_check"))
        elif event.event_type == "resource_node_discovery":
            print("resource node discovery")
            self.handle_resource_node_discovery()
            
    def handle_storage(self):
        """
        Handles the storage event to monitor and manage resources.
        """
        
        # Process resource production for each node
        for resource_node, workers in self.worker_distribution.items():
            # Calculate extraction rate based on the number of workers
            extraction_rate = resource_node.base_extraction_rate * len(workers)
            extracted_amount = min(resource_node.capacity, extraction_rate)
            
            # Add extracted resources to storage, respecting capacity
            resource = self.inventory[resource_node.resource_type]
            available_capacity = resource.capacity - resource.amount
            actual_added = min(extracted_amount, available_capacity)
            resource.add(actual_added)
            
            # Update resource node capacity after extraction
            resource_node.capacity -= actual_added

        # Schedule the next storage event
        next_storage_time = self.time + 10  # Run storage checks every 10 time units
        heapq.heappush(self.event_queue, Event(next_storage_time, "storage_check"))

        #elif event.event_type == "interaction": #Score increase for childern 
            #self.handle_death()
        

    def handle_birth(self):
        new_child_id = self.unique_id
        new_child = Colonist(new_child_id, 0, 0, 0, 0)
        self.population += 1
        self.child_population += 1
        self.unique_id += 1
        self.Children[new_child_id] = new_child  # Add child to dictionary

        # Schedule an event for this new child to become an adult
        transition_time = self.time + np.random.exponential(self.adult_birth_rate)  # Transition from child to adult
        transition_event = Event(transition_time, "transition", new_child)
        heapq.heappush(self.children_queue, transition_event)

        # New birth event
        birth_time = self.time + np.random.exponential(self.child_birth_rate)  
        birth_event = Event(birth_time, "birth")
        heapq.heappush(self.event_queue, birth_event)

    def handle_transitions(self, colonist):
        if colonist.id not in self.Children and colonist.id not in self.Adults and colonist.id not in self.Elders:
            return
        age_state = colonist.age # 0 is child & 1 is adult

        transition_actions = {
            0: self.transition_child_to_adult,  # Transition from child (0) to adult (1)
            1: self.transition_adult_to_elder,   # Transition from adult (1) to elder (2)
            2: self.transition_elder_death      # Transition from elder (2) to death (3)
        }
        # Get the appropriate function for the current state
        if age_state in transition_actions:
            transition_actions[age_state](colonist)
        else:
            print(f"Invalid age state: {age_state}")

    def transition_child_to_adult(self, colonist):
        # Handle the transition from child to adult
        if colonist.id not in self.Children:
            return
        print("transition c to a, ", colonist)
        colonist.age_up()
        self.child_population -= 1
        self.adult_population += 1
        # Remove from Children and add to Adults
        del self.Children[colonist.id]
        self.Adults[colonist.id] = colonist

        #Work
        node_with_fewest_workers = min(self.worker_distribution, key=lambda node: len(self.worker_distribution[node]))
        # Assign the colonist to this node
        self.worker_distribution[node_with_fewest_workers].append(colonist.id)

        # Schedule the colonist to eventually become an elder
        transition_time = self.time + np.random.exponential(self.elder_birth_rate)  # Transition from adult to elder
        transition_event = Event(transition_time, "transition", colonist)
        heapq.heappush(self.adults_queue, transition_event)

    def transition_adult_to_elder(self, colonist):
        # Handle the transition from adult to elder
        if colonist.id not in self.Adults:
            return
        print("transition a to e, ", colonist)
        colonist.age_up()
        self.adult_population -= 1
        self.elder_population += 1
        # Remove from Adults and add to Elders
        del self.Adults[colonist.id]
        self.remove_worker_from_resource_node(colonist)  
        colonist.current_job = None
        self.Elders[colonist.id] = colonist

        # Schedule the colonist to eventually die
        death_time = self.time + np.random.exponential(self.elder_death_rate)  # Death event for elder
        death_event = Event(death_time, "transition", colonist)
        heapq.heappush(self.elders_queue, death_event)
        

    def transition_elder_death(self, colonist):
        if colonist.id not in self.Elders:
            return
        print("transition e to d, ", colonist)
        self.elder_population -= 1
        self.population -= 1
        print("colonist:",self.Elders)
        del self.Elders[colonist.id]


    def handle_production(self, resource_node: ResourceNode):
        workers = self.worker_distribution.get(resource_node, [])
        avg_skill_score = self.calculate_average_skill(resource_node.resource_type, workers)
        production = resource_node.extract_resources(len(workers), avg_skill_score)
        # Add the produced resources to inventory
        self.inventory[resource_node.resource_type].amount += production
        #print(f"Produced {production:.2f} of {resource_node.resource_type} from {resource_node}.")

        # Increase worker skill score
        for worker_id in workers:  
            worker = self.Adults[worker_id] 
            worker.gain_skill(resource_node.resource_type, len(workers))

        # Check if the resource node is depleted
        if resource_node.current_capacity <= 0:
            print(f"ResourceNode {resource_node} is depleted and will be removed.")
            self.remove_resource_node(resource_node)
        else:
            # If the resource node is not empty, add the event to the queue
            production_time = self.time + np.random.exponential(self.resource_production_rate)
            produce_event = Event(production_time, "produce", resource_node)
            heapq.heappush(self.event_queue, produce_event)



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


    def handle_consumption(self, resource_type: ResourceType):
        time_since_last_consumption = self.time - self.last_consumption_time[resource_type]
        resource = self.inventory[resource_type]
        consumed = resource.consume(self.population, time_since_last_consumption)
        self.last_consumption_time[resource_type] = self.time
        # Check for potential deaths due to resource shortages
        
        next_consumption_time = self.time + np.random.gamma(self.gamma_shape, self.gamma_scale)
        heapq.heappush(self.event_queue, Event(next_consumption_time, "consume", resource_type))

    def handle_death(self, colonist: Colonist):
        if colonist.id not in self.Children and colonist.id not in self.Adults and colonist.id not in self.Elders:
            return
        print("handle death", colonist)
        if colonist.age == 0:  # Child
            self.child_population -= 1
            
            del self.Children[colonist.id]
        elif colonist.age == 1:  # Adult
            self.adult_population -= 1
            del self.Adults[colonist.id]
            self.remove_worker_from_resource_node(colonist)
        elif colonist.age == 2:  # Elder
            self.elder_population -= 1
            del self.Elders[colonist.id]
        self.population -= 1
        if self.population == 0:
            print("[Simulation Ended] All colonists have perished.")
            self.simulation_duration = self.time  # Stop simulation immediately
        

    def handle_resource_node_discovery(self):
        new_resource_type = np.random.choice([ResourceType.WATER, ResourceType.OXYGEN])  
        new_capacity = np.random.randint(2000, 10000)  # Random capacity range
        new_base_rate = np.random.uniform(10, 15)  # Random base extraction rate
        
        # Create the new resource node
        new_resource_node = ResourceNode(new_resource_type, base_extraction_rate=new_base_rate, capacity=new_capacity)
        self.worker_distribution[new_resource_node] = []

        # Add the node to the colony's list of resource nodes
        self.resource_nodes.append(new_resource_node)
        
        #Reallocate Workers?

        # Schedule next discovery event
        next_discovery_time = self.time + np.random.exponential(self.discovery_rate) 
        discovery_event = Event(next_discovery_time, "resource_node_discovery", None)
        heapq.heappush(self.event_queue, discovery_event)


    def schedule_death_events(self, risk_event: bool = False):
        print("scheduale death event")
        """Schedules death events based on resource depletion or random factors."""
        for colonist in list(self.Adults.values()) + list(self.Children.values()) + list(self.Elders.values()):
            death_probability = 0.001  # Base probability
            if colonist.age == 0:  # Child
                death_probability = 0.001
            elif colonist.age == 1:  # Adult
                death_probability = 0.001
            elif colonist.age == 2:  # Elder
                death_probability = 0.01  # Elders have a higher base rate
            for resource in self.inventory.values():
                if resource.amount == 0:
                    death_probability += 0.2  # High risk due to depletion
                elif resource.amount < 500:
                    death_probability += 0.05  # Moderate risk for low resources
            if risk_event:
                death_probability *= 1.25
            # Schedule death event based on probability
            time_to_death = np.random.exponential(1 / death_probability)
            death_time = self.time + time_to_death

            # Schedule the death event
            death_event = Event(death_time, "death", colonist)
            heapq.heappush(self.event_queue, death_event)
        
    def schedule_risk_events(self, interval: int = 100, lambda_rate: float = 0.5):
        print("schedule risk event")
        num_risks = np.random.poisson(lambda_rate)  # Number of risk events in the interval
        for _ in range(num_risks):
            # Randomly distribute the events within the interval
            risk_time = self.time + np.random.uniform(0, interval)
            risk_event = Event(risk_time, "risk")
            heapq.heappush(self.event_queue, risk_event)

    def handle_risk_event(self):
        print("Risk event")
        
        # Randomly deplete resources to simulate risk impact
        for resource_type, resource in self.inventory.items():
            depletion_amount = np.random.uniform(0, 100)
            resource.amount = max(0, resource.amount - depletion_amount)

        # Schedule death events due to resource depletion
        self.schedule_death_events()


    def print_colony_stats(self):
        """Prints out the current stats of the colony."""
        # Population statistics
        print(f"Time: {self.time}")
        print(f"Population: {self.population}")
        print(f"Children: {self.child_population}, Adults: {self.adult_population}, Elders: {self.elder_population}")
        print(f"Days Survived: {self.days_survived}")
    
        # Resource inventory
        for resource_type, resource in self.inventory.items():
            print(f"{resource_type.name}: {resource.amount:.2f} (Consumption Rate: {resource.base_consumption_rate})")
    
        # Resource node production and capacity
        for node in self.resource_nodes:
            print(f"Node {node.resource_type}: Capacity: {node.current_capacity:.2f}, Production Rate: {node.base_extraction_rate}")

        # Worker distribution by resource node
        for node, workers in self.worker_distribution.items():
            print(f"Node {node.resource_type} has {len(workers)} workers.")
    
        print("-" * 40)  

def main():
    colony = Colony(initial_population=50)
    colony.simulate()

if __name__ == "__main__":
    main()