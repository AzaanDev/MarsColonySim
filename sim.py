import numpy as np
import heapq
from typing import Dict, List
import scipy.stats as stats
from resource_types import ResourceType
from resource_node import ResourceNode
from resources import Resource
from colonist import Colonist
from matplotlib.table import Table
import numpy as np
import heapq
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any
import time
import matplotlib.pyplot as plt
from matplotlib.table import Table


def generate_elder_death_time(c=1.0, scale=10.0):
    years_to_death = stats.gompertz.rvs(c, scale=scale)
    
    # Ensure non-negative lifetime
    years_to_death = max(0, years_to_death)
    
    # Convert years to months
    months_to_death = max(0, int(years_to_death * 12))
    return months_to_death 



class Colony:
    def __init__(self, initial_population: int = 50, months_of_reserve: int = 12, policy_level: int = 4):
        self.current_tick = 0 # Each tick is a month
        self.simulation_duration = 3600  # 100 years (1200 months)

        # Population management
        self.population = initial_population
        self.unique_id = initial_population
        self.child_population = 0
        self.adult_population = initial_population
        self.elder_population = 0
        self.months_survived = 0

        # Rates 
        self.birth_rate = 1
        self.death_rate = 0.001 
        self.resource_node_discovery_rate = 0.01

        self.reactive_behavior = self.set_policy(policy_level)
        #Policy Level
        # Colonist management
        self.Children = {} 
        self.Adults = {
            i: Colonist(
                id=i,
                age_bracket=1,
                age= np.random.randint(216, 720),  # Random age in months for adults (18 to 60 years)
                gender=0 if i < initial_population // 2 else 1,   # 0 for male, 1 for female (50/50 chance)
                water_score=np.random.randint(50, 91),
                food_score=np.random.randint(50, 91),
                oxygen_score=np.random.randint(50, 91),
                reactive_behavior=self.reactive_behavior,
            )
            for i in range(initial_population)
        }
        self.Elders = {}    

        # Helper Structures FOR VALID ADULTS FOR MATING ONLY, Slightly more memory for faster execution instead of searching
        self.males = []
        self.females = []
        self.pregnant_females = []
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
                recycling_efficiency=0.9,
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
                recycling_efficiency=0.9,
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
        self.water_node = ResourceNode(ResourceType.WATER, base_extraction_rate=5, capacity=1000)
        self.food_node = ResourceNode(ResourceType.FOOD, base_extraction_rate=5, capacity=1000) # Farming
        self.oxygen_node = ResourceNode(ResourceType.OXYGEN, base_extraction_rate=5, capacity=1000)
        self.resource_nodes = [self.water_node, self.food_node, self.oxygen_node]
        self.population_history = [] 
        self.initialize_age_transitions()


    def simulate(self):
        end_reason = "Completed Full Simulation"
        stability_window = 120
        stability_threshold = 0.01

        while self.current_tick < self.simulation_duration and self.population > 0:
            self.population_history.append(self.population)
            #self.print_colony_stats()
            depletion_estimates = self.estimate_time_until_resource_depletion()

            # Assign Work - Every Tick, Each Agent decides what to do based on the current 
            self.step_children_elder()
            worker_distribtuion, non_working_adults = self.step_adults(depletion_estimates)

            # Update Inventory Production - Consumption  
            self.resource_production(worker_distribtuion)
            d = self.resource_consumption()

            # New Births
            self.process_births()
            self.process_pregnancy()

            # Age transitions
            self.process_age_transitions()

            # New Resource Node Discovery
            self.resource_node_discovery(non_working_adults)
            risk_occurs = self.process_disaster()

            if self.current_tick >= stability_window:
                recent_population_change = abs(self.population_history[-1] - self.population_history[-stability_window])
                if recent_population_change < stability_threshold:
                    print(f"Simulation stopped at tick {self.current_tick} due to stability.")
                    end_reason = "Population Stabilized"
                    break


            # Procress Deaths at start of tick, before any consumption or production
            early_end = self.process_deaths(d)
            self.current_tick += 1
            if early_end:
                end_reason = "Population Died Out"
                break

        return end_reason

# Agent Functions

    def set_policy(self, policy_level: int) -> float:
        if policy_level == 1:
            return 0.0
        elif policy_level == 2:
            return 0.25
        elif policy_level == 3:
            return 0.5
        elif policy_level == 4:
            return 0.75
    
    def step_children_elder(self):
        if not self.Elders or not self.Children:  
            return
        
        for child in self.Children.values():
            elder = np.random.choice(list(self.Elders.values()))
            child.food_skill.gain_experience(1 * elder.food_skill.value/100)
            child.water_skill.gain_experience(1 * elder.water_skill.value/100)
            child.oxygen_skill.gain_experience(1 * elder.oxygen_skill.value/100)


    def step_adults(self, sorted_depletion_estimates):
        worker_distribution = {resource_node: [] for resource_node in self.resource_nodes}
        non_working_adults = 0 
        # Adults step in a random order
        adult_list = list(self.Adults.values())
        np.random.shuffle(adult_list)
    
        for adult in adult_list:
            if adult.pregnant:
                continue

            assigned = False
            # Determine whether the adult will work on their primary task or contribute to the resource that runs out first
            if np.random.uniform(0, 1) <  adult.reactive_behavior:
                for resource_type, remaining_months in sorted_depletion_estimates.items():
                    for resource_node in self.resource_nodes:
                        if resource_node.resource_type == resource_type:
                            # Prevent Overallocation using estimate remaining_capacity based on the current number of workers
                            remaining_capacity = resource_node.current_capacity - (resource_node.base_extraction_rate * len(worker_distribution[resource_node]))
                            if remaining_capacity > 0:
                                worker_distribution[resource_node].append(adult.id)
                                assigned = True
                                break
                    if assigned:
                        break
            else:
                # Work on the primary task only
                primary_task = adult.determine_highest_score()
                for resource_node in self.resource_nodes:
                    if resource_node.resource_type == primary_task:
                        remaining_capacity = resource_node.current_capacity - (resource_node.base_extraction_rate * len(worker_distribution[resource_node]))
                        if remaining_capacity > 0:
                            worker_distribution[resource_node].append(adult.id)
                            assigned = True
                            break
            if not assigned:
                non_working_adults += 1
        return worker_distribution, non_working_adults
        
# Resource Functions

    def estimate_time_until_resource_depletion(self):
        depletion_estimates = {}
        for resource_type, resource in self.inventory.items():
            remaining_months = resource.amount / (self.population * resource.base_consumption_rate)
            depletion_estimates[resource_type] = remaining_months
        sorted_depletion_estimates = dict(sorted(depletion_estimates.items(), key=lambda item: item[1]))
        return sorted_depletion_estimates

    def resource_consumption(self):
        deficits = {}
        for resource_type, resource in self.inventory.items():
            t, d = resource.consume(self.population, 1)  
            if d < 0:  # If there's a deficit, log it
                deficits[resource_type] = abs(d)
        return deficits

    def resource_production(self, worker_distribtuion):
        nodes_to_remove = []
        for resource_node, workers in worker_distribtuion.items():
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
            if node in self.resource_nodes:
                self.resource_nodes.remove(node)

    def resource_node_discovery(self, non_working_adults):
        # Modify resource node discovery based on non-working adults
        discovery_probability = self.resource_node_discovery_rate * non_working_adults 
        if np.random.uniform() < discovery_probability:
            # Find the resource type with the least amount in inventory
            resource_amounts = {
                resource_type: resource.amount 
                for resource_type, resource in self.inventory.items()
            }
        
            # Sort resource types by their current amount
            sorted_resource_types = sorted(
                resource_amounts.items(), 
                key=lambda x: x[1]
            )
        
            # Choose the resource type with the least amount
            chosen_resource_type = sorted_resource_types[0][0]
        
            # Create a new resource node for the chosen resource type
            if chosen_resource_type == ResourceType.WATER:
                resource_node = ResourceNode(ResourceType.WATER, base_extraction_rate=5, capacity=np.random.randint(1000, 5000))
            elif chosen_resource_type == ResourceType.FOOD:
                resource_node = ResourceNode(ResourceType.FOOD, base_extraction_rate=5, capacity=np.random.randint(1000, 5000))
            elif chosen_resource_type == ResourceType.OXYGEN:
                resource_node = ResourceNode(ResourceType.OXYGEN, base_extraction_rate=5, capacity=np.random.randint(1000, 5000))
        
            # Add the new resource node
            self.resource_nodes.append(resource_node)
        

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
        rate = 2 * (len(self.females) / len(self.Adults)) * (len(self.males)/ (len(self.Adults) - 1)) * self.birth_rate
        if np.random.uniform() > rate:
            return  

        female_id = np.random.choice(self.females)
        female = self.Adults[female_id]
        female.pregnant = True
        self.pregnant_females.append(female.id)  
        self.females.remove(female.id)
        birth_tick = self.current_tick + 9 
        heapq.heappush(self.pregnancy_queue, (birth_tick, female.id))  

    def process_pregnancy(self):
        while self.pregnancy_queue and self.pregnancy_queue[0][0] <= self.current_tick:
            birth_tick, colonist_id = heapq.heappop(self.pregnancy_queue)
            self.pregnant_females.remove(colonist_id)  
            self.females.append(colonist_id)

            colonist = self.Adults[colonist_id] 
            colonist.pregnant = False
            child_id = self.unique_id
            self.unique_id += 1 
            self.population += 1
            self.child_population += 1
            child = Colonist(id=child_id, gender=np.random.randint(0, 2), reactive_behavior=self.reactive_behavior)
            self.Children[child_id] = child  
            transition_tick = self.current_tick + 216
            heapq.heappush(self.children_queue, (transition_tick, child_id))

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
            if colonist_id not in self.Adults:
                continue
            adult = self.Adults.pop(colonist_id)  
            if adult.pregnant:
                delayed_transition_tick = self.current_tick + 9
                heapq.heappush(self.adults_queue, (delayed_transition_tick, colonist_id))
                self.Adults[colonist_id] = adult  
                continue
            
            adult.age = 720  
            adult.age_bracket = 2  
            self.adult_population -= 1
            self.elder_population += 1
            self.Elders[colonist_id] = adult  
            if adult.gender == 0:
                self.males.remove(colonist_id) 
            elif adult.gender == 1:
                self.females.remove(colonist_id) 

            life_expectancy_ticks = generate_elder_death_time()
            transition_tick = self.current_tick + life_expectancy_ticks
            heapq.heappush(self.elders_queue, (transition_tick, colonist_id))

        # Elders Death
        while self.elders_queue and self.elders_queue[0][0] <= self.current_tick:
            transition_tick, colonist_id = heapq.heappop(self.elders_queue)
            elder = self.Elders.pop(colonist_id)  
            elder.age = transition_tick - self.current_tick  
            self.elder_population -= 1
            self.population -= 1

    def process_deaths(self, deficits):
        if self.population <= 0:
            return True  
        if self.inventory[ResourceType.WATER].amount == 0 or self.inventory[ResourceType.OXYGEN].amount == 0:
            return True

        # Base death rate
        death_rate = 0.001
        if deficits:
                for resource_type, deficit in deficits.items():
                    # Scale death rate increase based on deficit severity
                    # Example: Each unit of deficit adds 0.01 to the death rate
                        death_rate += 0.001 * deficit / (self.population)  
                
        #print(death_rate)
        # Calculate the number of deaths using an exponential distribution
        expected_deaths = int(self.population * death_rate)
        number_of_deaths = min(expected_deaths, self.population)  # Ensure deaths don't exceed population
        
        # Process deaths
        all_colonists = list(self.Children.keys()) + list(self.Adults.keys()) + list(self.Elders.keys())
        for _ in range(number_of_deaths):
            if self.population <= 0:
                break  # No one left to process

            # Select a random colonist to die
            chosen_id = np.random.choice(all_colonists)
            all_colonists.remove(chosen_id)

            # Remove the chosen colonist from the appropriate group
            if chosen_id in self.Children:
                del self.Children[chosen_id]
                self.children_queue = [
                    (tick, cid) for tick, cid in self.children_queue if cid != chosen_id
                ]
                heapq.heapify(self.children_queue)
                self.child_population -= 1
            elif chosen_id in self.Adults:
                del self.Adults[chosen_id]
                self.adults_queue = [
                    (tick, cid) for tick, cid in self.adults_queue if cid != chosen_id
                ]
                heapq.heapify(self.adults_queue)
                if chosen_id in self.males:
                    self.males.remove(chosen_id)
                elif chosen_id in self.females:
                    self.females.remove(chosen_id)
                self.pregnancy_queue = [
                    (tick, cid) for tick, cid in self.pregnancy_queue if cid != chosen_id
                ]
                heapq.heapify(self.pregnancy_queue)
                self.adult_population -= 1
            elif chosen_id in self.Elders:
                del self.Elders[chosen_id]
                self.elders_queue = [
                    (tick, cid) for tick, cid in self.elders_queue if cid != chosen_id
                ]
                heapq.heapify(self.elders_queue)
                self.elder_population -= 1

            # Reduce the total population
            self.population -= 1
        # Return False to indicate the simulation continues
        return False

    def process_disaster(self):
        disaster_probability = 0.02
        if np.random.uniform(0, 1) > disaster_probability:
            return False  

        resource_types = list(self.inventory.keys())
        selected_resource_type = np.random.choice(resource_types)
        resource = self.inventory[selected_resource_type]
        amount_to_destroy = np.random.uniform(resource.amount * 0.5, resource.amount * 0.7)
        resource.amount = max(0, resource.amount - amount_to_destroy)
        return True



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

def compare_policies_with_common_random_numbers(
    policies=[1, 2, 3, 4], 
    num_runs=1, 
    initial_population=50, 
    seed=42
):
    # Use a single random number generator with a fixed seed
    rng = np.random.default_rng(seed)
    
    # Generate a single set of seeds to use for ALL policy runs
    common_seeds = rng.integers(0, 2**32 - 1, size=num_runs, dtype=np.uint32)
    
    policy_results = []
    
    for policy_level in policies:
        policy_run_results = []
        
        for run_seed in common_seeds:
            np.random.seed(run_seed)
            
            colony = Colony(initial_population=initial_population, policy_level=policy_level)
            end_reason = colony.simulate()
            
            policy_run_results.append({
                'policy_level': policy_level,
                'seed': run_seed,
                'final_population': colony.population,
                'years_survived': colony.current_tick // 12,
                'end_reason': end_reason,
                'final_resources': {
                    resource_type.name: resource.amount 
                    for resource_type, resource in colony.inventory.items()
                }
            })
        
        policy_results.extend(policy_run_results)
    
    df = pd.DataFrame(policy_results)
    return df

def plot_population_survival(policy_results_df):
    plt.figure(figsize=(15, 10))
    
    # Survival Analysis
    plt.subplot(2, 1, 1)
    survival_summary = policy_results_df.groupby('policy_level')['years_survived'].agg(['mean', 'std'])
    
    survival_summary.plot(kind='bar', y='mean', yerr='std', ax=plt.gca(), 
                           capsize=5, rot=0, 
                           color=['blue', 'green', 'red', 'purple'])
    plt.title('Colony Survival Years by Policy Level')
    plt.xlabel('Policy Level')
    plt.ylabel('Years Survived')
    
    # Success Rate Analysis
    plt.subplot(2, 1, 2)
    success_rate = policy_results_df[policy_results_df['end_reason'] != 'Population Died Out']
    success_summary = success_rate.groupby('policy_level').size() / 10 * 100
    
    success_summary.plot(kind='bar', ax=plt.gca(), 
                          color=['blue', 'green', 'red', 'purple'])
    plt.title('Colony Success Rate by Policy Level')
    plt.xlabel('Policy Level')
    plt.ylabel('Success Rate (%)')
    
    plt.tight_layout()
    plt.savefig('policy_survival_analysis.png')
    plt.close()

def create_policy_comparison_table(policy_results_df):
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    
    # Prepare data for the table
    summary = policy_results_df.groupby('policy_level').agg({
        'final_population': ['mean', 'std'],
        'years_survived': ['mean', 'std']
    })
    
    # Count success and failure
    success_counts = policy_results_df[policy_results_df['end_reason'] != 'Population Died Out']
    success_rate = success_counts.groupby('policy_level').size() / policy_results_df.groupby('policy_level').size() * 100
    
    # Prepare table data
    table_data = [
        ['Policy Level', 'Avg Final Pop', 'Pop Std Dev', 'Avg Years', 'Years Std Dev', 'Success Rate (%)']
    ]
    
    for policy_level in sorted(policy_results_df['policy_level'].unique()):
        row = [
            policy_level,
            f"{summary.loc[policy_level, ('final_population', 'mean')]:.2f}",
            f"{summary.loc[policy_level, ('final_population', 'std')]:.2f}",
            f"{summary.loc[policy_level, ('years_survived', 'mean')]:.2f}",
            f"{summary.loc[policy_level, ('years_survived', 'std')]:.2f}",
            f"{success_rate.get(policy_level, 0):.2f}"
        ]
        table_data.append(row)
    
    # Create the table
    plt.table(cellText=table_data, 
              loc='center', 
              cellLoc='center', 
              colWidths=[0.15]*6)
    plt.title('Policy Comparison Summary')
    plt.tight_layout()
    plt.savefig('policy_comparison_table.png')
    plt.close()

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.table import Table

def run_multiple_simulations_with_confidence_interval(
    num_runs=10, 
    initial_population=50, 
    policy_level=4,  
    confidence_level=0.95,
    precision=0.1
):
    results = []
    successful_ticks = []
    np.random.seed(42)  # Initial random seed
    
    ci_info = {
        'mean': None,
        'variance': None,
        'confidence_interval': None,
        'relative_width': 5000,
        'successful_runs': 0
    }
    
    # Run simulations until we reach the specified number of runs
    while ci_info['relative_width'] > precision:
        current_seed = 42 + len(results)
        np.random.seed(int(time.time()))
        
        # Simulate colony process
        colony = Colony(initial_population=initial_population, policy_level=policy_level)
        end_reason = colony.simulate()
        
        if end_reason != "Population Died Out":
            successful_ticks.append(colony.current_tick)
            results.append({
                'seed': current_seed,  # Store just the seed number
                'final_population': colony.population,
                'years_survived': colony.current_tick // 12,
                'end_reason': end_reason,
                'final_resources': {
                    resource_type.name: resource.amount 
                    for resource_type, resource in colony.inventory.items()
                },
                'population_breakdown': {
                    'children': colony.child_population,
                    'adults': colony.adult_population,
                    'elders': colony.elder_population
                }
            })
            
            # Update confidence interval calculations with each successful run
            if len(successful_ticks) > 1:
                mean_ticks = np.mean(successful_ticks)
                std_error = stats.sem(successful_ticks)
                variance_ticks = np.var(successful_ticks, ddof=1)  # Using n-1 for sample variance
                
                confidence_interval = stats.t.interval(
                    confidence_level,
                    len(successful_ticks) - 1,
                    loc=mean_ticks,
                    scale=std_error
                )
                
                interval_width = confidence_interval[1] - confidence_interval[0]
                relative_width = interval_width / mean_ticks if mean_ticks != 0 else float('inf')
                
                ci_info.update({
                    'mean': mean_ticks,
                    'variance': variance_ticks,
                    'confidence_interval': confidence_interval,
                    'relative_width': relative_width,
                    'successful_runs': len(successful_ticks)
                })
                
                print(f"Run {len(results)}/{num_runs}")
                print(f"Confidence Interval: {confidence_interval}")
                print(f"Relative Width: {relative_width:.4f}")
                print(f"Variance: {variance_ticks:.2f}")
                print(f"Successful Runs: {len(successful_ticks)}\n")
    
    if len(successful_ticks) < 2:
        raise ValueError("Not enough successful runs to calculate confidence intervals")
        
    return results, ci_info

def plot_simulation_results(results):
    if not results:
        raise ValueError("No simulation results available")
    
    # Create figure for simulation results
    fig1 = plt.figure(figsize=(15, len(results) * 0.6 + 2)) 
    ax1 = fig1.add_subplot(111)
    ax1.axis('off')
    
    # Define the column labels for simulation results
    results_labels = [
        'Population', 'Years Survived',
        'Children', 'Adults', 'Elders'
    ]
    
    # Extract data for simulation results
    results_data = [
        [
            str(result['final_population']), 
            str(result['years_survived']), 
            str(result['population_breakdown']['children']), 
            str(result['population_breakdown']['adults']),
            str(result['population_breakdown']['elders'])
        ] for i, result in enumerate(results)
    ]
    
    # Create the simulation results table
    results_table = ax1.table(
        cellText=results_data,
        colLabels=results_labels,
        loc='center',
        cellLoc='center',
        colColours=['lightgrey'] * len(results_labels)
    )
    
    # Format the results table
    results_table.auto_set_font_size(False)
    results_table.set_fontsize(10)  
    results_table.scale(1.5, 2)  
    
    ax1.set_title('Simulation Results with Variance Reduction', pad=20)
    plt.subplots_adjust(top=0.85, bottom=0.05) 
    plt.savefig("test.png",bbox_inches='tight')
    return fig1

def plot_statistical_summary(ci_info):
    if not ci_info['confidence_interval']:
        raise ValueError("No confidence interval information available")
    
    # Create figure for statistical summary
    fig2 = plt.figure(figsize=(8, 4))
    ax2 = fig2.add_subplot(111)
    ax2.axis('off')
    
    # Define the statistical summary data
    summary_labels = ['Metric', 'Value']
    summary_data = [
        ['Mean Ticks', f"{ci_info['mean']:.2f}"],
        ['Variance', f"{ci_info['variance']:.2f}"],
        ['Confidence Interval', f"({ci_info['confidence_interval'][0]:.2f}, {ci_info['confidence_interval'][1]:.2f})"],
        ['Relative Width', f"{ci_info['relative_width']:.4f}"],
        ['Successful Runs', str(ci_info['successful_runs'])]
    ]
    
    # Create the summary table
    summary_table = ax2.table(
        cellText=summary_data,
        colLabels=summary_labels,
        loc='center',
        cellLoc='center',
        colColours=['lightgrey'] * 2
    )
    
    # Format the summary table
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(9)
    summary_table.scale(1.2, 1.5)
    
    ax2.set_title('Statistical Summary with Variance Reduction', pad=20)
    plt.tight_layout()
    
    return fig2

def main():
    results, ci_info = run_multiple_simulations_with_confidence_interval(
        num_runs=100,
        initial_population=50, 
        policy_level=4,
        precision=0.1
    )
    
    print("\nFinal Confidence Interval Results:")
    print(f"Mean Ticks: {ci_info['mean']:.2f}")
    print(f"Variance: {ci_info['variance']:.2f}")
    print(f"Confidence Interval: {ci_info['confidence_interval']}")
    print(f"Final Precision Achieved: {ci_info['relative_width']:.4f}")
    print(f"Successful Runs: {ci_info['successful_runs']}")
    
    # Create and display simulation results
    fig1 = plot_simulation_results(results)
    plt.figure(fig1.number)
    plt.show(block=False)
    
    # Create and display statistical summary in a separate window
    fig2 = plot_statistical_summary(ci_info)
    plt.figure(fig2.number)
    plt.show()

if __name__ == "__main__":
    main()

        '''
       Policy Level  Years until Stabilized
0             3               92.333333
1             4               90.500000
''''