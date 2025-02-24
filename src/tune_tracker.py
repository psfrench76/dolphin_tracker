from random import shuffle

from track import run_tracking_and_evaluation
import click
import yaml
import random
import os
import math

# TODO: Intermediate logging every generation

@click.command()
@click.option('--dataset', required=True, help="Path to the dataset directory.")
@click.option('--model', required=True, help="Path to the model file.")
@click.option('--run_name', required=True, help="Name of the run for saving evolved configurations.")
@click.option('--tracker', required=True, help="Path to the baseline tracker configuration file.")
@click.option('--ps', default=5, help="Population size.")
@click.option('--gen', default=3, help="Number of generations.")
@click.option('--mrate', default=0.2, help="Mutation rate.")
@click.option('--mrange', default=0.1, help="Mutation range.")
@click.option('--dist', default='uniform', help="Mutation distribution.")
@click.option('--np', default=2, help="Number of parents.")
def main(dataset, model, run_name, tracker, ps, gen, mrate, mrange, dist, np):
    print(f"Running tracker from tune_tracker.py")
    evolve_tracker(dataset, model, run_name, tracker, population_size=ps, generations=gen, mutation_rate=mrate,
                   mutation_range=mrange, distribution=dist, num_parents=np)


class Tracker:

    def __init__(self, config, id, generation, model, dataset, distribution='uniform'):
        if isinstance(config, str):
            cfg = yaml.safe_load(config)
        elif isinstance(config, dict):
            cfg = config
        else:
            raise ValueError("Config must be a string or a dictionary")

        self.mins = {}
        self.maxs = {}

        with open('dolphin_tracker/cfg/tracking_hyperparameters.yaml', 'r') as file:
            hyp_cfg_dict = yaml.safe_load(file)
            hyp_dict = {}
            fixed_dict = {}
            for key, (min_val, max_val) in hyp_cfg_dict['hyp'].items():
                hyp_dict[key] = cfg[key]
                self.mins[key] = min_val
                self.maxs[key] = max_val

            for key in hyp_cfg_dict['fixed']:
                fixed_dict[key] = (cfg[key])

        self.hyp = hyp_dict
        self.fixed = fixed_dict
        self.distribution = distribution
        self.model = model
        self.dataset = dataset
        self.fitness = None
        self.tracker_file = None
        self.mutated_params = {}
        self.id = id
        self.crossover = {}
        self.generation = generation
        self.parents = []

    def write_yaml(self, filename):
        with open(filename, 'w') as f:
            params = self.hyp.copy()
            params.update(self.fixed)
            yaml.dump(params, f)
        self.tracker_file = filename

    def mutate_hyperparameters(self, mutation_rate, mutation_range):
        self.fitness = None
        for key, value in self.hyp.items():
            if random.random() < mutation_rate:
                mutation_limit = value * mutation_range
                if self.distribution == 'uniform':
                    mutation = random.uniform(-mutation_limit, mutation_limit)
                elif self.distribution == 'normal':
                    mutation = random.gauss(0, mutation_limit / 2)
                else:
                    raise ValueError(f"Unsupported distribution type: {self.distribution}")

                if isinstance(value, float):
                    new_value = max(min(value + mutation, self.maxs[key]), self.mins[key])
                    self.hyp[key] = new_value
                    self.mutated_params[key] = self.hyp[key]
                elif isinstance(value, int):
                    new_value = int(max(min(value + mutation, self.maxs[key]), self.mins[key]))
                    if new_value != value:
                        self.hyp[key] = new_value
                        self.mutated_params[key] = self.hyp[key]

    def evaluate(self, output):
        if self.fitness is None:
            results = run_tracking_and_evaluation(self.dataset, self.model, output, self.tracker_file)
            mota = results['mota'].values[0]
            idf1 = results['idf1'].values[0]
            motp = results['motp'].values[0]
            self.fitness = float(mota + idf1 + (1 - motp))
        return self.fitness


class GeneticAlgorithm:

    def __init__(self, config_path, model, dataset, run_name, output, population_size=10, generations=20,
                 mutation_rate=0.1, mutation_range=0.1, distribution='uniform', num_parents=2,
                 initial_temperature=1.0, temperature_threshold=0.01):
        with open('dolphin_tracker/cfg/settings.yaml', 'r') as file:
            self.settings = yaml.safe_load(file)

        self.config_path = config_path
        self.model = model
        self.dataset = dataset
        self.run_name = run_name
        self.output = output
        self.population_size = population_size
        self.num_generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_range = mutation_range
        self.distribution = distribution
        self.num_parents = num_parents

        self.yaml_dir = str(os.path.join(self.settings['tracker_config_dir'], 'evolved', self.run_name))
        self.population = []
        self.generations = []
        self.trackers = []
        self.nans = set()
        self.population_probabilities = {}
        self.global_id_counter = 0
        self.generation_number = 1
        self.initial_temperature = initial_temperature
        self.temperature_threshold = temperature_threshold
        self.cooling_rate = self.calculate_cooling_rate()

        self.initialize_population()

    def calculate_cooling_rate(self):
        return (self.temperature_threshold / self.initial_temperature) ** (1 / (self.num_generations - 1))

    def temperature(self, generation=None):
        if generation is None:
            generation = self.generation_number
        return self.initial_temperature * (self.cooling_rate ** (generation - 1))

    def new_id(self):
        this_id = self.global_id_counter
        self.global_id_counter += 1
        return this_id

    def generation_dir(self):
        directory = str(os.path.join(self.yaml_dir, f"gen{self.generation_number}"))
        os.makedirs(directory, exist_ok=True)
        return directory

    def initialize_population(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.population = []
        print(
            f"Initializing Population of size {self.population_size}; Generation {self.generation_number} of {self.num_generations}")
        for i in range(self.population_size):
            tracker = Tracker(config, self.new_id(), self.generation_number, self.model, self.dataset,
                              self.distribution)
            tracker.mutate_hyperparameters(self.mutation_rate, self.mutation_range)
            self.population.append(tracker.id)
            self.trackers.append(tracker)

            child_file = os.path.join(self.generation_dir(), f"{tracker.id}.yaml")
            tracker.write_yaml(child_file)

        self.save_generation()

    def evaluate_population(self):
        print(f"\n-------------------------------------------\n"
              f"Evaluating Generation {self.generation_number - 1} of {self.num_generations}\n"
              f"-------------------------------------------\n")
        for i, tracker_id in enumerate(self.population):
            print(f"\nEvaluating tracker {tracker_id}, member {i} of {len(self.population)} in generation "
                  f"{self.generation_number - 1} of {self.num_generations}\n")
            self.trackers[tracker_id].evaluate(self.output)

    def save_generation(self):
        self.generations.append([tracker_id for tracker_id in self.population])
        self.generation_number += 1

    def get_probabilities(self, subpopulation):
        fitness_values = [self.trackers[tracker_id].fitness for tracker_id in subpopulation]
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        max_fitness = max_fitness if max_fitness != min_fitness else min_fitness + 1

        normalized_fitness = [(f - min_fitness) / (max_fitness - min_fitness) for f in fitness_values]
        temperature = self.temperature(self.generation_number - 1)
        exp_values = [math.exp((f - 1) / temperature) for f in normalized_fitness]
        for i, e in enumerate(exp_values):
            if math.isnan(e):
                self.nans.add(subpopulation[i])
                exp_values[i] = 0
        sum_exp_values = sum(exp_values)
        probabilities = [exp_val / sum_exp_values for exp_val in exp_values]

        print(f"\nPotential parent ids: {subpopulation}")
        print(f"Fitness values: {fitness_values}")
        print(f"Normalized fitness: {normalized_fitness}")
        print(f"Exp values: {exp_values}")
        print(f"Probabilities: {probabilities}")

        return probabilities

    def pop_parent(self, potential_parents, best=False):
        parent = None
        shuffle(potential_parents)
        if best:
            parent = max(potential_parents, key=lambda x: self.trackers[x].fitness)
        else:
            probabilities = self.get_probabilities(potential_parents)
            while parent is None:
                for tracker_id, p in zip(potential_parents, probabilities):
                    if random.random() < p:
                        parent = tracker_id
                        break
        potential_parents.remove(parent)
        return parent

    def select_parents(self):
        print(f"Selecting parents for generation {self.generation_number}")
        selected_parents = set()
        potential_parents = self.population.copy()

        # For the sake of logging
        probabilities = self.get_probabilities(potential_parents)
        self.population_probabilities[self.generation_number - 1] = zip(potential_parents.copy(), probabilities.copy())

        # Always keep best 2
        for _ in range(min(self.num_parents, 2)):
            selected_parents.add(self.pop_parent(potential_parents, best=True))

        print(f"Selected parents: {selected_parents}")
        print(f"Temperature: {self.temperature(self.generation_number - 1)}")

        # If we need more parents, select more probabilistically through simulated annealing
        while len(selected_parents) < self.num_parents:
            selected_parents.add(self.pop_parent(potential_parents, best=False))
            print(f"Selected parents: {selected_parents}")

        return list(selected_parents)

    def crossover(self, parent1, parent2):
        child_config = {}
        child_number = self.new_id()

        crossover_details = {}

        for key in parent1.hyp.keys():
            chosen_parent = random.choice([parent1, parent2])
            child_config[key] = chosen_parent.hyp[key]
            crossover_details[key] = {
                'parent': parent1.id if chosen_parent == parent1 else parent2.id,
                'value': child_config[key]
            }

        child_config.update(parent1.fixed)
        child = Tracker(child_config, child_number, self.generation_number, self.model, self.dataset, self.distribution)
        child.crossover = crossover_details
        child.parents = [parent1.id, parent2.id]

        child.mutate_hyperparameters(self.mutation_rate, self.mutation_range)

        child_file = os.path.join(self.generation_dir(), f"{child_number}.yaml")
        child.write_yaml(child_file)
        return child

    def run(self):
        while self.generation_number <= self.num_generations:
            self.evaluate_population()

            parents = self.select_parents()
            self.population = parents.copy()
            while len(self.population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(self.trackers[parent1], self.trackers[parent2])
                self.population.append(child.id)
                self.trackers.append(child)
            self.save_generation()

        self.evaluate_population()
        best_tracker = max(self.population, key=lambda x: self.trackers[x].fitness)
        self.save_evolutions()
        return self.trackers[best_tracker]

    def save_evolutions(self):
        population_history = {'generations': {}}
        for generation_number, generation in enumerate(self.generations, start=1):
            generation_details = {
                'temperature': self.temperature(generation_number),
                'children': [],
                'parents': []
            }
            if generation_number in self.population_probabilities:
                generation_details['population_probabilities'] = {tracker_id: prob for tracker_id, prob in
                                                                  self.population_probabilities[generation_number]}

            for tracker_id in generation:
                tracker = self.trackers[tracker_id]
                if tracker.generation == generation_number:
                    generation_details['children'].append({
                        'id': tracker.id,
                        'fitness': tracker.fitness,
                        'crossover': tracker.crossover,
                        'mutations': tracker.mutated_params,
                        'parents': tracker.parents
                    })
                else:
                    generation_details['parents'].append({
                        'id': tracker.id,
                        'fitness': tracker.fitness,
                        'generation': tracker.generation
                    })

            population_history['generations'][generation_number] = generation_details

        evolutions_file = os.path.join(self.output, 'evolutions.yaml')
        with open(evolutions_file, 'w') as f:
            yaml.dump(population_history, f)


def evolve_tracker(dataset, model, run_name, baseline_tracker, population_size, generations, mutation_rate,
                   mutation_range, distribution, num_parents, initial_temperature=1.0, temperature_threshold=0.01):
    output = os.path.join("output", run_name)

    ga = GeneticAlgorithm(config_path=baseline_tracker,
                          model=model,
                          dataset=dataset,
                          run_name=run_name,
                          output=output,
                          population_size=population_size,
                          generations=generations,
                          mutation_rate=mutation_rate,
                          mutation_range=mutation_range,
                          distribution=distribution,
                          num_parents=num_parents,
                          initial_temperature=initial_temperature,
                          temperature_threshold=temperature_threshold)
    best_tracker = ga.run()
    original_tracker_file = best_tracker.tracker_file
    best_tracker_file = os.path.join(output, "best_tracker.yaml")
    best_tracker.write_yaml(best_tracker_file)

    results = run_tracking_and_evaluation(dataset, model, output, best_tracker_file)

    print(f"\nEvolution finished. Best results:\n{results}")
    print(f"Best tracker: {best_tracker.id} from generation {best_tracker.generation}. Fitness: {best_tracker.fitness}")
    print(f"Ran {generations} generations with {population_size} trackers per generation.")
    print(f"Baseline tracker: {baseline_tracker}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    if ga.nans:
        print(f"\nWARNING: NaNs were encountered in the fitness function for the following trackers: {ga.nans}")
    print(f"\nOriginal best tracker configuration saved to: {original_tracker_file}")
    print(f"Best tracker configuration saved to: {best_tracker_file}")
    print(f"Evolutions details saved to: {os.path.join(output, 'evolutions.yaml')}")


if __name__ == '__main__':
    main()
