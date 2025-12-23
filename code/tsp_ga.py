#*********************************************************************#
# Overarching Question: "How does the obj. function value change with 
# iteration number on TSPs of various sizes?"
# - want to record length of best tour seen so far at each iteration, 
#   graph that as y vs. x values of iteration number, and try to
#   characterize how it changes 
#
# Also, want to vary: 
#   - number of cities
#   - population size
#   - number of solutions carried over between generations
#*********************************************************************#

from os import getcwd, path
from sys import maxsize
import numpy as np
import pandas as pd
import pygad

np.set_printoptions(threshold=maxsize)
SEED = 560
rng = np.random.default_rng(seed=SEED)


def generate_initial_pop(initial_pop_size, num_unique_nodes, guess_fname):
    guess_length = num_unique_nodes + 1 
    initial_pop = np.empty((initial_pop_size, guess_length), 
                           dtype=np.uint32)

    if guess_fname != None: 
        best_guess = np.load(guess_fname, allow_pickle=True)[0]
        initial_pop[0] = np.copy(best_guess)
        next_guess_index = 1
    else: 
        next_guess_index = 0
    

    for guess_index in range(next_guess_index, initial_pop_size): 
        nodes = np.arange(num_unique_nodes)
        current_guess = rng.choice(nodes, size=num_unique_nodes, replace=False)
        endpoint = np.copy(current_guess[0])
        current_guess = np.append(current_guess, endpoint)
        initial_pop[guess_index] = np.copy(current_guess)

        #initial_pop[guess_index][0:num_unique_nodes] = np.copy(current_guess)
        #initial_pop[guess_index][num_unique_nodes] = np.copy(endpoint)
        #for gene_index in range(len(best_guess)):
        #    initial_pop[guess_index][gene_index] = current_guess[gene_index]
    #print(initial_pop)
    return initial_pop


def fitness_function(ga_instance, solution, solution_index): 
    """Evaluate objective function, impose constraints

    The objective function that we are trying to minimize is total
    distance travelled. However, the fitness function is required to
    return high values for more optimal solutions. The constraints are
    that the path must start and end at the same node, and every other
    city must be visited exactly once. 
    """
    #print(f"fitness function params: solution = {solution}")
    #print(f"fitness function params: solution_index = {solution_index}")
    # ensure starting point and ending point are the same
    #print(solution.shape)
    if solution[0] != solution[-1]:
        """print(f"ERROR")
        print(f"{solution_index}:\t{solution}")"""
        print("INFEASIBLE SOL")
        print(f"{solution_index}:\t{solution}")
        return float("-inf")
    
    # next, check that all other nodes are visited once 
    remaining_nodes = solution[1:-1]
    unique_nodes = np.unique(remaining_nodes)

    if len(remaining_nodes) != len(unique_nodes): 
        """print(f"ERROR")
        print(f"{solution_index}:\t{solution}")"""
        print("INFEASIBLE SOL")
        print(f"{solution_index}:\t{solution}")
        return float("-inf")
    
    # at this point, solution is feasible; evaluate total (scaled) distance
    distance = 0
    current_index = 0
    next_index = current_index + 1

    while next_index < len(solution): 
        current_node = int(solution[current_index])
        next_node = int(solution[next_index])
        distance += dist_matrix[current_node][next_node]
        current_index += 1
        next_index += 1 

    # PyGAD maximizes fitness; returning this value ensures that shorter
    # paths yield greater fitness values than longer paths without having 
    # to keep track of scaling factors, etc. 
    """print(f"{solution_index}:\t{solution}\t\t{distance}")"""
    return -1 * distance


def crossover_func(parents, offspring_size, ga_instance):
#def crossover_func(parents, offspring_size):
    """
    Decided to use partially matched crossover after reading pg. 6-7 in
    chapter 14 of Optimization: a Gentle Introduction. This crossover
    operator will prevent offspring from containing duplicate values 
    which violate the constraints of the TSP. Also referenced the 
    description here: 
    https://en.wikipedia.org/wiki/Crossover_(evolutionary_algorithm)#Partially_mapped_crossover_(PMX)
    """
    offspring = np.empty(offspring_size, dtype=np.uint32)
    num_genes = offspring_size[1]
    for offspring_index in range(offspring_size[0]):
        parent_indices = rng.integers(0, len(parents), 2)
        #@note trying this to make sure that the same element isn't chosen twice 
        #@note not sure if I should worry about case where different parents happen to be identical...probably not? 
        while parent_indices[0] == parent_indices[1]:
            parent_indices = rng.integers(0, len(parents), 2)
        #print(f"PARENT INDICES: {parent_indices}")
        p0 = parents[parent_indices[0]]
        #print(f"p0:\t{p0}")
        p1 = parents[parent_indices[1]]
        #print(f"p1:\t{p1}")
        tmp_offspring = [-1] * num_genes
        #print(tmp_offspring)
        
        slice_start = rng.integers(0, num_genes)
        slice_end = rng.integers(slice_start, num_genes)
       
        #print("slice start index: " + str(slice_start))
        #print("slice end index: " + str(slice_end))

        # copy initial segment from p0
        for i in range(slice_start, slice_end + 1): 
            tmp_offspring[i] = p0[i]

        # ensure that if the start/end is populated, it matches on both ends
        if tmp_offspring[0] != -1:
            tmp_offspring[-1] = tmp_offspring[0]
        elif tmp_offspring[-1] != -1: 
            tmp_offspring[0] = tmp_offspring[-1]

        #print(tmp_offspring)

        # copy genes from p1 that which occur in this segment but which are
        # not yet included in the offspring
        for i in range(slice_start, slice_end + 1): 
            # this is m in the description of the algorithm on Wikipedia 
            curr_p1_gene = p1[i]

            if curr_p1_gene in tmp_offspring: 
                #print(str(curr_p1_gene) + " already occurs in offspring")
                continue

            else: 
                # this is n in the description of the algorithm on Wikipedia
                corresponding_offspring_gene = tmp_offspring[i]
                indices = np.where(p1==corresponding_offspring_gene)[0]
                index = indices[0]
            
                # in the offspring, copy m into the position where n is in p1
                # if that position is not already occupied 
                if tmp_offspring[index] == -1: 
                    tmp_offspring[index] = curr_p1_gene 
                    # ensure start/endpoints match 
                    if index == 0 or index == num_genes - 1: 
                        tmp_offspring[(num_genes - 1) - index] = curr_p1_gene
                else: 
                    # the place where n is in p1 is occupied in offspring
                    new_index = np.where(p1==tmp_offspring[index])[0][0]
                    tmp_offspring[new_index] = curr_p1_gene
                    # ensure start/endpoints match 
                    if new_index == 0 or new_index == num_genes - 1: 
                        tmp_offspring[(num_genes - 1) - new_index] = curr_p1_gene
        #print(tmp_offspring)
        # fill in remaining genes from p1 which have not appeared in offspring 
        unused_p1_genes = np.array([gene for gene in p1 \
                                    if gene not in tmp_offspring])
        #print(unused_p1_genes)
        for gene in unused_p1_genes:
            if -1 in tmp_offspring:
                first_unused_index = tmp_offspring.index(-1)
                tmp_offspring[first_unused_index] = gene 
                if first_unused_index == 0: 
                    tmp_offspring[-1] = gene 
            else:
                break
        #print(tmp_offspring)

        tmp_offspring = np.array(tmp_offspring)
        #for i in range(num_genes):
        #    offspring[offspring_index][i] = tmp_offspring[i]
        #print(f"offspring:\t{tmp_offspring}")
        offspring[offspring_index] = np.ndarray.copy(tmp_offspring)
       
    return offspring


def mutation_func(offspring, ga_instance):
    """For the time being, rather than selecting genes to swap, if an
    offspring is to undergo mutation, I will select a single (random) 
    gene in that offspring, remove it from the offspring, and insert it
    into a random index of that offspring. After this operation, I will 
    ensure that the offspring is still a feasible solution. 
    """
    #print(offspring)
    num_genes = len(offspring[0])

    for i in range(len(offspring)):
        tmp_offspring = np.ndarray.copy(offspring[i])
        if rng.random() < ga_instance.mutation_probability:
            #print(offspring[i])
            indices = rng.integers(0, num_genes, 2)
            initial_index = indices[0]
            final_index = indices[1]
            affected_gene = np.copy(tmp_offspring[initial_index])
            tmp_offspring = np.delete(tmp_offspring, initial_index)
            
            if initial_index == 0: 
                tmp_offspring[-1] = tmp_offspring[0]
            elif initial_index == num_genes - 1: 
                tmp_offspring[0] = tmp_offspring[-1]
            
            tmp_offspring = np.insert(tmp_offspring, final_index, 
                                      affected_gene)

            if final_index == 0: 
                tmp_offspring[-1] = tmp_offspring[0]
            elif final_index == num_genes - 1:
                tmp_offspring[0] = tmp_offspring[-1]
            #print(offspring)
            offspring[i] = np.ndarray.copy(tmp_offspring)
            #print(offspring[i])
    
    #print(offspring)
    return offspring

def on_gen(ga_instance):
    print(f"Generation : {ga_instance.generations_completed}", flush=True)
    print(f"Fitness of the best solution : {ga_instance.best_solution()[1]}", flush=True)


def main(): 
    """Entrypoint for program"""
    global dist_matrix 
    #global scaling_factor
    #global rng
    #sample_sizes = [10, 100, 1000, 10000, 13509]
    sample_sizes = [10]
    initial_pop_sizes = [20, 200, 2000, 20000] 
    num_trials = 5
   
    parent_dir = path.dirname(getcwd())
    data_dir = f"{parent_dir}/data"
    dataset_name = "usa13509"
    unit = "mi"
    extension = ".npy"
    ortools_solver_time_lim = 300

    for num_nodes in sample_sizes:
        #print(f"SAMPLE SIZE = {num_nodes}")
        dist_matrix_fname = f"{data_dir}/{dataset_name}_dist_matrix_{unit}_"\
                            f"{num_nodes}_nodes_{SEED}{extension}"

        dist_matrix = np.load(dist_matrix_fname, allow_pickle=True)
        #print(dist_matrix)

        initial_pop_size = 2000 #num_nodes * 100
        num_generations = 50 
        # top 10% of organisms will mate 
        num_parents_mating = round(0.10 * initial_pop_size)
        parent_selection_type = "rws" # roulette wheel selection
        #keep_parents = -1 # keep default (-1) for now; has no effect if keep_elitism = 1
        # keep n best solutions in the next generation 
        keep_elitism = 1 # keep default (1) for now 
        mutation_probability = 0.1
        save_best_solutions = False
        save_solutions = False
    
        guess_fname = f"{data_dir}/{dataset_name}_ortools_best_guess_{num_nodes}_nodes_{SEED}.npy"
        distance_array = np.empty((num_trials, num_generations), dtype=np.uint32)
        distance_csv = f"{data_dir}/{dataset_name}_{num_nodes}_nodes_{initial_pop_size}_init_pop_1e-1_mut_prob.csv"
    
        for trial_index in range(num_trials):
            #initial_pop = generate_initial_pop(initial_pop_size, num_nodes, guess_fname)     
            initial_pop = generate_initial_pop(initial_pop_size, num_nodes, None)            
            #print(f"initial_pop shape: {initial_pop.shape}")
            ga_instance = pygad.GA(num_generations=num_generations, 
                                num_parents_mating=num_parents_mating, 
                                fitness_func=fitness_function,
                                initial_population=initial_pop, 
                                gene_type=np.uint32,
                                parent_selection_type=parent_selection_type,
                                #keep_parents=keep_parents,
                                keep_elitism=keep_elitism,
                                crossover_type=crossover_func,
                                mutation_type=mutation_func,
                                mutation_probability=mutation_probability,
                                on_generation=on_gen,
                                save_best_solutions=save_best_solutions, 
                                save_solutions=save_solutions)
            
            ga_instance.run()
            
            print(ga_instance.best_solution())
            approx_distance = ga_instance.best_solution()[1] * -1
            #print(f"distance: {approx_distance}")
            #ga_instance.plot_fitness()
            #print(f"generation where best fitness spotted: {ga_instance.best_solution_generation}")

            for sol_index in range(num_generations):
                distance_array[trial_index][sol_index] = -1 * ga_instance.best_solutions_fitness[sol_index]
            #print(distance_array[trial_index])

            print(ga_instance.best_solutions_fitness)
        #print(distance_array)
        df = pd.DataFrame(distance_array)
        df.to_csv(distance_csv)

if __name__ == "__main__":
    main()

