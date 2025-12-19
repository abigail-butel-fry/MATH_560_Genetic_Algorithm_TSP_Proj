import numpy as np
import sys 

def solve_tsp(dist_matrix, num_nodes_remaining, distance): 
    num_nodes = len(dist_matrix)
    min_distance = sys.maxsize 
    shortest_path = []
    current_distance = 0 







def main(): 
    sample_size = 10
    #scaling_factor = 100
    seed = 560

    #fname = f"usa13509_dist_matrix_mi_{sample_size}_nodes_{scaling_factor}_{seed}.npy"
    fname = f"usa13509_dist_matrix_mi_{sample_size}_nodes_{seed}.npy"

    dist_matrix = np.load(fname, allow_pickle=True)

if __name__ == "__main__": 
    main()