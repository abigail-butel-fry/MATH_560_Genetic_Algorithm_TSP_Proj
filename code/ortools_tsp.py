"""This file contains code used to compute distance matrices from .tsp
files (with EDGE_WEIGHT_TYPE EUC_2D). It also uses Google's OR-Tools to
solve the TSP. 
Source: 
https://developers.google.com/optimization/routing/tsp#complete_programs

This code will be used to get a baseline for what reasonable solutions
may look like in order to assess the quality of the solutions that the
genetic algorithm code generates.
"""

from geopy import distance
import numpy as np
from os import getcwd, path

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

seed = 560
rng = np.random.default_rng(seed=seed)

def read_tsp_file(fname, num_header_rows, num_nodes_file, num_nodes_sample):
    """Read coordinates from .tsp file (downloaded from TSPLIB)

    Parameters
    ----------
    fname : str
        The name of the .tsp file to be parsed.
    num_header_tows : int 
        The number of rows at the top of the .tsp file which contain
        text-based metadata. 
    num_nodes_file : int 
        The total number of entries (i.e. node names/numbers and
        coordinate pairs) in the .tsp file. 
    num_nodes_sample : int 
        The number of entries that should be randomly selected from the
        file and returned. 

    Returns
    -------
    np.ndarray
        Array where each entry is a tuple containing the name/number, 
        latitude, and longitude for a given node. 
    """
    coordinates = np.loadtxt(fname,
                             dtype=np.dtype({"names": ["id", "lat", "lon"],
                                             "formats": ["u4", "d", "d"]}),
                             skiprows=num_header_rows,
                             ndmin=1,
                             max_rows=num_nodes_file)
    
    # scale coordinates to get valid lat/lon values (decimal degrees)
    for index, _ in enumerate(coordinates): 
        coordinates[index]["lat"] *= 1e-4 
        coordinates[index]["lon"] *= 1e-4 

    random_points = rng.choice(coordinates, size=num_nodes_sample,
                               replace=False)
    
    return np.reshape(random_points, (num_nodes_sample,))


def compute_dist_matrix(coordinate_array):
    """Create array of distances between every pair in coordinate_array

    In this 2-d array, the entry in the i-th row and j-th column will
    correspond to the distance (in miles) between the i-th and j-th
    nodes in coordinate_array. These distances will be rounded to the
    nearest integer because Google's OR-Tools routing solver expects
    integer data. 

    @note
    http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/TSPFAQ.html
    https://stackoverflow.com/questions/49753792/calculating-distances-in-tsplib
    """
    num_nodes = len(coordinate_array)
    distances = np.empty(shape=(num_nodes, num_nodes), dtype=np.uint32)
    
    for i in range(num_nodes):
        node_i_coords = (coordinate_array[i]["lat"], coordinate_array[i]["lon"])

        # since the distances are symmetric, only need to compute 
        # distances on/above the diagonal to populate entire matrix
        for j in range(i, num_nodes):
            node_j_coords = (coordinate_array[j]["lat"], coordinate_array[j]["lon"])
            
            dist = distance.great_circle(node_i_coords, node_j_coords).miles
            rounded_dist = round(dist)

            distances[i][j] = rounded_dist
            distances[j][i] = rounded_dist
    
    return distances


def create_data_model(fname):
    """Stores the data for the problem.
    """
    data = {}
    data["distance_matrix"] = np.load(fname, allow_pickle=True)
    data["num_vehicles"] = 1
    data["depot"] = 0 # index of start/end city
    return data


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()} miles")
    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += f" {manager.IndexToNode(index)} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f" {manager.IndexToNode(index)}\n"
    plan_output += f"Route distance: {route_distance} miles\n"
    print(plan_output)


def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes


def main():
    """Entry point of the program."""
    sample_sizes=[10]
    #sample_sizes = [10, 100, 1000, 10000, 13509]
    #scaling_factor = 100
    parent_dir = path.dirname(getcwd())
    tsp_fname = f"{parent_dir}/usa13509.tsp"
    data_dir = f"{parent_dir}/data"
    print(tsp_fname)
    time_lim_sec = 300

    #************************************************************************#
    #          LOAD COORDINATES; LOAD OR COMOPUTE DISTANCE MATRICES          #
    #************************************************************************#

    for size in sample_sizes: 
        coord_fname = f"{data_dir}/usa13509_coords_{size}_nodes_{seed}.npy"
        dist_mat_fname = f"{data_dir}/usa13509_dist_matrix_mi_{size}_nodes_{seed}.npy"

        # first, check whether a .npy file containing these coordinates exists
        if path.exists(coord_fname) and path.getsize(coord_fname) > 0: 
            coords = np.load(coord_fname, allow_pickle=True)
        else: # if it doesn't exist, create it
            coords = read_tsp_file(tsp_fname, num_header_rows=9, 
                                   num_nodes_file=13509, 
                                   num_nodes_sample=size)
            np.save(coord_fname, coords) 

        # next, determine whether a file containing a distance matrix exists
        # for this set of coordinates
        if path.exists(dist_mat_fname) and path.getsize(dist_mat_fname) > 0:
            dist_matrix = np.load(dist_mat_fname, allow_pickle=True)
        else: # if it doesn't exist, create it
            dist_matrix = compute_dist_matrix(coords)
            np.save(dist_mat_fname, dist_matrix)

        #*************************************************************#
        #          FOR CURRENT SET OF POINTS, RUN THE SOLVER          #
        #*************************************************************#
        # Instantiate the data problem.
        data = create_data_model(dist_mat_fname) 

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.time_limit.seconds = time_lim_sec 


        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            print_solution(manager, routing, solution)
            first_route = np.array(get_routes(solution, routing, manager)[0])
            first_distance = np.array(solution.ObjectiveValue())
            first_guess = np.empty(2, dtype=object)
            first_guess[0] = first_route.copy()
            first_guess[1] = first_distance.copy()
            print(first_guess)
            guess_fname = f"{data_dir}/usa13509_ortools_guess_{size}_nodes_{seed}_PATH_CHEAPEST_ARC.npy"
            np.save(guess_fname, first_guess)


        # Try guided local search strategy.
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = time_lim_sec #@note not sure if this needs to be repeated?
        #search_parameters.log_search = True

         # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            print_solution(manager, routing, solution)
            new_route = np.array(get_routes(solution, routing, manager)[0])
            new_distance = np.array(solution.ObjectiveValue())
            new_guess = np.empty(2, dtype=object)
            new_guess[0] = new_route.copy()
            new_guess[1] = new_distance.copy()
            print(new_guess)
           #@note write route information, overall distance to a file
            guess_fname = f"{data_dir}/usa13509_ortools_guess_{size}_nodes_{seed}_{time_lim_sec}_sec.npy"
            np.save(guess_fname, new_guess)

        print("next iteration")


if __name__ == "__main__":
    main()



    