"""This file contains functions to generate maps to display TSP data

Source: https://towardsdatascience.com/geopandas-101-plot-any-data-with
        -a-latitude-and-longitude-on-a-map-98e01944b972
"""

import pandas as pd
import matplotlib.pyplot as plt
import descartes
import geopandas as gpd
from os import getcwd, path 
from shapely.geometry import Point, Polygon

import numpy as np

#from os import getcwd
SEED = 560

def plot_lat_lon(coords_fname, route_fname=None, seed=560): 
    # create basic US map 
    parent_dir = path.dirname(getcwd())
    #cwd = getcwd()
    shp_file_path = f"{parent_dir}/tl_2023_us_state/tl_2023_us_state.shp"

    country_map = gpd.read_file(shp_file_path)
    fig, ax = plt.subplots(figsize=(30, 25))
    country_map.plot(ax=ax, color="lightgray")
    #ax.set_xlim(-173, -64)
    #ax.set_ylim(14, 72)
    ax.set_xlim(-125, -65)
    ax.set_ylim(24, 50)

    # load and plot coordinates from current sample 
    coords = np.load(coords_fname, allow_pickle=True)
    # make longitudes negative so they're plotted in western hemisphere
    coords["lon"] *= -1
    num_coords = len(coords)
    #coord_indices = np.arange(num_coords)

    df = pd.DataFrame(coords)
    
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )

    print(gdf.head())
    gdf.plot(aspect=1, 
            ax=ax, 
            markersize=30, 
            color='red', 
            marker='o')
    print(gdf["geometry"])
    
    if route_fname != None: 
        route = np.load(route_fname, allow_pickle=True)[0]
        print(route)
        for node_index in range(num_coords):
            current_node = route[node_index]
            next_node = route[node_index + 1]
            # can't use gdf["geometry"][current_node] because Point 
            # objects aren't iterable
            current_point = (gdf.iloc[current_node]["lon"], 
                             gdf.iloc[current_node]["lat"])
            next_point = (gdf.iloc[next_node]["lon"], 
                          gdf.iloc[next_node]["lat"])
            plt.annotate(
                "",                   # text (empty)
                xy=next_point,        # arrow head
                xytext=current_point, # arrow tail 
                arrowprops=dict(arrowstyle='->', color='black', lw=1)
            )

    plt.savefig(f"{parent_dir}/maps/us13509_map_{num_coords}_pts_seed_{SEED}")


def plot_distance_vs_generation(csv_fname, plot_fname, plot_title): 
    data_df = pd.read_csv(csv_fname)
    data = data_df.iloc[:, 1:].to_numpy()
    print(data)

    num_trials = data.shape[0]
    #x_values = np.arange(num_trials)
    
    plt.figure()
    plt.title(plot_title)
    plt.xlabel("Generation")
    plt.ylabel("Distance (miles)")
    #plt.yticks(np.arange(4700, 6700, 100))
    #plt.ylim((4700, 6700))

    for trial_index in range(num_trials): 
        y_values = data[trial_index]
        plt.plot(y_values, label=f"trial {trial_index}")

    plt.legend()
    plt.savefig(plot_fname)
    
    
    
    

def main():
    sample_sizes = [10, 100, 1000, 10000, 13509]
    parent_dir = path.dirname(getcwd())
    data_dir = f"{parent_dir}/data"
    dataset = f"{data_dir}/usa13509"
    plot_dir = f"{parent_dir}/plots"
    
    """
    # have created basic plot of US; need to plot points and save files
    for size in sample_sizes: 
        coord_fname = f"{data_dir}/usa13509_coords_{size}_nodes_{SEED}.npy"
        guess_fname = f"{data_dir}/usa13509_ortools_best_guess_{size}_nodes_{SEED}.npy"
        plot_lat_lon(coord_fname, guess_fname, seed=SEED)
    """
    tour_len = 10
    init_pop_sizes = [20, 200, 2000, 20000]
    for pop_size in init_pop_sizes: 
        data_fname = f"{dataset}_{tour_len}_nodes_{pop_size}_init_pop_1e-1_mut_prob.csv"
        plot_fname = f"{plot_dir}/dist_vs_gen_{tour_len}_nodes_init_pop_{pop_size}.png"
        plot_title = f"Distance vs. Generation Number (Initial Pop. Size = {pop_size})"
        plot_distance_vs_generation(data_fname, plot_fname, plot_title)

if __name__ == "__main__":
    main()