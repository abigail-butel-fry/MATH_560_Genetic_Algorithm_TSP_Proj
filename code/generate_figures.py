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

def plot_lat_lon(fname, seed=560): 
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
    coords = np.load(fname, allow_pickle=True)
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
    
    for index, row in gdf.iterrows(): 
        #print(f"index = {index}")
        print(f"xy=({row["lon"]}, {row["lat"]})")
        label = str(index)
        #print(row["geometry"])
        #textcoords=?
        ax.annotate(label, xy=(row["lon"], row["lat"]),
                    fontsize=15, weight="bold")
        
    
    plt.savefig(f"{parent_dir}/maps/us13509_map_{num_coords}_pts_seed_{seed}")
    

def main():
    #sample_sizes = [10, 100, 1000, 10000, 13509]
    sample_sizes = [10]
    #scaling_factor = 100
    seed = 560
    parent_dir = path.dirname(getcwd())
    
    # have created basic plot of US; need to plot points and save files
    for size in sample_sizes: 
        #coord_fname = f"usa13509_coords_{size}_nodes_{scaling_factor}_{seed}.npy"
        coord_fname = f"{parent_dir}/data/usa13509_coords_{size}_nodes_{seed}.npy"
        plot_lat_lon(coord_fname, seed)

    #@todo write function to print route on a map


if __name__ == "__main__":
    main()