# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 14:35:15 2025

@author: KDVLabFerro
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 10:38:54 2025

@author: KDVLabFerro
"""



from pylablib.devices import Thorlabs # import the device libraries
from pylablib.devices import uc480
import pandas as pd
#import numpy as np # import numpy for saving
import matplotlib.pyplot as plt
#from pylablib.devices import uc480
import numpy as np
from PIL import Image
import time
import os
import serial
from scipy.optimize import curve_fit
import pims
from skimage.filters import unsharp_mask
import cv2
from sklearn.neighbors import KDTree 
from collections import defaultdict


'''This code will compare the radius distribution function between the first and last 3 runs of a training 
set to compare them.'''


def findNNdf(df, rad):
    '''
    This code uses a KD tree to quickly find how many points are within
    a given distance of each point's neighbor points. Should use rad>droplet radius
    to account for any errors in finding the position.
    
    The function returns the original DataFrame with added columns for nearest neighbor indices,
    number of nearest neighbors, and the fraction of nearest neighbors with a given number of neighbors.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing the x and y coordinates along with any other columns
    rad : float
        The radius distance within which nearest neighbors are found

    Returns:
    df : pandas.DataFrame
        The original DataFrame with added columns for nearest neighbors information
    nn_df : pandas.DataFrame
        DataFrame containing the nearest neighbor indices, number of neighbors, and fraction of neighbors
    '''
    
    # Extract x and y coordinates from the DataFrame
    points = df[['x', 'y']].values
    
    # Build the KDTree with the given points
    tree = KDTree(points, leaf_size=2)
    
    # Find the nearest neighbors within the given radius
    nn_indices = tree.query_radius(points, r=rad)
    
    # Map the indices to particle numbers using the 'particle' column
    particle_numbers = df['particle'].values
    nn_particle_numbers = [particle_numbers[i] for i in nn_indices]
    
    # Calculate the number of nearest neighbors for each point (excluding itself)
    numNN = np.array([len(i) - 1 for i in nn_indices])  # Subtract 1 to exclude the point itself
    
    
    # Add the nearest neighbor data to the original DataFrame
    df['nearest_neighbors'] = nn_particle_numbers
    df['num_neighbors'] = numNN
    


    
    #return df, nn_fraction_df
    return df
def filter_neighbors_by_size(info):
    '''
    This function updates the 'nearest_neighbors' column of the DataFrame by filtering the neighbors 
    based on their distances and sizes. The distance between a particle and its neighbors is compared 
    to the sum of their 'size' values. Only neighbors that are within this distance are retained.
    
    Parameters:
    df : pandas.DataFrame
        The DataFrame containing particle data, including 'x', 'y', 'size', and 'nearest_neighbors'
    x_col : str, optional
        The name of the column representing the x-coordinate (default is 'x')
    y_col : str, optional
        The name of the column representing the y-coordinate (default is 'y')
    size_col : str, optional
        The name of the column representing the particle's size (default is 'size')
    rad_col : str, optional
        The name of the column representing the nearest neighbors' particle numbers (default is 'nearest_neighbors')
    particle_col : str, optional
        The name of the column representing the particle identifier (default is 'particle')
    
    Returns:
    df : pandas.DataFrame
        The DataFrame with the updated 'nearest_neighbors' column, filtered by size-distance criteria
    '''
    df=info
    # Get the particle positions and sizes
    particles = df['particle'].values
    sizes = df['size'].values
    x_coords = df['x'].values
    y_coords = df['y'].values
    
    # Loop through each particle
    updated_neigh_list = []
    numneigh=[]
    for i, particle_id in enumerate(particles):
        
        # Get the list of neighboring particles' IDs
        neighbors = df.loc[i, 'nearest_neighbors']
        
        # Get the current particle's position and size
        x_i, y_i, size_i = x_coords[i], y_coords[i], sizes[i]
        
        # Initialize the list of valid neighbors for this particle
        valid_neighbors = []
        
        # Check each neighbor
        for neighbor_id in neighbors:
            
            #This will skip calling itself it's own neighbour
            if neighbor_id == particle_id:
                continue
            # Get the index of the neighbor
            neighbor_idx = df[df['particle'] == neighbor_id].index[0]
            
            # Get the neighbor's position and size
            x_j, y_j, size_j = x_coords[neighbor_idx], y_coords[neighbor_idx], sizes[neighbor_idx]
            
            # Calculate the distance between the particle and its neighbor
            distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            
            # Check if the distance is less than or equal to the sum of their sizes
            if distance <= size_i + size_j:
                #print('MAX DIS=' + str(distance))
                valid_neighbors.append(neighbor_id)
        
        # Append the valid neighbors' particle IDs to the updated list
        updated_neigh_list.append(valid_neighbors)
        numneigh.append(len(valid_neighbors))
        
    # Replace the 'nearest_neighbors' column with the filtered list of valid neighbors
    df['nearest_neighbors'] = updated_neigh_list
    df['num_neighbors'] = numneigh
    
    return df


total_particles = 0
all_distances = []
total_particles2 = 0
all_distances2 = []


prefix='*.tif'
directories=['G:/Experiments/dec 12/','G:/Experiments/dec 16/']
for directory in directories:
    runs2=[]
    runs=[]
    y_values=[]
    x_values=[]
    ds=150
    i=300
    trainamp=150
    total_runs = 1
    runfinal=500
    depths=[1200,1400]
    depths=[1200]
    widths=[-200,0,100]
    widths=[-200]
    #dss=[100,125,150,175,200]
    dss=[100]#,125,150,175,200]

    #dss=[175,200]
    
    grouped_runs = defaultdict(list)

    for ds in dss:
        for depth in depths:
            for width in widths:
                y = 30 + 0.3833333 * depth    
                x = 640 + 0.3833333 * -width
                
                for r in range(0, runfinal + 1):
                    folder_path = directory + '/training at' + str(ds) + 'from' + str(depth) + ',' + str(width) + 'run' + str(r) + '/'

                    if os.path.exists(folder_path):
                        key = (ds, depth, width)
                        run_tuple = (
                            'training at-' + str(ds) + 'from' + str(depth) + ',' + str(width) + 'run' + str(r) + '/',
                            'training at-' + str(ds) + 'from' + str(depth) + ',' + str(width) + 'run' + str(0) + '/',
                            r,
                            depth,
                            width,
                            ds
                        )
                        grouped_runs[key].append(run_tuple)

    # Separate first 3s and last 3s
    first_three_runs = []
    last_three_runs = []

    for key, group in grouped_runs.items():
        if len(group) >= 6:
            first_three_runs.extend(group[:3])
            last_three_runs.extend(group[-3:])
        else:
            first_three_runs.extend(group[:3])
            last_three_runs.extend(group[-3:])

    # Combine them into a single list of tuples
    runs= first_three_runs 
    runs2 = last_three_runs

    
    
    
    
    #Finding the radial distribution fn for the first three runs
    for run,run0,r,depth,width,dis in runs:  
        droplets=pd.read_csv(directory+run+'extrainfo.csv')
        drop_pos=droplets[['particle', 'x', 'y','size']]
        drop_pos=findNNdf(drop_pos,85)
        
        #ONLY LOOK AT THE SMALL REGION NEAR DROPLET
        drop_pos = drop_pos[(drop_pos['y'] >= 150) & (drop_pos['y'] <= 250) & 
                      (drop_pos['x'] >= 200) & (drop_pos['x'] <= 400)]

        
        total_particles2 += len(drop_pos) 
        print(run)
        # Set the particle column as the index for faster lookup (optional)
        drop_indexed = drop_pos.set_index('particle')
        
        #Load list of mobile particles
        drop_mob=pd.read_csv(directory+run0+'mobiledropletstot.csv')
        drop_mob = drop_mob['particle'].tolist()
        
        drop_pos = drop_pos[
            (drop_pos['particle'].isin(drop_mob))
                ]
        
        for idx, row in drop_pos.iterrows():
            particle_id = row['particle']
            x1, y1 = row['x'], row['y']
            neighbours = row['nearest_neighbors']
            
            for neighbour_id in neighbours:
                # Get the coordinates of the neighbour
                if neighbour_id in drop_indexed.index:
                    x2, y2 = drop_indexed.loc[neighbour_id, ['x', 'y']]
                    # Calculate Euclidean distance
                    dist = (np.sqrt((x2 - x1)**2 + (y2 - y1)**2))/19 #this is in average droplet diameters
                    all_distances.append(dist)

    
    #doing the same for the final three runs
    for run,run0,r,depth,width,dis in runs2:  
        droplets=pd.read_csv(directory+run+'extrainfo.csv')
        drop_pos=droplets[['particle', 'x', 'y','size']]
        drop_pos=findNNdf(drop_pos,85)
        
        #ONLY LOOK AT THE SMALL REGION NEAR FROPLET
        drop_pos = drop_pos[(drop_pos['y'] >= 150) & (drop_pos['y'] <= 250) & 
                      (drop_pos['x'] >= 200) & (drop_pos['x'] <= 400)]

        
        
        total_particles += len(drop_pos) 

        # Set the particle column as the index for faster lookup (optional)
        drop_indexed = drop_pos.set_index('particle')
        
        #Load list of mobile particles
        drop_mob=pd.read_csv(directory+run0+'mobiledropletstot.csv')
        drop_mob = drop_mob['particle'].tolist()
        
        
        drop_pos = drop_pos[
            (drop_pos['particle'].isin(drop_mob))
                ]
        
        for idx, row in drop_pos.iterrows():
            particle_id = row['particle']
            x1, y1 = row['x'], row['y']
            neighbours = row['nearest_neighbors']
            
            for neighbour_id in neighbours:
                # Get the coordinates of the neighbour
                if neighbour_id in drop_indexed.index:
                    x2, y2 = drop_indexed.loc[neighbour_id, ['x', 'y']]
                    # Calculate Euclidean distance
                    dist = (np.sqrt((x2 - x1)**2 + (y2 - y1)**2))/19 #this is in average droplet diameters
                    all_distances2.append(dist)





# Normalize both sets of distances
# First set (for the 'runs' group)
counts1, bins1 = np.histogram(all_distances, bins=70)
counts1[0] = 0  # Avoid zero count at the first bin
bin_centers1 = 0.5 * (bins1[1:] + bins1[:-1])
epsilon = 1e-8
r_squared1 = bin_centers1**2 + epsilon
normalized_counts1 = (counts1 / r_squared1) / total_particles

# Second set (for the 'runs2' group)
counts2, bins2 = np.histogram(all_distances2, bins=70)
counts2[0] = 0  # Avoid zero count at the first bin
bin_centers2 = 0.5 * (bins2[1:] + bins2[:-1])
r_squared2 = bin_centers2**2 + epsilon
normalized_counts2 = (counts2 / r_squared2) / total_particles2

# Plot both histograms on the same figure
plt.figure(figsize=(6, 4))

# Plot the first (maroon) line
plt.plot(bin_centers1, normalized_counts1, color='maroon', linewidth=3, label='Initial')

# Plot the second (green) line
plt.plot(bin_centers2, normalized_counts2, color='green', linewidth=3, label='Final')

# Labels and title
plt.xlabel('r (in droplet diameters)')
plt.ylabel('Probability of neighbour/r^2')
plt.title('Initial g(r)')

# Show legend
plt.legend()

# Make sure everything fits without overlap
plt.tight_layout()

# Display the plot
plt.show()
