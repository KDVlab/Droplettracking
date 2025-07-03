# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 03:47:01 2025

@author: MemoryPC
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
def preprocess_img(frame,x0,xf,y0,yf):                                                                    #CHANGE#
    frame = frame[y0:yf,x0:xf]
    frame = unsharp_mask(frame, radius = 2, amount = 5)
    frame *= 255.0/frame.max()
    return frame

def filter_neighbors_by_size(info,tol):
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
        x_i, y_i, size_i = x_coords[i], y_coords[i], sizes[i]*tol
        
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


numneigh=[]
allangles=[]
numneigh2=[]
allangles2=[]

#This loads up all the relevant experimental runs according to how I've saved them
#Note that the final runs are known to be in a trained state because I only stop exps once the system has trained


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
    #depths=[1200]
    widths=[-200,0,100]
    #widths=[-200]
    
    dss=[100,125,150,175,200]
    #dss=[200]

    #dss=[175,200]
    dire='F:/Experiments/aug9/'
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
    
    
    
    
    #Finding radial distribution of bonds for the initial, untrained system
    for run,run0,r,depth,width,dis in runs:    
        print(run)
        droplets=pd.read_csv(directory+run+'extrainfo.csv')
        drop_nn=droplets[['particle', 'x', 'y','size']]
        
        drop_nn=findNNdf(drop_nn,30)
        drop_nn=filter_neighbors_by_size(drop_nn,1.15)

        #droplets = droplets.merge(drop_nn[['particle', 'num_neighbors', 'nearest_neighbors']], on='particle', how='left')
        #drops = drops.loc[:, ~droplets.columns.str.contains('^Unnamed')]
        drop_nn.to_csv(directory+run+'nearest_neighbours.csv', index=False)
        #print('DONE')
        
        #Load list of mobile particles
        drop_mob=pd.read_csv(directory+run0+'mobiledropletstot.csv')
        drop_mob = drop_mob['particle'].tolist()
        
        angles = []  # List to store the angles
        numneigh.append(drop_nn['num_neighbors'].mean())
        # Create a lookup dict for quick position access
        position_dict = drop_nn.set_index('particle')[['x', 'y']].to_dict('index')

        position_dict = drop_nn.set_index('particle')[['x', 'y']].to_dict('index')
        '''
        #ONLY LOOK AT THE SMALL REGION NEAR FROPLET
        drop_nnfil = drop_nn[(drop_nn['y'] >= 150) & (drop_nn['y'] <= 250) & 
                      (drop_nn['x'] >= 200) & (drop_nn['x'] <= 400)]
        '''
        
        #ONly look at the mobile droplets:
            
        drop_nnfil = drop_nn[
            (drop_nn['particle'].isin(drop_mob))
                ]
        for idx, row in drop_nnfil.iterrows():
            particle_id = row['particle']
            x1, y1 = row['x'], row['y']
            
            neighbors = row['nearest_neighbors']
            
            for neighbor_id in neighbors:
                if neighbor_id in position_dict:
                    x2, y2 = position_dict[neighbor_id]['x'], position_dict[neighbor_id]['y']
                    
                    # Compute angle of line from (x1, y1) to (x2, y2)
                    angle = np.arctan2(y2 - y1, x2 - x1)  # in radians
                    # angle_deg = np.degrees(angle)      # if you want degrees
                    
                    angles.append(angle)  # or append angle_deg
                    allangles.append(angle)
        angles = np.array(angles)


        # Set number of bins (e.g., 36 bins = 10° per bin)
        num_bins = 36

        # Create histogram
        counts, bin_edges = np.histogram(angles, bins=num_bins, range=(-np.pi, np.pi))

        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        

    #Doing the same for the final three runs
    
    for run,run0,r,depth,width,dis in runs2:    
        
        droplets=pd.read_csv(directory+run+'extrainfo.csv')
        drop_nn=droplets[['particle', 'x', 'y','size']]
        
        drop_nn=findNNdf(drop_nn,30)
        drop_nn=filter_neighbors_by_size(drop_nn,1.1)

        #droplets = droplets.merge(drop_nn[['particle', 'num_neighbors', 'nearest_neighbors']], on='particle', how='left')
        #drops = drops.loc[:, ~droplets.columns.str.contains('^Unnamed')]
        drop_nn.to_csv(directory+run+'nearest_neighbours.csv', index=False)
        angles = []  # List to store the angles
        numneigh2.append(drop_nn['num_neighbors'].mean())
        # Create a lookup dict for quick position access
        position_dict = drop_nn.set_index('particle')[['x', 'y']].to_dict('index')

        position_dict = drop_nn.set_index('particle')[['x', 'y']].to_dict('index')
        
        '''
        #ONLY LOOK AT THE SMALL REGION NEAR DROPLET
        drop_nnfil = drop_nn[(drop_nn['y'] >= 150) & (drop_nn['y'] <= 250) & 
                      (drop_nn['x'] >= 200) & (drop_nn['x'] <= 400)]
        '''
        
        #ONly look at the mobile droplets(from a pre-existing file containing just mobile droplet information):
        #Load list of mobile particles
        drop_mob=pd.read_csv(directory+run0+'mobiledropletstot.csv')
        drop_mob = drop_mob['particle'].tolist()
            
        drop_nnfil = drop_nn[
            (drop_nn['particle'].isin(drop_mob))]
                
        for idx, row in drop_nnfil.iterrows():
            particle_id = row['particle']
            x1, y1 = row['x'], row['y']
            
            neighbors = row['nearest_neighbors']
            
            for neighbor_id in neighbors:
                if neighbor_id in position_dict:
                    x2, y2 = position_dict[neighbor_id]['x'], position_dict[neighbor_id]['y']
                    
                    # Compute angle of line from (x1, y1) to (x2, y2)
                    angle = np.arctan2(y2 - y1, x2 - x1)  # in radians
                    # angle_deg = np.degrees(angle)      # if you want degrees
                    
                    angles.append(angle)  # or append angle_deg
                    allangles2.append(angle)
        angles = np.array(angles)


        # Set number of bins (e.g., 36 bins = 10° per bin)
        num_bins = 36

        # Create histogram
        counts, bin_edges = np.histogram(angles, bins=num_bins, range=(-np.pi, np.pi))

        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  
#%%        
        
#Plotting average of original


avgnumneigh = np.mean(numneigh)
# Compute standard deviation
std_dev = np.std(numneigh)
# Compute standard error
std_error = std_dev / np.sqrt(len(numneigh))



anglesall = np.array(allangles)


# Set number of bins (e.g., 36 bins = 10° per bin)
num_bins = 36

# Create histogram
counts, bin_edges = np.histogram(anglesall, bins=num_bins, range=(-np.pi, np.pi))

# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot as radial histogram (polar plot)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# Bars: plot as radii for each bin center
bars = ax.bar(bin_centers, counts, width=(2 * np.pi / num_bins), bottom=0.0, align='center', edgecolor='black', color='maroon')

# Optional formatting
ax.set_theta_zero_location('E')  # 0 degrees at the top
ax.set_theta_direction(-1)       # Clockwise
ax.set_xticklabels([])
#ax.set_ylim(0, 350)
plt.title("Average untrianed Direction of Bond Angles")
plt.tight_layout()

#plt.text(-0.7, 500, 
#         'N= ' + str(round(avgnumneigh, 2)) + ' ± ' + str(round(std_error, 2)), 
#         fontsize=25, ha='center', va='bottom')

plt.savefig(directory+'anisotropy in untrained bonds.png', dpi=50)  # You can change the filename and dpi as needed



plt.show()
            





#plotitng average of final
anglesall2 = np.array(allangles2)


avgnumneigh2 = np.mean(numneigh2)
# Compute standard deviation
std_dev2 = np.std(numneigh2)
# Compute standard error
std_error2 = std_dev2 / np.sqrt(len(numneigh2))


# Create histogram
counts, bin_edges = np.histogram(anglesall2, bins=num_bins, range=(-np.pi, np.pi))

# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot as radial histogram (polar plot)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# Bars: plot as radii for each bin center
bars = ax.bar(bin_centers, counts, width=(2 * np.pi / num_bins), bottom=0.0, align='center',edgecolor='black', color='green')

# Optional formatting
ax.set_theta_zero_location('E')  # 0 degrees at the top
ax.set_theta_direction(-1)       # Clockwise
ax.set_xticklabels([])

#ax.set_ylim(0, 350)

plt.title("Average Trained Direction of Bond Angles")
plt.tight_layout()

#plt.text(-0.7, 500, 
#         'N= ' + str(round(avgnumneigh2, 2)) + ' ± ' + str(round(std_error2, 2)), 
#         fontsize=25, ha='center', va='bottom')

plt.savefig(directory+'anisotropy in trained bonds.png', dpi=500)
plt.show()

