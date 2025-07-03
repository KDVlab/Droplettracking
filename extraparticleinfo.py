# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 03:24:27 2025

@author: MemoryPC
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:47:14 2025

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
from pathlib import Path



def preprocess_img(frame,x0,xf,y0,yf):                                                                    #CHANGE#
    frame = frame[y0:yf,x0:xf]
    frame = unsharp_mask(frame, radius = 2, amount = 5)
    frame *= 255.0/frame.max()
    return frame



def create_subpixel_circular_mask(image, center, radius):
    """Create a binary mask for a circle with sub-pixel precision."""
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    mask = ((x - center[0])**2 + (y - center[1])**2) <= radius**2
    return mask.astype(np.uint8)


def find_largest_subpixel_circle(image, center, minrad):
    """
    Find the largest circle that fits at the specified center without touching
    any white regions, using sub-pixel adjustments to optimize the circle's placement.

    Args:
        image (numpy.ndarray): Binary image where 255 represents white (obstacles) and 0 is the background.
        center (tuple): Initial estimate of the circle's center (x, y).
        minrad (float): Minimum radius to start with.

    Returns:
        tuple: Largest valid circle's radius and adjusted center.
    """
    # Initialize variables
    radius = minrad
    max_radius = min(image.shape[0], image.shape[1]) / 2.0
    largest_radius = 0.0
    best_center = center
    step_size = 0.01  # Sub-pixel step size for center adjustment
    max_iterations = 10  # Limit iterations to prevent infinite loops
    
    while radius <= max_radius:
        # Create the mask for the current circle
        mask = create_subpixel_circular_mask(image, best_center, radius)

        # Apply the mask to the image
        masked_image = mask * image

        if np.sum(masked_image) > 0:  # Circle touches a white region
            
            # Attempt to adjust the center in sub-pixel steps
            found_valid_position = False
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue  # Skip no-movement
                    candidate_center = (best_center[0] + dx, best_center[1] + dy)
                    mask = create_subpixel_circular_mask(image, candidate_center, radius)
                    masked_image = mask * image
                    
                    if np.sum(masked_image) == 0:  # Found a valid position
                        best_center = candidate_center
                        found_valid_position = True
                        break
                if found_valid_position:
                    break

            if not found_valid_position:
                # No valid center found, stop growing the circle
                break

        # Update the largest valid radius and grow the circle
        largest_radius = radius-step_size
        radius += step_size

    return largest_radius, best_center



def plot_pixels_around_peak(image, peaks, peak_index, distance=5):
    """
    Plot the pixels around a random peak. The area is a square of side 21 (10 pixels in each direction)
    
    :param image: The image from which to extract the region.
    :param peaks: Array of detected peaks (coordinates).
    :param peak_index: The index of the random peak to consider.
    :param distance: Distance in pixels from the peak to extract the region (default is 10).
    """
    # Get the coordinates of the selected peak
    #print(peaks)
    #peak=peaks[peak_index]
    #peak_x = peak[1]  # First column (x)
    #peak_y = peak[0]
    particle_row = peaks[peaks['particle'] == peak_index]

    # Get the 'x' and 'y' values, and ensure they are integers
    peak_x = int(particle_row['x'].values[0])  # Get the first (and only) value for 'x' as an integer
    peak_y = int(particle_row['y'].values[0])  # Get the first (and only) value for 'y' as an integer



    # Calculate the bounds of the region around the peak
    y_min = max(0,int( peak_y - distance))
    y_max = min(image.shape[0],int( peak_y + distance + 1) ) # +1 because slice is exclusive on the upper bound
    x_min = max(0, int(peak_x - distance))
    x_max = min(image.shape[1], int(peak_x + distance + 1))

    # Extract the region of the image around the peak
    region = image[y_min:y_max, x_min:x_max]

    # Plot the region around the peak
    '''
    plt.imshow(region, cmap='gray')
    plt.title(f'Pixels Around Peak {peak_index} at ({peak_y}, {peak_x})')
    plt.axis('off')
    plt.show()
    '''
    return region


def fill_center_with_black(image, radius=6):
    """
    Fill the center of an image with a black circle.

    :param image: 2D NumPy array representing the image.
    :param radius: Radius of the circle to fill (default is radius 5).
    :return: Modified image with the center filled with a black circle.
    """
    # Calculate the center of the image
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2

    # Loop through the image and apply the black circle condition
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Calculate the distance from the center
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            # If the distance is less than the radius, fill with black (0 for grayscale)
            if distance <= radius:
                image[y, x] = 0  # Set pixel to black

    return image


def binarize_and_count(image, threshold):
    # Load the image
    

    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:  # Check if it's a color image (3 channels)
        grayscale_image = np.mean(image, axis=2)  # Convert to grayscale by averaging the color channels
    else:
        grayscale_image = image  # It's already grayscale

    # Binarize the image: 1 if pixel > threshold, else 0
    binary_image = grayscale_image > threshold

    # Count how many pixels are above the threshold (i.e., how many True values in the binary image)
    num_pixels_above_threshold = np.sum(binary_image)
    '''
    # Display the binarized image
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Binarized Image (Threshold = {threshold})')
    plt.axis('off')
    plt.show()
    '''
    
    # Return the count of pixels above the threshold
    return num_pixels_above_threshold,binary_image


def show_region(image, threshold, peaks,distance, ind):
    #print(ind)
    region = plot_pixels_around_peak(image, peaks, ind, distance) 
    num,binary = binarize_and_count(region, threshold)
    return region, binary

def inverse_2x2(matrix):
    """
    Finds the inverse of a 2x2 matrix.
    
    """
    
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    det = a * d - b * c
    if det != 0:
        inverse_matrix = np.array([[d, -b],
                                   [-c, a]]) / det
        return inverse_matrix
    else:
       
        return None

def distvector(ref,part2,df1):
    '''
    Returns the vect between the positions of the two particles you provided
    '''
    dx=df1[df1['particle']==part2]['x'].values-df1[df1['particle']==ref]['x'].values
    dy=df1[df1['particle']==part2]['y'].values-df1[df1['particle']==ref]['y'].values
    vec=np.array([dx,dy])
    return vec

def straintensor(early, late, ref,neighbours):
     '''
     Finds the strain tensor 'E' according to the equations on
     https://docs.lammps.org/fix_nonaffine_displacement.html
     '''
     X = np.zeros((2,2))
     Y = np.eye(2)
     skipped=0
     for n in neighbours:
         Rn_i=distvector(ref,n,early)
         Rn_f=distvector(ref,n,late)
         Xn = np.outer(Rn_f, Rn_i)
         #print(ref,n)
         Yn = np.outer(Rn_i, Rn_i)
         if Rn_i.size == 0 or Rn_f.size == 0:
            skipped=skipped+1
            #print('SKIPPED:'+str(n))
            continue
        
         X=X+Xn
         Y=Y+Yn
     
     #print('Skipped '+ str(skipped) + 'out of ' + str(len(neighbours)))   
     invY=inverse_2x2(Y)
     E=X@invY  
     
     return E
def findNN(points,rad):
	'''
	(Courtesy of Hamza)This code uses a KD tree to quickly find how many points within
	a given distance of each point neigbour points. Should use rad>droplet radius
	to account for any errors in finding th;e position
	returns the indices of the nearest neighbours as well as the number of nearest neighbours
	'''
    
	tree = KDTree(points, leaf_size=2)
	nn_indices = tree.query_radius(points, r=rad) #Find KD tree which also gives point indices
	numNN = np.array([len(i)-1 for i in nn_indices]) #Find the number of nearest neighbours per site
	#find fractions of nearest neighbours with given number of neighbours
	fractionNN = [None]*8
	for j in np.arange(0,8,1):
		fractionNN[j] = np.count_nonzero(numNN==j)/len(numNN)
	return nn_indices, numNN, fractionNN

def d2min(trajectories,early,late,region):
        
        preall = trajectories[trajectories['frame'] == early]
        pre = preall[['particle', 'x', 'y']]
        pre.reset_index(drop=True, inplace=True)

        postall = trajectories[trajectories['frame'] ==late]
        post = postall[['particle', 'x', 'y']]
        post.reset_index(drop=True, inplace=True)

        #This finds all of the particles which you have in both of the frames
        particlenums1 = pre.particle.unique()
        particlenums2 = post.particle.unique()
        particlenums = np.intersect1d(particlenums1, particlenums2)

        
        '''
        This will find all of the particles which are within our region of interest
        If you don't want to crop it, set these limits to be the edges of your image.
        relpartnums saves the particle numbers of all of these particles and relparts
        saves their indices in the early frame
        '''
        

        #THIS WILL BE UNNESSESSARY IF NOT CROPPING
        
        particlenums2 = post['particle'].unique()
      
        relpartnums=np.intersect1d( particlenums2, particlenums)
        
        
        relparts= []
        
        for value in relpartnums:
            indices = pre.index[pre['particle'] == value].tolist()
            relparts.extend(indices)

        '''
        The following finds the particles in the neighbourhood of each particle in the early
        frame. The neighbourhood is defined by the second arguement in 'findNN' (currently it's 28).
        then, fin finds just keeps all of that data for the particles in our region of interest. 
        '''                                                                        
        dfN = pre[['x', 'y']].copy()
        indices, numNN, fractionNN = findNN(dfN, region)
        df_fractionNN = pd.DataFrame(fractionNN, columns=['Fraction of NN'])
        # Label each row with the corresponding index
        df_fractionNN.index.name = 'Index'
        pre['neighbours']=indices
        Fin= pre[pre['particle'].isin(relpartnums)]

        
        '''
        Here's where the magic happens:folliwing the math from the link
        that I had earlier in this code, I calculate D2min and save it in the
        Fin dataframe for each particle. 
        '''
        Fin['D2min'] = ''   
        Fin['E'] = ''   
        for i in relpartnums:
            '''
            First I need to convert the indices of the relevant particle's 
            neighbourhood to their particle numbers
            '''
            nin = pre.loc[pre['particle'] == i, 'neighbours'].iloc[0]
            neighbours = []
            for n in nin:
                neighbour = pre.loc[n, 'particle']
                neighbours.append(neighbour)
            
            #Use the function I made to calculate the strain tensor for this particle
            E=straintensor(pre,post,i,neighbours)
            
            #Sum over it's neighbours and add up the non affine contributuon from each
            D2min=0
            for n in neighbours:
                Rfn=distvector(i,n,post)
                Rin=distvector(i,n,pre)
                ERi=E@Rin
                Rt=Rfn-ERi
                ncont=np.linalg.norm(Rt)
                #print(ncont)
                D2min=D2min+ncont
                #print(D2min)
            
            Fin.loc[Fin['particle'] == i, 'D2min'] = D2min
            Fin.loc[Fin['particle'] == i, 'E'] = np.array2string(E)
            #print(E)
        #Fin.to_csv(directory+run+ '/D2mininfo.csv') 
        #plt.scatter(Fin['x'], Fin['y'], c=Fin['D2min'], cmap='viridis')
        #plt.xlabel('x')
        #plt.ylabel('y')
        
        
        #plt.title('D2min')
        #plt.colorbar(label='D2min')
        #plt.savefig(directory+run+'/D2min')
        #plt.show()
        return Fin
 


def safe_read_csv(filepath, retries=3, delay=30):
    filepath = Path(filepath)  # ensure it's a Path object

    if not filepath.exists():
        raise FileNotFoundError(f"File does not exist: {filepath}")

    for attempt in range(retries):
        try:
            return pd.read_csv(filepath)
        except OSError as e:
            print(f"Error reading {filepath} (attempt {attempt + 1}): {e}")
            time.sleep(delay)

    # After final attempt, raise error to stop execution
    raise RuntimeError(f"Failed to read {filepath} after {retries} attempts.")


      
#%%    
'''This code will take trajectories that have already been found and do extra analysis on it to 
distinguish the large and small droplets and calculate their D2min over the run'''

prefix='*.tif'
directories=['G:/Experiments/dec 12/']

for directory in directories:
    
    runs2=[]
    y_values=[]
    x_values=[]
    ds=150
    i=300
    trainamp=150
    total_runs = 1
    runfinal=500
    depths=[1400]
    widths=[-200,0,100]
    #widths=[100]    
    dss=[75,100,125,150,175,200]
    dss=[200]
    dire='F:/Experiments/aug9/'
    for ds in dss:
        for depth in depths:
            for width in widths:
                #usually 30 (y), usually 650 (x):
                y=30+0.3833333*depth    
                x=640+0.3833333*-width
                
                
                
                for r in range (0,runfinal+1):

                        
                    folder_path = directory+ '/training at' + str(ds) + 'from' + str(depth) + ',' + str(width) + 'run' + str(r) + '/'
                    
                    # Check if the directory exists
                    if os.path.exists(folder_path):
                        # If the directory exists, append to the lists
                        
                        # Second format for 'runs' and 'runs2'
                        runs2.append(('training at-' + str(ds) + 'from' + str(depth) + ',' + str(width) + 'run' + str(r) + '/', r, depth,width,ds))
                        y_values.append(y)
                        x_values.append(x)
                    else:
                        # If directory doesn't exist, move on to the next width
                        continue
             
    #runs2=runs2[160:]    
    
    for run,r,depth,width,dis in runs2:    
        
        
        #for ru in range (0,runfinal+1):

             
            #run= '/training at-' + str(dis) + 'from' + str(depth) + ',' + str(width) + 'run' + str(ru) + '/'
            # Check if the directory exists
            if os.path.exists(directory+run):
                
                droplets=pd.read_csv(directory+run+'dropletsinfo.csv')
                
                
            start=0 
            xadd=300
            yadd=200
            
            
            ypos=30+0.3833333*depth    
            xpos=640+0.3833333*-width
            
            x0=int(xpos-xadd)
            xf=int(xpos+xadd)
            y0=int(ypos-yadd)
            yf=int(ypos+yadd)
            x0 = max(x0, 0)
            xf = max(xf, 0)
            y0 = max(y0, 0)
            yf = max(yf, 0)
            
            #print('starting' + run)
            image_sequence = pims.ImageSequence(os.path.join(directory, run, prefix))

            frames = [preprocess_img(frame,x0,xf,y0,yf) for frame in image_sequence]
            '''
            print(run)
            plt.imshow(frames[len(frames)-1])
            plt.show()
            '''
            
            #Prepares the particle information
            img_example=frames[len(frames)-1]
            drops=droplets[['particle', 'x', 'y']]
            drops['particle']=drops['particle'].astype(int)
            drops['x']=drops['x'].astype(int)
            drops['y']=drops['y'].astype(int)
            
            
            
            threshold_radius = 8.9  # Define a threshold radius for classification
            size = []  # To store the sizes of the circles
            big_circles = []  # To store big circles
            small_circles = []  # To store small circles

            # Assuming peaks is a list of peak positions, e.g., [(x1, y1), (x2, y2), ...]
            for particle in drops['particle']:  # Replace `peaks` with your actual list of peaks
                #ind = 2
                img, bina = show_region(img_example, 95 , drops, 15, particle)  # Adjust the `show_region` function to your case

                # Preprocessing (e.g., fill the center with black)
                newbina = fill_center_with_black(bina, 5)
                newbina = newbina.astype(np.uint8)

                # Find the largest circle that can fit without touching any white region
                largest_radius, center = find_largest_subpixel_circle(newbina, (img.shape[1] // 2, img.shape[0] // 2), 5)

                # Classify the circle based on the threshold radius
                if largest_radius > threshold_radius:
                    big_circles.append((center, largest_radius,particle))
                    drops.loc[drops['particle'] == particle, 'size'] = 10.6
                else:
                    small_circles.append((center, largest_radius,particle))
                    drops.loc[drops['particle'] == particle, 'size'] = 8.8

                size.append(largest_radius)

            # Create an image template to overlay circles over the original image
            image_template = img_example.copy()  # Make a copy of the original image

            # Plot big circles (in green) over the image_template
            for center, radius,index in big_circles:
                theta = np.linspace(0, 2 * np.pi, 100)
                particle_row = drops[drops['particle'] == index]

                # Get the 'x' and 'y' values, and ensure they are integers
                x = int(particle_row['x'].values[0])+ radius * np.cos(theta)  # Get the first (and only) value for 'x' as an integer
                y = int(particle_row['y'].values[0]) + radius * np.sin(theta) # Get the first (and only) value for 'y' as an integer


                #x = peaks[index,1] + radius * np.cos(theta)
                #y = peaks[index,0] + radius * np.sin(theta)
                points = np.array([np.vstack((x, y)).T], dtype=np.int32)
                cv2.polylines(image_template, points, isClosed=True, color=(0, 255, 0), thickness=1)  # Green for big circles

            # Plot small circles (in red) over the image_template
            for center, radius,index in small_circles:
                theta = np.linspace(0, 2 * np.pi, 100)
                
                particle_row = drops[drops['particle'] == index]

                # Get the 'x' and 'y' values, and ensure they are integers
                x = int(particle_row['x'].values[0])+ radius * np.cos(theta)  # Get the first (and only) value for 'x' as an integer
                y = int(particle_row['y'].values[0]) + radius * np.sin(theta) # Get the first (and only) value for 'y' as an integer
                points = np.array([np.vstack((x, y)).T], dtype=np.int32)
                cv2.polylines(image_template, points, isClosed=True, color=(255, 0, 0), thickness=1)  # Red for small circles




                
            '''
            plt.imshow(image_template, cmap='gray')
           
            plt.axis('off')
            plt.show()

            plt.imshow(frames[0], cmap='gray')

            
            plt.axis('off')
            plt.show()
            '''
            
            drops.to_csv(directory+run+'extrainfo.csv', index=False)
            
#%%            
            
            #This section does D2min
            
            '''This first part simple seperates out the relevant information from a larger dataframe
            which contains data from all of the runs. If you have trajectory data for just the relevant
            run, simply load that as df3'''
            for ru in range (0,runfinal+1):

                 
                folder_path = directory+ '/training at' + str(dis) + 'from' + str(depth) + ',' + str(width) + 'run' + str(ru) + '/'
                # Check if the directory exists
                if os.path.exists(folder_path):
                    runfin=ru
                    #print(ru)
            
            
            #This following code figures out how to crop the large dataset to get out just the data for 1 run
            runf='training at'  + str(dis) + 'from' +str(depth)+','+str(width)+ 'run' + str(runfin)+'/'
            filepath = directory + runf+ 'TrackingOilDropletsByFrame/oildroplettrajectories_data.csv'
            t = safe_read_csv(filepath)
            
            endframes=pd.read_csv(directory + runf+ 'TrackingOilDropletsByFrame/endframes.csv')
            index = endframes.index[endframes['amp'] == r].tolist()
            endindex = int(index[0])
            end=endframes.loc[endindex, 'endframe']
            index = t.index[t['frame'] == end].tolist()   
            tnew = t.loc[t.frame < end-1]
            indexi = endframes.index[endframes['amp'] == r - 1].tolist()

            # Set endi and start based on whether indexi has values
            if indexi:
                endindexi = int(indexi[0])
                endi = endframes.loc[endindexi, 'endframe']
                  # Set start to endi if indexi is found
            else:
                endi = 0
                
               
            
            df3=tnew.loc[tnew.frame > endi]
            
            
             
                
            end= df3['frame'].max()-3
            start= df3['frame'].min()
            half=int((end-start)/2)
            
            #Here, you can choose to look at the D2min between two desired points along the runs
            D=d2min(df3,start,end,25)

                
            drops = drops.merge(D[['particle', 'D2min']], on='particle', how='left')
            drops.to_csv(directory+run+'extrainfo.csv', index=False)
            print(str(run))



            
            
