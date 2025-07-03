# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:49:51 2024

@author: KDVLabFerro
"""






import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.filters import unsharp_mask
import pims
from PIL import Image, ImageDraw
import tifffile as tiff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from skimage.filters import unsharp_mask
from skimage.feature import match_template
from skimage.feature import peak_local_max
import pandas as pd
import pims
import trackpy as tp
from scipy.optimize import curve_fit
from derivative import dxdt
import multiprocessing
from numpy import diff
import imageio
import seaborn as sns
from sklearn.neighbors import KDTree 

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
     for n in neighbours:
         Rn_i=distvector(ref,n,early)
         Rn_f=distvector(ref,n,late)
         Xn = np.outer(Rn_f, Rn_i)
         print(ref,n)
         Yn = np.outer(Rn_i, Rn_i)
         if Rn_i.size == 0 or Rn_f.size == 0:
            print('SKIPPED:'+str(n))
            continue
        
         X=X+Xn
         Y=Y+Yn
     invY=inverse_2x2(Y)
     E=X@invY  
     return E
     
#Make this the directory where you keep the files containing your experiments
directory='D:/Experiments/apr5/'


'''
All of this below is how I load up all of my runs subsequently. 
Since I am doing this for a bunch of videos, I find it easiest to
make a list of all of my folder names and then cycle through. If you're
just doing it for a few videos, might be better to take this out and replace
it with: runs=['the list of your file names']

'''

runs=[]


depths=[800]
dss=[10,20,30,40,50,60,70,80]
total_runs = 1
vels=[10]


for vel in vels:
    for depth in depths:
        
        for i in range(1, total_runs + 1):
            for ds in dss:
               runs.append('2d=' +str(ds)+' Intruder_in_glass_' + str(vel) + 'from' + str(depth)+',0' + 'run' + str(i)+'/') 
'''              
depths=[1010]
dss=[20,40,60,80]
total_runs = 1
vels=[2,5,10,15,20,30]


for vel in vels:
    for depth in depths:
        
        for i in range(1, total_runs + 1):
            for ds in dss:
               runs.append('2d=' +str(ds)+' Intruder_in_glass_' + str(vel) + 'from' + str(depth)+',0' + 'run' + str(i)+'/') 

'''
                            
runs=['2 Intruder_in_glass_0.5from600,-1300run0/']                                   
for run in runs:
    '''
    This reads the file with all of the trajectories of my oil droplets
    be sure to change the directory it reads from if you save this data 
    somewhere different. It's just important that it has all of the droplet
    positions for all of the frames in your experiment
    '''                       
    trajectories = pd.read_csv(directory+run+ '/TrackingOilDropletsByFrame/oildroplettrajectories_data.csv')
    '''
    D2min is calculated between 2 frames, here I seperate out
    just the data for the first frame (pre) and the final frame (post), keeping 
    just the relevant columns and then I reset the index. Be sure to input what 
    frames you want to be the initial and final
    '''
    preall = trajectories[trajectories['frame'] == 18000]
    pre = preall[['particle', 'x', 'y']]
    pre.reset_index(drop=True, inplace=True)

    postall = trajectories[trajectories['frame'] == trajectories['frame'].max()-1]
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
    
    #For this exp: 800(150:550,450:850),1000(250:650,450:850),1200(300:700,450:850),1400(400:800,450:850),1600(500:900,450:850) 1800(550:950,450:850)

    #This will find all of the particles which are within our region of interest
    x_min = 450
    x_max = 850
    y_min = 150
    y_max = 550
    pre_filtered = pre[(pre['x'] >= x_min) & (pre['x'] <= x_max) & (pre['y'] >= y_min) & (pre['y'] <= y_max)]
    fil_parts1 = pre_filtered['particle'].unique()
    particlenums2 = post['particle'].unique()
    almpartnums = np.intersect1d(fil_parts1, particlenums2)
    relpartnums=np.intersect1d(almpartnums, particlenums)
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
    indices, numNN, fractionNN = findNN(dfN, 45)
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
            print(ncont)
            D2min=D2min+ncont
            print(D2min)
        
        Fin.loc[Fin['particle'] == i, 'D2min'] = D2min
        Fin.loc[Fin['particle'] == i, 'E'] = np.array2string(E)
        print(E)
    Fin.to_csv(directory+run+ '/D2mininfo.csv') 
    plt.scatter(Fin['x'], Fin['y'], c=Fin['D2min'], cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('D2min')
    plt.colorbar(label='D2min')
    #plt.savefig(directory+run+'/D2min')
    plt.show()

