# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 03:16:15 2025

@author: MemoryPC
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:58:12 2025

@author: KDVLabFerro
"""

        # -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:48:36 2023

@author: marzu
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

'''@pims.pipeline
def preprocess_img(frame):                                                                    #CHANGE#
    frame = frame[0:1200,0:1200]
    frame = unsharp_mask(frame, radius = 2, amount = 5)
    frame *= 255.0/frame.max()
    return frame'''

def FindDroplets(img):
   subpix = tp.locate(img, diameter=((11,11)) , percentile = 99, threshold = 200/255, separation = 14)
   peaks = subpix[['y', 'x']].values  # find our peaks
   return peaks

def process_frame(frame_idx, frame, template, thresh):
    result, coord = MatchTemplate(frame, template, thresh)
    subpixelcoords = Parabolic_Fit(coord, result)
    coord = np.column_stack((coord, len(coord) * [frame_idx]))
    subpixeldf = pd.DataFrame(data=subpixelcoords, columns=['y', 'x'])
    df1 = pd.DataFrame(data=coord, columns=['y1', 'x1', 'frame'])
    dfcombined = pd.concat([df1, subpixeldf], axis=1)
    return dfcombined

#This function does the same method as the FindDroplets function but will store the position information in a dataframe


def MatchTemplate(img, template, thresh):
   match = match_template(img, template, pad_input = True)
  # x,y  = ij[::-1]
   peaks = peak_local_max(match,min_distance=8,threshold_rel=thresh) # find our peaks
   return match, peaks

def Match2Template(img, template1, template2, thresh):
   map1 = match_template(img, template1, pad_input=True)
   map2 = match_template(img, template2, pad_input=True)
   match = map1+map2
  # x,y  = ij[::-1]
   peaks = peak_local_max(match,min_distance=10,threshold_rel=thresh) # find our peaks
   
   return match, peaks

#This function does the same method as the MatchTemplate function but will store the position information in a dataframe

def labelimg(num, img, template, thresh):
    global features
    features = pd.DataFrame()
    match, peaks = MatchTemplate(img, template, thresh)

    for i in range(len(peaks)):
        features = features.append({
            'y': peaks[:, 0][i],
            'x': peaks[:, 1][i],
            'frame': num,
        }, ignore_index=True)
    
    return features

def Parabola(data,a,x0,y0):
    x = data
    x0 = float(x0)
    y0 = float(y0)
    a = float(a)
    func = - a*((x - x0)**2) + y0
    return func.ravel()


@pims.pipeline
def preprocess_img(frame,x0,xf,y0,yf):                                                                    #CHANGE#
    frame = frame[y0:yf,x0:xf]
    frame = unsharp_mask(frame, radius = 2, amount = 5)
    frame *= 255.0/frame.max()
    return frame


def Parabolic_Fit(coords,match):
    """Takes discrete coordinates from template matching along with the intensity match, returns an array of the subpixel coordinates"""
    subpixelcoords = np.array([])
    xpeaks = np.array([])
    ypeaks = np.array([])
    for i in range(len(coords)):
        #Peak in the x-direction
        initial_guess=(0.1,coords[i][1],1.0)  
        x = np.arange(coords[i][1]-2,coords[i][1]+3)
        popt, pcov = curve_fit(Parabola,x,match[coords[i][0],x],p0=initial_guess)
        xpeaks = np.append(xpeaks,popt[1])
        #Peaks in y-direction
        initial_guess=(0.1,coords[i][0],1.0)  
        x = np.arange(coords[i][0]-2,coords[i][0]+3)
        popt, pcov = curve_fit(Parabola,x,match[x,coords[i][1]],p0=initial_guess)
        ypeaks = np.append(ypeaks,popt[1])
    subpixelcoords = np.stack((ypeaks,xpeaks),axis=-1)
    return subpixelcoords

plt.rc('image', cmap='gray')



directories=['F:/Experiments/july10/']
#directory = 'D:/Experiments/july 2nd/ff droplet in glass at 5DEG (5 microns-s) 07-02_1(2)/'

for directory in directories:


    prefix= '*.tif'




    #FILL VALUES


    #n is the smoothing number
    n=3
    #keep only 1 every 'r' frames
    r=5

#[600,900,1200,1500,1800
    '''
    runs=[]
    widths=[-1300]
    x=2600
    total_runs = 1
    total_pushes =4
    depths=[600,900,1200,1500,1800]
    for current_run in range(1, total_runs + 1):
        for depth in depths:
            for width in widths:
                 for i in range(0,total_pushes):
                             if ((current_run - 1) * total_pushes + i) % 2 == 0:
                                     runs.append('2 Intruder_in_glass_0.5from' + str(depth) + ',' + str(width) + 'run' + str((current_run - 1) * total_pushes + i) + '/')
    '''
    
    runs=[]  
    y_values=[]
    x_values=[]
    i=2
    x=100
    total_runs = 1
    total_pushes =15
    depths=[900,1200,1500,1800,2100]
    widths=[400]
    
    
    for depth in depths:
        for width in widths:
                #runs.append('2 Intruder_in_glass_'+str(0)+'from'+str(depth)+','+str(width)+'run'+str(i)+'/')
                for r in range (10,300,10):
                    y=20+0.3833333*depth    
                    x=640+0.3833333*-width

                    runs.append('2 Intruder_in_glass_-'+str(r)+'from'+str(depth)+','+str(width)+'run'+str(i)+'/')
                    runs.append('2 Intruder_in_glass_'+str(r)+'from'+str(depth)+','+str(width)+'run'+str(i)+'/')
                    y_values.append(y)
                    x_values.append(x)
                    y_values.append(y)
                    x_values.append(x)
                    
    runsinfo = pd.DataFrame({'Run': runs, 'Y': y_values,'X':x_values})
                    
                    
    #This is for leaving out the first few runs
    #runs = runs[80:]
    #This is for leaving out a specific run  (exc) which should have been there...
    #exc=188
    #runs = runs[:exc] + runs[exc + 1:]

    '''
    x_positions = [1000, 2000]
    x_velocities = [1, 5, 20]
    depths = [1000]
    total_runs = 2
    total_pushes = 40  # can be a max of about 60 with 50um push distance)
    runs=[]

    for current_run in range(1, total_runs + 1):
        for depth in depths:
           
            for i in range(total_pushes):
                runs.append('2 Intruder_in_glass_5from' + str(depth) + 'run' + str((current_run-1)*total_pushes+i) + '/')
    '''
    
    
    
        #%%
    for run in runs:
        
        thresh = 0.8
        start=0 
        xadd=300
        yadd=200
        
        filtered = runsinfo[runsinfo['Run'] == run]
        xpos=filtered.iloc[0,2]
        ypos=filtered.iloc[0,1]
        
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
        # Only load every 5th frame
        #frames = preprocess_img(image_sequence[::r])
        #for k in range (n):t
         #       imgplot.append(plt.imshow(frames[k]))
          #      plt.show()
            
        def createFolder(dire):
            if not os.path.exists(dire+'/Imagesub'):
                os.mkdir(dire+'/Imagesub')
        def createFolder3(dire):
            if not os.path.exists(dire+'/TrackingOilDropletsByFrame'):
                os.mkdir(dire+'/TrackingOilDropletsByFrame')
        
        
        createFolder3(directory + run)
        createFolder(directory+run)

        arr=[]
        arr2=[]

          
        template1 = imageio.imread(directory+'template1.tif')
        template2 = imageio.imread(directory+'template2.tif')
         
        start=0 
        end=len(frames)-1  
    
                                                   
                 #CHANGE (all the template matching)#
        plt.imshow(frames[3])
        #Take one image that has a clear view of a single droplet and crop the image to be a single droplet


        plt.show()
        

        
        img_example = frames[60]
        match, peaks = Match2Template(img_example,template1,template2, thresh)


        fig, ([ax1, ax2]) = plt.subplots(ncols=2, figsize=(8, 3))

        ax1.imshow(img_example)

        ax2.imshow(img_example)
        ax2.set_axis_off()
        ax2.set_title('image')

        ax2.plot(peaks[:,1], peaks[:,0], 'x', markeredgecolor='blue', markerfacecolor='none', markersize=3)

        plt.show()
        '''
        '''
        video_path = directory + run + '/imagesubtraction.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (frames[0].shape[1], frames[0].shape[0]))
        
        
        #%%
        #s=200
     
        for i in range(start,end-n,n):
                    
                    images2 = cv2.absdiff(frames[i], frames[i + n]) 
                    images2 = images2.astype(np.uint8)
                    images2 = np.clip(images2, 0, 255)
                    tiff.imsave(directory +run+ '/Imagesub/subtractedimage' + str(i) + '.tif', images2)
                    arr.append(images2)
                    #video_writer.write(images2)


                    #print('Done', i, 'of', len(frames))
                    
        
        #video_writer.release()
        #norm=(end-start)/n
        #totalsum = np.zeros_like(arr2[0])
        #for x in arr2:
        #    totalsum += x
        print('Done image sub')   

        #totalsum = np.log1p(totalsum)        
        #plt.savefig(directory+ 'ff path.png')  
        #plt.plot(totalsum) 
        #plt.savefig(directory+ 'ff path.png')  
        #imgplot=[]
        #imgplot.append(plt.imshow(totalsum))
        #plt.show()



        #The belpw code will show the  subtracted images
        #imgplot=[]
        #for k in range (len(arr)):
        #        imgplot.append(plt.imshow(arr2[k]))
        #        plt.show()

            



        #%%


        #%%



        #Remember to change directory, normalization factor, frame boundaries                                                                #CHANGE#

        '''
        #Perform template matching on all frames and store positions in a csv
        '''
        

        results = np.zeros((10**7, 3))
                                                                                #CHANGE#
            
        df2 = pd.DataFrame()
        for idx, frame in enumerate(frames[start:end]):
                    result, coord = Match2Template(frame,template1,template2, thresh)
                    subpixelcoords = Parabolic_Fit(coord,result)
                    coord = np.column_stack((coord,len(coord)*[idx]))
                    subpixeldf = pd.DataFrame(data = subpixelcoords,columns=['y','x'])
                    df1 = pd.DataFrame(data = coord, columns = ['y1','x1','frame'])
                    dfcombined = pd.concat([df1,subpixeldf],axis=1)
                    df2 = pd.concat([df2,dfcombined])
                    print('done position finding for frame:' + str(idx))
        print('done position finding')
        results = df2[['frame', 'y', 'x']]
        results.to_csv(directory+run+'position_data.csv')

  
        

        

        #%%
        '''
        #This code is designed to track droplet positions over time using the position information
        #already collected in the previous script
        '''
        t1 = tp.link_df(results, search_range = 8, adaptive_stop=1, adaptive_step=0.97, memory=10)
        t = tp.filter_stubs(t1,10)


        t['dx']= 0
        t['df']= 0
         
        t['dx'] = t.groupby('particle')['x'].diff(n)
        t['df'] = t.groupby('particle')['frame'].diff(n).fillna(0).astype(int)
        t['dx/df'] = t['dx'] / t['df']

        t['dy'] = t.groupby('particle')['y'].diff(n)
        t['df'] = t.groupby('particle')['frame'].diff(n).fillna(0).astype(int)
        t['dy/df'] = t['dy'] / t['df']
        print('dome velocity smoothing')
        t['dx'] = t['dx'].fillna(0)
        t['dy'] = t['dy'].fillna(0)
          
        


        trajectories = t

        trajectories.to_csv(directory+run+'/TrackingOilDropletsByFrame/oildroplettrajectories_data.csv')

        
        
#%%       
        
        #The below code will make a overlay of the trajectory information with the raw images
        #This is very slow and can get very busy if there are several droplets being tracked
        
        trajectories=pd.read_csv(directory+run+'/TrackingOilDropletsByFrame/oildroplettrajectories_data.csv')
        '''

        
            
        def plotTraj(traj, k, directory, frames):
            plt.ion()

            
            plt.xlim(50,1250)
            plt.ylim(150,400)
            # Plot trajectories
            trajectories_fig = tp.plot_traj(traj[traj.frame <= k], colorby='particle', cmap=mpl.cm.winter, superimpose=frames[k],plot_style={'linewidth':0.5})

            # Save the figure as an image

            trajectories_fig.figure.savefig(directory + '/TrackingOilDropletsByFrame/oiltrajectories' + str(k) + '.png', dpi=600)

            # Read the saved image
            image = cv2.imread(directory  + '/TrackingOilDropletsByFrame/oiltrajectories' + str(k) + '.png')
            image = image.astype(np.uint8)
            image = np.clip(image, 0, 255)
            
            # Close the plot to prevent too many figures open
            plt.close('all')
            
        
        
        '''
        
        

        
     
            #%%
        def gaussian(x, A, mu, sigma):
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

       
       
        def createFolder(dire):
            if not os.path.exists(dire+'/heatmap'):
                os.mkdir(dire+'/heatmap')
              
        createFolder(directory+run)
        
        df3 = pd.read_csv(directory+run + '/TrackingOilDropletsByFrame/oildroplettrajectories_data.csv')
        


            #%%
        def gaussian(x, A, mu, sigma):
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
        def fill_nan_with_avg(lst):
            filled_list = lst.copy()
            for i in range(len(lst)):
                if np.isnan(lst[i]):
                    left_idx = i - 1
                    right_idx = i + 1
                    
                    while left_idx >= 0 and np.isnan(lst[left_idx]):
                        left_idx -= 1
                    while right_idx < len(lst) and np.isnan(lst[right_idx]):
                        right_idx += 1
                    
                    if left_idx >= 0 and right_idx < len(lst):
                        avg = (lst[left_idx] + lst[right_idx]) / 2.0
                        filled_list[i] = avg
                    elif left_idx >= 0:
                        filled_list[i] = lst[left_idx]
                    elif right_idx < len(lst):
                        filled_list[i] = lst[right_idx]
            return filled_list

        def veltracking (df,b,a):
            
            droplets = df.loc[df.frame == a]
            frm2 = df.loc[df.frame == b]
            
            #This will pick out the 3 frames around the points of interest so we only deal with those
            early = df[(df['frame'] >= b - 1) & (df['frame'] <= b + 1)]
            late = df[(df['frame'] >= a - 1) & (df['frame'] <= a + 1)]

            xpos = []
            ypos = []
            ixpos = []
            iypos = []
            droplets['xdistance'] = 0
            droplets['ydistance'] = 0
            droplets['distance'] = 0
            
            #this will use frames around frame a and b to fill in missing data from surroundings
            
            particlenums1 = early.particle.unique()
            particlenums2 = late.particle.unique()
            particlenums = np.intersect1d(particlenums1, particlenums2)
            
            #new#
            missing = set(particlenums) - set(droplets.particle.unique()) 
            add1 = df.loc[(df.frame == a+1) & df.particle.isin(missing)]
            missing1 = set(missing) - set(add1.particle.unique())
            add2 = df.loc[(df.frame == a-1) & df.particle.isin(missing1)]
            droplets = pd.concat([droplets, add1, add2], ignore_index=True)

            
            missingf = set(particlenums) - set(frm2.particle.unique()) 
            addf1 = df.loc[(df.frame == b+1) & df.particle.isin(missingf)]
            missingf1 = set(missingf) - set(addf1.particle.unique())
            addf2 = df.loc[(df.frame == b-1) & df.particle.isin(missingf1)]
            frm2 = pd.concat([frm2, addf1, addf2], ignore_index=True)
      
            for i in particlenums:
                x_pos = droplets[(droplets.particle == i)].x.iat[0]
                y_pos = droplets[(droplets.particle == i)].y.iat[0]
                ix_pos = frm2[(frm2.particle == i) ].x.iat[0]
                iy_pos = frm2[(frm2.particle == i)].y.iat[0]
                xpos.append(x_pos)
                ypos.append(y_pos)
                ixpos.append(ix_pos)
                iypos.append(iy_pos)
                droplets.loc[droplets.particle == i, 'xdistance'] = x_pos - ix_pos
                droplets.loc[droplets.particle == i, 'ydistance'] = y_pos - iy_pos
            
                 
            #droplets['distance']= (droplets.xdistance**2+droplets.ydistance**2)
            
            droplets['distance']=np.sqrt(droplets['xdistance']**2+droplets['ydistance']**2)
            
            
            maximum = droplets['distance'].max()
            print(str(maximum))
            maximum=30
            
            #The below code will plot the droplets frame by frame if you want
            #plt.ylim([-0.2,6])
            #plt.xlim([0,1300])
            #plt.xlabel('Distance from intruder (pixels')
            #plt.ylabel('Velocity (pixels/frame')
            #plt.title('Velocity of oildroplets in glass  (5deg, 50um_s)')
            #plt.scatter(droplets.distance, np.sqrt(droplets.dy**2+droplets.dx**2), s= 5)
            #plt.savefig(directory+run+'/heatmap/velocitiesforframe')
            #plt.figure()

            dropletsf10 = droplets[droplets['distance'] > droplets['distance'].quantile(0.9)]
            dropletsf10.to_csv(directory+run+ 'top10percent.csv')
            cumulativespeedf = dropletsf10['distance'].sum()
            cumulativespeedf_df = pd.DataFrame({'Cumulative Speed': [cumulativespeedf]})
            cumulativespeedf_df.to_csv(directory+run+ 'normcumulativespeedf10.csv')


            
            morethan10=droplets[droplets.distance>10]

            morethan10.to_csv(directory+run+ 'dispmorethan10.csv') 

            
            cumulativespeed10 = morethan10['distance'].sum()
            cumulativespeed10_df = pd.DataFrame({'Cumulative Speed': [cumulativespeed10]})
            cumulativespeed10_df.to_csv(directory+run+ 'normcumulativespeed10.csv')
  
            fast20=droplets[droplets.distance>20]
            fast20.to_csv(directory+run+ 'fasterthan20.csv')
            cumulativespeed20 = fast20['distance'].sum()
            cumulativespeed20_df = pd.DataFrame({'Cumulative Speed': [cumulativespeed20]})
            cumulativespeed20_df.to_csv(directory+run+ 'normcumulativespeed20.csv')
            
  
            '''
            
            return droplets
            '''

        def savecontour2(df3,a, b):
            droplets = df3
            df3['vel'] = np.sqrt(df3['dx/df']**2 + df3['dy/df']**2)
            #df3.to_csv(directory + run + '/oildroplettrajectories_data.csv')

            df3 = df3[(df3['frame'] >= a) & (df3['frame'] <= b)]
            
            for i in df3.particle.unique():  # Iterate over unique particles
                pathlength = df3.loc[df3['particle'] == i, 'vel'].sum(skipna=True)
                pathx=df3.loc[df3['particle'] == i, 'dx/df'].sum(skipna=True)
                pathy=df3.loc[df3['particle'] == i, 'dy/df'].sum(skipna=True)
                df3.loc[df3['particle'] == i, 'pathlength'] = pathlength
            frm2 = df3.loc[df3.frame == b]
            frm2.to_csv(directory + run + '/pathlengths.csv')
            print(frm2)

            # Filtering and saving based on pathlength thresholds
            thresholds = [20,10,5,3,2,1,0.5]
            for threshold in thresholds:
                fast_threshold = frm2[frm2['pathlength'] > threshold]
                fast_threshold.to_csv(directory + run + f'FFMOTIONpathlengthmorethan{threshold}.csv')
                cumulative_pathlength = fast_threshold['pathlength'].sum()
                cumulative_pathlength_df = pd.DataFrame({'Cumulative Vel': [cumulative_pathlength]})
                cumulative_pathlength_df.to_csv(directory + run + f'/FFMOTIONcpathlength{threshold}.csv')
            '''
            # Plotting
            plt.scatter(fast_threshold['particle'], fast_threshold['pathlength'], label=f'Path Length > {threshold}')
            plt.xlabel('Particle Number')
            plt.ylabel('Path Length')
            plt.title('Particle Number vs. Path Length')
            plt.legend()
            plt.show()
            '''
            return frm2 

            #%%
            

        savecontour2(df3,start+n,end-n)
    
        droplets= veltracking(df3,start+n,end-n)   
    