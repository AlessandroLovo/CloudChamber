# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:52:44 2019

@author: ale57
"""

import scipy
import sys
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
from shutil import copyfile

#filename = sys.argv[1]
#filename = 'example.png'
#matrix = scipy.misc.imread(filename)
def get_trace(filename):
    '''
    USAGE: trace,white_perc = get_trace(filename)
    
    trace      > np.array with the coordinates of white pixels
    white_perc > percentage of white pixels
    '''
    
    matrix = np.array(imageio.imread(filename))
    
    xdim,ydim = matrix.shape
    
    trace = []
    pixel_count = 0
    first_time = True
    
    for i,x in enumerate(matrix):
        for j,y in enumerate(x):
            if(y > 0):
                if first_time:
                    first_y = y
                    first_time = False
                if first_y != y:
                    print(filename,': non saturated frame', i,j,y)
                trace += [[i,j]]
                pixel_count += 1
                
    trace = np.array(trace)
    white_perc = pixel_count*100/xdim/ydim
    #plt.scatter(trace[:,0],trace[:,1], marker='+')

    return trace,white_perc

#def centroid(trace):
    

    