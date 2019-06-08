# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 23:46:57 2019

@author: ale57
"""

import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt

class Trace():
    
    def __init__(self,folder,filename):
        
        # estimators
        self.lenght = -1.0
        self.thickness = -1.0
        self.density = -1.0
        self.curvature = -1.0
        
        # e.g filename = mean_280519-video7_000-007_opened_cc02.png
        prefix, dot, name = str(filename).partition('-') # = mean_280519, -, video7_000-007_opened_cc02.png
        segment, dot, suffix1 = name.partition('-') # = video7_000, -, 007_opened_cc02.png
        frame, dot, suffix2 = suffix1.partition('_') # = 007, _, opened_cc02.png
        
        self.frame_name = segment + '-' + frame
        self.filename = filename
        
        
        
        image = Image.open(str(folder)+str(filename))
        self.matrix = np.asarray(image.convert('L'))
        
        xdim,ydim = self.matrix.shape
        
        trace = []
        pixel_count = 0
        first_time = True
        
        for i,x in enumerate(self.matrix):
            for j,y in enumerate(x):
                if(y > 0):
                    if first_time:
                        first_y = y
                        first_time = False
                    if first_y != y:
                        print(filename,': non saturated frame', i,j,y)
                    trace += [[i,j]]
                    pixel_count += 1
        
        self.trace = np.array(trace)
        self.white_number = pixel_count
        self.white_perc = pixel_count*100/xdim/ydim
        self.max_linear_lenght = np.sqrt((np.max(self.trace[:,0] - np.min(self.trace[:,0])))**2 + (np.max(self.trace[:,1] - np.min(self.trace[:,1])))**2)
        self.centroid = [0,0]
        self.xy_variance = [0,0]
        self.direction_from_variance = 0
        self.lenght_from_variance = 0
        self.inertia_tensor = [[0,0],[0,0]]
        self.inertia_eigenvalues = [0,0]
        self.inertia_eigenvectors = [[0,0],[0,0]]
        self.eccentricity_from_inertia = 0
        
    def scatter_trace(self):
        self.compute_variance()
        points = [[self.trace[0,0],self.trace[0,1]]]
        points.append([self.trace[0,0] + self.max_linear_lenght/2*self.direction_from_variance[0],
                       self.trace[0,1] + self.max_linear_lenght/2*self.direction_from_variance[1]])
        points = np.array(points)
        print(points)
        plt.figure()
        plt.scatter(self.trace[:,0],self.trace[:,1])
        plt.plot(points[:,0],points[:,1])
        plt.show()
        
    def compute_basics(self):
        self.compute_centroid()
        self.compute_variance()
        
    
    def compute_centroid(self):
        xm = np.mean(self.trace[:,0])
        ym = np.mean(self.trace[:,1])
        self.centroid[0] = xm
        self.centroid[1] = ym
    
    def compute_variance(self):
        xvar = np.var(self.trace[:,0])
        yvar = np.var(self.trace[:,1])
        self.lenght_from_variance = np.sqrt(xvar + yvar)
        self.xy_variance = [xvar,yvar]
        self.direction_from_variance = [np.sqrt(xvar/(xvar + yvar)),np.sqrt(yvar/(xvar + yvar))]
    
    def compute_inertia(self):
        if self.centroid == [0,0]:
            self.compute_centroid()
            
        Ixx = 0
        Iyy = 0
        Ixy = 0
        for point in self.trace:
            Ixx += np.power(point[0] - self.centroid[0],2)
            Iyy += np.power(point[1] - self.centroid[1],2)
            Ixy += (point[0] - self.centroid[0])*(point[1] - self.centroid[1])
        Ixx = float(Ixx)/self.white_number
        Iyy = float(Iyy)/self.white_number
        Ixy = float(Ixy)/self.white_number
        
        self.inertia_tensor = [[Ixx,Ixy],[Ixy,Iyy]]
        
        self.inertia_eigenvalues, self.inertia_eigenvectors = np.linalg.eig(self.inertia_tensor)
        self.eccentricity_from_inertia = np.max(self.inertia_eigenvalues)/np.min(self.inertia_eigenvalues)
        
    def compute_estimators(self):
        self.compute_inertia()
        

            