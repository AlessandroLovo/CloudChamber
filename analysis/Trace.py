# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 23:46:57 2019

@author: ale57
"""

import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
from tqdm import tqdm

def lt(p,A,n):
    t = (p/4.0 - np.sqrt((p/4.0)**2 - n*A))/n
    l = A/t
    return l,t

class Trace():
    
    def __init__(self,folder,filename):
        
        # estimators
        self.lenght = -1.0
        self.extra_lenght = 0.
        self.linear_lenght = 0.
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
        self.density_matrix = np.zeros_like(self.matrix,dtype=float)
        
        self.perimeter_points = []
        self.components = []
        self.components_centroids = []
        self.components_extremals = []
        
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
        points.append([self.trace[0,0] + self.max_linear_lenght*self.direction_from_variance[0],
                       self.trace[0,1] + self.max_linear_lenght*self.direction_from_variance[1]])
        points = np.array(points)
        #print(points)
        plt.figure()
        plt.scatter(self.trace[:,0],self.trace[:,1],marker='+')
        plt.scatter(self.perimeter_points[:,0],self.perimeter_points[:,1],marker='^',color='red')
        plt.scatter(self.components_centroids[:,0],self.components_centroids[:,1],marker='o',color='green',s=500)
        plt.scatter(self.components_extremals[:,:,0].flatten(),self.components_extremals[:,:,1].flatten(),marker='o',color='orange',s=500)
        #plt.contour(self.density_matrix.T,levels=[0.25,0.5,0.75,1])
        #plt.plot(points[:,0],points[:,1])
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
        
    
        
    def compute_estimators(self, radius=4, thr=0.5, tolerance=6, max_components_distance=8, high_thr = 0.5):
        
        r = int(radius)
        perimeter = 0
        self.perimeter_points = []
        components_points = []
        area = 0
        for i,x in tqdm(enumerate(self.matrix)):
            for j,y in enumerate(x):
                q = 0
                count = 0
                #square kernel
                k_min = max(0,i - r)
                k_max = min(i + r + 1,self.matrix.shape[0])
                l_min = max(0,j - r)
                l_max = min(j + r + 1,self.matrix.shape[1])
                for k in range(k_min,k_max):
                    for l in range(l_min,l_max):
                        # restrict to inscribed circle
                        if (k - i)**2 + (l - j)**2 > r**2:
                            continue
                        count += 1
                        if self.matrix[k,l] > 0:
                            q += 1
                if q == 0:
                    continue
                int_thr = int(thr*count)
                int_high_thr = int(high_thr*count)
                
                if (q <= int_thr and q >= int_thr - tolerance):
                    perimeter += 1
                    self.perimeter_points.append([i,j])
                if q >= int_thr:
                    area += 1
                if q >= int_high_thr:
                    components_points.append([i,j])
                self.density_matrix[i,j] = q*1.0/count
        
        self.perimeter_points = np.array(self.perimeter_points)
    
        self.components = []
        self.components_centroids = []
        
        for p in components_points:
            found = False
            for i,t in enumerate(self.components):
                for p1 in t:
                    if p1 == p:
                        continue
                    if (p1[0] - p[0])**2 + (p1[1] - p[1])**2 < max_components_distance**2:
                        self.components[i].append([p[0],p[1]])
                        found = True
                        break
                if found:
                    break
            if not found:
                self.components.append([[p[0],p[1]]])
                
        # check components again
        if len(self.components) > 1:
            n_components = len(self.components) + 1
            while len(self.components) < n_components:
                n_components = len(self.components)
                print('Checking components')
                for i,c in enumerate(self.components):
                    for i1,c1 in enumerate(self.components):
                        found = False
                        if i >= i1:
                            continue
                        for p in c:
                            for p1 in c1:
                                if (p1[0] - p[0])**2 + (p1[1] - p[1])**2 < max_components_distance**2:
                                    print(i1,n_components)
                                    self.components.remove(self.components[i1])
                                    self.components[i] += c1
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                    
                    
        # compute missing distance pieces
        self.components_extremals = []
        for c in self.components:
            i_best = 0
            j_best = 0
            d = 0
            for i,p in enumerate(c):
                for j,p1 in enumerate(c):
                    if i <= j:
                        break
                    if (p1[0] - p[0])**2 + (p1[1] - p[1])**2 > d:
                        i_best = i
                        j_best = j
                        d = (p1[0] - p[0])**2 + (p1[1] - p[1])**2
            
            self.components_extremals.append([c[i_best],c[j_best]])
        
        self.components_extremals = np.array(self.components_extremals)

        self.extra_lenght = 0.
        if len(self.components_centroids) > 1:
            d1 = 0
            for i,c in enumerate(self.components_extremals):
                d = 10**9 # just a big number
                if i == len(self.components_extremals) - 1:
                    break
                for i1,c1 in enumerate(self.components_extremals):
                    if i >= i1:
                        continue
                    for p in c:
                        for p1 in c1:
                            if (p1[0] - p[0])**2 + (p1[1] - p[1])**2 < d:
                                d = (p1[0] - p[0])**2 + (p1[1] - p[1])**2
                            if (p1[0] - p[0])**2 + (p1[1] - p[1])**2 > d1:
                                d1 = (p1[0] - p[0])**2 + (p1[1] - p[1])**2
                print(d)
                self.extra_lenght += np.sqrt(d)
                
            self.linear_lenght = np.sqrt(d1)
        
        else:
            c = self.components_extremals[0]
            self.linear_lenght = np.sqrt((c[0][0] - c[1][0])**2 + (c[0][1] - c[1][1])**2)
        
        for i,c in enumerate(self.components):
            self.components[i] = np.array(c)
            xm = np.mean(self.components[i][:,0])
            ym = np.mean(self.components[i][:,1])
            self.components_centroids.append([xm,ym])
        
        self.components_centroids = np.array(self.components_centroids)
            
                
        
        self.lenght, self.thickness = lt(perimeter,area,len(self.components_centroids))
        self.lenght += self.extra_lenght
        
        self.curvature = self.lenght/self.linear_lenght
                
        return perimeter, area, len(self.components_centroids), self.lenght, self.thickness, self.extra_lenght, self.curvature
                
                            
                            
                            
                            
                            
                            
                            