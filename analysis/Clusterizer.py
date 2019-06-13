#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:59:36 2019

@author: alessandro
"""

import Particle
from Trace import Trace

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
import os



def load_Clusterizer(name):
    filename = name + '.npy'
    c = Clusterizer(name,[],True)
    c.slim_particles = np.load(filename)
    c.values = np.stack(c.slim_particles['values'])
    return c

def Join_Clusterizers(c_list,name):
    c = Clusterizer(name,[],True)
    c.slim_particles = np.concatenate([c1.slim_particles for c1 in c_list])
    c.values = np.stack(c.slim_particles['values'])
    return c

class Clusterizer:
    
    def __init__(self,name,particles,slim):
        
        self.name = name
        
        self.old_index = 0
        self.old_j = 0
        
        
        if len(particles) != 0:
            if slim:
                self.slim_particles = np.array(particles,dtype=Particle.slim_particle_dtype)
            else:
                slim_particles = [p.slim() for p in particles]
                self.slim_particles = np.array(slim_particles,dtype=Particle.slim_particle_dtype)
    
            self.values = np.stack(self.slim_particles['values'])
        else:
            self.slim_particles = []
            self.values = []
            
    def save(self,name=''):
        if len(self.slim_particles) == 0:
            print(self.name+' is empty: not saving')
            return
        if name == '':
            name = self.name
        np.save(name,self.slim_particles)
    
    def add_particles(self,particles,slim):
        if slim:
            slim_particles = particles
        else:
            slim_particles = [p.slim() for p in particles]
        slim_particles = np.array(slim_particles,dtype=Particle.slim_particle_dtype)
        
        self.slim_particles = np.concatenate(self.slim_particles,slim_particles)
        
        self.values = np.stack(self.slim_particles['values'])
        
    
    def plot_simple(self,key_x,key_y):
        plt.figure()
        plt.title(self.name)
        plt.xlabel(key_x)
        plt.ylabel(key_y)
        plt.scatter(self.values[:,Particle.what_index(key_x)],
                            self.values[:,Particle.what_index(key_y)],marker='o',color='black')
        plt.show()
        
        
    
    def plot(self):
        '''
        Before running this function type
        
            %matplotlib notebook
            
        Use the lists to select axis
        
        Click near a data point to select it:
            press 'f' to view its frame picture
            press 't' to view its trace analysis (takes a bit of time)
            press 'n' to go to the next trace of that particle
            press 'w' to close all figures windows 
        '''
        
        def graph(c,key_x,key_y):
            global ax
            fig, ax = plt.subplots()
            ax.set_title(c.name)
            ax.set_xlabel(key_x)
            ax.set_ylabel(key_y)
            
            #x = np.arange(0,10)
            #y = x**2
            #ax.scatter(x,y)
            ax = plt.suptitle('')
            ax = plt.scatter(c.values[:,Particle.what_index(key_x)],
                            c.values[:,Particle.what_index(key_y)],marker='o',color='black')
        
        
        
            def onclick(event):
                ix = event.xdata
                iy = event.ydata
                
                #ax = plt.scatter(ix,iy,color='orange')
                c.old_j = 0
                ax = plt.scatter(c.values[c.old_index,Particle.what_index(key_x)],
                            c.values[c.old_index,Particle.what_index(key_y)],marker='o',color='black')
                d = 100.0
                for i,p in enumerate(c.values):
                    d1 = (ix - p[Particle.what_index(key_x)])**2 + (iy - p[Particle.what_index(key_y)])**2
                    if d > d1:
                        c.old_index = i
                        d = d1
                
                ax = plt.suptitle('particle %d: frame %d/%d' % (c.slim_particles[c.old_index][0],
                                                                c.old_j + 1, len(c.slim_particles[c.old_index][2])))
                ax = plt.scatter(c.values[c.old_index,Particle.what_index(key_x)],
                            c.values[c.old_index,Particle.what_index(key_y)],marker='o',color='orange')
                
            def onpress(event):
                path = c.slim_particles[c.old_index][1]
                filename = c.slim_particles[c.old_index][2][c.old_j]
                #os.system('killall eog')
                
                if event.key == 'n':    # next trace of this particle
                    c.old_j = (c.old_j + 1) % len(c.slim_particles[c.old_index][2])
                    ax = plt.suptitle('particle %d: frame %d/%d' % (c.slim_particles[c.old_index][0],
                                                                c.old_j + 1, len(c.slim_particles[c.old_index][2])))
                    os.system('eog '+path+filename+' &')
                elif event.key == 'f': # show frame
                    os.system('eog '+path+filename+' &')
                elif event.key == 't': # show trace analysis
                    t = Trace(path,filename)
                    t.compute_estimators()
                    t.scatter_trace('temp.png')
                    os.system('eog temp.png &')
                elif event.key == 'w':
                    ax = plt.suptitle('')
                    os.system('killall eog')
                    ax = plt.scatter(c.values[c.old_index,Particle.what_index(key_x)],
                            c.values[c.old_index,Particle.what_index(key_y)],marker='o',color='black')
                    
                
                
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            cid2 = fig.canvas.mpl_connect('key_press_event', onpress)
        
        
        ipywidgets.interact(graph, c = ipywidgets.fixed(self), key_x = Particle.keys, key_y = Particle.keys)
        
        
        
        
        
        
        
        
        
        
        
        