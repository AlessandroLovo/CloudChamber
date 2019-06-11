#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:59:36 2019

@author: alessandro
"""

import Particle

import numpy as np
import matplotlib.pyplot as plt

class Clusterizer:
    
    def __init__(self,name,particles,slim):
        
        self.name = name
        
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
    
    def add_particles(self,particles,slim):
        if slim:
            slim_particles = particles
        else:
            slim_particles = [p.slim() for p in particles]
        slim_particles = np.array(slim_particles,dtype=Particle.slim_particle_dtype)
        
        self.slim_particles = np.concatenate(self.slim_particles,slim_particles)
        
        self.values = np.stack(self.slim_particles['values'])
        
    
    def plot(self,key_x,key_y):
        plt.figure()
        plt.title(self.name)
        plt.xlabel(key_x)
        plt.ylabel(key_y)
        plt.scatter(self.values[:,Particle.what_index(key_x)],self.values[:,Particle.what_index(key_y)],marker='o',color='black')
        
        plt.show()