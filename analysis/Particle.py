#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:18:06 2019

@author: alessandro
"""

from Trace import Trace
from Trasher import Trigger

import os
from tqdm import tqdm
import numpy as np

class Particle:
    
    def __init__(self, ID, traces):
        self.ID = ID
        self.traces = traces
        self.lenght = 0
        self.thickness = 0
        self.n_components = 0
        self.curvature = 0
        self.persistence = 1
        self.traces_names = []
        self.isready = False
        
    def Average(self):
        lenght = 0
        thickness = 0
        n_components = 0
        curvature = 0
        self.traces_names = []
        
        for t in self.traces:
            if t.lenght < 0:
                t.compute_estimators()
            lenght += t.lenght
            thickness += t.thickness
            n_components += t.n_components
            curvature += t.curvature
            self.traces_names.append(t.filename)
        
        self.persistence = len(self.traces)
        self.lenght = lenght*1./len(self.traces)
        self.thickness = thickness*1./len(self.traces)
        self.n_components = n_components*1./len(self.traces)
        self.curvature = curvature*1./len(self.traces)
        
        self.isready = True
        
    def values(self):
        if not self.isready:
            self.Average()
        return [self.lenght,self.thickness,self.n_components,self.curvature,self.persistence,self.traces_names]
        
        

        
def Join_Particles(folder,overlap_thr=0.5,autotrigger=True,eccentricity_thr=10,verbose=True):
    ID = 1
    particles = []
    
    for filename in tqdm(os.listdir(folder)):
        
        if not filename.startswith('mean_'):
            continue
        # e.g filename = mean_280519-video7_000-007_opened_cc02.png
        prefix, dot, name = str(filename).partition('-') # = mean_280519, -, video7_000-007_opened_cc02.png
        segment, dot, suffix1 = name.partition('-') # = video7_000, -, 007_opened_cc02.png
        frame, dot, suffix2 = suffix1.partition('_') # = 007, _, opened_cc02.png
        frameID = int(frame)
        previous_name = segment+('-%03d' % (frameID - 1))
        
        t = Trace(folder,filename)
        if autotrigger:
            if not Trigger(t,eccentricity_thr=eccentricity_thr,verbose=verbose):
                continue
        
        found = False
        for p in particles:
            if p.traces[-1].frame_name == previous_name:
                diff_matrix = np.abs(p.traces[-1].matrix - t.matrix)
                
                # compute white_number for diff_matrix
                n_whites = 0
                for x in diff_matrix:
                    for y in x:
                        if y > 0:
                            n_whites += 1
                
                overlap = 1.0 - n_whites/(p.traces[-1].white_number + t.white_number)
                if verbose:
                    print('\n'+previous_name+'-'+t.frame_name+': overlap = '+str(overlap))
                if overlap >= overlap_thr:
                    found = True
                    p.traces.append(t)
                    break
        
        if not found:
            particles.append(Particle(ID,[t]))
            ID += 1
            
    return particles, ID - 1
            
        