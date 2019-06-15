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
import time
from multiprocessing import Pool

slim_particle_dtype = [('ID',int),('path','U200'),('names',list),('values',np.ndarray)]
keys = ['persistence','lenght','thickness','n_components','curvature']

def what_index(key):
    for i,k in enumerate(keys):
        if k == key:
            return i
    print('index '+str(key)+' not found')
    return len(keys) # in order to raise an out of range error

class Particle:
    
    def __init__(self, path, ID, traces):
        self.path = path
        self.ID = ID
        self.traces = traces
        self.lenght = 0
        self.thickness = 0
        self.n_components = 0
        self.curvature = 0
        self.persistence = 1
        self.traces_names = []
        
        self.slim_particle = 0
        self.isready = False
        
    def Average(self):
        lenght = 0
        thickness = 0
        n_components = 0
        curvature = 0
        self.traces_names = []
        
        for t in self.traces:
            if t.lenght < 0:
                time = t.compute_estimators()
            if time < 0:
                self.traces.remove(t)
                continue
            lenght += t.lenght
            thickness += t.thickness
            n_components += t.n_components
            curvature += t.curvature
            self.traces_names.append(t.filename)
        
        if len(self.traces) == 0:
            # invalid particle
            return -1
        self.persistence = len(self.traces)
        self.lenght = lenght*1./len(self.traces)
        self.thickness = thickness*1./len(self.traces)
        self.n_components = n_components*1./len(self.traces)
        self.curvature = curvature*1./len(self.traces)
        
        self.slim_particle = (self.ID,self.path,self.traces_names,np.array([self.persistence,self.lenght,self.thickness,self.n_components,self.curvature]))
        
        self.isready = True
        
        return 0
        
    def slim(self):
        if not self.isready:
            r = self.Average()
        if r == 0:
            return self.slim_particle
        else:
            return []
        
        

        
def Join_Particles(particles=[],start_ID=1,folder='./',overlap_thr=0.5,autotrigger=True,eccentricity_thr=10,verbose=True):    
    
    ID = start_ID
    
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
            particles.append(Particle(folder,ID,[t]))
            ID += 1
            
    return particles, ID - start_ID

def Big_iteration(particles=[],path='./',subdirectory='trigger_thr0.005_cl9_op3/cc_filtered_2den0.81_gausrad2.0/',
                  slim=True,overlap_thr=0.5,autotrigger=True,eccentricity_thr=10,verbose=True,video_name=''):
    
    start_time = time.time()
    if len(particles) == 0:
        ID = 1
    else:
        ID = particles[0].ID + 1
    
    for folder in os.listdir(path):
        subfolder = folder+'/'+subdirectory
        if not os.path.exists(path+subfolder):
            if verbose:
                print(subfolder+' not found')
            continue
        if verbose:
            print('found '+subfolder)
        if video_name != '':
            video, dot, segment = folder.partition('_')
            if video != video_name:
                continue
        
        p, n = Join_Particles(particles=[],start_ID=ID,folder=path+subfolder,overlap_thr=overlap_thr,
                              autotrigger=autotrigger,eccentricity_thr=eccentricity_thr,verbose=verbose)
        
        if n == 0:
            continue
        # check for overlap between consequent segments
        min_l = min(10,len(particles))
        for q in particles[-min_l:]:
            if len(q.traces_names) == 0:
                continue
            last_trace_name = q.traces[-1].filename
            prefix, dot, name = str(last_trace_name).partition('-') # = mean_280519, -, video7_000-007_opened_cc02.png
            segment, dot, suffix1 = name.partition('-') # = video7_000, -, 007_opened_cc02.png
            last_frame, dot, suffix2 = suffix1.partition('_') # = 007, _, opened_cc02.png
            if last_frame == '048':
                last_videoID,dot,last_segment_ID = segment.partition('_') # video7, _, 000
                min_m = min(10,len(p) - 1)
                for r in p[:min_m]:
                    if len(r.traces) == 0:
                        continue
                    first_trace_name = r.traces[0].filename
                    prefix, dot, name = str(first_trace_name).partition('-') # = mean_280519, -, video7_000-007_opened_cc02.png
                    segment, dot, suffix1 = name.partition('-') # = video7_000, -, 007_opened_cc02.png
                    first_frame, dot, suffix2 = suffix1.partition('_') # = 007, _, opened_cc02.png
                    if first_frame == '001':
                        first_videoID,dot,first_segment_ID = segment.partition('_') # video7, _, 000
                        if (first_videoID == last_videoID and int(first_segment_ID) == int(last_segment_ID) + 1):
                            last_t = q.traces[-1]
                            first_t = r.traces[0]
                            diff_matrix = np.abs(last_t.matrix - first_t.matrix)
                            
                            # compute white_number for diff_matrix
                            n_whites = 0
                            for x in diff_matrix:
                                for y in x:
                                    if y > 0:
                                        n_whites += 1
                            
                            overlap = 1.0 - n_whites/(last_t.white_number + first_t.white_number)
                            if verbose:
                                print('Possible joint over segments detected: overlap = '+str(overlap))
                            if overlap >= overlap_thr:
                                if verbose:
                                    print('\n joining particles at the beginning of '+segment)
                                n -= 1
                                p.remove(r)
                                q.traces += r.traces
        
        particles += p
        ID += n
    
    if slim:
        if verbose:
            print(('\nSlimming %d particles' % ID))
        pool = Pool(6) # 6 threads
        slim_particles = pool.map(Particle.slim,particles)
        # remove invalid particles
        for h in slim_particles:
            if len(h) == 0:
                slim_particles.remove(h)
                ID -= 1
        
        particles = slim_particles
        
        
        
    end_time = time.time()
    delta_t = int(end_time - start_time)
    hs = int(delta_t/3600)
    mins = int((delta_t - hs*3600)/60)
    secs = int(delta_t - hs*3600 - mins*60)
    print('Total time: {0} h {1} min {2} s'.format(hs,mins,secs))
    
    return particles, ID - 1, slim
        
            
        