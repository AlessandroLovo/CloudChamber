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

color_list = ['black','yellow','green']

def color(n):
    if type(n) == int:
        return color_list[n]
    elif type(n) == np.ndarray:
        r = []
        for nn in n:
            r.append(color_list[nn])
        return np.array(r)
    elif type(n) == list:
        r = []
        for nn in n:
            r.append(color_list[nn])
        return r
    else:
        raise TypeError


def load_Clusterizer(path,name):
    filename = path + name + '.npy'
    c = Clusterizer(name,[],True)
    c.path = path
    c.slim_particles = np.load(filename)
    c.values = np.stack(c.slim_particles['values'])
    c.labels = np.zeros(len(c.values),dtype=int)
    if os.path.exists(path+name+'_labels.npy'):
        c.labels = np.load(path+name+'_labels.npy')
    return c

def join_Clusterizers(c_list,name):
    c = Clusterizer(name,[],True)
    c.slim_particles = np.concatenate([c1.slim_particles for c1 in c_list])
    c.labels = np.concatenate([c1.labels for c1 in c_list])
    c.values = np.stack(c.slim_particles['values'])
    return c

def plot_Clusterizers(c_list,color_list):
    def graph(c_l,color_l,key_x,key_y):
        fig, ax = plt.subplots()
        ax.set_xlabel(key_x)
        ax.set_ylabel(key_y)
        for i,c in enumerate(c_l):
            ax = plt.scatter(c.values[:,Particle.what_index(key_x)],
                            c.values[:,Particle.what_index(key_y)],marker='o',alpha=0.5,color=color_l[i],label=c.name)
        ax = plt.legend()
    
    ipywidgets.interact(graph, c_l = ipywidgets.fixed(c_list), color_l = ipywidgets.fixed(color_list),
                        key_x = Particle.keys, key_y = Particle.keys)
    
def alpha_selector(v,thrs=[0,50,15,10,2]):
    if v[0] < thrs[0]:  # persistence
        return False
    if v[1] > thrs[1]:  # lenght
        return False
    if v[2] < thrs[2]:  # thickness
        return False
    if v[3] > thrs[3]:  # n_components
        return False
    if v[4] > thrs[4]:  # curvature
        return False
    return True
    
def sub_Clusterizer(clusterizer_in,newname,func=alpha_selector):
    particles = []
    for i,v in enumerate(clusterizer_in.values):
        if func(v):
            particles.append(clusterizer_in.slim_particles[i])
    c = Clusterizer(newname,particles,False)
    return c

class Clusterizer:
    
    def __init__(self,name,particles,slim):
        
        self.name = name
        
        self.path = './'
        
        self.old_index = 0
        self.old_j = 0
        self.same_frame = []
        
        
        
        if len(particles) != 0:
            if slim:
                self.slim_particles = np.array(particles,dtype=Particle.slim_particle_dtype)
            else:
                slim_particles = [p.slim() for p in particles]
                self.slim_particles = np.array(slim_particles,dtype=Particle.slim_particle_dtype)
    
            self.values = np.stack(self.slim_particles['values'])
            self.labels = np.zeros(len(self.values),dtype=int)
        else:
            self.slim_particles = []
            self.values = []
            self.labels = []
            
    def save(self,path='',name=''):
        if len(self.slim_particles) == 0:
            print(self.name+' is empty: not saving')
            return
        if name == '':
            name = self.name
        if path == '':
            path = self.path
        np.save(path+name,self.slim_particles)
        np.save(path+name+'_labels',self.labels)
    
    def add_particles(self,particles,slim):
        if slim:
            slim_particles = particles
        else:
            slim_particles = [p.slim() for p in particles]
        slim_particles = np.array(slim_particles,dtype=Particle.slim_particle_dtype)
        
        self.slim_particles = np.concatenate(self.slim_particles,slim_particles)
        
        self.values = np.stack(self.slim_particles['values'])
        
    
    def plot_simple(self,key_x,key_y,colour='black'):
        plt.figure()
        plt.title(self.name)
        plt.xlabel(key_x)
        plt.ylabel(key_y)
        plt.scatter(self.values[:,Particle.what_index(key_x)],
                            self.values[:,Particle.what_index(key_y)],marker='o',color=colour)
        plt.show()
        
        
    
    def plot(self):
        '''
        Before running this function type
        
            %matplotlib notebook
            
        Use the lists to select axis
        
        Click near a data point to select it:
            press 'f' to view its pre-processed Frame picture
            press 't' to view its Trace analysis (takes a bit of time)
            press 'n' to go to the Next trace of that particle
            press 'r' to Recall the selected particle after you changed the axis
            press 'e' to view its frame from the vidEo
            press 'a' to highlight All components in the same video frame of the selected particle
            press 'w' to close (Waste) all figures windows 
            press 'b' to change the laBel of the selected particle
            press 's' to Save the capacitor
        '''
        
        def graph(c,key_x,key_y):
            global ax
            fig, ax = plt.subplots()
            ax.set_title(c.name)
            ax.set_xlabel(key_x)
            if key_x != key_y:
                ax.set_ylabel(key_y)
                ax = plt.scatter(c.values[:,Particle.what_index(key_x)],
                                c.values[:,Particle.what_index(key_y)],marker='o',color=color(c.labels))
            
            else:
                ax.set_ylabel('counts')
                ax = plt.hist(c.values[:,Particle.what_index(key_x)],bins=30,histtype='step')
            ax = plt.suptitle('')
        
            def onclick(event):
                if key_x != key_y:
                    ix = event.xdata
                    iy = event.ydata
                    
                    #ax = plt.scatter(ix,iy,color='orange')
                    c.old_j = 0
                    if c.old_index in c.same_frame:
                        ax = plt.scatter(c.values[c.old_index,Particle.what_index(key_x)],
                                c.values[c.old_index,Particle.what_index(key_y)],marker='o',color='red')
                    else:
                        ax = plt.scatter(c.values[c.old_index,Particle.what_index(key_x)],
                                c.values[c.old_index,Particle.what_index(key_y)],marker='o',color=color(c.labels)[c.old_index])
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
                if key_x != key_y:
                    path = c.slim_particles[c.old_index][1]
                    filename = c.slim_particles[c.old_index][2][c.old_j]
                    if not os.path.exists(path+filename):
                        # try going to next segment
                        prefix,dot,suffix = path.partition('/trigger') # .../video7_000, /trigger, _thr0.005_cl9_op3/cc_filtered_2den0.81_gausrad2.0/
                        next_segment_ID = int(prefix[-3:]) + 1
                        path = prefix[:-3]+('%03d' % next_segment_ID)+dot+suffix
                    #os.system('killall eog')
                    
                    if event.key == 'n':    # next trace of this particle
                        c.old_j = (c.old_j + 1) % len(c.slim_particles[c.old_index][2])
                        ax = plt.suptitle('particle %d: frame %d/%d' % (c.slim_particles[c.old_index][0],
                                                                    c.old_j + 1, len(c.slim_particles[c.old_index][2])))
                        
                    elif event.key == 'r':  # recall last particle when changing view
                        ax = plt.suptitle('particle %d: frame %d/%d' % (c.slim_particles[c.old_index][0],
                                                                    c.old_j + 1, len(c.slim_particles[c.old_index][2])))
                        ax = plt.scatter(c.values[c.old_index,Particle.what_index(key_x)],
                                c.values[c.old_index,Particle.what_index(key_y)],marker='o',color='orange')
                        #os.system('eog '+path+filename+' &')
                    elif event.key == 'f': # show frame
                        os.system('eog '+path+filename+' &')
                    elif event.key == 't': # show trace analysis
                        t = Trace(path,filename)
                        t.compute_estimators()
                        t.scatter_trace('temp.png')
                        os.system('eog temp.png &')
                    elif event.key == 'a':
                        ax = plt.scatter(c.values[:,Particle.what_index(key_x)],
                                c.values[:,Particle.what_index(key_y)],marker='o',color=color(c.labels))
                        prefix1,dot,suffix1 = filename.partition('_')   # mean, _, 280519-video7_000-009_opened_cc10.png
                        prefix2,dot,suffix2 = suffix1.partition('_')    # 280519-video7, _, 000-009_opened_cc10.png
                        prefix3,dot,suffix3 = suffix2.partition('_')    # 000-009, _, opened_cc10.png
                        beginnung = prefix1+dot+prefix2+dot+prefix3+dot # mean_280519-video7_000-009_
                        for i,p in enumerate(c.slim_particles):
                            for name in p[2]:
                                if name.startswith(beginnung):
                                    c.same_frame.append(i)
                        for i in c.same_frame:
                            ax = plt.scatter(c.values[i,Particle.what_index(key_x)],
                                c.values[i,Particle.what_index(key_y)],marker='o',color='red')
                    elif event.key == 'e':
                        raw_path = '/media/alessandro/DATA/tesi/Nebbia/raw_data/'
                        prefix,dot,suffix = path.partition('/trigger') # .../video7_000, /trigger, _thr0.005_cl9_op3/cc_filtered_2den0.81_gausrad2.0/
                        pattume,dot,suffix2 = prefix.partition('pre-processed_data/') # .../, pre-processed_data/, 280519/video7_000
                        add_path = suffix2[:-4]+'/frames/'     # 280519/video7/frames/
                        prefix1,dot,suffix3 = filename.partition('-')   # mean_280519, -, video7_000-009_opened_cc10.png
                        prefix2,dot,data = prefix1.partition('_')       # mean, _, 280519
                        segment,dot,suffix4 = suffix3.partition('-')    # video7_000, -, 009_opened_cc10.png
                        frame,dot,suffix5 = suffix4.partition('_')      # 009, _, opened_cc10.png
                        frame_name = 'outvid-'+data+'-'+segment+'-'+frame+'.png'
                        os.system('eog '+raw_path+add_path+frame_name+' &')                    
                    elif event.key == 'w':
                        ax = plt.suptitle('')
                        os.system('killall eog')
                        ax = plt.scatter(c.values[:,Particle.what_index(key_x)],
                                c.values[:,Particle.what_index(key_y)],marker='o',color=color(c.labels))
                        c.same_frame = []
                    elif event.key == 'b':
                        c.labels[c.old_index] = (c.labels[c.old_index] + 1) % len(color_list)
                        ax = plt.scatter(c.values[:,Particle.what_index(key_x)],
                                c.values[:,Particle.what_index(key_y)],marker='o',color=color(c.labels))
                    elif event.key == 's':
                        c.save()
                    
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            cid2 = fig.canvas.mpl_connect('key_press_event', onpress)
        
        
        ipywidgets.interact(graph, c = ipywidgets.fixed(self), key_x = Particle.keys, key_y = Particle.keys)
        
        
        
        
        
        
        
        
        
        
        
        