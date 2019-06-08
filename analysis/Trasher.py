# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 23:34:37 2019

@author: ale57
"""

from Trace import Trace

from shutil import copyfile
import os

  
class Trasher():
    
    def __init__(self, raw_path,junk_path,interesting_path,copy_junk = True):
        self.raw_path = raw_path
        self.junk_path = junk_path
        self.interesting_path = interesting_path
        self.copy_junk = copy_junk
        self.junk_count = 0
        self.interesting_count = 0
        
        self.white_perc_thr = 3
        self.eccentricity_thr = 30
    
    
    def move_file(self, filename,good):
        if good:
            copyfile(self.raw_path + filename,self.interesting_path + filename)
            print(filename, ' is interesting')
        elif self.copy_junk:
            copyfile(self.raw_path + filename,self.junk_path + filename)
            print(filename, ' is junk')
    
    def is_good(self, filename):
        t = Trace(self.raw_path,filename)
        
        if t.white_perc > self.white_perc_thr:
            print('too crowded: ',t.white_perc)
            return False
        t.compute_inertia()
        if t.eccentricity_from_inertia < self.eccentricity_thr:
            print('too spherical ',t.eccentricity_from_inertia)
            return False
        
        self.interesting_count += 1
        return True
    
    def trash(self):
        total_count = 0
        self.interesting_count = 0
        for filename in os.listdir(self.raw_path):
            if not filename.endswith('.png'):
                continue
            good = self.is_good(filename)
            self.move_file(filename,good)
            total_count += 1
    
        self.junk_count = total_count - self.interesting_count
        
        print(self.junk_count,' junk frames')
        print(self.interesting_count,' interesting frames')
        
    def clear_directories(self):
        for filename in os.listdir(self.junk_path):
            if not filename.endswith('.png'):
                continue
            os.remove(self.junk_path + filename)
        for filename in os.listdir(self.interesting_path):
            if not filename.endswith('.png'):
                continue
            os.remove(self.interesting_path + filename)