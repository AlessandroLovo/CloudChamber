#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:46:02 2019

@author: alessandro
"""

import os
import sys
import time

from ConnectedComponents6_5 import ConnectedComponents

path = sys.argv[1]  # = ../../raw_data/280519/frames/
data=sys.argv[2]    # = 280519
video=sys.argv[3]   # video7

start = 0
end = 0
if len(sys.argv) == 6:
    start = int(sys.argv[4])
    end = int(sys.argv[5])

start_time = time.time()
signals = 0
total_cc = 0
segments = 0
for i in range(start,end):
    segment = video+('_%03d' %i)
    print('Analyzing '+segment)
    partial_signals, partial_cc = ConnectedComponents(path,data,segment,is_slim=True,verbose=False)
    
    signals += partial_signals
    total_cc += partial_cc
    segments += 1
    
stop_time = time.time()
delta_t = int(stop_time - start_time)
hs = int(delta_t/3600)
mins = int((delta_t - hs*3600)/60)
secs = int(delta_t - hs*3600 - mins*60)

print('\n\n'+str(segments)+' segments analized: ')
print('Total nuber of analyzed frames: '+str(signals))
print('Total number of Connected Components: '+str(total_cc))
print('Average time required for one segment to be analyzed: '+str(delta_t/segments))
print('Total time: {0} h {1} min {2} s'.format(hs,mins,secs))