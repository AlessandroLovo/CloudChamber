#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:01:54 2019

@author: alessandro
"""

# to be run on RaspberryPi


import os
import sys
import time
import numpy as np
from PIL import Image

import TriggerFunctions
import ExtractFrames


path = sys.argv[1]
data = sys.argv[2]
max_time = int(sys.argv[3])
video_time = 10 #[s]
discriminant_thr = 0.004
min_triggered_frames = 4

if not path.endswith('/'):
    path += '/'
output_folder = path+data+'/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

start_time = time.time()
partial_time = start_time


count = 0
while (partial_time - start_time) < max_time:
    triggered_frames  = 0
    video = 'video_%03d' % count
    #acquire video
    os.system('echo raspivid -md 4 -o {0}{1}.h264 -t {2}000 -ISO 600 -co 100 -br 80 -fps 24 -sa -20'.format(output_folder,video,video_time))
    os.system('raspivid -md 4 -o {0}{1}.h264 -t {2}000 -ISO 600 -co 100 -br 80 -fps 24 -sa -20'.format(output_folder,video,video_time))
    #extract frames
    ExtractFrames.ExtractFrames(folder=path,data=data,do_segment=False,framestep=10,only_one_video=True,video_name=video)
    
    
    input_folder = output_folder+'frames/'
    run = data+'-'+video+'-'
    n_frames = TriggerFunctions.TotalFrames(input_folder, 'outvid-'+run)
    image_prototype=Image.open(str(input_folder)+'outvid-'+run+'001.png')
    matrix_prototype=np.asarray(image_prototype.convert('L'))
    print('\n\n'+video+": number of frames: "+str(n_frames))
    matrix_mean, matrix_var = TriggerFunctions.TotalMeanVar(input_folder, matrix_prototype,'outvid-'+run,  n_frames)
    
    #loop on frames
    for file_raw in os.listdir(input_folder):
        if not file_raw.endswith('.png'):
            continue
        if file_raw.endswith('-001.png'):
            continue
        raw_image = Image.open(str(input_folder)+file_raw)
        matrix_raw=np.asarray(raw_image.convert('L'))
        if not TriggerFunctions.ImageSelectingByDiscrimination(matrix_raw/255., matrix_mean, matrix_var, discriminant_thr, verbose = False):
                continue
        else:
            triggered_frames += 1
            
    #clean up
    os.system('rm -r {0}'.format(input_folder))
    os.system('rm {0}{1}.mp4'.format(output_folder,video))
    
    if triggered_frames < min_triggered_frames:
        os.system('rm {0}{1}.h264'.format(output_folder,video))
    
    t = partial_time    
    partial_time = time.time()
    process_time = partial_time - t
    print('Triggered frames: {0}'.format(triggered_frames))
    print('Time required for processing one segment : {0}'.format(process_time))
    
    count += 1