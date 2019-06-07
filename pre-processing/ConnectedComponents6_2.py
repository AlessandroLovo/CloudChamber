import numpy as np
from PIL import Image
import os
import sys
#import math
#import cv2
import time
import matplotlib as mpl
mpl.use('Agg')
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
#import scipy
#from scipy import ndimage
import Filters
import TriggerFunctions
from tqdm import tqdm

path = sys.argv[1]
data=sys.argv[2]
video=sys.argv[3]

is_slim = True # save only final results

if not path.endswith('/'):
    path += '/'

#parameters
'''
denoising1_bkg = 5
dilation_bkg = 6
denoising1 = 10
denoising2= 0.81
denoising2_sub= 0.0078
max_pool = 2
gauss_radius= 2.0
cc_thr= 3000
discriminant_thr=0.004
'''

nebbia_path = '/home/gaiag/'
local_path = '/media/alessandro/DATA/tesi/Nebbia/pre-processed_data/'

def ConnectedComponents(path, data, video, is_slim, outpath='local_path', discriminant_thr=0.004, denoising1_bkg = 5, dilation_bkg = 6,
                        denoising1 = 10, denoising2= 0.81, denoising2_sub= 0.0078, max_pool = 2, gauss_radius= 2.0, cc_thr= 3000):
    
    run=data+'-'+video+'-'
    
    output_path=nebbia_path+str(data)+'/'+str(video)+'/trigger_thr'+str(discriminant_thr)+'/'
    if outpath == 'local_path':
        output_path=local_path+str(data)+'/'+str(video)+'/trigger_thr'+str(discriminant_thr)+'/'
    input_folder = path#+'frames/'
    output_folder_mean = output_path+'means_filtered_2den'+str(denoising2)+'_gausrad'+str(gauss_radius)+'/'
    output_folder_cc = output_path+'cc_filtered_2den'+str(denoising2)+'_gausrad'+str(gauss_radius)+'/'
    if not os.path.exists(output_folder_mean):
            os.makedirs(output_folder_mean)
    if not os.path.exists(output_folder_cc):
            os.makedirs(output_folder_cc)
    
    #log_book
    log_path = str(output_path)+'log_book'+'_means_filtered_2den'+str(denoising2)+'_gausrad'+str(gauss_radius)+'.txt'
    
    total_number_cc = 0
    signals = 0
    start = time.time()
    
    #log_text=open(log_path, "w")
    #log_text.write("Copmuting total number of frames:\n")
    #log_text.close()
    n_frames = TriggerFunctions.TotalFrames(input_folder, 'outvid-'+run)
    image_prototype=Image.open(str(input_folder)+'outvid-'+run+'001.png')
    matrix_prototype=np.asarray(image_prototype.convert('L'))
    print("Number of frames: "+str(n_frames)+'\n')
    
    log_text=open(log_path, "w")
    log_text.write('video ID' +str(run)+'\n')
    log_text.write('denoising1_bkg '+str(denoising1_bkg)+'\n')
    log_text.write('dilation_bkg '+str(dilation_bkg)+'\n')
    log_text.write('denoising1 '+str(denoising1)+'\n')
    log_text.write('denoising2 '+str(denoising2)+'\n')
    log_text.write('denoising2_sub '+str(denoising2_sub)+'\n')
    log_text.write('max_pool '+str(max_pool)+'\n')
    log_text.write('gauss_radius '+str(gauss_radius)+'\n')
    log_text.write('cc_thr '+str(cc_thr)+'\n')
    log_text.write('discriminant_thr '+str(discriminant_thr)+'\n')
    log_text.write("Number of frames: "+str(n_frames)+'\n')
    log_text.close()
    
    #normalized mean and variance matrices
    matrix_mean, matrix_var = TriggerFunctions.TotalMeanVar(input_folder, matrix_prototype,'outvid-'+run,  n_frames)
    
    if not is_slim:
        plt.matshow(matrix_var)
        plt.colorbar()
        plt.savefig(str(output_path)+'var_all.png')  
    
        plt.matshow(matrix_mean)
        plt.colorbar()
        plt.savefig(str(output_path)+'mean_all.png')
        
    #prepare mean
    n_backgrounds=0
    start_partial=time.time()
    output_folder = output_path+'pooled/'
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    print('Summing up bkg frames')
    for file in tqdm(os.listdir(input_folder)):
        if (file.startswith('outvid-'+run)):
            if file.endswith('-001.png'):
                continue
            file_name, dot, extension = file.partition('.')
            file_name1, dot, file_name2 = file_name.partition('-')
            file_name3, dot, file_name4 = file_name2.partition('-')
            file_name5, dot, file_name6 = file_name4.partition('-')
            '''
            number=int(file_name6)
            if(number%2):
                continue
            '''
            
            #background manipulation before subtracting
            Filters.Denoising1(input_folder, file, output_folder, 'bkg_denoising1.png', denoising1_bkg)
            Filters.Dilation(output_folder, 'bkg_denoising1.png', output_folder, 'bkg_dilationed.png', dilation_bkg)
            raw_pooled = Image.open(str(output_folder)+'bkg_dilationed.png')
            raw_matrix = np.asarray(raw_pooled.convert('L'))
            if n_backgrounds == 0:
                matrix_sum_pooled=np.zeros_like(raw_matrix)

            matrix_sum_pooled = np.add(matrix_sum_pooled*1., raw_matrix*1.)
            
            n_backgrounds += 1
    
    image_sum = Image.fromarray(np.uint8(matrix_sum_pooled*1.0/n_backgrounds))
    image_sum.save(str(output_folder)+'bkg_mean.png')
    n_backgrounds -= 1        
    
        
    #loop on frames
    for file_raw in tqdm(os.listdir(input_folder)):
        if not file_raw.endswith('.png'):
            continue
        if file_raw.endswith('-001.png'):
            continue
        raw_image = Image.open(str(input_folder)+file_raw)
        matrix_raw=np.asarray(raw_image.convert('L'))
        raw_file_name, dot, extension = file_raw.partition('.')
        raw_file_name1, dot, raw_file_name2 = raw_file_name.partition('-')
        raw_file_name3, dot, raw_file_name4 = raw_file_name2.partition('-')
        raw_file_name5, dot, raw_file_name6 = raw_file_name4.partition('-')
        if raw_file_name5 != video:
            continue
        #selecting via discriminant value
        if not TriggerFunctions.ImageSelectingByDiscrimination(matrix_raw/255., matrix_mean, matrix_var, discriminant_thr, verbose = True):
            continue
        print(raw_file_name5)
            
        signals+=1
        log_text=open(log_path, "a")
        log_text.write('Selected file number '+str(signals)+': '+str(file_raw)+'\n')
        log_text.close()
        print('Selected file number '+str(signals)+': '+str(file_raw)+'\n')
    
        start_partial = time.time()
        
        Filters.Denoising1(input_folder, file_raw, output_folder, 'raw_denoising1.png', denoising1)
        raw_pooled = Image.open(str(output_folder)+'raw_denoising1.png')
        frame_matrix = np.asarray(raw_pooled.convert('L'))
        Filters.Dilation(output_folder, 'raw_denoising1.png', output_folder, 'raw_dilationed.png', dilation_bkg)
        raw_dilationed = Image.open(str(output_folder)+'raw_dilationed.png')
        matrix_dilationed = np.asarray(raw_dilationed.convert('L'))
        if frame_matrix.shape != matrix_sum_pooled.shape or frame_matrix.shape != matrix_dilationed.shape:
            print('Error: shapes do not match')
            return
        
        #bkg subtraction
        subtracted_matrix = frame_matrix + (matrix_dilationed - matrix_sum_pooled)*1.0/n_backgrounds
        image_frame = Image.fromarray(np.uint8(subtracted_matrix))
        image_frame.save(str(output_folder)+'subtracted.png')
        Filters.Denoising2(output_folder, 'subtracted.png', output_folder, 'sub_denoising2.png', denoising2_sub)
        Filters.ReducingResolution(output_folder, 'sub_denoising2.png', output_folder, 'pool_'+str(file_raw), max_pool)
        Filters.Denoising2(output_folder, 'pool_'+str(file_raw), output_folder_mean,
                           'mean_'+str(raw_file_name2)+'_den2.png', denoising2)
        
        number_cc = Filters.Labeling(output_folder_mean, 'mean_'+str(raw_file_name2)+'_den2.png', output_folder_cc,
                                     'labeled_'+str(raw_file_name2)+'.png', gauss_radius, cc_thr, verbose=True)
            
        total_number_cc = total_number_cc + number_cc
        stop_partial=time.time()
        delta_partial = stop_partial-start_partial
        delta_integrated = stop_partial-start
        log_text=open(log_path, "a")
        log_text.write("Time required for one frame to be analyzed: "+str(delta_partial)+' sec \n'+
                           "Integrated time from the beginning: "+str(delta_integrated)+' sec \n')
        log_text.close()
        print("Time required for one frame to be analyzed: "+str(delta_partial)+' sec \n'+
                           "Integrated time from the beginning: "+str(delta_integrated)+' sec \n')
        
    if is_slim:
        os.system('rm -r '+output_folder_mean+'../pooled')
        os.system('rm -r '+output_folder_mean)
        os.system('rm '+output_folder_cc+'label*')
        
        
    stop=time.time()
    delta=stop-start
    log_text=open(log_path, "a")
    log_text.write("Total time for the execution: "+str(delta)+' sec \n'+
                   "Total number of frames triggered for the analysis: "+str(signals)+'\n'+
                   "Total number of cc collected in the video: "+str(total_number_cc)+'\n')
    log_text.close()
    print("Total time for the execution: "+str(delta)+' sec \n'+
          "Total number of frames triggered for the analysis: "+str(signals)+'\n'+ 
          "Total number of cc collected in the video: "+str(total_number_cc)+'\n')


if __name__ == '__main__':
    path = sys.argv[1]  #e.g. ../../raw_data/280519/frames/
    data=sys.argv[2]    #e.g. 280519
    video=sys.argv[3]   #e.g. video7_000
    
    is_slim = False # save only final results
    
    if not path.endswith('/'):
        path += '/'
    
    ConnectedComponents(path, data, video, is_slim)