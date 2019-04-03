import numpy as np
from PIL import Image
import os
import math
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
import scipy
from scipy import ndimage
from tqdm import tqdm


#TOTAL FRAMES: counting total frames
def TotalFrames(folder, verbose=False):
        
	n=0
	for file in os.listdir(folder):
                if (file.endswith('-1.png') or file.endswith('-2.png') or file.endswith('-3.png') or file.endswith('-4.png')):
                        continue
                n+=1
	print('Total number of images processed: '+str(n))
	return n


#MEAN AND VARIANCE MATRIX: normalized mean and variance matrix of frames pixel from a single video
#input: frames folder, frames matrix prototype, number of frames
def TotalMeanVar(folder_in, matrix_prototype, n):
        
    matrix_sum = np.zeros_like(matrix_prototype)
    matrix_sumsq = np.zeros_like(matrix_prototype)

    #loop on frames
    for file in os.listdir(folder_in):
        if not file.endswith('.png'):
            continue
        if (file.endswith('-1.png') or file.endswith('-2.png') or file.endswith('-3.png') or file.endswith('-4.png')):
            continue
        image = Image.open(str(folder_in)+file)
        matrix = np.asarray(image.convert('L'), dtype=np.int16)*1.
        matrix_sum = np.add(matrix_sum*1., matrix)
        matrix_sumsq = np.add(matrix_sumsq*1., np.square(matrix))
		
        #compute mean and variance
    matrix_var = np.subtract(matrix_sumsq*1./n, np.square(matrix_sum*1./n))/255.
    matrix_mean =  matrix_sum*1./n/255.
    
    return matrix_mean, matrix_var


#IMAGE SELECTION BY DISCRIMINATION: select relevant frames with discriminant bigger than threshold value
#input: image, mean matrix, variance matrix, threshold value
def ImageSelectingByDiscrimination(matrix, matrix_mean, matrix_var, thr, verbose=False):
	#computing some discriminant value
    matrix_d = np.square(np.subtract(matrix, matrix_mean))
    discriminant = sum(sum(matrix_d))*1./sum(sum(matrix_var))
    if verbose:
        print('Discriminant: '+str(discriminant))
    if discriminant > thr:
        return True
    else:
        return False
