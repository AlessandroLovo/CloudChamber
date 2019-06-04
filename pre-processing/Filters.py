import numpy as np
from PIL import Image
import os
import math
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
import scipy
from scipy import ndimage

#denoising 1:                                                                                                                                   
def Denoising1(inputpath, inputfile, outputpath, outputfile, denoising1, verbose=False):
    if verbose:
        print('Denoising 1\n')
    image_in = cv2.imread(str(inputpath)+inputfile)
    image_out = cv2.fastNlMeansDenoising(image_in, None, 30.0, 21, denoising1)
    image_out = ndimage.uniform_filter(image_out, size=denoising1)
    cv2.imwrite(str(outputpath)+outputfile, image_out)

#dilation:
def Dilation(inputpath, inputfile, outputpath, outputfile, dilation, verbose=False):
    if verbose:
        print('Dilation \n')
    image_in = cv2.imread(str(inputpath)+inputfile)
    kernel = np.ones((dilation,dilation),np.uint8)
    image_out = cv2.dilate(image_in,kernel,iterations = 1)
    cv2.imwrite(str(outputpath)+outputfile, image_out)

#subtract background:
def BackgroundSubtraction(inputpath, inputfile, inputbkg, outputpath, outputfile, verbose=False):
    if verbose:
        print('Subtracting background\n')
    image_in = Image.open(str(inputpath)+inputfile)
    image_bkg = Image.open(str(inputpath)+inputbkg)
    matrix_bkg = np.asarray(image_bkg.convert('L'))
    matrix_in = np.asarray(image_in.convert('L'))
    matrix_sub = cv2.subtract(matrix_in, matrix_bkg)
    image_sub = Image.fromarray(np.uint8(matrix_sub))
    image_sub.save(str(outputpath)+outputfile)

#denoising 2:                                                                                                                                   
def Denoising2(inputpath, inputfile, outputpath, outputfile, denoising2, verbose=False):
    if verbose:
        print('Denoising 2\n')
    image_in = Image.open(str(inputpath)+inputfile)
    matrix_in = np.asarray(image_in.convert('L'))/255.
    matrix_in_reshaped4 = matrix_in.reshape(1, matrix_in.shape[0], matrix_in.shape[1], 1)
    if verbose:
        print('Image loaded \nImage dimensions: '+str(matrix_in.shape))
    matrix_out_reshaped4 = (matrix_in_reshaped4 > denoising2).astype(int) #binary
    matrix_out = matrix_out_reshaped4.reshape(matrix_out_reshaped4.shape[1], matrix_out_reshaped4.shape[2])
    image_out = Image.fromarray(np.uint8(matrix_out)*255)
    image_out.save(str(outputpath)+outputfile)


#reducing resolution (with cnn):                                                                                                                                     
def ReducingResolution(inputpath, inputfile, outputpath, outputfile, max_pool, verbose=False):
    if verbose:
        print('Reducing Resolution \n')
    image_in = Image.open(str(inputpath)+inputfile)
    matrix_in = np.asarray(image_in.convert('L'))/255.
    matrix_in_reshaped4 = matrix_in.reshape(1, matrix_in.shape[0], matrix_in.shape[1], 1)
    model=Sequential()
    model.add(MaxPooling2D(input_shape=(matrix_in.shape[0], matrix_in.shape[1], 1), pool_size=(max_pool,max_pool)))
    matrix_pooled_reshaped4 = model.predict(matrix_in_reshaped4, batch_size=1, verbose=1)
    matrix_pooled = matrix_pooled_reshaped4.reshape(matrix_pooled_reshaped4.shape[1], matrix_pooled_reshaped4.shape[2])
    image_pooled = Image.fromarray(np.uint8(matrix_pooled)*255)
    image_pooled.save(str(outputpath)+outputfile)
    if verbose:
        print('Image dimension after pooling: '+str(matrix_pooled.shape))
        print('Number of white pixels after pooling: '+str(sum(sum(matrix_pooled))))
	

#labeling:
def Labeling(inputpath, inputfile, outputpath, outputfile, gauss_radius, cc_thr, verbose=False):
    if verbose:
        print('Labeling \n')
    file,dot,extension = inputfile.partition('.')
    image_in = scipy.misc.imread(str(inputpath)+inputfile, flatten=True) # gray-scale image
    image_out = ndimage.gaussian_filter(image_in, gauss_radius)

    s = ndimage.generate_binary_structure(2,2)
    labeled, nr_objects = ndimage.label(image_out, structure=s)
    labels = np.unique(labeled)
    labeled = np.searchsorted(labels, labeled)
    if verbose:
        print("Number of objects is %d " % nr_objects)
    cv2.imwrite(str(outputpath)+outputfile, labeled)

    #projecting on the trace:
    total_number_cc=0
    for i in range(nr_objects):
        i+=1
        projector=(labeled==i).astype(int)
        cc_size=sum(sum(projector))
        if cc_size<cc_thr:
            continue
        total_number_cc+=1
        if verbose:
            print('Dimension of the connencted component '+str(i)+': '+str(sum(sum(projector))))
        matrix=projector*image_in #image_out to have gray-scale
        cv2.imwrite(str(outputpath)+file+'_cc'+str(i)+'.png', matrix)
        if verbose:
            print('Total number of selected connected components: '+str(total_number_cc))
    return total_number_cc
